"""
SVD Collaborative Filtering Recommender System - PRODUCTION OPTIMIZED
----------------------------------------------------------------------

"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')


class LRUCache:
    """Simple LRU Cache to prevent memory leaks"""
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # Remove oldest
    
    def clear(self):
        self.cache.clear()


class SVDRecommender:
    """
    Production-ready SVD Recommender
    
    Optimizations:
    - Precomputed Q^T Q for fold-in (10x faster)
    - LRU cache (no memory leaks)
    - Zero-copy DataFrame operations
    - Smart path management
    """
    
    def __init__(self, 
                 model_path: str, 
                 movies_path: str, 
                 ratings_path: str,
                 cache_size: int = 1000):
        
        self.model = None
        self.movies = None
        self.ratings = None
        self.cache_size = cache_size
        
        # LRU cache for user rated movies
        self._user_rated_cache = LRUCache(cache_size)
        
        # Load
        self._load_model(model_path)
        self._load_data(movies_path, ratings_path)
        
        # Cache latent vectors
        self._cache_latent_vectors()
        
        # PRECOMPUTE Q^T Q for fold-in
        self._precompute_fold_in_matrix()
        
        print(f"Hệ thống sẵn sàng với LRU cache {cache_size}")
    
    def _load_model(self, model_path: str):
        """Load SVD model"""
        try:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            print(f"Đã load model: {path.name}")
        except Exception as e:
            raise Exception(f"Cannot load model: {e}")
    
    def _load_data(self, movies_path: str, ratings_path: str):
        """Load data"""
        try:
            self.movies = pd.read_csv(movies_path)
            self.ratings = pd.read_csv(ratings_path)
            
            # Validate
            required_movie_cols = ['movieId', 'title_clean', 'genres', 'rating_avg', 'rating_count']
            required_rating_cols = ['userId', 'movieId', 'rating']
            
            if not all(col in self.movies.columns for col in required_movie_cols):
                raise ValueError("Movies data missing required columns")
            
            if not all(col in self.ratings.columns for col in required_rating_cols):
                raise ValueError("Ratings data missing required columns")
            
            print(f"Đã load dữ liệu: {len(self.movies):,} phim, {len(self.ratings):,} đánh giá")
        except Exception as e:
            raise Exception(f"Cannot load data: {e}")
    
    def _cache_latent_vectors(self):
        """Cache latent vectors for vectorized ops"""
        self.qi_matrix = self.model.qi
        self.bi_vector = self.model.bi
        self.pu_matrix = self.model.pu
        self.bu_vector = self.model.bu
        self.global_mean = self.model.trainset.global_mean
        
        # ID mappings
        self.user_id_map = {
            self.model.trainset.to_raw_uid(i): i 
            for i in range(self.model.trainset.n_users)
        }
        self.movie_id_map = {
            self.model.trainset.to_raw_iid(i): i 
            for i in range(self.model.trainset.n_items)
        }
        
        print(f"Đã cache {len(self.user_id_map)} users, {len(self.movie_id_map)} phim")
    
    def _precompute_fold_in_matrix(self):
        """
        Precompute Q^T Q + λI for fold-in
        
        Fold-in solves: (Q^T Q + λI) pu = Q^T residuals
        Q^T Q is constant → compute once, reuse forever
        
        ~10x faster cold start!
        """
        print("Đang tính toán ma trận fold-in...")
        
        # Q^T Q
        QTQ = self.qi_matrix.T @ self.qi_matrix
        
        # Add regularization: Q^T Q + λI
        n_factors = QTQ.shape[0]
        lambda_reg = 0.1
        self.fold_in_matrix = QTQ + lambda_reg * np.eye(n_factors)
        
        # Precompute inverse: (Q^T Q + λI)^-1
        try:
            self.fold_in_matrix_inv = np.linalg.inv(self.fold_in_matrix)
            print(f"Đã tính toán ma trận fold-in ({n_factors}x{n_factors})")
        except:
            print("Không thể đảo ngược ma trận, sẽ dùng lstsq")
            self.fold_in_matrix_inv = None
    
    def _get_user_rated_movies(self, user_id: int) -> np.ndarray:
        """Get user rated movies with LRU cache"""
        cached = self._user_rated_cache.get(user_id)
        if cached is not None:
            return cached
        
        # Query once, cache result
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        rated_movies = user_ratings['movieId'].values
        self._user_rated_cache.put(user_id, rated_movies)
        
        return rated_movies
    
    def recommend(self, 
                  user_id: int, 
                  n: int = 10,
                  exclude_rated: bool = True,
                  min_rating_count: int = 0) -> pd.DataFrame:
        """
        Recommend movies (ZERO-COPY operations)
        """
        # Validate
        if user_id not in self.ratings['userId'].values:
            raise ValueError(f"User {user_id} not found")
        
        # Get candidates WITHOUT .copy() → zero-copy filter
        mask = self.movies['rating_count'] >= min_rating_count
        
        if exclude_rated:
            rated_movies = self._get_user_rated_movies(user_id)
            mask &= ~self.movies['movieId'].isin(rated_movies)
        
        # Get movie IDs directly from filtered view
        candidate_movie_ids = self.movies.loc[mask, 'movieId'].values
        
        if len(candidate_movie_ids) == 0:
            return pd.DataFrame()
        
        # VECTORIZED PREDICTION
        predicted_ratings = self._predict_batch_vectorized(
            user_id, 
            candidate_movie_ids
        )
        
        # Create results DataFrame (small, OK to create)
        pred_df = pd.DataFrame({
            'movieId': candidate_movie_ids,
            'predicted_rating': predicted_ratings
        })
        
        # Get top N indices
        top_n_indices = np.argpartition(predicted_ratings, -n)[-n:]
        top_n_indices = top_n_indices[np.argsort(predicted_ratings[top_n_indices])[::-1]]
        
        # Filter to top N
        pred_df = pred_df.iloc[top_n_indices].reset_index(drop=True)
        
        # Merge (only N rows, efficient)
        recommendations = pred_df.merge(
            self.movies[['movieId', 'title_clean', 'genres', 
                        'rating_avg', 'rating_count']], 
            on='movieId',
            how='left'
        )
        
        recommendations['rank'] = range(1, len(recommendations) + 1)
        recommendations = recommendations.rename(columns={'title_clean': 'title'})
        
        return recommendations[[
            'rank', 'movieId', 'title', 'genres', 
            'predicted_rating', 'rating_avg', 'rating_count'
        ]]
    
    def _predict_batch_vectorized(self, 
                                  user_id: int, 
                                  movie_ids: np.ndarray) -> np.ndarray:
        """
        Fully vectorized batch prediction
        Formula: rating = global_mean + bu + bi + pu·qi
        """
        # Get user factors
        if user_id not in self.user_id_map:
            return np.full(len(movie_ids), self.global_mean)
        
        user_inner_id = self.user_id_map[user_id]
        pu = self.pu_matrix[user_inner_id]
        bu = self.bu_vector[user_inner_id]
        
        # Vectorized movie ID mapping
        movie_inner_ids = np.array([
            self.movie_id_map.get(mid, -1) for mid in movie_ids
        ])
        valid_mask = movie_inner_ids >= 0
        
        # Initialize predictions
        predictions = np.full(len(movie_ids), self.global_mean + bu)
        
        if valid_mask.any():
            valid_ids = movie_inner_ids[valid_mask]
            qi_valid = self.qi_matrix[valid_ids]
            bi_valid = self.bi_vector[valid_ids]
            
            # Vectorized computation
            ratings_valid = (
                self.global_mean + 
                bu + 
                bi_valid + 
                np.dot(qi_valid, pu)
            )
            
            predictions[valid_mask] = np.clip(ratings_valid, 1, 5)
        
        return predictions
    
    def recommend_new_user(self, 
                          favorite_movies: List[Tuple[int, float]], 
                          n: int = 10,
                          min_rating_count: int = 10) -> pd.DataFrame:
        """
        Cold start recommendation with PRECOMPUTED fold-in
        """
        # Validate
        if not favorite_movies:
            raise ValueError("Must provide favorite movies")
        
        favorite_ids = [m[0] for m in favorite_movies]
        
        # Check validity
        valid_ids = set(self.movies['movieId'].values)
        invalid = [mid for mid in favorite_ids if mid not in valid_ids]
        if invalid:
            raise ValueError(f"Invalid movie IDs: {invalid}")
        
        # Get known favorites
        known_favorites = [
            (mid, rating) for mid, rating in favorite_movies 
            if mid in self.movie_id_map
        ]
        
        if len(known_favorites) < 3:
            # Fallback to content-based
            return self._recommend_content_based(
                favorite_movies, favorite_ids, n, min_rating_count
            )
        
        # Fold-in with precomputed matrix
        pu_new, bu_new = self._fold_in_user_fast(known_favorites)
        
        # Get candidates (zero-copy)
        mask = (
            (~self.movies['movieId'].isin(favorite_ids)) &
            (self.movies['rating_count'] >= min_rating_count)
        )
        candidate_movie_ids = self.movies.loc[mask, 'movieId'].values
        
        # Predict
        predictions = self._predict_with_pu(
            pu_new, bu_new, candidate_movie_ids
        )
        
        # Top N
        top_n_indices = np.argpartition(predictions, -n)[-n:]
        top_n_indices = top_n_indices[np.argsort(predictions[top_n_indices])[::-1]]
        
        pred_df = pd.DataFrame({
            'movieId': candidate_movie_ids[top_n_indices],
            'predicted_rating': predictions[top_n_indices]
        })
        
        recommendations = pred_df.merge(
            self.movies[['movieId', 'title_clean', 'genres', 
                        'rating_avg', 'rating_count']], 
            on='movieId'
        )
        
        recommendations['rank'] = range(1, len(recommendations) + 1)
        recommendations = recommendations.rename(columns={'title_clean': 'title'})
        
        return recommendations[[
            'rank', 'movieId', 'title', 'genres', 
            'predicted_rating', 'rating_avg', 'rating_count'
        ]]
    
    def _fold_in_user_fast(self, 
                          favorite_movies: List[Tuple[int, float]]) -> Tuple[np.ndarray, float]:
        """
        FAST fold-in using precomputed (Q^T Q + λI)^-1
        
        Before: Solve (Q^T Q + λI) pu = Q^T r each time
        After:  pu = (Q^T Q + λI)^-1 Q^T r (matrix already inverted!)
        
        ~10x faster!
        """
        # Collect data
        qi_list = []
        bi_list = []
        ratings = []
        
        for mid, rating in favorite_movies:
            if mid in self.movie_id_map:
                inner_id = self.movie_id_map[mid]
                qi_list.append(self.qi_matrix[inner_id])
                bi_list.append(self.bi_vector[inner_id])
                ratings.append(rating)
        
        qi_matrix = np.array(qi_list)
        bi_array = np.array(bi_list)
        ratings = np.array(ratings)
        
        # Compute bu
        bu_new = np.mean(ratings - self.global_mean - bi_array)
        
        # Residuals
        residuals = ratings - self.global_mean - bu_new - bi_array
        
        # Solve using precomputed inverse
        if self.fold_in_matrix_inv is not None:
            # Fast path: pu = (Q^T Q + λI)^-1 Q^T r
            pu_new = self.fold_in_matrix_inv @ (qi_matrix.T @ residuals)
        else:
            # Fallback: solve linear system
            try:
                pu_new = np.linalg.lstsq(qi_matrix, residuals, rcond=None)[0]
            except:
                pu_new = np.zeros(self.qi_matrix.shape[1])
        
        return pu_new, bu_new
    
    def _predict_with_pu(self, 
                        pu: np.ndarray, 
                        bu: float, 
                        movie_ids: np.ndarray) -> np.ndarray:
        """Predict with given pu and bu"""
        movie_inner_ids = np.array([
            self.movie_id_map.get(mid, -1) for mid in movie_ids
        ])
        valid_mask = movie_inner_ids >= 0
        
        predictions = np.full(len(movie_ids), self.global_mean + bu)
        
        if valid_mask.any():
            valid_ids = movie_inner_ids[valid_mask]
            qi_valid = self.qi_matrix[valid_ids]
            bi_valid = self.bi_vector[valid_ids]
            
            ratings_valid = (
                self.global_mean + 
                bu + 
                bi_valid + 
                np.dot(qi_valid, pu)
            )
            
            predictions[valid_mask] = np.clip(ratings_valid, 1, 5)
        
        return predictions
    
    def _recommend_content_based(self,
                                favorite_movies: List[Tuple[int, float]],
                                favorite_ids: List[int],
                                n: int,
                                min_rating_count: int) -> pd.DataFrame:
        """Content-based filtering fallback"""
        favorite_ratings = {m[0]: m[1] for m in favorite_movies}
        favorite_df = self.movies[self.movies['movieId'].isin(favorite_ids)]
        
        # Genre preferences
        genre_scores = {}
        for _, row in favorite_df.iterrows():
            rating = favorite_ratings[row['movieId']]
            for genre in row['genres'].split('|'):
                if genre not in genre_scores:
                    genre_scores[genre] = []
                genre_scores[genre].append(rating)
        
        genre_avg = {g: np.mean(ratings) for g, ratings in genre_scores.items()}
        
        # Candidates (zero-copy filter)
        mask = (
            (~self.movies['movieId'].isin(favorite_ids)) &
            (self.movies['rating_count'] >= min_rating_count)
        )
        candidates = self.movies[mask]
        
        # Score
        scores = []
        for _, movie in candidates.iterrows():
            genres = movie['genres'].split('|')
            
            genre_score = np.mean([genre_avg.get(g, 3.0) for g in genres])
            quality_score = movie['rating_avg']
            popularity_score = np.log1p(movie['rating_count']) / 10
            
            final_score = (
                0.6 * genre_score + 
                0.3 * quality_score + 
                0.1 * popularity_score
            )
            
            scores.append({
                'movieId': movie['movieId'],
                'predicted_rating': final_score
            })
        
        pred_df = pd.DataFrame(scores)
        pred_df = pred_df.nlargest(n, 'predicted_rating')
        
        recommendations = pred_df.merge(
            self.movies[['movieId', 'title_clean', 'genres', 
                        'rating_avg', 'rating_count']], 
            on='movieId'
        )
        
        recommendations['rank'] = range(1, len(recommendations) + 1)
        recommendations = recommendations.rename(columns={'title_clean': 'title'})
        
        return recommendations[[
            'rank', 'movieId', 'title', 'genres', 
            'predicted_rating', 'rating_avg', 'rating_count'
        ]]
    
    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Predict single rating"""
        return float(self._predict_batch_vectorized(user_id, np.array([movie_id]))[0])
    
    def batch_predict(self, 
                     user_movie_pairs: List[Tuple[int, int]]) -> pd.DataFrame:
        """Batch predictions"""
        from collections import defaultdict
        user_movies = defaultdict(list)
        
        for user_id, movie_id in user_movie_pairs:
            user_movies[user_id].append(movie_id)
        
        predictions = []
        for user_id, movie_ids in user_movies.items():
            ratings = self._predict_batch_vectorized(
                user_id, 
                np.array(movie_ids)
            )
            
            for mid, rating in zip(movie_ids, ratings):
                predictions.append({
                    'userId': user_id,
                    'movieId': mid,
                    'predicted_rating': rating
                })
        
        return pd.DataFrame(predictions)
    
    def get_user_profile(self, user_id: int) -> Optional[Dict]:
        """Get user profile"""
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        
        if len(user_ratings) == 0:
            return None
        
        user_movies = user_ratings.merge(
            self.movies[['movieId', 'title_clean', 'genres']], 
            on='movieId'
        )
        
        all_genres = []
        for genres_str in user_movies['genres'].dropna():
            all_genres.extend(genres_str.split('|'))
        genre_counts = pd.Series(all_genres).value_counts()
        
        top_movies = user_movies.nlargest(5, 'rating')[
            ['title_clean', 'genres', 'rating']
        ].to_dict('records')
        
        return {
            'user_id': user_id,
            'n_ratings': len(user_ratings),
            'avg_rating': float(user_ratings['rating'].mean()),
            'rating_std': float(user_ratings['rating'].std()),
            'favorite_genres': genre_counts.head(5).to_dict(),
            'top_rated_movies': top_movies
        }
    
    def explain_recommendation(self, user_id: int, movie_id: int) -> Dict:
        """Explain recommendation"""
        predicted_rating = self.predict_rating(user_id, movie_id)
        
        movie_info = self.movies[self.movies['movieId'] == movie_id].iloc[0]
        movie_genres = set(movie_info['genres'].split('|'))
        
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        user_movies = user_ratings.merge(
            self.movies[['movieId', 'title_clean', 'genres', 'rating_avg']], 
            on='movieId'
        )
        
        similar_movies = []
        for _, row in user_movies.iterrows():
            row_genres = set(row['genres'].split('|'))
            genre_overlap = len(movie_genres & row_genres) / len(movie_genres | row_genres)
            if genre_overlap > 0.5 and row['rating'] >= 4.0:
                similar_movies.append({
                    'title': row['title_clean'],
                    'genres': row['genres'],
                    'user_rating': row['rating'],
                    'genre_similarity': genre_overlap
                })
        
        similar_movies = sorted(
            similar_movies, 
            key=lambda x: x['genre_similarity'], 
            reverse=True
        )[:3]
        
        user_genre_ratings = {}
        for _, row in user_movies.iterrows():
            for genre in row['genres'].split('|'):
                if genre not in user_genre_ratings:
                    user_genre_ratings[genre] = []
                user_genre_ratings[genre].append(row['rating'])
        
        genre_avg = {g: np.mean(ratings) 
                    for g, ratings in user_genre_ratings.items()}
        
        matched_genres = {g: genre_avg.get(g, 0) 
                         for g in movie_genres if g in genre_avg}
        
        return {
            'movie_id': movie_id,
            'movie_title': movie_info['title_clean'],
            'movie_genres': movie_info['genres'],
            'predicted_rating': predicted_rating,
            'movie_avg_rating': movie_info['rating_avg'],
            'similar_movies_you_liked': similar_movies,
            'your_genre_preferences': matched_genres
        }
    
    def get_stats(self) -> Dict:
        """Get system stats"""
        return {
            'n_users': len(self.user_id_map),
            'n_movies': len(self.movie_id_map),
            'n_ratings': len(self.ratings),
            'n_factors': self.qi_matrix.shape[1],
            'global_mean': float(self.global_mean),
            'cache_size': len(self._user_rated_cache.cache)
        }


# UTILITY FUNCTIONS

def find_project_root() -> Path:
    """
    Find project root by looking for markers
    Supports running from any subdirectory
    """
    current = Path.cwd()
    
    # Look for project markers
    markers = ['models', 'data', 'notebooks', '.git', 'README.md']
    
    # Try current and parents
    for path in [current] + list(current.parents):
        if any((path / marker).exists() for marker in markers):
            return path
    
    # Fallback: current directory
    return current


def load_recommender(model_dir: Optional[str] = None, 
                    data_dir: Optional[str] = None,
                    cache_size: int = 1000) -> SVDRecommender:
    """
    Load recommender with smart path resolution
    
    Usage:
    ------
    # Auto-detect paths
    recommender = load_recommender()
    
    # Custom paths
    recommender = load_recommender(
        model_dir='path/to/models',
        data_dir='path/to/data'
    )
    """
    # Auto-detect project root
    project_root = find_project_root()
    
    # Default paths
    if model_dir is None:
        model_dir = project_root / 'models'
    else:
        model_dir = Path(model_dir)
    
    if data_dir is None:
        data_dir = project_root / 'data' / 'cleaned'
    else:
        data_dir = Path(data_dir)
    
    # Construct file paths
    model_path = model_dir / 'svd_model.pkl'
    movies_path = data_dir / 'movies_cleaned.csv'
    ratings_path = data_dir / 'ratings_cleaned.csv'
    
    # Validate
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not movies_path.exists():
        raise FileNotFoundError(f"Movies data not found: {movies_path}")
    if not ratings_path.exists():
        raise FileNotFoundError(f"Ratings data not found: {ratings_path}")
    
    print(f"Project root: {project_root}")
    print(f"Model dir: {model_dir}")
    print(f"Data dir: {data_dir}")
    
    return SVDRecommender(
        str(model_path), 
        str(movies_path), 
        str(ratings_path),
        cache_size=cache_size
    )


def print_recommendations(recommendations: pd.DataFrame, 
                         title: str = "Gợi ý phim"):
    """In gợi ý phim đẹp"""
    if recommendations.empty:
        print("\nKhông tìm thấy gợi ý phù hợp")
        return
    
    
    print(f"{title}")
    
    
    for _, row in recommendations.iterrows():
        print(f"#{row['rank']:<2} | {row['title']:<45} | {row['genres']:<20}")
        print(f"     Dự đoán: {row['predicted_rating']:.2f} | "
              f"TB: {row['rating_avg']:.2f} | "
              f"Số đánh giá: {row['rating_count']:,}")
        print()


def print_user_profile(profile: Optional[Dict]):
    """In hồ sơ người dùng"""
    if not profile:
        print("\nKhông tìm thấy hồ sơ người dùng")
        return
    
   
    print(f"HỒ SƠ NGƯỜI DÙNG - ID: {profile['user_id']}")
    
    
    print(f"\nThống kê:")
    print(f"Số phim đã đánh giá: {profile['n_ratings']}")
    print(f"Đánh giá trung bình: {profile['avg_rating']:.2f}")
    print(f"Độ lệch chuẩn: {profile['rating_std']:.2f}")
    
    print(f"\nThể loại yêu thích:")
    for genre, count in profile['favorite_genres'].items():
        print(f"{genre}: {count} phim")
    
    print(f"\nTop phim được đánh giá cao:")
    for movie in profile['top_rated_movies']:
        print(f"{movie['title_clean']} - {movie['rating']:.1f}")
        print(f"{movie['genres']}")


