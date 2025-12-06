"""
Neural Collaborative Filtering Recommender System
Production-Ready Implementation with Performance Optimization
Version: 1.1.0 (Optimized)
"""

import pickle
import numpy as np
import pandas as pd
from tensorflow import keras
from functools import lru_cache
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class NCFRecommender:
    """
    Neural Collaborative Filtering Recommender System
    Optimized với LRU Cache cho hiệu năng cao
    
    Example Usage:
    --------------
    >>> # Initialize
    >>> rec = NCFRecommender.load()  # Tự động load từ project root
    >>> 
    >>> # Get recommendations (cached)
    >>> recs = rec.recommend(user_id=1, n=10)
    >>> print(recs)
    """
    
    def __init__(self, model, user_encoder, movie_encoder, ratings_df, movies_df):
        """Khởi tạo với các components đã load"""
        self.model = model
        self.user_encoder = user_encoder
        self.movie_encoder = movie_encoder
        self.ratings = ratings_df
        self.movies = movies_df
        
        # Pre-compute và cache các thông tin thường dùng
        self.all_movie_ids = movies_df['movieId'].values
        self.all_movie_indices = movie_encoder.transform(self.all_movie_ids)
        
        # Cache user-movie mappings
        self._build_user_movie_cache()
        
        # Cache movie info dict
        self.movie_info_dict = movies_df.set_index('movieId').to_dict('index')

        # Precompute quick membership sets for O(1) checks
        self.user_set = set(self.ratings['userId'].unique())
        self.movie_set = set(self.movies['movieId'].unique())
        
        print(f"NCF Recommender sẵn sàng!")
        print(f"   - Số users: {ratings_df['userId'].nunique():,}")
        print(f"   - Số movies: {len(movies_df):,}")
        print(f"   - Số ratings: {len(ratings_df):,}")
        print(f"   - Cache đã được khởi tạo")
    
    
    def _build_user_movie_cache(self):
        """Xây dựng cache mapping user -> rated movies"""
        print("Đang xây dựng user-movie cache...")
        self.user_rated_movies = {}
        
        for user_id in self.ratings['userId'].unique():
            rated = self.ratings[self.ratings['userId'] == user_id]['movieId'].values
            self.user_rated_movies[user_id] = set(rated)
        
        print(f"Đã cache {len(self.user_rated_movies):,} users")
    
    
    @classmethod
    def load(cls, 
             model_path='../models/neural_cf_model.h5',
             user_encoder_path='../models/neural_cf_user_encoder.pkl',
             movie_encoder_path='../models/neural_cf_movie_encoder.pkl',
             ratings_path='../data/cleaned/ratings_cleaned.csv',
             movies_path='../data/cleaned/movies_cleaned.csv'):
        """
        Load tất cả components và tạo recommender instance
        Đường dẫn mặc định từ project root
        
        Parameters:
        -----------
        model_path : str, default='models/neural_cf_model.h5'
            Đường dẫn đến Keras model (từ project root)
        user_encoder_path : str
            Đường dẫn đến user encoder
        movie_encoder_path : str
            Đường dẫn đến movie encoder
        ratings_path : str
            Đường dẫn đến ratings CSV
        movies_path : str
            Đường dẫn đến movies CSV
        
        Returns:
        --------
        NCFRecommender instance
        """
        print("Đang khởi tạo NCF Recommender...")

        # Resolve project root reliably (two levels up from src)
        project_root = Path(__file__).resolve().parent.parent

        # Convert to Path objects and try to resolve fallback locations
        model_path = Path(model_path)
        user_encoder_path = Path(user_encoder_path)
        movie_encoder_path = Path(movie_encoder_path)
        ratings_path = Path(ratings_path)
        movies_path = Path(movies_path)

        # If provided paths don't exist, try resolving relative to project_root
        if not model_path.exists():
            candidate = project_root / 'models' / model_path.name
            if candidate.exists():
                model_path = candidate

        if not user_encoder_path.exists():
            candidate = project_root / 'models' / user_encoder_path.name
            if candidate.exists():
                user_encoder_path = candidate

        if not movie_encoder_path.exists():
            candidate = project_root / 'models' / movie_encoder_path.name
            if candidate.exists():
                movie_encoder_path = candidate

        if not ratings_path.exists():
            candidate = project_root / 'data' / 'cleaned' / ratings_path.name
            if candidate.exists():
                ratings_path = candidate

        if not movies_path.exists():
            candidate = project_root / 'data' / 'cleaned' / movies_path.name
            if candidate.exists():
                movies_path = candidate

        # Load model + encoders with safe fallback
        try:
            # Load model
            print(f"Đang load model từ {model_path}...")
            model = keras.models.load_model(str(model_path), compile=False)
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )

            # Load encoders
            print(f"Đang load encoders...")
            with open(user_encoder_path, 'rb') as f:
                user_encoder = pickle.load(f)
            with open(movie_encoder_path, 'rb') as f:
                movie_encoder = pickle.load(f)

            # Load data
            print(f"Đang load dữ liệu...")
            ratings_df = pd.read_csv(ratings_path)
            movies_df = pd.read_csv(movies_path)

            print(f"Đã load {len(ratings_df):,} ratings và {len(movies_df):,} movies")
            return cls(model, user_encoder, movie_encoder, ratings_df, movies_df)

        except Exception as e:
            # If any component missing or fails to load, provide a lightweight stub
            print(f"[NCF Fallback] Không thể load NCF model/encoders: {e}")
            print("[NCF Fallback] Trả về StubNCFRecommender — sẽ bỏ qua NCF trong hybrid.")

            # Load data so stub has access to movies/ratings
            try:
                ratings_df = pd.read_csv(ratings_path)
            except Exception:
                ratings_df = pd.DataFrame(columns=['userId', 'movieId', 'rating'])
            try:
                movies_df = pd.read_csv(movies_path)
            except Exception:
                movies_df = pd.DataFrame(columns=['movieId', 'title_clean', 'genres', 'rating_avg', 'rating_count'])

            class StubNCFRecommender:
                """A minimal stub with the same public API used by the hybrid system.

                Methods return safe defaults so evaluation and hybrid logic can continue
                without raising when NCF artifacts are absent.
                """
                def __init__(self, ratings_df, movies_df):
                    self.ratings = ratings_df
                    self.movies = movies_df
                    # minimal sets
                    self.user_set = set(self.ratings['userId'].unique()) if not self.ratings.empty else set()
                    self.movie_set = set(self.movies['movieId'].unique()) if not self.movies.empty else set()

                def recommend(self, user_id, n=10, exclude_rated=True, return_details=True, use_cache=True, return_details_kw=False, **kwargs):
                    # Return empty DataFrame with expected columns
                    cols = ['rank', 'movieId', 'title_clean', 'title', 'genres', 'predicted_rating', 'ncf_score']
                    return pd.DataFrame(columns=cols)

                def predict_rating(self, user_id, movie_id):
                    # Return neutral rating
                    return 3.0

                # keep method name compatibility
                def get_user_profile(self, user_id, top_n=5):
                    return {}

            return StubNCFRecommender(ratings_df, movies_df)
    
    
    @lru_cache(maxsize=1000)
    def _get_user_rated_movies_cached(self, user_id):
        """Cache danh sách phim user đã rate (immutable)"""
        return frozenset(self.user_rated_movies.get(user_id, set()))
    
    
    @lru_cache(maxsize=500)
    def _get_user_index_cached(self, user_id):
        """Cache user index encoding"""
        try:
            return self.user_encoder.transform([user_id])[0]
        except:
            return None
    
    
    def recommend(self, user_id, n=10, exclude_rated=True, 
                  min_rating_count=10, return_details=True, use_cache=True):
        """
        Gợi ý phim cho user (với optimization)
        
        Parameters:
        -----------
        user_id : int
            User ID cần gợi ý
        n : int, default=10
            Số lượng phim gợi ý
        exclude_rated : bool, default=True
            Loại bỏ phim đã xem
        min_rating_count : int, default=10
            Số lượng ratings tối thiểu cho phim ứng viên
        return_details : bool, default=True
            Trả về DataFrame chi tiết hoặc chỉ movie IDs
        use_cache : bool, default=True
            Sử dụng cache cho hiệu năng cao
        
        Returns:
        --------
        DataFrame hoặc list of movie IDs
        """
        
        # Validate user tồn tại (fast set membership)
        if user_id not in self.user_set:
            raise ValueError(f"Không tìm thấy User ID {user_id} trong database")
        
        # Get user index (cached)
        if use_cache:
            user_idx = self._get_user_index_cached(user_id)
            if user_idx is None:
                raise ValueError(f"Không thể encode user ID {user_id}")
        else:
            user_idx = self.user_encoder.transform([user_id])[0]
        
        # Get rated movies (cached)
        if use_cache:
            rated_movie_ids = self._get_user_rated_movies_cached(user_id)
        else:
            rated_movie_ids = set(
                self.ratings[self.ratings['userId'] == user_id]['movieId'].values
            )
        
        # Xây dựng candidate mask (vectorized)
        candidate_mask = np.ones(len(self.all_movie_ids), dtype=bool)
        
        # Loại bỏ phim đã xem
        if exclude_rated:
            candidate_mask &= ~np.isin(self.all_movie_ids, list(rated_movie_ids))
        
        # Lọc theo số lượng ratings
        if min_rating_count > 0:
            candidate_mask &= (self.movies['rating_count'].values >= min_rating_count)
        
        # Lấy danh sách candidates
        candidate_movie_ids = self.all_movie_ids[candidate_mask]
        candidate_movie_indices = self.all_movie_indices[candidate_mask]
        
        if len(candidate_movie_ids) == 0:
            print("Không có phim nào để gợi ý")
            return pd.DataFrame() if return_details else []
        
        # Predict ratings (batch prediction cho hiệu năng)
        user_array = np.full(len(candidate_movie_ids), user_idx, dtype=np.int32)
        predictions = self.model.predict(
            [user_array, candidate_movie_indices],
            batch_size=2048,  # Tăng batch size
            verbose=0
        ).flatten()
        
        # Denormalize: [0,1] -> [1,5] (vectorized)
        predictions = np.clip(predictions * 4 + 1, 1.0, 5.0)
        
        # Lấy top N (dùng argpartition cho hiệu năng cao)
        if n < len(predictions):
            # argpartition nhanh hơn argsort khi chỉ cần top N
            top_indices = np.argpartition(predictions, -n)[-n:]
            top_indices = top_indices[np.argsort(predictions[top_indices])[::-1]]
        else:
            top_indices = np.argsort(predictions)[::-1]
        
        top_movie_ids = candidate_movie_ids[top_indices]
        top_predictions = predictions[top_indices]
        
        # Trả về simple list nếu yêu cầu
        if not return_details:
            return top_movie_ids.tolist()
        
        # Xây dựng kết quả chi tiết
        results = pd.DataFrame({
            'rank': range(1, len(top_movie_ids) + 1),
            'movieId': top_movie_ids,
            'predicted_rating': top_predictions
        })

        # Merge với movie info
        results = results.merge(
            self.movies[['movieId', 'title_clean', 'genres', 
                        'rating_avg', 'rating_count']],
            on='movieId',
            how='left'
        )

        # Standardize columns for hybrid
        # Ensure both title_clean and title exist
        if 'title' not in results.columns:
            results['title'] = results['title_clean']

        # Add model-specific score alias
        results['ncf_score'] = results['predicted_rating']

        # Return consistent column order
        return results[[
            'rank', 'movieId', 'title_clean', 'title', 'genres',
            'predicted_rating', 'ncf_score', 'rating_avg', 'rating_count'
        ]]
    
    
    @lru_cache(maxsize=1000)
    def predict_rating(self, user_id, movie_id):
        """
        Dự đoán rating cho cặp user-movie cụ thể (cached)
        
        Parameters:
        -----------
        user_id : int
        movie_id : int
        
        Returns:
        --------
        float: Predicted rating [1.0 - 5.0]
        """
        
        # Validate
        if user_id not in self.user_set:
            raise ValueError(f"Không tìm thấy user {user_id}")
        if movie_id not in self.movie_set:
            raise ValueError(f"Không tìm thấy movie {movie_id}")
        
        # Encode (sử dụng cache)
        user_idx = self._get_user_index_cached(user_id)
        movie_idx = self.movie_encoder.transform([movie_id])[0]
        
        # Predict - FIX: Sử dụng numpy arrays thay vì list of lists
        user_array = np.array([user_idx], dtype=np.int32)
        movie_array = np.array([movie_idx], dtype=np.int32)
        
        pred = self.model.predict(
            [user_array, movie_array],  # Đúng format cho model
            verbose=0
        )[0][0]
        
        # Denormalize
        pred = float(np.clip(pred * 4 + 1, 1.0, 5.0))
        
        return pred
    
    
    @lru_cache(maxsize=500)
    def get_user_profile(self, user_id, top_n=5):
        """
        Lấy thông tin profile của user (cached)
        
        Parameters:
        -----------
        user_id : int
        top_n : int, default=5
            Số lượng phim được rate cao nhất
        
        Returns:
        --------
        dict với thống kê user
        """
        
        if user_id not in self.ratings['userId'].values:
            raise ValueError(f"Không tìm thấy user {user_id}")
        
        # Lấy ratings của user (sử dụng cached set nếu có thể)
        user_ratings = self.ratings[self.ratings['userId'] == user_id].copy()
        user_ratings = user_ratings.merge(
            self.movies[['movieId', 'title_clean', 'genres']],
            on='movieId'
        )
        
        # Top movies
        top_movies = user_ratings.nlargest(top_n, 'rating')[
            ['movieId', 'title_clean', 'genres', 'rating']
        ].to_dict('records')
        
        # Thống kê genres (vectorized)
        all_genres = []
        for genres_str in user_ratings['genres'].values:
            all_genres.extend(genres_str.split('|'))
        
        from collections import Counter
        genre_counts = Counter(all_genres)
        top_genres = [g for g, _ in genre_counts.most_common(5)]
        
        return {
            'user_id': user_id,
            'total_ratings': len(user_ratings),
            'avg_rating': float(user_ratings['rating'].mean()),
            'rating_std': float(user_ratings['rating'].std()),
            'top_movies': top_movies,
            'favorite_genres': top_genres
        }
    
    
    def batch_recommend(self, user_ids, n=10, show_progress=True, n_jobs=1):
        """
        Gợi ý cho nhiều users cùng lúc (optimized)
        
        Parameters:
        -----------
        user_ids : list of int
        n : int, default=10
        show_progress : bool, default=True
        n_jobs : int, default=1
            Số lượng parallel jobs (future: multiprocessing)
        
        Returns:
        --------
        dict: {user_id: recommendations_df}
        """
        
        results = {}
        total = len(user_ids)
        
        print(f"Đang tạo gợi ý cho {total} users...")
        
        for i, user_id in enumerate(user_ids, 1):
            if show_progress and i % 50 == 0:
                print(f"   Đã xử lý {i}/{total} users ({i/total*100:.1f}%)...", end='\r')
            
            try:
                recs = self.recommend(user_id, n=n, return_details=True, use_cache=True)
                results[user_id] = recs
            except Exception as e:
                if show_progress:
                    print(f"User {user_id} thất bại: {e}")
                results[user_id] = None
        
        if show_progress:
            print(f"Hoàn thành {total}/{total} users                    ")
        
        return results
    
    
    @lru_cache(maxsize=200)
    def similar_movies_by_embedding(self, movie_id, n=10):
        """
        Tìm phim tương tự dựa trên embedding (cached)
        
        Parameters:
        -----------
        movie_id : int
        n : int, default=10
        
        Returns:
        --------
        DataFrame với các phim tương tự
        """
        
        if movie_id not in self.movies['movieId'].values:
            raise ValueError(f"❌ Không tìm thấy movie {movie_id}")
        
        # Lấy embeddings
        movie_embedding_layer = self.model.get_layer('movie_embedding')
        embeddings = movie_embedding_layer.get_weights()[0]
        
        # Lấy target embedding
        movie_idx = self.movie_encoder.transform([movie_id])[0]
        target_emb = embeddings[movie_idx].reshape(1, -1)
        
        # Tính cosine similarity (vectorized)
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(target_emb, embeddings)[0]
        
        # Lấy top N (loại bỏ chính nó)
        top_indices = np.argsort(similarities)[::-1][1:n+1]
        top_movie_ids = self.movie_encoder.inverse_transform(top_indices)
        
        results = pd.DataFrame({
            'rank': range(1, len(top_movie_ids) + 1),
            'movieId': top_movie_ids,
            'similarity': similarities[top_indices]
        })
        
        results = results.merge(
            self.movies[['movieId', 'title_clean', 'genres', 
                        'rating_avg', 'rating_count']],
            on='movieId'
        )
        
        return results
    
    
    def clear_cache(self):
        """Xóa tất cả LRU cache"""
        self._get_user_rated_movies_cached.cache_clear()
        self._get_user_index_cached.cache_clear()
        self.predict_rating.cache_clear()
        self.get_user_profile.cache_clear()
        self.similar_movies_by_embedding.cache_clear()
        print("Đã xóa toàn bộ cache")
    
    
    def cache_info(self):
        """Hiển thị thông tin cache"""
        print("Thông tin Cache:")
        print(f"   - User rated movies: {self._get_user_rated_movies_cached.cache_info()}")
        print(f"   - User indices: {self._get_user_index_cached.cache_info()}")
        print(f"   - Predict ratings: {self.predict_rating.cache_info()}")
        print(f"   - User profiles: {self.get_user_profile.cache_info()}")
        print(f"   - Similar movies: {self.similar_movies_by_embedding.cache_info()}")
    
    
    def evaluate_user(self, user_id, test_size=0.2, random_state=42):
        """
        Đánh giá chất lượng gợi ý cho user cụ thể
        
        Parameters:
        -----------
        user_id : int
        test_size : float, default=0.2
        random_state : int, default=42
        
        Returns:
        --------
        dict với evaluation metrics
        """
        
        user_ratings = self.ratings[self.ratings['userId'] == user_id].copy()
        
        if len(user_ratings) < 10:
            return {'error': 'User có quá ít ratings để đánh giá'}
        
        # Chia train/test
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(
            user_ratings, test_size=test_size, random_state=random_state
        )
        
        # Predict test ratings
        predictions = []
        actuals = []
        
        for _, row in test.iterrows():
            try:
                pred = self.predict_rating(user_id, row['movieId'])
                predictions.append(pred)
                actuals.append(row['rating'])
            except:
                continue
        
        if len(predictions) == 0:
            return {'error': 'Không có predictions hợp lệ'}
        
        # Tính metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        
        return {
            'user_id': user_id,
            'n_train': len(train),
            'n_test': len(test),
            'rmse': float(rmse),
            'mae': float(mae)
        }
    
    
    def __repr__(self):
        return (f"NCFRecommender("
                f"users={self.ratings['userId'].nunique():,}, "
                f"movies={len(self.movies):,}, "
                f"ratings={len(self.ratings):,})")

# CONVENIENCE FUNCTIONS
def quick_recommend(user_id, n=10):
    """
    Quick one-liner để lấy gợi ý
    
    Example:
    --------
    >>> recs = quick_recommend(user_id=1, n=10)
    """
    rec = NCFRecommender.load()
    return rec.recommend(user_id, n=n)


def demo():
    """Chạy demo của recommender system"""
    
    print("="*80)
    print("NCF RECOMMENDER SYSTEM - DEMO (OPTIMIZED)")
    print("="*80)
    
    # Load recommender (từ project root)
    rec = NCFRecommender.load()
    
    # Test user
    test_user = 1
    
    # 1. User profile
    print(f"\n{'='*80}")
    print(f"1. THÔNG TIN USER - User {test_user}")
    print(f"{'='*80}")
    profile = rec.get_user_profile(test_user)
    print(f"Tổng số ratings: {profile['total_ratings']}")
    print(f"Rating trung bình: {profile['avg_rating']:.2f}")
    print(f"Thể loại yêu thích: {', '.join(profile['favorite_genres'])}")
    print(f"\nTop phim đã rate cao:")
    for movie in profile['top_movies']:
        print(f"   {movie['rating']} {movie['title_clean']}")
    
    # 2. Recommendations
    print(f"\n{'='*80}")
    print(f"2. GỢI Ý - Top 10 cho User {test_user}")
    print(f"{'='*80}")
    recs = rec.recommend(test_user, n=10)
    for _, row in recs.iterrows():
        print(f"#{row['rank']:<2} {row['title_clean']:<40} "
              f"Dự đoán: {row['predicted_rating']:.2f} "
              f"(Avg: {row['rating_avg']:.2f}, N={row['rating_count']})")
    
    # 3. Similar movies
    print(f"\n{'='*80}")
    print(f"3. PHIM TƯƠNG TỰ - Giống với Toy Story (movie 1)")
    print(f"{'='*80}")
    similar = rec.similar_movies_by_embedding(1, n=5)
    for _, row in similar.iterrows():
        print(f"#{row['rank']} {row['title_clean']:<40} "
              f"Độ tương đồng: {row['similarity']:.3f}")
    
    # 4. Predict rating
    print(f"\n{'='*80}")
    print(f"4. DỰ ĐOÁN RATING")
    print(f"{'='*80}")
    pred = rec.predict_rating(test_user, 1)
    print(f"User {test_user} sẽ đánh giá Movie 1: {pred:.2f} ⭐")
    
    # 5. Cache info
    print(f"\n{'='*80}")
    print(f"5. THÔNG TIN CACHE")
    print(f"{'='*80}")
    rec.cache_info()
    
    print(f"\n{'='*80}")
    print("DEMO HOÀN TẤT!")
    print(f"{'='*80}")


#

if __name__ == "__main__":
    demo()