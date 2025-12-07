"""
Content-Based Movie Recommender System
Module này chứa ContentBasedRecommender class để gợi ý phim dựa trên nội dung.
Sử dụng TF-IDF và Cosine Similarity.

"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import Optional, List, Union
import warnings
warnings.filterwarnings('ignore')


class ContentBasedRecommender:
    """
    Content-Based Movie Recommender sử dụng TF-IDF và Cosine Similarity
    
    Attributes:
    -----------
    model_path : str
        Đường dẫn đến thư mục chứa models
    project_root : str
        Đường dẫn đến thư mục gốc project
    top_k_indices : np.ndarray
        Top-K similar movie indices cho mỗi phim
    top_k_scores : np.ndarray
        Top-K similarity scores cho mỗi phim
    movie_indices : dict
        Mapping từ movieId sang matrix index
    movies : pd.DataFrame
        DataFrame chứa thông tin phim
    K : int
        Số lượng similar movies được lưu trữ
    
    Methods:
    --------
    recommend(movie_id, movie_title, n, verbose)
        Gợi ý phim dựa trên 1 phim
    recommend_multi(movie_ids, n, verbose)
        Gợi ý phim dựa trên nhiều phim (Cold Start)
    get_movie_info(movie_id, movie_title)
        Lấy thông tin chi tiết của 1 phim
    search_movies(query, limit)
        Tìm kiếm phim theo tên
    """
    
    def __init__(self, model_path: str = None):
        """
        Khởi tạo ContentBasedRecommender
        
        Parameters:
        -----------
        model_path : str
            Đường dẫn đến thư mục chứa models 
            Nếu None, tự động tìm từ thư mục gốc project
        """
        # Tự động detect project root
        if model_path is None:
            # Lấy đường dẫn của file này
            current_file = os.path.abspath(__file__)
            current_dir = os.path.dirname(current_file)
            
            # Kiểm tra xem đang chạy từ đâu
            # Nếu file này ở trong src/ thì lùi 1 cấp
            if os.path.basename(current_dir) == 'src':
                project_root = os.path.dirname(current_dir)
            else:
                # Nếu không, coi current_dir là project root
                project_root = current_dir
            
            model_path = os.path.join(project_root, 'models')
            
            # Fallback: Nếu không tìm thấy models/, thử tìm từ working directory
            if not os.path.exists(model_path):
                cwd = os.getcwd()
                model_path = os.path.join(cwd, 'models')
                project_root = cwd
        else:
            # Nếu user truyền vào model_path, tính project_root từ đó
            project_root = os.path.dirname(model_path)
        
        self.model_path = model_path
        self.project_root = project_root
        self.top_k_indices = None
        self.top_k_scores = None
        self.movie_indices = None
        self.movies = None
        self.K = None
        
        # Load model tự động khi khởi tạo
        self._load_model()
    
    
    def _load_model(self):
        """Load model và dữ liệu từ file"""
        try:
            # Load content-based model
            model_file = os.path.join(self.model_path, 'content_based_model.pkl')
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            self.top_k_indices = model['top_k_indices']
            self.top_k_scores = model['top_k_scores']
            self.movie_indices = model['movie_indices']
            self.K = model['K']
            
            # Load movies data - SỬA ĐƯỜNG DẪN
            movies_file = os.path.join(self.project_root, 'data', 'cleaned', 'movies_cleaned.csv')
            self.movies = pd.read_csv(movies_file)

            # Ensure numeric dtypes for keys and ratings to avoid object dtype issues
            if 'movieId' in self.movies.columns:
                self.movies['movieId'] = pd.to_numeric(self.movies['movieId'], errors='coerce')
            if 'rating_avg' in self.movies.columns:
                self.movies['rating_avg'] = pd.to_numeric(self.movies['rating_avg'], errors='coerce')
            if 'rating_count' in self.movies.columns:
                self.movies['rating_count'] = pd.to_numeric(self.movies['rating_count'], errors='coerce')
            # Drop any rows without a valid movieId
            if 'movieId' in self.movies.columns:
                self.movies = self.movies[self.movies['movieId'].notna()]
                try:
                    self.movies['movieId'] = self.movies['movieId'].astype(int)
                except Exception:
                    pass
            
            print(f"Đã load Content-Based Model thành công!")
            print(f"  - Số phim: {len(self.movies):,}")
            print(f"  - Top-K: {self.K}")
            print(f"  - Matrix shape: {self.top_k_indices.shape}")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Không tìm thấy file model!\n"
                f"Vui lòng chạy notebook Task 4 để train model trước.\n"
                f"Error: {e}"
            )
        except Exception as e:
            raise Exception(f"Lỗi khi load model: {e}")
    
    
    def get_movie_info(self, 
                       movie_id: Optional[int] = None, 
                       movie_title: Optional[str] = None) -> Optional[pd.Series]:
        """
        Lấy thông tin chi tiết của 1 phim
        
        Parameters:
        -----------
        movie_id : int, optional
            ID của phim
        movie_title : str, optional
            Tên phim (tìm kiếm gần đúng)
        
        Returns:
        --------
        pd.Series or None: Thông tin phim
        """
        if movie_title:
            # Tìm theo title
            matches = self.movies[
                self.movies['title_clean'].str.contains(movie_title, case=False, na=False)
            ]
            if len(matches) == 0:
                print(f"Không tìm thấy phim: '{movie_title}'")
                return None
            return matches.iloc[0]
        
        elif movie_id:
            # Tìm theo ID
            if movie_id not in self.movie_indices:
                print(f"Movie ID {movie_id} không tồn tại")
                return None
            idx = self.movie_indices[movie_id]
            return self.movies.iloc[idx]
        
        else:
            print("Vui lòng cung cấp movie_id hoặc movie_title")
            return None
    
    
    def search_movies(self, query: str, limit: int = 10) -> pd.DataFrame:
        """
        Tìm kiếm phim theo tên
        
        Parameters:
        -----------
        query : str
            Từ khóa tìm kiếm
        limit : int
            Số lượng kết quả tối đa (default: 10)
        
        Returns:
        --------
        pd.DataFrame: Danh sách phim tìm thấy
        """
        matches = self.movies[
            self.movies['title_clean'].str.contains(query, case=False, na=False)
        ].head(limit)
        
        if len(matches) == 0:
            print(f"Không tìm thấy phim nào với từ khóa: '{query}'")
            return pd.DataFrame()
        
        print(f"Tìm thấy {len(matches)} phim:")
        for idx, row in matches.iterrows():
            print(f"  [{row['movieId']}] {row['title_clean']} - {row['genres']}")
        
        return matches[['movieId', 'title_clean', 'genres', 'rating_avg', 'rating_count']]
    
    
    def recommend(self, 
                  movie_id: Optional[int] = None,
                  movie_title: Optional[str] = None,
                  n: int = 10,
                  verbose: bool = True) -> Optional[pd.DataFrame]:
        """
        Gợi ý phim dựa trên 1 phim (Content-Based Filtering)
        
        Parameters:
        -----------
        movie_id : int, optional
            ID của phim cần tìm phim tương tự
        movie_title : str, optional
            Tên phim (alternative to movie_id)
        n : int
            Số lượng phim gợi ý (default: 10)
        verbose : bool
            Hiển thị thông tin chi tiết (default: True)
        
        Returns:
        --------
        pd.DataFrame: Danh sách phim gợi ý
        
        Examples:
        ---------
        >>> recommender = ContentBasedRecommender()
        >>> recs = recommender.recommend(movie_id=1, n=5)
        >>> recs = recommender.recommend(movie_title="Toy Story", n=10)
        """
        
        # Tìm movie info
        movie_info = self.get_movie_info(movie_id=movie_id, movie_title=movie_title)
        if movie_info is None:
            return None
        
        movie_id = movie_info['movieId']
        
        # Kiểm tra movie_id có trong mapping không
        if movie_id not in self.movie_indices:
            print(f"Movie ID {movie_id} không tồn tại trong dataset")
            return None
        
        # Lấy index của movie trong matrix
        idx = self.movie_indices[movie_id]
        
        if verbose:
           
            print(f"PHIM GỐC:")
    
            print(f"  Tên:        {movie_info['title_clean']}")
            print(f"  Thể loại:   {movie_info['genres']}")
            print(f"  Rating:     {movie_info['rating_avg']:.2f}/5.0")
            print(f"  Lượt rate:  {movie_info['rating_count']:.0f}")
        
        # Lấy top-K similar movies đã tính sẵn
        similar_indices = self.top_k_indices[idx]
        similar_scores = self.top_k_scores[idx]
        
        # Lấy top N
        top_n_indices = similar_indices[:n]
        top_n_scores = similar_scores[:n]
        
        # Tạo DataFrame kết quả
        recommendations = self.movies.iloc[top_n_indices].copy()
        recommendations['similarity_score'] = top_n_scores
        # Normalize/alias column names used by other recommenders
        # Ensure consistent score column name for hybrid merging
        recommendations['content_score'] = recommendations['similarity_score']
        # Keep both title_clean and title for compatibility
        recommendations['title'] = recommendations['title_clean']
        recommendations['rank'] = range(1, n+1)
        
        # Sắp xếp lại columns
        result_cols = ['rank', 'movieId', 'title_clean', 'title', 'genres', 
                   'rating_avg', 'rating_count', 'similarity_score', 'content_score']
        recommendations = recommendations[result_cols]
        
        if verbose:
           
            print(f"TOP {n} PHIM TƯƠNG TỰ:")
            print(f"{'Rank':<5} {'Title':<45} {'Genres':<25} {'Rating':<8} {'Similarity':<10}")
            print("-" * 100)
            
            for _, row in recommendations.iterrows():
                # Truncate title và genres nếu quá dài
                title = row['title_clean'][:42] + "..." if len(row['title_clean']) > 45 else row['title_clean']
                genres = row['genres'][:22] + "..." if len(row['genres']) > 25 else row['genres']
                
                print(f"#{row['rank']:<4} {title:<45} {genres:<25} "
                      f"{row['rating_avg']:.2f}   {row['similarity_score']:.3f}")
            
            # Thống kê
            print(f"\nThống kê:")
            print(f"  - Similarity score trung bình: {recommendations['similarity_score'].mean():.3f}")
            print(f"  - Similarity score cao nhất:   {recommendations['similarity_score'].max():.3f}")
            print(f"  - Rating trung bình:            {recommendations['rating_avg'].mean():.2f}/5.0")
        
        return recommendations
    
    
    def recommend_multi(self, 
                       movie_ids: List[int],
                       n: int = 10,
                       verbose: bool = True) -> Optional[pd.DataFrame]:
        """
        Gợi ý phim dựa trên nhiều phim yêu thích (Cold Start Solution)
        
        Parameters:
        -----------
        movie_ids : list of int
            Danh sách movie_id yêu thích
        n : int
            Số lượng phim gợi ý (default: 10)
        verbose : bool
            Hiển thị thông tin chi tiết (default: True)
        
        Returns:
        --------
        pd.DataFrame: Danh sách phim gợi ý
        
        Examples:
        ---------
        >>> recommender = ContentBasedRecommender()
        >>> recs = recommender.recommend_multi([1, 2, 3], n=10)
        """
        
        if not movie_ids or len(movie_ids) == 0:
            print("Vui lòng cung cấp ít nhất 1 movie_id")
            return None
        
        if verbose:
           
            print(f"GỢI Ý DỰA TRÊN {len(movie_ids)} PHIM YÊU THÍCH:")
           
        
        # Tính average similarity
        all_scores = np.zeros(len(self.movies))
        valid_count = 0
        input_indices = []
        
        for movie_id in movie_ids:
            if movie_id not in self.movie_indices:
                if verbose:
                    print(f"Movie ID {movie_id} không tồn tại, bỏ qua...")
                continue
            
            idx = self.movie_indices[movie_id]
            input_indices.append(idx)
            movie_title = self.movies.iloc[idx]['title_clean']
            
            if verbose:
                genres = self.movies.iloc[idx]['genres']
                print(f"[{movie_id}] {movie_title} - {genres}")
            
            # Lấy similarity scores
            similar_indices = self.top_k_indices[idx]
            similar_scores = self.top_k_scores[idx]
            
            # Cộng scores vào all_scores
            all_scores[similar_indices] += similar_scores
            valid_count += 1
        
        if valid_count == 0:
            print("Không có phim hợp lệ nào!")
            return None
        
        # Tính average
        all_scores /= valid_count
        
        # Loại bỏ các phim đã có trong input
        all_scores[input_indices] = -1
        
        # Lấy top N
        top_n_indices = np.argsort(all_scores)[::-1][:n]
        top_n_scores = all_scores[top_n_indices]
        
        # Tạo DataFrame kết quả
        recommendations = self.movies.iloc[top_n_indices].copy()
        recommendations['avg_similarity'] = top_n_scores
        # Standardize columns
        recommendations['content_score'] = recommendations['avg_similarity']
        recommendations['title'] = recommendations['title_clean']
        recommendations['rank'] = range(1, n+1)
        
        result_cols = ['rank', 'movieId', 'title_clean', 'title', 'genres', 
                   'rating_avg', 'rating_count', 'avg_similarity', 'content_score']
        recommendations = recommendations[result_cols]
        
        if verbose:
           
            print(f"TOP {n} PHIM GỢI Ý:")
           
            print(f"{'Rank':<5} {'Title':<45} {'Genres':<25} {'Rating':<8} {'Similarity':<10}")
            print("-" * 100)
            
            for _, row in recommendations.iterrows():
                title = row['title_clean'][:42] + "..." if len(row['title_clean']) > 45 else row['title_clean']
                genres = row['genres'][:22] + "..." if len(row['genres']) > 25 else row['genres']
                
                print(f"#{row['rank']:<4} {title:<45} {genres:<25} "
                      f"{row['rating_avg']:.2f}   {row['avg_similarity']:.3f}")
            
            # Thống kê
            print(f"\nThống kê:")
            print(f"  - Similarity score trung bình: {recommendations['avg_similarity'].mean():.3f}")
            print(f"  - Rating trung bình:            {recommendations['avg_similarity'].mean():.2f}/5.0")
        
        return recommendations
    
    
    def get_stats(self) -> dict:
        """
        Lấy thống kê về model
        
        Returns:
        --------
        dict: Thống kê model
        """
        return {
            'n_movies': len(self.movies),
            'top_k': self.K,
            'matrix_shape': self.top_k_indices.shape,
            'genres_available': sorted(
                set('|'.join(self.movies['genres'].dropna()).split('|'))
            ),
            'avg_rating': self.movies['rating_avg'].mean(),
            'total_ratings': self.movies['rating_count'].sum()
        }


# UTILITY FUNCTIONS (cho quick usage)

def quick_recommend(movie_id: Optional[int] = None,
                   movie_title: Optional[str] = None,
                   n: int = 10,
                   model_path: str = None) -> Optional[pd.DataFrame]:
    """
    Quick recommendation function (không cần khởi tạo class)
    
    Parameters:
    -----------
    movie_id : int, optional
        ID của phim
    movie_title : str, optional
        Tên phim
    n : int
        Số lượng gợi ý (default: 10)
    model_path : str, optional
        Đường dẫn model (default: auto-detect)
    
    Returns:
    --------
    pd.DataFrame: Danh sách phim gợi ý
    
    Examples:
    ---------
    >>> from content_based_recommender import quick_recommend
    >>> recs = quick_recommend(movie_title="Toy Story", n=5)
    """
    recommender = ContentBasedRecommender(model_path=model_path)
    return recommender.recommend(movie_id=movie_id, movie_title=movie_title, n=n)


# MAIN - FOR TESTING

if __name__ == "__main__":
    print("CONTENT-BASED RECOMMENDER - TEST")
    
    # Khởi tạo recommender
    recommender = ContentBasedRecommender()
    
    # Test 1: Recommend từ 1 phim
    print("TEST 1: GỢI Ý TỪ 1 PHIM")
    recs = recommender.recommend(movie_title="Toy Story", n=5)
    
    # Test 2: Recommend từ nhiều phim
    print("TEST 2: GỢI Ý TỪ NHIỀU PHIM")
    recs = recommender.recommend_multi([1, 2, 3], n=5)
    
    # Test 3: Search movies
    print("TEST 3: TÌM KIẾM PHIM")
    results = recommender.search_movies("Star Wars", limit=5)
    
    # Test 4: Get stats
    print("TEST 4: THỐNG KÊ MODEL")
    stats = recommender.get_stats()
    print(f"  Số phim:        {stats['n_movies']:,}")
    print(f"  Top-K:          {stats['top_k']}")
    print(f"  Rating TB:      {stats['avg_rating']:.2f}/5.0")
    print(f"  Tổng ratings:   {stats['total_ratings']:,.0f}")
    
    print("\nTất cả tests hoàn thành!")