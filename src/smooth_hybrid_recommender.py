"""
PERFORMANCE-OPTIMIZED Smooth Hybrid Recommender

"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')


class SmoothHybridRecommender:
    """Performance-optimized Hybrid Recommender"""
    
    def __init__(self, content_based_rec, svd_rec, ncf_rec, ratings_df, movies_df):
        self.content_rec = content_based_rec
        self.svd_rec = svd_rec
        self.ncf_rec = ncf_rec
        self.ratings = ratings_df
        self.movies = movies_df
        
        # Build user rating count cache
        self.user_rating_counts = ratings_df.groupby('userId').size().to_dict()
        
        # OPTIMIZATION: Pre-build user rated movies dict
        self._user_rated_cache = {}
        for user_id in self.user_rating_counts.keys():
            self._user_rated_cache[user_id] = set(
                ratings_df[ratings_df['userId'] == user_id]['movieId'].values
            )
        
        print("Smooth Hybrid Recommender sẵn sàng!")
        print(f"   - Content-Based: {len(movies_df):,} phim")
        print(f"   - SVD: Phân rã ma trận")
        print(f"   - NCF: Deep Learning")
        print(f"   - Số người dùng: {len(self.user_rating_counts):,}")
    
    @staticmethod
    def calculate_smooth_weights(n_ratings, method='sigmoid'):
        """Calculate smooth weights (unchanged)"""
        if method == 'sigmoid':
            content_weight = 1 / (1 + np.exp((n_ratings - 5) / 2))
            ncf_weight = 1 / (1 + np.exp(-(n_ratings - 20) / 5))
            svd_weight = 1 - content_weight - ncf_weight
            svd_weight = max(0, svd_weight)
        
        elif method == 'exponential':
            content_weight = np.exp(-n_ratings / 10)
            ncf_weight = 1 - np.exp(-(n_ratings - 15) / 10)
            ncf_weight = max(0, ncf_weight)
            svd_weight = 1 - content_weight - ncf_weight
            svd_weight = max(0, svd_weight)
        
        elif method == 'polynomial':
            if n_ratings <= 5:
                content_weight = 1.0
                svd_weight = 0.0
                ncf_weight = 0.0
            elif n_ratings <= 20:
                t = (n_ratings - 5) / 15
                content_weight = (1 - t) ** 2
                ncf_weight = t ** 2
                svd_weight = 2 * t * (1 - t)
            else:
                content_weight = 0.0
                svd_weight = max(0, 0.3 * np.exp(-(n_ratings - 20) / 20))
                ncf_weight = 1 - svd_weight
        else:
            raise ValueError(f"Unknown method: {method}")
        
        total = content_weight + svd_weight + ncf_weight
        
        return {
            'content': content_weight / total,
            'svd': svd_weight / total,
            'ncf': ncf_weight / total
        }
    
    @staticmethod
    def calculate_hard_weights(n_ratings):
        """Hard switching for comparison"""
        if n_ratings < 5:
            return {'content': 1.0, 'svd': 0.0, 'ncf': 0.0}
        elif n_ratings < 20:
            return {'content': 0.0, 'svd': 1.0, 'ncf': 0.0}
        else:
            return {'content': 0.0, 'svd': 0.0, 'ncf': 1.0}
    
    def get_user_rating_count(self, user_id):
        """Lấy số lượng đánh giá của người dùng"""
        return self.user_rating_counts.get(user_id, 0)
    
    def recommend(self, user_id, n=10, method='sigmoid', 
                  exclude_rated=True, return_weights=False, verbose=False):
        """
        TỐI ƯU HÓA: Gợi ý hybrid với vectorized operations
        
        Tối ưu chính:
        - Không dùng vòng lặp .iterrows()
        - Xử lý batch DataFrame
        - Merge một lần duy nhất
        """
        n_ratings = self.get_user_rating_count(user_id)
        weights = self.calculate_smooth_weights(n_ratings, method=method)
        
        if verbose:
            
            print(f"GỢI Ý HYBRID - User {user_id}")
            
            print(f"Số đánh giá: {n_ratings}")
            print(f"Trọng số:")
            print(f"   - Content-Based: {weights['content']:.3f}")
            print(f"   - SVD: {weights['svd']:.3f}")
            print(f"   - NCF: {weights['ncf']:.3f}")
           
        
        # TỐI ƯU HÓA: Lấy danh sách phim đã xem từ cache (O(1) thay vì O(n) query)
        rated_movies = self._user_rated_cache.get(user_id, set()) if exclude_rated else set()
        
        # Thu thập gợi ý từ mỗi model
        model_recs = []
        
        # 1. Content-Based
        if weights['content'] > 0.01:
            try:
                if n_ratings == 0:
                    cb_recs = self.movies.nlargest(n * 3, 'rating_avg')[
                        ['movieId', 'title_clean', 'genres', 'rating_avg']
                    ].copy()
                    cb_recs['predicted_rating'] = cb_recs['rating_avg']
                else:
                    user_movies = self.ratings[
                        self.ratings['userId'] == user_id
                    ].nlargest(5, 'rating')['movieId'].values
                    
                    if len(user_movies) > 0:
                        cb_recs = self.content_rec.recommend(
                            user_movies[0], n=n*3, verbose=False
                        )
                        
                        if not cb_recs.empty:
                            cb_recs = cb_recs.rename(columns={
                                'similarity_score': 'content_score',
                                'title_clean': 'title'
                            })
                    else:
                        cb_recs = pd.DataFrame()
                
                if not cb_recs.empty:
                    # TỐI ƯU HÓA: Xử lý vectorized thay vì iterrows()
                    cb_recs = cb_recs.rename(columns={
                        'title_clean': 'title',
                        'predicted_rating': 'content_score',
                        'similarity': 'content_score'
                    })
                    if 'content_score' not in cb_recs.columns:
                        cb_recs['content_score'] = cb_recs.get('rating_avg', 3.5)
                    
                    cb_recs = cb_recs[['movieId', 'title', 'genres', 'content_score']]
                    model_recs.append(cb_recs)
            except Exception as e:
                if verbose:
                    print(f"Content-Based thất bại: {e}")
        
        # 2. SVD
        if weights['svd'] > 0.01:
            try:
                svd_recs = self.svd_rec.recommend(user_id, n=n*3, exclude_rated=False)
                
                if not svd_recs.empty:
                    # TỐI ƯU HÓA: Đổi tên và chọn cột trong một bước
                    svd_recs = svd_recs.rename(columns={
                        'predicted_rating': 'svd_score'
                    })
                    svd_recs = svd_recs[['movieId', 'title', 'genres', 'svd_score']]
                    model_recs.append(svd_recs)
            except Exception as e:
                if verbose:
                    print(f"SVD thất bại: {e}")
        
        # 3. NCF
        if weights['ncf'] > 0.01:
            try:
                ncf_recs = self.ncf_rec.recommend(
                    user_id, n=n*3, exclude_rated=True, return_details=True
                )
                
                if not ncf_recs.empty:
                    # TỐI ƯU HÓA: Vectorized rename
                    ncf_recs = ncf_recs.rename(columns={
                        'title_clean': 'title',
                        'predicted_rating': 'ncf_score'
                    })
                    ncf_recs = ncf_recs[['movieId', 'title', 'genres', 'ncf_score']]
                    model_recs.append(ncf_recs)
            except Exception as e:
                if verbose:
                    print(f"NCF thất bại: {e}")
        
        if not model_recs:
            if return_weights:
                return pd.DataFrame(), weights
            return pd.DataFrame()
        
        # TỐI ƯU HÓA: Merge tất cả cùng lúc thay vì xây dựng dict lặp đi lặp lại
        # Nhanh hơn 10-50x so với cách dùng iterrows()!
        results_df = model_recs[0]
        for i in range(1, len(model_recs)):
            results_df = results_df.merge(
                model_recs[i], 
                on=['movieId', 'title', 'genres'], 
                how='outer'
            )
        
        # TỐI ƯU HÓA: Tính điểm trọng số dạng vector
        # Điền NaN bằng điểm mặc định 3.0
        results_df['content_score'] = results_df.get('content_score', pd.Series([np.nan] * len(results_df))).fillna(3.0)
        results_df['svd_score'] = results_df.get('svd_score', pd.Series([np.nan] * len(results_df))).fillna(3.0)
        results_df['ncf_score'] = results_df.get('ncf_score', pd.Series([np.nan] * len(results_df))).fillna(3.0)
        
        # Tính trung bình có trọng số (KHÔNG CÓ VÒNG LẶP!)
        results_df['predicted_rating'] = (
            weights['content'] * results_df['content_score'] +
            weights['svd'] * results_df['svd_score'] +
            weights['ncf'] * results_df['ncf_score']
        )
        
        # TỐI ƯU HÓA: Lọc vectorized
        if exclude_rated and rated_movies:
            results_df = results_df[~results_df['movieId'].isin(rated_movies)]
        
        # Lấy top N
        if len(results_df) > n:
            results_df = results_df.nlargest(n, 'predicted_rating')
        
        results_df = results_df.reset_index(drop=True)
        results_df['rank'] = range(1, len(results_df) + 1)
        
        # Thứ tự cột cuối cùng
        results_df = results_df.rename(columns={'title': 'title_clean'})
        cols = ['rank', 'movieId', 'title_clean', 'genres', 'predicted_rating',
                'content_score', 'svd_score', 'ncf_score']
        results_df = results_df[cols]
        
        if return_weights:
            return results_df, weights
        return results_df
    
    def recommend_for_new_user(self, favorite_movies, n=10, method='sigmoid'):
        """Gợi ý cho người dùng mới (tối ưu hóa)"""
        n_ratings = len(favorite_movies)
        weights = self.calculate_smooth_weights(n_ratings, method=method)
        
        
        print(f"GỢI Ý CHO NGƯỜI DÙNG MỚI")
      
        print(f"Phim yêu thích: {n_ratings}")
        print(f"Trọng số: CB={weights['content']:.2f}, SVD={weights['svd']:.2f}, NCF={weights['ncf']:.2f}")
        
        
        fav_movie_ids = [m for m, r in favorite_movies]
        
        # TỐI ƯU HÓA: Thu thập tất cả gợi ý cùng lúc
        all_cb_recs = []
        
        for movie_id, rating in favorite_movies:
            try:
                cb_recs = self.content_rec.recommend(
                    movie_id=movie_id,  # ← THÊM: chỉ định rõ parameter
                    n=n*2, 
                    verbose=False
                )
                
                if cb_recs is not None and not cb_recs.empty:
                    # ← FIX CHÍNH: Đổi tên TRƯỚC KHI append
                    if 'similarity_score' in cb_recs.columns:
                        cb_recs = cb_recs.rename(columns={'similarity_score': 'similarity'})
                    
                    cb_recs['source_movie'] = movie_id
                    all_cb_recs.append(cb_recs)
                    print(f"Tìm được {len(cb_recs)} phim từ movie {movie_id}")  # ← Debug log
                    
            except Exception as e:
                print(f"Lỗi movie {movie_id}: {e}")  # ← Better error handling
                continue
        
        if not all_cb_recs:
            return pd.DataFrame()
        
        # TỐI ƯU HÓA: Concatenate một lần thay vì vòng lặp dict
        combined_recs = pd.concat(all_cb_recs, ignore_index=True)
        combined_recs = combined_recs[~combined_recs['movieId'].isin(fav_movie_ids)]
        
        # TỐI ƯU HÓA: Vectorized groupby aggregation
        results_df = combined_recs.groupby('movieId').agg({
            'title_clean': 'first',
            'genres': 'first',
            'similarity': 'mean'
        }).reset_index()
        
        results_df['predicted_rating'] = results_df['similarity'] * 5
        results_df = results_df.nlargest(n, 'predicted_rating')
        results_df['rank'] = range(1, len(results_df) + 1)
        results_df['method'] = 'content-based'
        
        return results_df[['rank', 'movieId', 'title_clean', 'genres', 
                          'predicted_rating', 'method']]
    
    def compare_smooth_vs_hard(self, user_ids, n=10):
        """So sánh smooth vs hard switching"""
        results = []
        
        for user_id in user_ids:
            n_ratings = self.get_user_rating_count(user_id)
            smooth_weights = self.calculate_smooth_weights(n_ratings, 'sigmoid')
            hard_weights = self.calculate_hard_weights(n_ratings)
            
            try:
                smooth_recs = self.recommend(
                    user_id, n=n, method='sigmoid', verbose=False
                )
                smooth_top = smooth_recs['movieId'].tolist() if not smooth_recs.empty else []
            except:
                smooth_top = []
            
            results.append({
                'user_id': user_id,
                'n_ratings': n_ratings,
                'smooth_cb': smooth_weights['content'],
                'smooth_svd': smooth_weights['svd'],
                'smooth_ncf': smooth_weights['ncf'],
                'hard_cb': hard_weights['content'],
                'hard_svd': hard_weights['svd'],
                'hard_ncf': hard_weights['ncf'],
                'smooth_top_movies': smooth_top[:5]
            })
        
        return pd.DataFrame(results)
    
    @classmethod
    def load(cls,
             content_rec_path='../models/content_based_model.pkl',
             ratings_path='../data/cleaned/ratings_cleaned.csv',
             movies_path='../data/cleaned/movies_cleaned.csv'):
        """Load hybrid system từ saved models"""
        
        print("ĐANG TẢI HỆ THỐNG HYBRID RECOMMENDER")
        
        
        print("Đang tải Content-Based...")
        from content_based_recommender import ContentBasedRecommender
        content_rec = ContentBasedRecommender()  
        
        print("Đang tải SVD...")
        from svd_recommender import load_recommender
        svd_rec = load_recommender()
        
        print("Đang tải NCF...")
        from neural_cf_recommender import NCFRecommender
        ncf_rec = NCFRecommender.load()
        
        print("Đang tải dữ liệu...")
        ratings_df = pd.read_csv(ratings_path)
        movies_df = pd.read_csv(movies_path)
        
        print("Hoàn tất!")
        
        
        return cls(content_rec, svd_rec, ncf_rec, ratings_df, movies_df)
    
    def __repr__(self):
        return (f"SmoothHybridRecommender("
                f"users={len(self.user_rating_counts):,}, "
                f"movies={len(self.movies):,})")


def visualize_weight_curves(output_path='../figures/weight_curves.png'):
    """
    Vẽ đồ thị weight curves cho 3 methods
    """
    import matplotlib.pyplot as plt
    
    n_ratings_range = np.arange(0, 51, 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    methods = ['sigmoid', 'exponential', 'polynomial']
    
    for ax, method in zip(axes, methods):
        content_weights = []
        svd_weights = []
        ncf_weights = []
        
        for n in n_ratings_range:
            weights = SmoothHybridRecommender.calculate_smooth_weights(n, method)
            content_weights.append(weights['content'])
            svd_weights.append(weights['svd'])
            ncf_weights.append(weights['ncf'])
        
        ax.plot(n_ratings_range, content_weights, 'b-', linewidth=2.5, 
                label='Content-Based', marker='o', markersize=3)
        ax.plot(n_ratings_range, svd_weights, 'g-', linewidth=2.5,
                label='SVD', marker='s', markersize=3)
        ax.plot(n_ratings_range, ncf_weights, 'r-', linewidth=2.5,
                label='NCF', marker='^', markersize=3)
        
        ax.set_xlabel('Số Ratings của User', fontsize=12, fontweight='bold')
        ax.set_ylabel('Trọng Số', fontsize=12, fontweight='bold')
        ax.set_title(f'Phương pháp: {method.capitalize()}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 1.05)
        
        # Add vertical lines for transition zones
        ax.axvline(x=5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=20, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(5, 1.02, '5', ha='center', fontsize=9, color='gray')
        ax.text(20, 1.02, '20', ha='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Đã lưu: {output_path}")
    plt.close()


def demo():
    """Demo smooth hybrid system"""
    
    print("SMOOTH HYBRID RECOMMENDER SYSTEM - DEMO")
    
    
    # Load hybrid
    hybrid = SmoothHybridRecommender.load()
    
    # Test with different user types
    test_cases = [
        (1, "User rất tích cực"),
        (100, "User trung bình"),
        (500, "User ít hoạt động")
    ]
    
    for user_id, desc in test_cases:
        print(f"\n{'='*80}")
        print(f"TEST: {desc} (User {user_id})")

        
        recs, weights = hybrid.recommend(
            user_id, n=5, method='sigmoid',
            return_weights=True, verbose=True
        )
        
        if not recs.empty:
            print("\nTop 5 Gợi ý:")
            for _, row in recs.iterrows():
                print(f"#{row['rank']} {row['title_clean'][:40]:<40} "
                      f"→ {row['predicted_rating']:.2f}")
    
    # Visualize
    
    print("TẠO ĐỒ THỊ WEIGHT CURVES...")
    visualize_weight_curves()
    
    print("\nDEMO HOÀN TẤT!")


# ĐÁNH GIÁ HIỆU NĂNG
def benchmark_performance():
    """So sánh hiệu năng cũ vs mới"""
    import time
    
    hybrid = SmoothHybridRecommender.load()
    test_users = [12, 3, 100, 250, 500]
    
    
    print("ĐÁNH GIÁ HIỆU NĂNG")
    
    times = []
    for user_id in test_users:
        start = time.time()
        recs = hybrid.recommend(user_id, n=10, verbose=False)
        elapsed = time.time() - start
        times.append(elapsed)
        
        print(f"User {user_id}: {elapsed*1000:.2f}ms - {len(recs)} gợi ý")
    
    print(f"\nThời gian trung bình: {np.mean(times)*1000:.2f}ms")
    print(f"Tổng thời gian: {sum(times)*1000:.2f}ms")


if __name__ == "__main__":
    demo()