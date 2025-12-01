
# Content-Based Recommendation Functions
# Generated automatically from Task 4

import pandas as pd
import numpy as np
import pickle

# Load model
with open('../models/content_based_model.pkl', 'rb') as f:
    cb_model = pickle.load(f)

top_k_indices = cb_model['top_k_indices']
top_k_scores = cb_model['top_k_scores']
movie_indices = cb_model['movie_indices']

# Load movies data
movies = pd.read_csv('../data/cleaned/movies_cleaned.csv')

def content_based_recommend(movie_id=None, movie_title=None, n=10, verbose=True):
    """
    Gợi ý phim dựa trên nội dung (Content-Based Filtering)

    Parameters:
    -----------
    movie_id : int
        ID của phim cần tìm phim tương tự
    movie_title : str
        Tên phim (alternative to movie_id)
    n : int
        Số lượng phim gợi ý (default=10)
    verbose : bool
        Hiển thị thông tin chi tiết

    Returns:
    --------
    DataFrame: Danh sách phim gợi ý
    """

    # Tìm movie index
    if movie_title:
        # Tìm theo title
        matches = movies[movies['title_clean'].str.contains(movie_title, case=False, na=False)]
        if len(matches) == 0:
            print(f"Không tìm thấy phim: '{movie_title}'")
            return None
        movie_id = matches.iloc[0]['movieId']
        if verbose:
            print(f"Tìm thấy: {matches.iloc[0]['title_clean']}")

    # Kiểm tra movie_id có trong mapping không
    if movie_id not in movie_indices:
        print(f"Movie ID {movie_id} không tồn tại trong dataset")
        return None

    # Lấy index của movie trong matrix
    idx = movie_indices[movie_id]

    # Lấy thông tin phim gốc
    movie_info = movies.iloc[idx]

    if verbose:
        print(f"\nPhim gốc:")
        print(f"   - Tên: {movie_info['title_clean']}")
        print(f"   - Thể loại: {movie_info['genres']}")
        print(f"   - Rating: {movie_info['rating_avg']:.2f}/5.0")
        print(f"   - Số lượt rate: {movie_info['rating_count']:.0f}")

    # Lấy top-K similar movies đã tính sẵn
    similar_indices = top_k_indices[idx]
    similar_scores = top_k_scores[idx]

    # Lấy top N
    top_n_indices = similar_indices[:n]
    top_n_scores = similar_scores[:n]

    # Tạo DataFrame kết quả
    recommendations = movies.iloc[top_n_indices].copy()
    recommendations['similarity_score'] = top_n_scores
    recommendations['rank'] = range(1, n+1)

    # Sắp xếp lại columns
    result_cols = ['rank', 'movieId', 'title_clean', 'genres', 'rating_avg', 
                   'rating_count', 'similarity_score']
    recommendations = recommendations[result_cols]

    if verbose:
        print(f"\nTop {n} phim tương tự:")
        print("-" * 100)
        for _, row in recommendations.iterrows():
            print(f"#{row['rank']:<2} | {row['title_clean']:<40} | {row['genres']:<30} | "
                  f"Rating: {row['rating_avg']:.2f} | Sim: {row['similarity_score']:.3f}")

    return recommendations


def content_based_recommend_multi(movie_ids, n=10, verbose=True):
    """
    Gợi ý phim dựa trên nhiều phim yêu thích (Cold Start)

    Parameters:
    -----------
    movie_ids : list
        List các movie_id yêu thích
    n : int
        Số lượng phim gợi ý
    verbose : bool
        Hiển thị thông tin chi tiết

    Returns:
    --------
    DataFrame: Danh sách phim gợi ý
    """

    if verbose:
        print(f"\nGợi ý dựa trên {len(movie_ids)} phim yêu thích:")

    # Tính average similarity
    all_scores = np.zeros(len(movies))
    valid_count = 0

    for movie_id in movie_ids:
        if movie_id not in movie_indices:
            if verbose:
                print(f"Movie ID {movie_id} không tồn tại, bỏ qua...")
            continue

        idx = movie_indices[movie_id]
        movie_title = movies.iloc[idx]['title_clean']

        if verbose:
            print(f"{movie_title}")

        # Lấy similarity scores
        similar_indices = top_k_indices[idx]
        similar_scores = top_k_scores[idx]

        # Cộng scores vào all_scores
        all_scores[similar_indices] += similar_scores
        valid_count += 1

    if valid_count == 0:
        print("Không có phim hợp lệ nào!")
        return None

    # Tính average
    all_scores /= valid_count

    # Loại bỏ các phim đã có trong input
    input_indices = [movie_indices[mid] for mid in movie_ids if mid in movie_indices]
    all_scores[input_indices] = -1

    # Lấy top N
    top_n_indices = np.argsort(all_scores)[::-1][:n]
    top_n_scores = all_scores[top_n_indices]

    # Tạo DataFrame kết quả
    recommendations = movies.iloc[top_n_indices].copy()
    recommendations['avg_similarity'] = top_n_scores
    recommendations['rank'] = range(1, n+1)

    result_cols = ['rank', 'movieId', 'title_clean', 'genres', 'rating_avg', 
                   'rating_count', 'avg_similarity']
    recommendations = recommendations[result_cols]

    if verbose:
        print(f"\nTop {n} phim gợi ý:")
        print("-" * 100)
        for _, row in recommendations.iterrows():
            print(f"#{row['rank']:<2} | {row['title_clean']:<40} | {row['genres']:<30} | "
                  f"Rating: {row['rating_avg']:.2f} | Sim: {row['avg_similarity']:.3f}")

    return recommendations

