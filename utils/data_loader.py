"""
Load và cache dữ liệu cho MovieLens App
"""

import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Tuple, Optional

from config.settings import (
    RATINGS_PATH,
    MOVIES_PATH,
    USERS_PATH,
    EVALUATION_RESULTS_PATH,
    DATASET_STATS_PATH,
    CLEANING_REPORT_PATH
)


@st.cache_data
def load_ratings() -> pd.DataFrame:
    """
    Load ratings dataset
    
    Returns:
        pd.DataFrame: Ratings data với columns [userId, movieId, rating, timestamp]
    """
    try:
        ratings = pd.read_csv(RATINGS_PATH)
        return ratings
    except FileNotFoundError:
        st.error(f"Không tìm thấy file: {RATINGS_PATH}")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi khi load ratings: {e}")
        st.stop()


@st.cache_data
def load_movies() -> pd.DataFrame:
    """
    Load movies dataset
    
    Returns:
        pd.DataFrame: Movies data với columns [movieId, title, genres, rating_avg, rating_count, ...]
    """
    try:
        movies = pd.read_csv(MOVIES_PATH)
        return movies
    except FileNotFoundError:
        st.error(f"Không tìm thấy file: {MOVIES_PATH}")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi khi load movies: {e}")
        st.stop()


@st.cache_data
def load_users() -> pd.DataFrame:
    """
    Load users dataset
    
    Returns:
        pd.DataFrame: Users data với columns [userId, age, gender, occupation, ...]
    """
    try:
        users = pd.read_csv(USERS_PATH)
        return users
    except FileNotFoundError:
        st.error(f"Không tìm thấy file: {USERS_PATH}")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi khi load users: {e}")
        st.stop()


@st.cache_data
def load_evaluation_results() -> Optional[pd.DataFrame]:
    """
    Load evaluation results (ranking metrics)
    
    Returns:
        pd.DataFrame hoặc None: Evaluation results nếu file tồn tại
    """
    try:
        if EVALUATION_RESULTS_PATH.exists():
            eval_results = pd.read_csv(EVALUATION_RESULTS_PATH)
            return eval_results
        else:
            return None
    except Exception as e:
        st.warning(f"Không thể load evaluation results: {e}")
        return None


@st.cache_data
def load_dataset_stats() -> Optional[pd.DataFrame]:
    """
    Load dataset statistics (from EDA)
    
    Returns:
        pd.DataFrame hoặc None: Dataset stats nếu file tồn tại
    """
    try:
        if DATASET_STATS_PATH.exists():
            stats = pd.read_csv(DATASET_STATS_PATH)
            return stats
        else:
            return None
    except Exception as e:
        st.warning(f"Không thể load dataset stats: {e}")
        return None


@st.cache_data
def load_cleaning_report() -> Optional[pd.DataFrame]:
    """
    Load cleaning report
    
    Returns:
        pd.DataFrame hoặc None: Cleaning report nếu file tồn tại
    """
    try:
        if CLEANING_REPORT_PATH.exists():
            report = pd.read_csv(CLEANING_REPORT_PATH)
            return report
        else:
            return None
    except Exception as e:
        st.warning(f"Không thể load cleaning report: {e}")
        return None


@st.cache_data
def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load tất cả dữ liệu cần thiết cho app
    
    Returns:
        Tuple: (ratings, movies, users, eval_results)
    """
    ratings = load_ratings()
    movies = load_movies()
    users = load_users()
    eval_results = load_evaluation_results()
    
    return ratings, movies, users, eval_results


@st.cache_data
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Backwards-compatible wrapper for older code that imports `load_data`.

    Returns:
        Tuple: (ratings, movies, users, eval_results)
    """
    return load_all_data()


def get_user_history(ratings: pd.DataFrame, user_id: int) -> pd.DataFrame:
    """
    Lấy lịch sử đánh giá của user
    
    Args:
        ratings (pd.DataFrame): Ratings dataset
        user_id (int): User ID
    
    Returns:
        pd.DataFrame: User's rating history
    """
    user_history = ratings[ratings['userId'] == user_id].copy()
    return user_history


def get_user_info(users: pd.DataFrame, user_id: int) -> Optional[pd.Series]:
    """
    Lấy thông tin của user
    
    Args:
        users (pd.DataFrame): Users dataset
        user_id (int): User ID
    
    Returns:
        pd.Series hoặc None: User info
    """
    user_data = users[users['userId'] == user_id]
    if len(user_data) > 0:
        return user_data.iloc[0]
    return None


def apply_cold_start_limit(
    user_history: pd.DataFrame, 
    mode: str, 
    cold_start_limit: int = 2, 
    warm_start_limit: int = 10
) -> pd.DataFrame:
    """
    Áp dụng cold start limit lên user history
    
    Args:
        user_history (pd.DataFrame): Full user history
        mode (str): Cold start mode ("Full Profile", "Cold Start (1-5)", "Warm Start (5-20)")
        cold_start_limit (int): Limit cho cold start mode
        warm_start_limit (int): Limit cho warm start mode
    
    Returns:
        pd.DataFrame: Limited user history
    """
    if mode == "Cold Start (1-5)":
        limit = min(cold_start_limit, len(user_history))
        return user_history.sort_values('timestamp').head(limit).copy()
    elif mode == "Warm Start (5-20)":
        limit = min(warm_start_limit, len(user_history))
        return user_history.sort_values('timestamp').head(limit).copy()
    else:  # Full Profile
        return user_history.copy()


def get_all_genres(movies: pd.DataFrame) -> list:
    """
    Lấy tất cả genres unique từ movies dataset
    
    Args:
        movies (pd.DataFrame): Movies dataset
    
    Returns:
        list: Sorted list of unique genres
    """
    all_genres = set()
    for genres_str in movies['genres'].dropna():
        all_genres.update(genres_str.split('|'))
    return sorted(list(all_genres))


def filter_movies_by_genre(movies: pd.DataFrame, genres: list) -> pd.DataFrame:
    """
    Lọc phim theo danh sách genres
    
    Args:
        movies (pd.DataFrame): Movies dataset
        genres (list): List of genres to filter
    
    Returns:
        pd.DataFrame: Filtered movies
    """
    if not genres:
        return movies
    
    filtered = movies[
        movies['genres'].apply(
            lambda x: any(g in str(x).split('|') for g in genres) if pd.notna(x) else False
        )
    ]
    return filtered


def filter_movies_by_rating(movies: pd.DataFrame, min_rating: float) -> pd.DataFrame:
    """
    Lọc phim theo rating trung bình tối thiểu
    
    Args:
        movies (pd.DataFrame): Movies dataset
        min_rating (float): Minimum average rating
    
    Returns:
        pd.DataFrame: Filtered movies
    """
    if 'rating_avg' not in movies.columns:
        return movies
    
    return movies[movies['rating_avg'] >= min_rating]


def get_user_ids(ratings: pd.DataFrame) -> list:
    """
    Lấy danh sách unique user IDs
    
    Args:
        ratings (pd.DataFrame): Ratings dataset
    
    Returns:
        list: Sorted list of user IDs
    """
    return sorted(ratings['userId'].unique())


def calculate_sparsity(ratings: pd.DataFrame, movies: pd.DataFrame, users: pd.DataFrame) -> float:
    """
    Tính độ thưa của ma trận user-movie
    
    Args:
        ratings (pd.DataFrame): Ratings dataset
        movies (pd.DataFrame): Movies dataset
        users (pd.DataFrame): Users dataset
    
    Returns:
        float: Sparsity (0-1)
    """
    n_users = len(users)
    n_movies = len(movies)
    n_ratings = len(ratings)
    
    if n_users == 0 or n_movies == 0:
        return 1.0
    
    sparsity = 1 - (n_ratings / (n_users * n_movies))
    return sparsity


def get_top_rated_movies(
    movies: pd.DataFrame, 
    min_rating_count: int = 100, 
    top_n: int = 10
) -> pd.DataFrame:
    """
    Lấy top phim được đánh giá cao
    
    Args:
        movies (pd.DataFrame): Movies dataset
        min_rating_count (int): Minimum number of ratings
        top_n (int): Number of top movies to return
    
    Returns:
        pd.DataFrame: Top rated movies
    """
    if 'rating_count' not in movies.columns or 'rating_avg' not in movies.columns:
        return pd.DataFrame()
    
    top_movies = movies[movies['rating_count'] >= min_rating_count]\
        .sort_values('rating_avg', ascending=False)\
        .head(top_n)
    
    return top_movies


def search_movies(movies: pd.DataFrame, query: str, max_results: int = 10) -> pd.DataFrame:
    """
    Tìm kiếm phim theo tên
    
    Args:
        movies (pd.DataFrame): Movies dataset
        query (str): Search query
        max_results (int): Maximum number of results
    
    Returns:
        pd.DataFrame: Search results
    """
    if not query or not query.strip():
        return pd.DataFrame()
    
    # Sanitize query
    query = query.strip()[:100]
    
    # Case-insensitive search
    results = movies[
        movies['title'].str.contains(query, case=False, na=False, regex=False)
    ].head(max_results)
    
    return results


def get_movie_by_id(movies: pd.DataFrame, movie_id: int) -> Optional[pd.Series]:
    """
    Lấy thông tin phim theo ID
    
    Args:
        movies (pd.DataFrame): Movies dataset
        movie_id (int): Movie ID
    
    Returns:
        pd.Series hoặc None: Movie info
    """
    movie_data = movies[movies['movieId'] == movie_id]
    if len(movie_data) > 0:
        return movie_data.iloc[0]
    return None


def get_movies_by_ids(movies: pd.DataFrame, movie_ids: list) -> pd.DataFrame:
    """
    Lấy thông tin nhiều phim theo list IDs
    
    Args:
        movies (pd.DataFrame): Movies dataset
        movie_ids (list): List of movie IDs
    
    Returns:
        pd.DataFrame: Movies data
    """
    return movies[movies['movieId'].isin(movie_ids)]