"""
utils/session_manager.py
Quản lý session state cho Real-Time Interaction
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Any


class SessionManager:
    """Class quản lý session state cho real-time interaction"""
    
    def __init__(self):
        """Initialize session manager"""
        self.keys = {
            'current_user_id': 'current_user_id',
            'session_ratings': 'session_ratings',
            'interaction_history': 'interaction_history',
            'original_recommendations': 'original_recommendations',
            'session_started': 'session_started'
        }
    
    def init_session(self, user_id: int) -> bool:
        """
        Initialize hoặc reset session cho user
        
        Args:
            user_id (int): User ID
        
        Returns:
            bool: True nếu user đã thay đổi (reset session)
        """
        # Check if session state keys exist
        if 'current_user_id' not in st.session_state:
            st.session_state.current_user_id = None
        
        # Check if user changed
        user_changed = st.session_state.current_user_id != user_id
        
        if user_changed:
            # User changed - full reset
            st.session_state.current_user_id = user_id
            self.reset_session()
            return True
        
        # User same - ensure all keys exist
        self._ensure_keys_exist()
        return False
    
    def _ensure_keys_exist(self):
        """Đảm bảo tất cả session keys tồn tại"""
        if 'session_ratings' not in st.session_state:
            st.session_state.session_ratings = {}
        
        if 'interaction_history' not in st.session_state:
            st.session_state.interaction_history = []
        
        if 'original_recommendations' not in st.session_state:
            st.session_state.original_recommendations = None
        
        if 'session_started' not in st.session_state:
            st.session_state.session_started = False
    
    def reset_session(self):
        """Reset session (giữ nguyên current_user_id)"""
        st.session_state.session_ratings = {}
        st.session_state.interaction_history = []
        st.session_state.original_recommendations = None
        st.session_state.session_started = False
    
    def start_session(self, original_recommendations: pd.DataFrame):
        """
        Bắt đầu session với baseline recommendations
        
        Args:
            original_recommendations (pd.DataFrame): Baseline recommendations
        """
        st.session_state.original_recommendations = original_recommendations.copy()
        st.session_state.session_started = True
    
    def is_session_started(self) -> bool:
        """Check session đã bắt đầu chưa"""
        return st.session_state.get('session_started', False)
    
    def get_session_ratings(self) -> Dict[int, float]:
        """Lấy dictionary session ratings (movieId -> rating)"""
        return st.session_state.get('session_ratings', {})
    
    def get_interaction_history(self) -> List[Dict[str, Any]]:
        """Lấy lịch sử tương tác"""
        return st.session_state.get('interaction_history', [])
    
    def get_original_recommendations(self) -> Optional[pd.DataFrame]:
        """Lấy baseline recommendations"""
        return st.session_state.get('original_recommendations', None)
    
    def add_rating(self, movie_id: int, rating: float, movie_title: str):
        """
        Thêm rating mới vào session
        
        Args:
            movie_id (int): Movie ID
            rating (float): Rating value (1-5)
            movie_title (str): Movie title
        """
        # Add to session ratings
        st.session_state.session_ratings[movie_id] = rating
        
        # Log to interaction history
        st.session_state.interaction_history.append({
            'timestamp': pd.Timestamp.now(),
            'action': 'rate',
            'movieId': movie_id,
            'title': movie_title,
            'rating': rating
        })
    
    def remove_rating(self, movie_id: int, movie_title: str):
        """
        Xóa rating khỏi session
        
        Args:
            movie_id (int): Movie ID
            movie_title (str): Movie title
        """
        # Remove from session ratings
        if movie_id in st.session_state.session_ratings:
            del st.session_state.session_ratings[movie_id]
        
        # Log to interaction history
        st.session_state.interaction_history.append({
            'timestamp': pd.Timestamp.now(),
            'action': 'remove',
            'movieId': movie_id,
            'title': movie_title,
            'rating': None
        })
    
    def get_rating(self, movie_id: int) -> Optional[float]:
        """Lấy rating của movie trong session (nếu có)"""
        return st.session_state.session_ratings.get(movie_id, None)
    
    def get_num_interactions(self) -> int:
        """Đếm số lượng tương tác mới trong session"""
        return len(st.session_state.session_ratings)
    
    def build_updated_profile(self, original_history: pd.DataFrame) -> pd.DataFrame:
        """
        Build updated user profile (original + session ratings)
        
        Args:
            original_history (pd.DataFrame): Original user rating history
        
        Returns:
            pd.DataFrame: Updated rating history
        """
        temp_ratings = original_history.copy()
        
        for movie_id, rating in st.session_state.session_ratings.items():
            new_rating = pd.DataFrame([{
                'userId': st.session_state.current_user_id,
                'movieId': movie_id,
                'rating': rating,
                'timestamp': int(pd.Timestamp.now().timestamp())
            }])
            temp_ratings = pd.concat([temp_ratings, new_rating], ignore_index=True)
        
        return temp_ratings
    
    def get_session_export_data(self) -> pd.DataFrame:
        """
        Export session ratings as DataFrame
        
        Returns:
            pd.DataFrame: Session ratings (movieId, rating)
        """
        data = [
            {'movieId': mid, 'rating': rating}
            for mid, rating in st.session_state.session_ratings.items()
        ]
        return pd.DataFrame(data)
    
    def get_history_export_data(self) -> pd.DataFrame:
        """
        Export interaction history as DataFrame
        
        Returns:
            pd.DataFrame: Full interaction history
        """
        if not st.session_state.interaction_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(st.session_state.interaction_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp', ascending=False)
    
    def calculate_overlap(self, updated_recs: pd.DataFrame) -> Dict[str, int]:
        """
        Tính overlap giữa original và updated recommendations
        
        Args:
            updated_recs (pd.DataFrame): Updated recommendations
        
        Returns:
            Dict: {'new_movies': int, 'removed_movies': int, 'overlap': int}
        """
        original_recs = self.get_original_recommendations()
        
        if original_recs is None or original_recs.empty:
            return {'new_movies': 0, 'removed_movies': 0, 'overlap': 0}
        
        original_ids = set(original_recs['movieId'].values[:10])
        updated_ids = set(updated_recs['movieId'].values[:10])
        
        new_movies = len(updated_ids - original_ids)
        removed_movies = len(original_ids - updated_ids)
        overlap = len(original_ids & updated_ids)
        
        return {
            'new_movies': new_movies,
            'removed_movies': removed_movies,
            'overlap': overlap
        }
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Lấy tóm tắt session hiện tại
        
        Returns:
            Dict: Session summary statistics
        """
        ratings = self.get_session_ratings()
        history = self.get_interaction_history()
        
        if not ratings:
            return {
                'num_interactions': 0,
                'avg_rating': 0.0,
                'num_high_ratings': 0,
                'num_low_ratings': 0
            }
        
        rating_values = list(ratings.values())
        
        return {
            'num_interactions': len(ratings),
            'avg_rating': sum(rating_values) / len(rating_values),
            'num_high_ratings': sum(1 for r in rating_values if r >= 4.0),
            'num_low_ratings': sum(1 for r in rating_values if r <= 2.0),
            'total_history_entries': len(history)
        }
    
    def get_session_genre_distribution(self, movies: pd.DataFrame) -> Dict[str, int]:
        """
        Phân tích phân bố thể loại trong session ratings
        
        Args:
            movies (pd.DataFrame): Movies dataset
        
        Returns:
            Dict: Genre -> count mapping
        """
        genre_counts = {}
        
        for movie_id in st.session_state.session_ratings.keys():
            movie = movies[movies['movieId'] == movie_id]
            if len(movie) > 0 and pd.notna(movie.iloc[0]['genres']):
                for genre in movie.iloc[0]['genres'].split('|'):
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        return genre_counts
    
    def clear_all_sessions(self):
        """Xóa toàn bộ session state (hard reset)"""
        for key in self.keys.values():
            if key in st.session_state:
                del st.session_state[key]


# Singleton instance
_session_manager = None

def get_session_manager() -> SessionManager:
    """
    Get singleton SessionManager instance
    
    Returns:
        SessionManager: Global session manager instance
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def init_session_state(user_id: int) -> None:
    """
    Backwards-compatible helper to initialize session state for a given user.

    Mirrors the older `init_session_state` API expected by `app.py`.
    """
    mgr = get_session_manager()
    mgr.init_session(user_id)