import streamlit as st
from typing import Dict, Any, List

from config.settings import (
    AVAILABLE_MODELS,
    DEFAULT_TOP_N,
    MIN_TOP_N,
    MAX_TOP_N,
    TOP_N_STEP,
    MIN_RATING_THRESHOLD,
    MAX_RATING_THRESHOLD,
    RATING_STEP,
    COLD_START_MODES
)

from utils.data_loader import (
    get_user_ids,
    get_user_history,
    get_user_info,
    apply_cold_start_limit,
    get_all_genres
)


def render_sidebar(ratings, movies, users) -> Dict[str, Any]:
    """Render the app sidebar and return selected parameters.

    Args:
        ratings (pd.DataFrame): Ratings dataframe
        movies (pd.DataFrame): Movies dataframe
        users (pd.DataFrame): Users dataframe

    Returns:
        Dict[str, Any]: Parameters used by the main app
    """
    st.sidebar.header("Tùy chọn")
    

    # User selection
    user_ids: List[int] = get_user_ids(ratings)
    if not user_ids:
        user_id = None
    else:
        user_id = st.sidebar.selectbox(
            "Chọn User",
            options=user_ids,
            index=0,
            help="Chọn User ID để xem lịch sử và gợi ý cá nhân hóa cho user đó."
        )

    # Model selection
    model_name = st.sidebar.selectbox(
        "Chọn Mô hình",
        options=list(AVAILABLE_MODELS.keys()),
        index=0,
        help="Chọn mô hình gợi ý bạn muốn sử dụng: Hybrid, SVD, NCF, hoặc Content-Based."
    )

    # Top-N
    top_n = st.sidebar.slider(
        "Số lượng gợi ý (Top N)",
        MIN_TOP_N,
        MAX_TOP_N,
        DEFAULT_TOP_N,
        step=TOP_N_STEP,
        help="Số lượng phim tối đa sẽ hiển thị trong danh sách gợi ý (Top-N)."
    )

    # Genres
    all_genres = get_all_genres(movies) if movies is not None else []
    selected_genres = st.sidebar.multiselect(
        "Thể loại",
        options=all_genres,
        default=[],
        help="Lọc gợi ý theo thể loại. Bỏ trống để không lọc theo thể loại."
    )

    # Minimum rating filter
    min_rating = st.sidebar.slider(
        "Min rating",
        MIN_RATING_THRESHOLD,
        MAX_RATING_THRESHOLD,
        MIN_RATING_THRESHOLD,
        step=RATING_STEP,
        help="Chỉ hiển thị phim có điểm trung bình (rating_avg) lớn hơn hoặc bằng giá trị này."
    )

    # Cold start mode
    cold_start_mode = st.sidebar.selectbox(
        "Cold start mode",
        options=list(COLD_START_MODES.keys()),
        index=0,
        help="Giới hạn số lượng đánh giá từ lịch sử user để mô phỏng Cold/Warm start."
    )

    # User history
    user_history_full = get_user_history(ratings, user_id) if user_id is not None else []
    user_history = apply_cold_start_limit(user_history_full, cold_start_mode) if user_id is not None else []

    # User info
    user_info = get_user_info(users, user_id) if user_id is not None else None

    return {
        'user_id': user_id,
        'model_name': model_name,
        'top_n': top_n,
        'selected_genres': selected_genres,
        'min_rating': min_rating,
        'cold_start_mode': cold_start_mode,
        'user_history': user_history,
        'user_history_full': user_history_full,
        'user_info': user_info,
        'all_genres': all_genres
    }