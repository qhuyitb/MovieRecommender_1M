"""
Cấu hình chung cho ứng dụng MovieLens Recommender
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
FIGURES_DIR = BASE_DIR / 'figures'
SRC_DIR = BASE_DIR / 'src'

# Data paths
RAW_DATA_DIR = DATA_DIR / 'raw'
CLEANED_DATA_DIR = DATA_DIR / 'cleaned'

RATINGS_PATH = CLEANED_DATA_DIR / 'ratings_cleaned.csv'
MOVIES_PATH = CLEANED_DATA_DIR / 'movies_cleaned.csv'
USERS_PATH = CLEANED_DATA_DIR / 'users_cleaned.csv'

# Report paths
DATASET_STATS_PATH = RAW_DATA_DIR / 'dataset_stats.csv'
CLEANING_REPORT_PATH = CLEANED_DATA_DIR / 'cleaning_report.csv'
EVALUATION_RESULTS_PATH = DATA_DIR / 'analysis' / 'ranking_results_with_hybrid.csv'
EDA_INSIGHTS_PATH = FIGURES_DIR / 'EDA_insights.txt'

# Figure paths
WEIGHT_CURVES_FIG = FIGURES_DIR / 'weight_curves.png'
SMOOTH_VS_HARD_FIG = FIGURES_DIR / 'smooth_vs_hard_comparison.png'

# ========== STREAMLIT PAGE CONFIGURATION ==========
PAGE_CONFIG = {
    "page_title": "Hệ Thống Gợi Ý Phim MovieLens",
    "page_icon": "🎬",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# APP CONFIGURATION
APP_TITLE = "Hệ Thống Gợi Ý Phim MovieLens"
APP_ICON = "🎬"
PAGE_LAYOUT = "wide"
SIDEBAR_STATE = "expanded"

# MODEL CONFIGURATION
AVAILABLE_MODELS = {
    "Hybrid-Smooth": "Smooth Hybrid Recommender",
    "SVD": "Singular Value Decomposition",
    "NCF": "Neural Collaborative Filtering",
    "Content-Based": "Content-Based Filtering"
}

# Model weights
DEFAULT_WEIGHTS = {
    'content': 0.33,
    'svd': 0.33,
    'ncf': 0.34
}

# RECOMMENDATION SETTINGS
DEFAULT_TOP_N = 10
MIN_TOP_N = 5
MAX_TOP_N = 50
TOP_N_STEP = 5

MIN_RATING_THRESHOLD = 0.0
MAX_RATING_THRESHOLD = 5.0
RATING_STEP = 0.5

# Cold start thresholds
COLD_START_THRESHOLD = 5
WARM_START_THRESHOLD = 20

# COLD START MODES
COLD_START_MODES = {
    "Full Profile": None,  # No limit
    "Cold Start (1-5)": 2,
    "Warm Start (5-20)": 10
}

# UI SETTINGS
# Metric card styling
METRIC_CARD_STYLE = """
.user-metric-card {
    background: #fff;
    border: 1.5px solid #e0e0e0;
    border-radius: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    padding: 1.1rem 0.5rem 0.7rem 0.5rem;
    margin-bottom: 0.5rem;
    text-align: center;
}
.user-metric-title {
    color: #22223b;
    font-weight: 600;
    font-size: 1.08rem;
    margin-bottom: 0.18rem;
}
.user-metric-value {
    color: #1976d2;
    font-size: 2.05rem;
    font-weight: bold;
}
"""

# GENRE CONFIGURATION
# Context-aware genre mapping
CONTEXT_GENRE_MAP = {
    "Vui vẻ": ["Comedy", "Animation", "Musical"],
    "Buồn": ["Drama", "Romance"],
    "Hồi hộp": ["Action", "Thriller", "Horror"],
    "Thư giãn": ["Comedy", "Romance", "Animation"],
    "Một mình": ["Drama", "Sci-Fi", "Thriller"],
    "Gia đình": ["Animation", "Adventure", "Comedy"],
    "Bạn bè": ["Action", "Comedy", "Adventure"],
    "Hẹn hò": ["Romance", "Comedy", "Drama"]
}

MOOD_OPTIONS = ["Không chọn", "Vui vẻ", "Buồn", "Hồi hộp", "Thư giãn"]
TIME_OPTIONS = ["Không chọn", "Sáng", "Trưa", "Chiều", "Tối"]
CONTEXT_OPTIONS = ["Không chọn", "Một mình", "Gia đình", "Bạn bè", "Hẹn hò"]

# EVALUATION METRICS
EVALUATION_METRICS = [
    'Precision@K',
    'Recall@K',
    'NDCG@K',
    'MAP@K',
    'ARR'
]

# CHART SETTINGS
CHART_HEIGHT = 250
CHART_COLOR_SCHEMES = {
    'content': '#FF6B6B',
    'svd': '#4ECDC4',
    'ncf': '#95E1D3',
    'hybrid': '#FFD93D'
}

# ACTIVITY LEVELS
def get_activity_level(n_ratings):
    """Xác định mức độ hoạt động dựa trên số lượng ratings"""
    if n_ratings < COLD_START_THRESHOLD:
        return "thấp"
    elif n_ratings < WARM_START_THRESHOLD:
        return "trung bình"
    else:
        return "cao"

def get_activity_explanation(n_ratings):
    """Giải thích mức độ hoạt động"""
    if n_ratings < COLD_START_THRESHOLD:
        return (
            "Lịch sử tương tác hạn chế → "
            "ưu tiên **Content-Based Filtering** "
            "(dựa trên độ tương đồng nội dung và thể loại)"
        )
    elif n_ratings < WARM_START_THRESHOLD:
        return (
            "Mức độ tương tác vừa phải → "
            "kết hợp cân bằng các chiến lược, "
            "nhấn mạnh **SVD (Collaborative Filtering)**"
        )
    else:
        return (
            "Lịch sử tương tác phong phú → "
            "ưu tiên **Neural Collaborative Filtering (NCF)** "
            "để học các mẫu hành vi phức tạp"
        )

# FIGURE FILES
EDA_FIGURE_FILES = [
    ('01_rating_distribution.png', '⭐ Phân Bố Rating'),
    ('02_top_genres.png', '🎭 Top Thể Loại Phổ Biến'),
    ('03_user_movie_heatmap.png', '🔥 Heatmap User-Movie'),
    ('04_top_movies.png', '🏆 Top Phim Được Đánh Giá'),
    ('05_ratings_over_time.png', '📅 Rating Theo Thời Gian'),
    ('06_rating_by_genre.png', '🎬 Rating Theo Thể Loại'),
]

# DATA QUALITY THRESHOLDS
EXCELLENT_QUALITY_THRESHOLD = 90  # %
GOOD_QUALITY_THRESHOLD = 80  # %

# High sparsity threshold
HIGH_SPARSITY_THRESHOLD = 0.95

# Minimum ratings per user for "rich" profile
RICH_PROFILE_THRESHOLD = 100

# SESSION STATE KEYS
SESSION_KEYS = {
    'current_user_id': 'current_user_id',
    'session_ratings': 'session_ratings',
    'interaction_history': 'interaction_history',
    'original_recommendations': 'original_recommendations',
    'session_started': 'session_started'
}

# VALIDATION
def validate_paths():
    """Kiểm tra các đường dẫn quan trọng có tồn tại không"""
    required_paths = [
        DATA_DIR,
        CLEANED_DATA_DIR,
        RATINGS_PATH,
        MOVIES_PATH,
        USERS_PATH
    ]
    
    missing_paths = []
    for path in required_paths:
        if not path.exists():
            missing_paths.append(str(path))
    
    if missing_paths:
        raise FileNotFoundError(
            f"Missing required paths:\n" + "\n".join(f"  - {p}" for p in missing_paths)
        )
    
    return True