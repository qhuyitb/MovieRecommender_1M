import streamlit as st
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import configuration
from config.settings import PAGE_CONFIG

# Import utilities
from utils.data_loader import load_data
from utils.model_loader import load_models
from utils.session_manager import init_session_state

# Import UI components
from ui.styles import load_custom_css, render_header, render_footer
from ui.sidebar import render_sidebar

# Import tab modules
from tabs.tab_recommendations import render_recommendations_tab
from tabs.tab_movie_explorer import render_movie_explorer_tab
from tabs.tab_user_profile import render_user_profile_tab
from tabs.tab_model_performance import render_model_performance_tab
from tabs.tab_advanced_analysis import render_advanced_analysis_tab
from tabs.tab_eda_insights import render_eda_insights_tab
from tabs.tab_realtime_interaction import render_realtime_interaction_tab

# PAGE CONFIG
st.set_page_config(**PAGE_CONFIG)

# APPLY CUSTOM STYLES
load_custom_css()

# LOAD DATA AND MODELS
@st.cache_resource
def initialize_app():
    """Initialize app by loading all models and data"""
    models = load_models()
    data = load_data()
    return models, data

try:
    models, data = initialize_app()
    content_rec, svd_rec, ncf_rec, hybrid_rec = models
    ratings, movies, users, eval_results = data
except Exception as e:
    st.error(f"❌ Lỗi khởi tạo ứng dụng: {str(e)}")
    st.stop()

# HEADER
render_header('🎬 Hệ Thống Gợi Ý Phim MovieLens')
st.markdown("**Hệ thống gợi ý kết hợp** Content-Based, SVD và Neural Collaborative Filtering")

# SIDEBAR
sidebar_params = render_sidebar(ratings, movies, users)

# Build settings dict expected by tab modules
settings = sidebar_params

# Build model map for tabs
model_map = {
    'Content-Based': content_rec,
    'SVD': svd_rec,
    'NCF': ncf_rec,
    'Hybrid-Smooth': hybrid_rec
}

# Expose user_id for session initialization
user_id = settings.get('user_id')

# INITIALIZE SESSION STATE
init_session_state(user_id)

# MAIN TABS
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🎬 Gợi Ý Phim",
    "🎞️ Khám Phá Phim",
    "👤 Hồ Sơ Người Dùng",
    "📊 Hiệu Suất Mô Hình",
    "🔍 Phân tích nâng cao",
    "📈 Insight Dữ Liệu",
    "💖 Tương Tác Real Time"
])

# TAB 1: RECOMMENDATIONS
with tab1:
    render_recommendations_tab(
        settings=settings,
        model_map=model_map,
        movies=movies,
        hybrid_rec=hybrid_rec
    )

# TAB 2: MOVIE EXPLORER
with tab2:
    render_movie_explorer_tab(
        movies=movies,
        content_rec=content_rec
    )

# TAB 3: USER PROFILE
with tab3:
    render_user_profile_tab(
        settings=settings,
        movies=movies,
        ratings=ratings,
        users=users
    )

# TAB 4: MODEL PERFORMANCE
with tab4:
    render_model_performance_tab(
        eval_results=eval_results
    )

# TAB 5: ADVANCED ANALYSIS
with tab5:
    render_advanced_analysis_tab(
        settings=settings,
        model_map=model_map,
        movies=movies,
        content_rec=content_rec,
        svd_rec=svd_rec,
        ncf_rec=ncf_rec,
        hybrid_rec=hybrid_rec
    )

# TAB 6: EDA INSIGHTS
with tab6:
    render_eda_insights_tab()

# TAB 7: REALTIME INTERACTION
with tab7:
    render_realtime_interaction_tab(
        settings=settings,
        movies=movies,
        content_rec=content_rec,
        svd_rec=svd_rec,
        ncf_rec=ncf_rec,
        hybrid_rec=hybrid_rec
    )

# FOOTER
render_footer()