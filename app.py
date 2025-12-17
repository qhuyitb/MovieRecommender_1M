import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import re
# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from content_based_recommender import ContentBasedRecommender
from svd_recommender import load_recommender as load_svd
from neural_cf_recommender import NCFRecommender
from smooth_hybrid_recommender import SmoothHybridRecommender

# Page config
st.set_page_config(
    page_title="Hệ Thống Gợi Ý Phim MovieLens",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
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
</style>
""", unsafe_allow_html=True)

# Cache model loading
@st.cache_resource
def load_models():
    """Load all recommender models once"""
    with st.spinner("Loading recommender models..."):
        content_rec = ContentBasedRecommender()
        svd_rec = load_svd()
        ncf_rec = NCFRecommender.load()
        hybrid_rec = SmoothHybridRecommender.load(
            ratings_path=os.path.join('data', 'cleaned', 'ratings_cleaned.csv'),
            movies_path=os.path.join('data', 'cleaned', 'movies_cleaned.csv')
        )
    return content_rec, svd_rec, ncf_rec, hybrid_rec

@st.cache_data
def load_data():
    """Load all datasets once"""
    ratings = pd.read_csv(os.path.join('data', 'cleaned', 'ratings_cleaned.csv'))
    movies = pd.read_csv(os.path.join('data', 'cleaned', 'movies_cleaned.csv'))
    users = pd.read_csv(os.path.join('data', 'cleaned', 'users_cleaned.csv'))
    
    # Load evaluation results if exists
    eval_results = None
    if os.path.exists('ranking_results_with_hybrid.csv'):
        eval_results = pd.read_csv('ranking_results_with_hybrid.csv')
    
    return ratings, movies, users, eval_results

# Load everything
content_rec, svd_rec, ncf_rec, hybrid_rec = load_models()
ratings, movies, users, eval_results = load_data()

# Header
st.markdown('<div class="main-header">🎬 Hệ Thống Gợi Ý Phim MovieLens</div>', unsafe_allow_html=True)
st.markdown("**Hệ thống gợi ý kết hợp** Content-Based, SVD và Neural Collaborative Filtering")

# ========== SIDEBAR ==========
st.sidebar.header("⚙️ Cài Đặt")

# User ID input
user_ids = sorted(ratings['userId'].unique())
user_id = st.sidebar.selectbox(
    "Chọn ID Người Dùng",
    options=user_ids,
    index=0,
    help="Chọn người dùng để nhận gợi ý phim cá nhân hóa"
)

# Model selection
model_name = st.sidebar.selectbox(
    "Mô Hình Gợi Ý",
    options=["Hybrid-Smooth", "SVD", "NCF", "Content-Based"],
    help="Chọn thuật toán để tạo gợi ý phim"
)

# Top-N slider
top_n = st.sidebar.slider(
    "Số Lượng Gợi Ý",
    min_value=5,
    max_value=50,
    value=10,
    step=5
)

# Genre filter
all_genres = set()
for genres_str in movies['genres'].dropna():
    all_genres.update(genres_str.split('|'))
all_genres = sorted(list(all_genres))

selected_genres = st.sidebar.multiselect(
    "Lọc Theo Thể Loại (tùy chọn)",
    options=all_genres,
    default=[]
)

# Min rating filter
min_rating = st.sidebar.slider(
    "Đánh Giá Trung Bình Tối Thiểu",
    min_value=0.0,
    max_value=5.0,
    value=0.0,
    step=0.5
)

# Get user data - Load full history first
user_history_full = ratings[ratings['userId'] == user_id].copy()
user_info = users[users['userId'] == user_id].iloc[0] if len(users[users['userId'] == user_id]) > 0 else None

# Cold Start Mode
st.sidebar.markdown("---")
st.sidebar.markdown("**🧪 Chế Độ Cold Start**")

cold_start_mode = st.sidebar.radio(
    "Giới hạn lịch sử quan sát",
    options=["Full Profile", "Cold Start (1-5)", "Warm Start (5-20)"],
    help="Mô phỏng kịch bản cold-start bằng cách giới hạn số rating hệ thống được biết"
)

# Apply cold start limit
if cold_start_mode == "Cold Start (1-5)":
    limit = min(2, len(user_history_full))
    user_history = user_history_full.sort_values('timestamp').head(limit).copy()
    st.sidebar.info(f"🔬 Chế độ Cold Start: Chỉ sử dụng {limit} rating đầu tiên")
elif cold_start_mode == "Warm Start (5-20)":
    limit = min(10, len(user_history_full))
    user_history = user_history_full.sort_values('timestamp').head(limit).copy()
    st.sidebar.info(f"🌡️ Chế độ Warm Start: Sử dụng {limit} rating đầu tiên")

else:
    user_history = user_history_full.copy()
    st.sidebar.success(f"✅ Full Profile: Sử dụng toàn bộ {len(user_history)} ratings")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Thống Kê Người Dùng:**")
st.sidebar.markdown(f"- Số đánh giá quan sát: {len(user_history)}")
st.sidebar.markdown(f"- Tổng đánh giá thực: {len(user_history_full)}")
if len(user_history) > 0:
    st.sidebar.markdown(f"- Đánh giá TB: {user_history['rating'].mean():.2f}")
    if user_info is not None:
        st.sidebar.markdown(f"- Tuổi: {user_info.get('age', 'N/A')}")
        st.sidebar.markdown(f"- Giới tính: {user_info.get('gender', 'N/A')}")


# ========== MAIN TABS ========== 
tab1, tab2, tab3, tab4, tab5, tab6,tab7 = st.tabs(["🎬 Gợi Ý Phim", "🎞️ Khám Phá Phim", "👤 Hồ Sơ Người Dùng", ". 💖 Tương Tác Real Time", "📊 Hiệu Suất Mô Hình","🔍 Phân tích nâng cao", "📈 Insight Dữ Liệu"])

# ========== TAB 1: RECOMMENDATIONS ==========
with tab1:
    st.header(f"Top {top_n} Phim Được Gợi Ý")
    
    # Select model
    model_map = {
        "SVD": svd_rec,
        "NCF": ncf_rec,
        "Content-Based": content_rec,
        "Hybrid-Smooth": hybrid_rec
    }
    
    selected_model = model_map[model_name]
    
    # Generate recommendations
    with st.spinner(f"Đang tạo gợi ý từ mô hình {model_name}..."):
        try:
            if model_name == "Content-Based":
                # Content-based needs seed items
                liked_movies = user_history[user_history['rating'] >= 4.0]['movieId'].values
                if len(liked_movies) == 0:
                    st.warning("⚠️ Người dùng chưa có phim nào được đánh giá cao (≥4.0). Không thể tạo gợi ý content-based.")
                    st.stop()
                seed_items = list(liked_movies)[:5]
                recs_df = selected_model.recommend_multi(seed_items, n=top_n*3, verbose=False)
                if recs_df is None or recs_df.empty:
                    st.error("Không thể tạo gợi ý phim")
                    st.stop()
                recommended_movies = recs_df['movieId'].values
            else:
                # CF models
                recs = selected_model.recommend(user_id, n=top_n*3, exclude_rated=True)
                if isinstance(recs, pd.DataFrame):
                    recommended_movies = recs['movieId'].values
                else:
                    recommended_movies = np.array(recs) if not isinstance(recs, np.ndarray) else recs
            
            # Filter by genre and rating
            filtered_movies = []
            for movie_id in recommended_movies:
                movie = movies[movies['movieId'] == movie_id]
                if len(movie) == 0:
                    continue
                movie = movie.iloc[0]
                
                # Check rating threshold
                if movie.get('rating_avg', 0) < min_rating:
                    continue
                
                # Check genre filter
                if selected_genres:
                    movie_genres = movie['genres'].split('|') if pd.notna(movie['genres']) else []
                    if not any(g in selected_genres for g in movie_genres):
                        continue
                
                filtered_movies.append(movie_id)
                if len(filtered_movies) >= top_n:
                    break
            
            if len(filtered_movies) == 0:
                st.warning("Không có phim nào phù hợp với bộ lọc. Thử điều chỉnh cài đặt.")
                st.stop()
            
            # Build results dataframe
            results = []
            for rank, movie_id in enumerate(filtered_movies, 1):
                movie = movies[movies['movieId'] == movie_id].iloc[0]
                results.append({
                    'Hạng': rank,
                    'ID Phim': movie_id,
                    'Tên Phim': movie['title'],
                    'Thể Loại': movie['genres'],
                    'Đánh Giá TB': movie.get('rating_avg', 0.0),
                    'Số Đánh Giá': movie.get('rating_count', 0)
                })
            
            results_df = pd.DataFrame(results)
            
            # Display results
            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Đánh Giá TB': st.column_config.NumberColumn(format="%.2f"),
                    'Số Đánh Giá': st.column_config.NumberColumn(format="%d")
                }
            )
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Tải Xuống Gợi Ý (CSV)",
                data=csv,
                file_name=f"{model_name}_goiy_user{user_id}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Lỗi khi tạo gợi ý: {str(e)}")
            st.stop()
    
    # Hybrid explanation
    if model_name == "Hybrid-Smooth":
        st.markdown("---")
        st.subheader("🔍 Giải Thích Mô Hình Kết Hợp")
        
        n_ratings = len(user_history)
        weights = hybrid_rec.calculate_smooth_weights(n_ratings, method='sigmoid')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig_content = go.Figure(go.Indicator(
                mode="gauge+number",
                value=weights['content'] * 100,
                title={'text': "Content-Based"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "#FF6B6B"}}
            ))
            fig_content.update_layout(height=250)
            st.plotly_chart(fig_content, use_container_width=True)
        
        with col2:
            fig_svd = go.Figure(go.Indicator(
                mode="gauge+number",
                value=weights['svd'] * 100,
                title={'text': "SVD"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "#4ECDC4"}}
            ))
            fig_svd.update_layout(height=250)
            st.plotly_chart(fig_svd, use_container_width=True)
        
        with col3:
            fig_ncf = go.Figure(go.Indicator(
                mode="gauge+number",
                value=weights['ncf'] * 100,
                title={'text': "NCF"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "#95E1D3"}}
            ))
            fig_ncf.update_layout(height=250)
            st.plotly_chart(fig_ncf, use_container_width=True)
        
        
        # Explanation text
        if n_ratings < 5:
            activity = "thấp"
            explanation = (
                "Lịch sử tương tác hạn chế → "
                "ưu tiên **Content-Based Filtering** "
                "(dựa trên độ tương đồng nội dung và thể loại)"
            )

        elif n_ratings < 20:
            activity = "trung bình"
            explanation = (
                "Mức độ tương tác vừa phải → "
                "kết hợp cân bằng các chiến lược, "
                "nhấn mạnh **SVD (Collaborative Filtering)**"
            )

        else:
            activity = "cao"
            explanation = (
                "Lịch sử tương tác phong phú → "
                "ưu tiên **Neural Collaborative Filtering (NCF)** "
                "để học các mẫu hành vi phức tạp"
            )

        st.info(
            f"**Mức Hoạt Động:** {n_ratings} đánh giá ({activity}) → {explanation}"
        )

        
    
    # Model comparison
    st.markdown("---")
    st.subheader("🔄 So Sánh Tất Cả Mô Hình")
    
    with st.spinner("Đang tạo gợi ý từ tất cả mô hình..."):
        comparison_results = {}
        
        for name, model in model_map.items():
            try:
                if name == "Content-Based":
                    liked_movies = user_history[user_history['rating'] >= 4.0]['movieId'].values
                    if len(liked_movies) == 0:
                        comparison_results[name] = []
                        continue
                    seed_items = list(liked_movies)[:5]
                    recs_df = model.recommend_multi(seed_items, n=5, verbose=False)
                    if recs_df is None or recs_df.empty:
                        comparison_results[name] = []
                    else:
                        comparison_results[name] = recs_df['movieId'].values[:5].tolist()
                else:
                    recs = model.recommend(user_id, n=5, exclude_rated=True)
                    if isinstance(recs, pd.DataFrame):
                        comparison_results[name] = recs['movieId'].values[:5].tolist()
                    else:
                        comparison_results[name] = (np.array(recs) if not isinstance(recs, np.ndarray) else recs)[:5].tolist()
            except:
                comparison_results[name] = []
        
        # Find common movies across models
        all_movie_ids = []
        for movie_ids in comparison_results.values():
            all_movie_ids.extend(movie_ids)
        
        movie_count = {}
        for mid in all_movie_ids:
            movie_count[mid] = movie_count.get(mid, 0) + 1
        
        # Display side by side
        cols = st.columns(4)
        for idx, (name, movie_ids) in enumerate(comparison_results.items()):
            with cols[idx]:
                st.markdown(f"**{name}**")
                if len(movie_ids) == 0:
                    st.caption("Không có gợi ý")
                else:
                    for mid in movie_ids:
                        movie = movies[movies['movieId'] == mid]
                        if len(movie) > 0:
                            title = movie.iloc[0]['title']
                            count = movie_count.get(mid, 1)
                            # Highlight common movies with color
                            if count >= 3:
                                st.markdown(f'<div style="background: linear-gradient(90deg, #ff6b6b, #ff8787); padding: 8px; border-radius: 5px; margin: 3px 0;"><b>🔥 {title}</b><br/><small>Được đề xuất bởi {count}/4 mô hình</small></div>', unsafe_allow_html=True)
                            elif count == 2:
                                st.markdown(f'<div style="background: linear-gradient(90deg, #ffd93d, #ffe66d); padding: 8px; border-radius: 5px; margin: 3px 0;"><b>⭐ {title}</b><br/><small>Được đề xuất bởi {count}/4 mô hình</small></div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div style="background: #e9ecef; padding: 8px; border-radius: 5px; margin: 3px 0; color: #495057;"><b>🔹 {title}</b><br/><small>Được đề xuất bởi 1/4 mô hình</small></div>', unsafe_allow_html=True)
        
        # Legend
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('🔥 **3+ mô hình** cùng gợi ý (đồng thuận cao)')
        with col2:
            st.markdown('⭐ **2 mô hình** cùng gợi ý (đồng thuận trung bình)')
        with col3:
            st.markdown('🔹 **1 mô hình** gợi ý (đồng thuận thấp)')
            
# ========== TAB 2: MOVIE EXPLORER ========== 
with tab2:
    st.header("🎞️ Khám Phá Phim")

    # --- Phần 1: Tìm kiếm phim ---
    st.subheader("🔎 Tìm Kiếm Phim")
    search_query = st.text_input("Nhập tên phim để tìm kiếm", "")
    filtered_movies = movies[movies['title'].str.contains(search_query, case=False, na=False)] if search_query else movies.head(0)
    selected_movie = None
    if not filtered_movies.empty:
        movie_titles = filtered_movies['title'].tolist()
        selected_title = st.selectbox("Chọn phim", movie_titles)
        selected_movie = filtered_movies[filtered_movies['title'] == selected_title].iloc[0]
        st.markdown(f"**Thể loại:** {selected_movie['genres']}")
        st.markdown(f"**Điểm trung bình:** {selected_movie.get('rating_avg', 0.0):.2f}")
        st.markdown(f"**Tổng lượt đánh giá:** {selected_movie.get('rating_count', 0)}")

    # --- Phần 2: Phim tương tự ---
    st.subheader("🎬 Phim Tương Tự (Content-Based)")
    if selected_movie is not None:
        with st.spinner("Đang tìm phim tương tự..."):
            try:
                recs_df = content_rec.recommend_multi([selected_movie['movieId']], n=10, verbose=True)
                if recs_df is not None and not recs_df.empty:
                    recs_df = recs_df[recs_df['movieId'] != selected_movie['movieId']]
                    # Đảm bảo các cột đúng tên, lấy avg_similarity nếu có, nếu không lấy similarity_score
                    similarity_col = 'avg_similarity' if 'avg_similarity' in recs_df.columns else ('similarity_score' if 'similarity_score' in recs_df.columns else None)
                    show_cols = ['title', 'genres', similarity_col, 'rating_avg', 'rating_count'] if similarity_col else ['title', 'genres', 'rating_avg', 'rating_count']
                    recs_df = recs_df[show_cols]
                    recs_df = recs_df.rename(columns={
                        'title': 'Tên Phim',
                        'genres': 'Thể Loại',
                        'avg_similarity': 'Điểm Tương Đồng',
                        'similarity_score': 'Điểm Tương Đồng',
                        'rating_avg': 'Đánh Giá TB',
                        'rating_count': 'Số Đánh Giá'
                    })
                    st.dataframe(recs_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Không tìm thấy phim tương tự.")
            except Exception as e:
                st.error(f"Lỗi khi tìm phim tương tự: {str(e)}")
    else:
        st.info("Hãy tìm và chọn một phim để xem gợi ý tương tự.")

    
    # --- Phần 2: Top phim được đánh giá cao

    st.subheader("🏆 Top Phim Được Đánh Giá Cao ")
    if 'rating_count' in movies.columns:
        top_movies = movies[movies['rating_count'] >= 100].sort_values('rating_avg', ascending=False).head(10)
        top_movies = top_movies[['title', 'genres', 'rating_avg', 'rating_count']]
        top_movies = top_movies.rename(columns={
            'title': 'Tên Phim',
            'genres': 'Thể Loại',
            'rating_avg': 'Đánh Giá TB',
            'rating_count': 'Số Đánh Giá'
        })
        st.dataframe(top_movies, use_container_width=True, hide_index=True)
    else:
        st.info("Không có dữ liệu số lượt đánh giá phim.")

    # --- Phần 3: Thống kê phim ---
    st.subheader("📊 Thống Kê Phim")
    col1, col2 = st.columns(2)
    with col1:
        # Phân bố thể loại - Pie chart
        genre_counts = {}
        for genres_str in movies['genres'].dropna():
            for genre in genres_str.split('|'):
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
        genre_df = pd.DataFrame(list(genre_counts.items()), columns=['Thể Loại', 'Số Lượng'])
        genre_df = genre_df.sort_values('Số Lượng', ascending=False)
        fig_genre = px.pie(genre_df, names='Thể Loại', values='Số Lượng', title='Phân Bố Thể Loại Phim', hole=0.3)
        st.plotly_chart(fig_genre, use_container_width=True)
    with col2:
        # Phân bố rating - Histogram with density curve
        if 'rating_avg' in movies.columns:
            fig_rating = px.histogram(
                movies,
                x='rating_avg',
                nbins=20,
                title='Phân Bố Điểm Trung Bình Phim',
                labels={'rating_avg': 'Điểm Trung Bình'},
                marginal='violin',
                histnorm='probability density'
            )
            fig_rating.update_traces(marker_color='#1976d2', opacity=0.7)
            fig_rating.update_layout(yaxis_title='Mật Độ Xác Suất')
            st.plotly_chart(fig_rating, use_container_width=True)
        else:
            st.info("Không có dữ liệu điểm trung bình phim.")

    # Violin plot điểm trung bình theo từng thể loại
    if 'genres' in movies.columns and 'rating_avg' in movies.columns:
        # Tách từng thể loại thành từng dòng
        genre_ratings = []
        for _, row in movies.iterrows():
            if pd.notna(row['genres']) and not pd.isna(row['rating_avg']):
                for genre in row['genres'].split('|'):
                    genre_ratings.append({'Thể Loại': genre, 'Đánh Giá TB': row['rating_avg']})
        genre_ratings_df = pd.DataFrame(genre_ratings)
        if not genre_ratings_df.empty:
            fig_violin = px.violin(
                genre_ratings_df,
                x='Thể Loại',
                y='Đánh Giá TB',
                box=True,
                points='all',
                title='Điểm Trung Bình Theo Thể Loại',
                color='Thể Loại',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_violin.update_layout(showlegend=False)
            st.plotly_chart(fig_violin, use_container_width=True)
        else:
            st.info("Không có dữ liệu để vẽ violin plot theo thể loại.")
    else:
        st.info("Không có dữ liệu thể loại hoặc điểm trung bình.")

    # Số lượng phim theo từng thập kỷ - Line chart
    if 'decade' in movies.columns:
        decade_counts = movies['decade'].value_counts().sort_index()
        fig_decade = px.bar(
            x=decade_counts.index.astype(str),
            y=decade_counts.values,
            labels={'x': 'Thập Kỷ', 'y': 'Số Lượng Phim'},
            title='Số Lượng Phim Theo Thập Kỷ',
            color=decade_counts.values,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_decade, use_container_width=True)
    else:
        st.info("Không có dữ liệu thập kỷ.")

# ========== TAB 3: USER PROFILE ==========
with tab3:
    st.header(f"Hồ Sơ Người Dùng: #{user_id}")
    
    if len(user_history) == 0:
        st.warning("⚠️ Người dùng này chưa có đánh giá nào trong dữ liệu.")
        st.stop()
    
    # User statistics
    st.subheader("📊 Thống Kê Người Dùng")
    st.markdown("""
    <style>
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
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f'''<div class="user-metric-card"><div class="user-metric-title">Tổng Đánh Giá</div><div class="user-metric-value">{len(user_history)}</div></div>''', unsafe_allow_html=True)

    with col2:
        st.markdown(f'''<div class="user-metric-card"><div class="user-metric-title">Đánh Giá Trung Bình</div><div class="user-metric-value">{user_history['rating'].mean():.2f}</div></div>''', unsafe_allow_html=True)

    with col3:
        genre_counts = {}
        for _, row in user_history.iterrows():
            movie = movies[movies['movieId'] == row['movieId']]
            if len(movie) > 0 and pd.notna(movie.iloc[0]['genres']):
                for genre in movie.iloc[0]['genres'].split('|'):
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
        most_genre = max(genre_counts, key=genre_counts.get) if genre_counts else "N/A"
        st.markdown(f'''<div class="user-metric-card"><div class="user-metric-title">Thể Loại Yêu Thích</div><div class="user-metric-value">{most_genre}</div></div>''', unsafe_allow_html=True)

    with col4:
        n_ratings = len(user_history)
        if n_ratings < 5:
            activity = "thấp"
        elif n_ratings < 20:
            activity = "trung bình"
        else:
            activity = "cao"
        st.markdown(f'''<div class="user-metric-card"><div class="user-metric-title">Mức Hoạt Động</div><div class="user-metric-value">{activity}</div></div>''', unsafe_allow_html=True)
    
    # Rating behavior
    st.subheader("📈 Hành Vi Đánh Giá")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rating distribution
        rating_dist = user_history['rating'].value_counts().sort_index()
        fig_rating = px.bar(
            x=rating_dist.index,
            y=rating_dist.values,
            labels={'x': 'Đánh Giá', 'y': 'Số Lượng'},
            title='Phân Bố Đánh Giá'
        )
        fig_rating.update_layout(showlegend=False)
        st.plotly_chart(fig_rating, use_container_width=True)
    
    with col2:
        # Genre breakdown
        genre_data = pd.DataFrame(list(genre_counts.items()), columns=['Thể Loại', 'Số Lượng'])
        genre_data = genre_data.sort_values('Số Lượng', ascending=False).head(10)
        
        fig_genre = px.pie(
            genre_data,
            names='Thể Loại',
            values='Số Lượng',
            title='Top 10 Thể Loại'
        )
        st.plotly_chart(fig_genre, use_container_width=True)
    
    # Timeline
    if 'timestamp' in user_history.columns:
        st.subheader("⏱️ Dòng Thời Gian Đánh Giá")
        
        timeline = user_history.copy()
        timeline['date'] = pd.to_datetime(timeline['timestamp'], unit='s')
        timeline = timeline.sort_values('date')
        timeline['cumulative'] = range(1, len(timeline) + 1)
        
        fig_timeline = px.line(
            timeline,
            x='date',
            y='cumulative',
            title='Tích Lũy Đánh Giá Theo Thời Gian',
            labels={'date': 'Ngày', 'cumulative': 'Tổng Đánh Giá'}
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Favorite movies
    st.subheader("⭐ Phim Được Người Dùng Đánh Giá Cao Nhất")
    
    top_rated = user_history.sort_values('rating', ascending=False).head(10)
    top_rated_display = []
    
    for _, row in top_rated.iterrows():
        movie = movies[movies['movieId'] == row['movieId']]
        if len(movie) > 0:
            movie = movie.iloc[0]
            top_rated_display.append({
                'Tên Phim': movie['title'],
                'Thể Loại': movie['genres'],
                'Đánh Giá Của Bạn': row['rating'],
                'Đánh Giá TB': movie.get('rating_avg', 0.0)
            })
    
    st.dataframe(
        pd.DataFrame(top_rated_display),
        use_container_width=True,
        hide_index=True
    )
    
    # --- Độ tương đồng người dùng ---
    st.subheader("🤝 Độ Tương Đồng Người Dùng")
    # Tạo dict userId -> set(movieId) để truy cập nhanh
    user_movie_map = ratings.groupby('userId')['movieId'].apply(set).to_dict()
    user_movies = user_movie_map.get(user_id, set())
    overlap_scores = []
    for uid, other_movies in user_movie_map.items():
        if uid == user_id:
            continue
        if not user_movies or not other_movies:
            continue
        overlap = len(user_movies & other_movies)
        union = len(user_movies | other_movies)
        percent = (overlap / union) * 100 if union > 0 else 0
        overlap_scores.append((uid, overlap, percent))
    # Lấy top-5 user giống nhất
    top_similar = sorted(overlap_scores, key=lambda x: (-x[1], -x[2]))[:5]
    if top_similar:
        sim_rows = []
        user_info_map = users.set_index('userId').to_dict('index')
        for uid, overlap, percent in top_similar:
            uinfo = user_info_map.get(uid, {})
            age = uinfo.get('age', 'N/A')
            gender = uinfo.get('gender', 'N/A')
            sim_rows.append({
                'User ID': uid,
                'Số phim chung': overlap,
                '% Trùng lặp': f"{percent:.1f}",
                'Tuổi': age,
                'Giới tính': gender
            })
        st.dataframe(
            pd.DataFrame(sim_rows),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Không tìm thấy user nào có phim chung với user này.")


# ========== TAB 4: Tương tác real-time ========== 



# ========== TAB 5: MODEL PERFORMANCE ========== 
with tab5:
    st.header("📊 Chỉ Số Hiệu Suất Mô Hình")
    if eval_results is None:
        st.warning("⚠️ Không tìm thấy kết quả đánh giá. Vui lòng chạy `run_ranking_evaluation.py` trước.")
        st.stop()
    # Overall metrics table
    st.subheader("📋 Kết Quả Đánh Giá")
    
    
    def highlight_best(s):
        """Highlight giá trị tốt nhất mỗi cột"""
        if s.name in ['Precision@K', 'Recall@K', 'NDCG@K', 'MAP@K', 'ARR']:
            is_max = s == s.max()
            return ['background-color: #90EE90' if v else '' for v in is_max]
        return ['' for _ in s]

    styled_df = eval_results.style.apply(highlight_best, axis=0)
    
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Precision@K': st.column_config.NumberColumn(format="%.4f"),
            'Recall@K': st.column_config.NumberColumn(format="%.4f"),
            'ARR': st.column_config.NumberColumn(format="%.2f"),
            'NDCG@K': st.column_config.NumberColumn(format="%.4f"),
            'MAP@K': st.column_config.NumberColumn(format="%.4f")
        }
    )
    
    
    # Visual comparisons
    st.subheader("📈 So Sánh Trực Quan")
    col1, col2 = st.columns(2)
    with col1:
        # Precision@K
        fig_prec = px.line(
            eval_results,
            x='K',
            y='Precision@K',
            color='model',
            markers=True,
            title='So Sánh Precision@K'
        )
        st.plotly_chart(fig_prec, use_container_width=True)
        # NDCG@K
        fig_ndcg = px.line(
            eval_results,
            x='K',
            y='NDCG@K',
            color='model',
            markers=True,
            title='So Sánh NDCG@K'
        )
        st.plotly_chart(fig_ndcg, use_container_width=True)
    with col2:
        # Recall@K
        fig_rec = px.line(
            eval_results,
            x='K',
            y='Recall@K',
            color='model',
            markers=True,
            title='So Sánh Recall@K'
        )
        st.plotly_chart(fig_rec, use_container_width=True)
        # MAP@K
        fig_map = px.bar(
            eval_results,
            x='model',
            y='MAP@K',
            color='K',
            barmode='group',
            title='MAP@K Theo Mô Hình'
        )
        st.plotly_chart(fig_map, use_container_width=True)
    
    
    # Model details
    with st.expander("🔍 Chi Tiết Mô Hình"):
        st.markdown("""
        **SVD (Singular Value Decomposition)**
        - Phương pháp phân rã ma trận
        - Học các yếu tố tiềm ẩn cho người dùng và phim
        - Nhanh và hiệu quả cho dữ liệu thưa
        
        **NCF (Neural Collaborative Filtering)**
        - Phương pháp deep learning
        - Kiến trúc multi-layer perceptron
        - Nắm bắt tương tác phi tuyến giữa người dùng và phim
        
        **Content-Based**
        - TF-IDF trên thể loại phim
        - Khớp độ tương đồng thể loại
        - Tốt cho phim mới (cold-start)
        
        **Hybrid-Smooth**
        - Phân bổ trọng số động (sigmoid)
        - Kết hợp cả ba mô hình
        - Thích ứng theo mức hoạt động người dùng
        - Công thức: w_content + w_svd + w_ncf = 1.0
        """)
    # Weight tuning visualization
    st.subheader("⚖️ Weight Tuning ")

    # Ảnh 1
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(
            'figures/weight_curves.png',
            caption="📈 Đường phân bổ trọng số Hybrid (w_content, w_svd, w_ncf)",
            use_container_width=True
        )

    st.subheader("📊 So Sánh Smooth vs Hard")
    
    # Ảnh 2
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(
            'figures/smooth_vs_hard_comparison.png',
            caption="⚡ So sánh hiệu quả khi dùng Smooth weights vs Hard weights",
            use_container_width=True
        )





    
    

# ========== TAB 6: ADVANCED ANALYSIS ========== 
with tab6:
    st.header("🔍 Phân Tích Nâng Cao")
    
    # ===== PHẦN 1: CHỈ SỐ ĐA DẠNG =====
    st.subheader(f"📊 Chỉ Số Đa Dạng Trong Top-{top_n} Gợi Ý")
    
    # Generate recommendations for analysis
    with st.spinner("Đang tạo gợi ý để phân tích..."):
        try:
            # Get recommendations from all models
            
            model_recs_analysis = {}
            
            for name, model in model_map.items():
                try:
                    if name == "Content-Based":
                        liked_movies = user_history[user_history['rating'] >= 4.0]['movieId'].values
                        if len(liked_movies) == 0:
                            model_recs_analysis[name] = []
                            continue
                        seed_items = list(liked_movies)[:5]
                        recs_df = model.recommend_multi(seed_items, n=top_n, verbose=False)
                        if recs_df is not None and not recs_df.empty:
                            model_recs_analysis[name] = recs_df['movieId'].values.tolist()
                        else:
                            model_recs_analysis[name] = []
                    else:
                        recs = model.recommend(user_id, n=top_n, exclude_rated=True)
                        if isinstance(recs, pd.DataFrame):
                            model_recs_analysis[name] = recs['movieId'].values.tolist()
                        else:
                            model_recs_analysis[name] = (np.array(recs) if not isinstance(recs, np.ndarray) else recs).tolist()
                except:
                    model_recs_analysis[name] = []
            
            # Calculate diversity metrics
            diversity_data = []
            
            for name, rec_list in model_recs_analysis.items():
                if len(rec_list) == 0:
                    continue
                
                # Get genres for recommended movies
                rec_genres = []
                unique_genres = set()
                for mid in rec_list:
                    movie = movies[movies['movieId'] == mid]
                    if len(movie) > 0 and pd.notna(movie.iloc[0]['genres']):
                        genres = movie.iloc[0]['genres'].split('|')
                        rec_genres.extend(genres)
                        unique_genres.update(genres)
                
                # Genre coverage
                total_genres = len(all_genres)
                genre_coverage = len(unique_genres) / total_genres if total_genres > 0 else 0
                
                # Intra-list diversity (genre entropy)
                from collections import Counter
                genre_counts = Counter(rec_genres)
                total = sum(genre_counts.values())
                entropy = 0
                if total > 0:
                    for count in genre_counts.values():
                        p = count / total
                        if p > 0:
                            entropy -= p * np.log2(p)
                
                # Novelty (% movies not in user history)
                user_rated = set(user_history['movieId'].values)
                novel_count = sum(1 for mid in rec_list if mid not in user_rated)
                novelty = novel_count / len(rec_list) if len(rec_list) > 0 else 0
                
                diversity_data.append({
                    'Mô Hình': name,
                    'Độ Phủ Thể Loại': f"{genre_coverage:.2%}",
                    'Entropy Thể Loại': f"{entropy:.2f}",
                    'Độ Mới Lạ': f"{novelty:.2%}",
                    'Số Thể Loại khác nhau': len(unique_genres)
                })
            
            if diversity_data:
                diversity_df = pd.DataFrame(diversity_data)
                st.dataframe(diversity_df, use_container_width=True, hide_index=True)
                
                st.info("""
                **Giải thích:**
                - **Độ Phủ Thể Loại**: % thể loại được đại diện trong gợi ý
                - **Entropy Thể Loại**: Đo độ đa dạng (càng cao càng đa dạng, max ≈ 3.5)
                - **Độ Mới Lạ**: % phim chưa được user đánh giá
                - **Số Thể Loại khác nhau**: Tổng số thể loại khác nhau trong gợi ý
                """)
            else:
                st.warning("Không có dữ liệu gợi ý để phân tích")
                
        except Exception as e:
            st.error(f"Lỗi khi phân tích đa dạng: {str(e)}")
    
    
    # ===== PHẦN 3: Phân Bố Thể Loại Trong Gợi Ý =====
    st.markdown("---")
    st.subheader(f"🎭 Phân Bố Thể Loại Trong Top-{top_n} Gợi Ý")
    st.caption(
        "Phân tích mang tính mô tả: đo lường mức độ xuất hiện của thể loại "
        "trong danh sách phim được mô hình đề xuất cho một người dùng cụ thể."
    )

    # Select genre to analyze
    selected_genre_analysis = st.selectbox(
        "Chọn thể loại để phân tích",
        options=sorted(all_genres),
        key="genre_analysis"
    )
    
    # Analyze genre performance
    with st.spinner(f"Đang phân tích phân bố thể loại trong gợi ý {selected_genre_analysis}..."):
        try:
            genre_movies = movies[movies['genres'].str.contains(selected_genre_analysis, na=False)]
            genre_movie_ids = set(genre_movies['movieId'].values)
            
            genre_performance = []
            
            for name, rec_list in model_recs_analysis.items():
                if len(rec_list) == 0:
                    continue
                
                # Count genre movies in recommendations
                genre_count = sum(1 for mid in rec_list if mid in genre_movie_ids)
                genre_ratio = genre_count / len(rec_list) if len(rec_list) > 0 else 0
                
                # Average rating of genre movies in recommendations
                genre_rec_ratings = []
                for mid in rec_list:
                    if mid in genre_movie_ids:
                        movie = movies[movies['movieId'] == mid]
                        if len(movie) > 0 and 'rating_avg' in movie.columns:
                            genre_rec_ratings.append(movie.iloc[0]['rating_avg'])
                
                avg_rating = np.mean(genre_rec_ratings) if genre_rec_ratings else 0
                
                genre_performance.append({
                    'Mô Hình': name,
                    f'Số Phim {selected_genre_analysis}': genre_count,
                    '% Trong Gợi Ý': f"{genre_ratio:.1%}",
                    'Đánh Giá TB': f"{avg_rating:.2f}"
                })
            
            if genre_performance:
                genre_perf_df = pd.DataFrame(genre_performance)
                st.dataframe(genre_perf_df, use_container_width=True, hide_index=True)
                
                # Visualization
                fig_genre_perf = px.bar(
                    genre_perf_df,
                    x='Mô Hình',
                    y=f'Số Phim {selected_genre_analysis}',
                    title=f'Số Phim {selected_genre_analysis} Trong Gợi Ý',
                    color='Mô Hình',
                    text=f'Số Phim {selected_genre_analysis}'
                )
                fig_genre_perf.update_traces(textposition='outside')
                st.plotly_chart(fig_genre_perf, use_container_width=True)
            else:
                st.info("Không có dữ liệu hiệu năng theo thể loại")
                
        except Exception as e:
            st.error(f"Lỗi khi phân tích theo thể loại: {str(e)}")
    
    # ===== PHẦN 4: PLAYGROUND TƯƠNG TÁC =====
    st.markdown("---")
    st.subheader("🎮 Playground Tương Tác")
    
    st.markdown("**Điều chỉnh trọng số Hybrid thủ công và xem kết quả:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        w_content_custom = st.slider(
            "Trọng số Content-Based",
            min_value=0.0,
            max_value=1.0,
            value=0.33,
            step=0.05,
            key="w_content"
        )
    
    with col2:
        w_svd_custom = st.slider(
            "Trọng số SVD",
            min_value=0.0,
            max_value=1.0,
            value=0.33,
            step=0.05,
            key="w_svd"
        )
    
    with col3:
        w_ncf_custom = st.slider(
            "Trọng số NCF",
            min_value=0.0,
            max_value=1.0,
            value=0.34,
            step=0.05,
            key="w_ncf"
        )
    
    # Normalize weights
    total_weight = w_content_custom + w_svd_custom + w_ncf_custom
    
    if total_weight > 0:
        w_content_norm = w_content_custom / total_weight
        w_svd_norm = w_svd_custom / total_weight
        w_ncf_norm = w_ncf_custom / total_weight
        
        st.info(f"""
        **Trọng số sau chuẩn hóa (tổng = 1.0):**
        - Content-Based: {w_content_norm:.2%}
        - SVD: {w_svd_norm:.2%}
        - NCF: {w_ncf_norm:.2%}
        """)
        
        # Generate custom hybrid recommendations
        if st.button("🚀 Tạo Gợi Ý Với Trọng Số Tùy Chỉnh", type="primary"):
            with st.spinner("Đang tạo gợi ý với trọng số tùy chỉnh..."):
                try:
                    # Get auto weights first
                    n_ratings = len(user_history)
                    auto_weights = hybrid_rec.calculate_smooth_weights(n_ratings, method='sigmoid')
                    
                    # Get predictions from each model manually
                    custom_scores = {}
                    
                    # Content-Based scores
                    try:
                        if n_ratings > 0:
                            liked_movies = user_history[user_history['rating'] >= 4.0]['movieId'].values
                            if len(liked_movies) > 0:
                                seed_items = list(liked_movies)[:5]
                                cb_recs = content_rec.recommend_multi(seed_items, n=50, verbose=False)
                                if cb_recs is not None and not cb_recs.empty:
                                    for _, row in cb_recs.iterrows():
                                        mid = row['movieId']
                                        score = row.get('avg_similarity', row.get('similarity_score', 0))
                                        if mid not in custom_scores:
                                            custom_scores[mid] = {'content': 0, 'svd': 0, 'ncf': 0}
                                        custom_scores[mid]['content'] = float(score) * 5  # Scale to 1-5
                    except Exception as e:
                        st.warning(f"Content-Based không khả dụng: {e}")
                    
                    # SVD scores
                    try:
                        svd_recs = svd_rec.recommend(user_id, n=50, exclude_rated=False)
                        if svd_recs is not None and not svd_recs.empty:
                            for _, row in svd_recs.iterrows():
                                mid = row['movieId']
                                score = row.get('predicted_rating', 3.0)
                                if mid not in custom_scores:
                                    custom_scores[mid] = {'content': 0, 'svd': 0, 'ncf': 0}
                                custom_scores[mid]['svd'] = float(score)
                    except:
                        pass
                    
                    # NCF scores
                    try:
                        ncf_recs = ncf_rec.recommend(user_id, n=50, exclude_rated=False, return_details=True)
                        if ncf_recs is not None and not ncf_recs.empty:
                            for _, row in ncf_recs.iterrows():
                                mid = row['movieId']
                                score = row.get('predicted_rating', 3.0)
                                if mid not in custom_scores:
                                    custom_scores[mid] = {'content': 0, 'svd': 0, 'ncf': 0}
                                custom_scores[mid]['ncf'] = float(score)
                    except:
                        pass
                    
                    # Calculate weighted scores
                    custom_results = []
                    user_rated = set(user_history['movieId'].values)
                    
                    for mid, scores in custom_scores.items():
                        if mid in user_rated:
                            continue
                        
                        # Custom weighted score
                        custom_score = (
                            w_content_norm * scores['content'] +
                            w_svd_norm * scores['svd'] +
                            w_ncf_norm * scores['ncf']
                        )
                        
                        movie = movies[movies['movieId'] == mid]
                        if len(movie) > 0:
                            movie = movie.iloc[0]
                            custom_results.append({
                                'movieId': mid,
                                'title': movie['title'],
                                'genres': movie['genres'],
                                'score': custom_score
                            })
                    
                    # Sort and get top 10
                    custom_results = sorted(custom_results, key=lambda x: x['score'], reverse=True)[:10]
                    
                    # Get auto recommendations
                    auto_recs = hybrid_rec.recommend(user_id, n=10, exclude_rated=True)
                    
                    # Display comparison
                    st.markdown("---")
                    col_custom, col_auto = st.columns(2)
                    
                    with col_custom:
                        st.markdown("**🎨 Gợi Ý Với Trọng Số Tùy Chỉnh**")
                        st.caption(f"Content: {w_content_norm:.1%} | SVD: {w_svd_norm:.1%} | NCF: {w_ncf_norm:.1%}")
                        
                        if custom_results:
                            custom_display = []
                            for i, item in enumerate(custom_results, 1):
                                custom_display.append({
                                    '#': i,
                                    'Tên Phim': item['title'],
                                    'Thể Loại': item['genres'],
                                    'Điểm': f"{item['score']:.2f}"
                                })
                            st.dataframe(
                                pd.DataFrame(custom_display),
                                use_container_width=True,
                                hide_index=True
                            )
                        else:
                            st.warning("Không tạo được gợi ý")
                    
                    with col_auto:
                        st.markdown("**🤖 Gợi Ý Với Trọng Số Tự Động**")
                        st.caption(f"Content: {auto_weights['content']:.1%} | SVD: {auto_weights['svd']:.1%} | NCF: {auto_weights['ncf']:.1%}")
                        
                        if not auto_recs.empty:
                            auto_display = []
                            for _, row in auto_recs.iterrows():
                                auto_display.append({
                                    '#': int(row['rank']),
                                    'Tên Phim': row['title_clean'],
                                    'Thể Loại': row['genres'],
                                    'Điểm': f"{row['predicted_rating']:.2f}"
                                })
                            st.dataframe(
                                pd.DataFrame(auto_display),
                                use_container_width=True,
                                hide_index=True
                            )
                    
                    # Calculate overlap
                    if custom_results:
                        custom_ids = {item['movieId'] for item in custom_results}
                        auto_ids = set(auto_recs['movieId'].values) if not auto_recs.empty else set()
                        overlap = len(custom_ids & auto_ids)
                        st.success(f"**Số phim trùng khớp:** {overlap}/10 ({overlap*10}%)")
                    
                except Exception as e:
                    st.error(f"Lỗi khi tạo gợi ý tùy chỉnh: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.warning("⚠️ Tổng trọng số phải > 0")
        
    
    
# ========== TAB 7: EDA INSIGHTS ========== 
with tab7:
    st.header("📈 Insight Dữ Liệu (EDA)")
    
    # ===== PHẦN 1: TỔNG QUAN DATASET =====
    st.subheader("📊 Tổng Quan Dataset")
    
    st.markdown("""
    <style>
    .insight-card {
        background: #fff;
        border: 1.5px solid #e0e0e0;
        border-radius: 10px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        padding: 1.1rem 0.5rem 0.7rem 0.5rem;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .insight-title {
        color: #22223b;
        font-weight: 600;
        font-size: 1.08rem;
        margin-bottom: 0.18rem;
    }
    .insight-value {
        color: #1976d2;
        font-size: 2.05rem;
        font-weight: bold;
    }
    
    .insight-removed-value {
        color: red;  
        font-size: 1rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("##### 📋 Thống Kê Dataset Gốc")
        try:
            dataset_stats = pd.read_csv('data/raw/dataset_stats.csv')
            
            # Parse stats dynamically
            stats_dict = dataset_stats.iloc[0].to_dict()
            
            # Display as metrics
            metric_cols = st.columns(3)
            
            with metric_cols[0]:
                total_movies = int(stats_dict.get('total_movies', 0))
                st.markdown(f"<div class='insight-card'><div class='insight-title'>Tổng Số Phim</div><div class='insight-value'>{int(stats_dict.get('total_movies', 0)):,}</div></div>", unsafe_allow_html=True)
            
            with metric_cols[1]:
                total_ratings = int(stats_dict.get('total_ratings', 0))
                st.markdown(f"<div class='insight-card'><div class='insight-title'>Tổng Đánh Giá</div><div class='insight-value'>{int(stats_dict.get('total_ratings', 0)):,}</div></div>", unsafe_allow_html=True)
            
            with metric_cols[2]:
                total_users = int(stats_dict.get('total_users', 0))
                st.markdown(f"<div class='insight-card'><div class='insight-title'>Tổng Users</div><div class='insight-value'>{total_users:,}</div></div>", unsafe_allow_html=True)
            
            # Additional metrics
            metric_cols2 = st.columns(2)
            with metric_cols2[0]:
                avg_rating = float(stats_dict.get('avg_rating', 0))
                st.markdown(f"<div class='insight-card'><div class='insight-title'>Rating Trung Bình</div><div class='insight-value'>{float(stats_dict.get('avg_rating', 0)):.2f}⭐</div></div>", unsafe_allow_html=True)
            
            with metric_cols2[1]:
                total_features = int(stats_dict.get('total_features', 0))
                st.markdown(f"<div class='insight-card'><div class='insight-title'>Số Features</div><div class='insight-value'>{total_features}</div></div>", unsafe_allow_html=True)
            
            # Calculate derived metrics
            if total_users > 0 and total_movies > 0:
                sparsity = 1 - (total_ratings / (total_users * total_movies))
                st.markdown(f"<div class='insight-card'><div class='insight-title'>Độ thưa ma trận</div><div class='insight-value'>{sparsity:.2%}</div></div>", unsafe_allow_html=True)
                
        except FileNotFoundError:
            st.warning("⚠️ Không tìm thấy file `data/raw/dataset_stats.csv`")
        except Exception as e:
            st.error(f"Lỗi khi đọc dataset stats: {e}")
    
    with col2:
        st.markdown("##### 🧹 Báo Cáo Làm Sạch Dữ Liệu")
        try:
            cleaning_report = pd.read_csv('data/cleaned/cleaning_report.csv')
            
            # Parse cleaning report dynamically
            report_dict = cleaning_report.iloc[0].to_dict()
            
            # Before/After comparison
            col_before, col_after = st.columns(2)
            
            with col_before:
                st.markdown(f"<div class='insight-card'><div class='insight-title'>📥 Trước Cleaning</div>"
                        f"<div>Phim: {int(report_dict.get('original_movies',0)):,}</div>"
                        f"<div>Ratings: {int(report_dict.get('original_ratings',0)):,}</div>"
                        f"<div>Users: {int(report_dict.get('original_users',0)):,}</div>"
                        f"</div>", unsafe_allow_html=True)
            
            with col_after:
                st.markdown(f"<div class='insight-card'><div class='insight-title'>✅ Sau Cleaning</div>"
                        f"<div>Phim: {int(report_dict.get('cleaned_movies',0)):,}</div>"
                        f"<div>Ratings: {int(report_dict.get('cleaned_ratings',0)):,}</div>"
                        f"<div>Users: {int(report_dict.get('cleaned_users',0)):,}</div>"
                        f"</div>", unsafe_allow_html=True)
            # Removed stats
            st.markdown("**🗑️ Đã Loại Bỏ:**")
            removed_cols = st.columns(3)
            
            with removed_cols[0]:
                st.markdown(f"<div class='insight-card'><div class='insight-title'>Phim Loại Bỏ</div>"
                        f"<div class='insight-value'>{int(report_dict.get('movies_removed',0)):,}</div>"
                        f"<div class='insight-removed-value'>-{float(report_dict.get('movies_removed_pct',0)):.1f}%</div></div>", unsafe_allow_html=True)
            
            with removed_cols[1]:
                st.markdown(f"<div class='insight-card'><div class='insight-title'>Ratings Loại Bỏ</div>"
                        f"<div class='insight-value'>{int(report_dict.get('ratings_removed',0)):,}</div>"
                        f"<div class='insight-removed-value'>-{float(report_dict.get('ratings_removed_pct',0)):.1f}%</div></div>", unsafe_allow_html=True)
            
            with removed_cols[2]:
                st.markdown(f"<div class='insight-card'><div class='insight-title'>Users Loại Bỏ</div>"
                        f"<div class='insight-value'>{int(report_dict.get('users_removed',0)):,}</div>"
                        f"<div class='insight-removed-value'>-{float(report_dict.get('users_removed_pct',0)):.1f}%</div></div>", unsafe_allow_html=True)
            
        except FileNotFoundError:
            st.warning("⚠️ Không tìm thấy file `data/cleaned/cleaning_report.csv`")
        except Exception as e:
            st.error(f"Lỗi khi đọc cleaning report: {e}")
    
    
    
    # ===== PHẦN 2: THƯ VIỆN BIỂU ĐỒ =====
    st.markdown("---")
    st.subheader("📸 Thư Viện Biểu Đồ EDA")
    
    # Define figure files dynamically
    figure_files = [
        ('01_rating_distribution.png', '⭐ Phân Bố Rating'),
        ('02_top_genres.png', '🎭 Top Thể Loại Phổ Biến'),
        ('03_user_movie_heatmap.png', '🔥 Heatmap User-Movie'),
        ('04_top_movies.png', '🏆 Top Phim Được Đánh Giá'),
        ('05_ratings_over_time.png', '📅 Rating Theo Thời Gian'),
        ('06_rating_by_genre.png', '🎬 Rating Theo Thể Loại'),
    ]
    
    # Check which figures exist
    available_figures = []
    figures_path = Path('figures')
    
    for filename, title in figure_files:
        filepath = figures_path / filename
        if filepath.exists():
            available_figures.append((str(filepath), title))
    
    if not available_figures:
        st.warning("⚠️ Không tìm thấy file hình trong thư mục `figures/`")
    else:
        # Display figures in tabs
        figure_tabs = st.tabs([title for _, title in available_figures])
        
        for idx, (filepath, title) in enumerate(available_figures):
            with figure_tabs[idx]:
                try:
                    # Tạo container với width giới hạn
                    col1, col2, col3 = st.columns([1, 3, 1])
                    
                    with col2:
                        st.image(filepath, use_container_width=True)
                    
                    # Add description based on filename
                    if '01_rating' in filepath:
                        st.caption("📊 Phân bố các mức rating từ 1-5 sao. Cho thấy xu hướng rating của users.")
                    elif '02_top_genres' in filepath:
                        st.caption("🎭 Các thể loại phim phổ biến nhất trong dataset.")
                    elif '03_user_movie' in filepath:
                        st.caption("🔥 Độ thưa của ma trận user-movie interaction.")
                    elif '04_top_movies' in filepath:
                        st.caption("🏆 Những phim có nhiều lượt đánh giá nhất.")
                    elif '05_ratings_over' in filepath:
                        st.caption("📅 Xu hướng rating theo thời gian.")
                    elif '06_rating_by' in filepath:
                        st.caption("🎬 So sánh rating trung bình giữa các thể loại.")
                        
                except Exception as e:
                    st.error(f"Không thể hiển thị hình: {e}")
    
    
    
    # EDA Insights text
    
    st.markdown("---")
    st.markdown("##### 💡 Insights Từ Phân Tích EDA")

    try:
        with open('figures/EDA_insights.txt', 'r', encoding='utf-8') as f:
            insights_text = f.read()
        
        # Split by VIZ markers to get sections
        
        viz_pattern = r'(VIZ \d+:.*?)(?=VIZ \d+:|$)'
        viz_sections = re.findall(viz_pattern, insights_text, re.DOTALL)
        
        # Get header (before first VIZ)
        header_match = re.search(r'^(.*?)(?=VIZ \d+:)', insights_text, re.DOTALL)
        if header_match:
            header_text = header_match.group(1).strip()
            if header_text:
                st.markdown(f"""
                <div style="border: 2px solid #2196F3; border-radius: 10px; padding: 15px; 
                            background-color: #E3F2FD; margin-bottom: 15px;">
                {header_text.replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)
        
        # Display each VIZ section in expander
        for viz_section in viz_sections:
            lines = viz_section.strip().split('\n')
            if lines:
                # Extract title (first line)
                title = lines[0].replace('VIZ', '📊 VIZ').strip()
                # Extract content (rest of lines)
                content = '\n'.join(lines[1:]).strip()
                
                if content:
                    with st.expander(title, expanded=False):
                        st.markdown(content)
        
    except FileNotFoundError:
        st.warning("⚠️ Không tìm thấy file `figures/EDA_insights.txt`")
    except Exception as e:
        st.error(f"Lỗi khi đọc insights: {e}")
    
    # ===== PHẦN 3: CHẤT LƯỢNG DỮ LIỆU =====
    st.markdown("---")
    st.subheader("Chất Lượng Dữ Liệu")
    
    try:
        cleaning_report = pd.read_csv('data/cleaned/cleaning_report.csv')
        report_dict = cleaning_report.iloc[0].to_dict()
        
        # Quality metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📊 Thống Kê Sau Cleaning")
            
            # Average ratings per movie/user
            avg_ratings_movie = float(report_dict.get('avg_ratings_per_movie', 0))
            avg_ratings_user = float(report_dict.get('avg_ratings_per_user', 0))
            
           
            st.markdown(f"""
                <div class='insight-card'>
                    <div class='insight-title'>Số lượt rating trung bình mỗi phim</div>
                    <div class='insight-value'>{avg_ratings_movie:.1f}</div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
                <div class='insight-card'>
                    <div class='insight-title'>Số lượt rating trung bình mỗi user</div>
                    <div class='insight-value'>{avg_ratings_user:.1f}</div>
                </div>
            """, unsafe_allow_html=True)
            # Min/Max ratings
            min_ratings_movie = int(report_dict.get('min_ratings_per_movie', 0))
            max_ratings_movie = int(report_dict.get('max_ratings_per_movie', 0))
            min_ratings_user = int(report_dict.get('min_ratings_per_user', 0))
            max_ratings_user = int(report_dict.get('max_ratings_per_user', 0))
            
            st.markdown(f"""
                <div class='insight-card'>
                    <div class='insight-title'>Range ratings/phim</div>
                    <div class='insight-value'>{min_ratings_movie} - {max_ratings_movie:,}</div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
                <div class='insight-card'>
                    <div class='insight-title'>Range ratings/user</div>
                    <div class='insight-value'>{min_ratings_user} - {max_ratings_user:,}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("##### 🔢 Feature Engineering")
            
            # TF-IDF features
            tfidf_features = int(report_dict.get('tfidf_features', 0))
            tfidf_sparsity = float(report_dict.get('tfidf_sparsity', 0))
            
           
            st.markdown(f"""
                <div class='insight-card'>
                    <div class='insight-title'>Số TF-IDF Features</div>
                    <div class='insight-value'>{tfidf_features:,}</div>
                </div>
            """, unsafe_allow_html=True)
           
            st.markdown(f"""
                <div class='insight-card'>
                    <div class='insight-title'>TF-IDF Sparsity</div>
                    <div class='insight-value'>{tfidf_sparsity:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
            
            
            # Quality score
            quality_score = 100 - float(report_dict.get('movies_removed_pct', 0))
            st.markdown(f"""
                <div class='insight-card'>
                    <div class='insight-title'>Chất Lượng Dữ Liệu</div>
                    <div class='insight-value'>{quality_score:.1f}%</div>
                    <div style="font-size:0.9rem; color: {"#08611A" if quality_score > 90 else "#07BF38"};">
                        {"Excellent" if quality_score > 90 else "Good"}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        
        # Data quality summary
        st.markdown("##### 🎯 Tóm Tắt Chất Lượng")
        
        summary_items = []
        
        # Check for missing values
        if report_dict.get('ratings_removed', 0) == 0:
            summary_items.append("✅ Không có missing values trong ratings")
        
        # Check for removed movies
        if report_dict.get('movies_removed', 0) > 0:
            summary_items.append(f"🗑️ Đã loại bỏ {int(report_dict['movies_removed'])} phim có ít tương tác")
        
        # Check sparsity
        original_movies = int(report_dict.get('original_movies', 1))
        original_users = int(report_dict.get('original_users', 1))
        original_ratings = int(report_dict.get('original_ratings', 0))
        sparsity = 1 - (original_ratings / (original_movies * original_users))
        
        if sparsity > 0.95:
            summary_items.append(f"⚠️ Ma trận rất thưa ({sparsity:.1%}) - phù hợp cho Collaborative Filtering")
        else:
            summary_items.append(f"✅ Mật độ dữ liệu tốt ({(1-sparsity):.1%})")
        
        # Check average ratings
        if avg_ratings_user >= 100:
            summary_items.append(f"✅ Users có lịch sử tương tác phong phú (TB: {avg_ratings_user:.0f} ratings/user)")
        else:
            summary_items.append(f"⚠️ Users có ít tương tác (TB: {avg_ratings_user:.0f} ratings/user)")
        
        for item in summary_items:
            st.info(item)
        
    except FileNotFoundError:
        st.warning("⚠️ Không tìm thấy cleaning report")
    except Exception as e:
        st.error(f"Lỗi khi phân tích chất lượng: {e}")
    
    # Data quality visualization
    if 'cleaning_report' in locals() and not cleaning_report.empty:
        st.markdown("##### 📊 Trực Quan Hóa Chất Lượng")
        
        # Create quality metrics chart
        quality_metrics = {
            'Metric': ['Phim Giữ Lại', 'Ratings Giữ Lại', 'Users Giữ Lại'],
            'Percentage': [
                100 - float(report_dict.get('movies_removed_pct', 0)),
                100 - float(report_dict.get('ratings_removed_pct', 0)),
                100 - float(report_dict.get('users_removed_pct', 0))
            ]
        }
        
        quality_df = pd.DataFrame(quality_metrics)
        
        fig_quality = px.bar(
            quality_df,
            x='Metric',
            y='Percentage',
            title='Tỷ Lệ Dữ Liệu Giữ Lại Sau Cleaning',
            labels={'Percentage': '% Giữ Lại'},
            color='Percentage',
            color_continuous_scale='Greens',
            text='Percentage'
        )
        fig_quality.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_quality.update_layout(showlegend=False, yaxis_range=[0, 105])
        
        st.plotly_chart(fig_quality, use_container_width=True)


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Hệ Thống Gợi Ý Phim MovieLens 1M | Xây dựng bằng Streamlit"
    "</div>",
    unsafe_allow_html=True
)

