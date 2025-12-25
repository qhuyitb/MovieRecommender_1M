"""
tabs/tab_user_profile.py
Tab 3: Hồ Sơ Người Dùng - User statistics and behavior analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px


def render_user_profile_tab(
    settings: dict,
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    users: pd.DataFrame
):
    """
    Render tab hồ sơ người dùng
    
    Args:
        settings: Dict từ sidebar
        movies: DataFrame phim
        ratings: DataFrame ratings (full)
        users: DataFrame users
    """
    user_id = settings['user_id']
    user_history = settings['user_history']
    
    st.header(f"Hồ Sơ Người Dùng: #{user_id}")
    
    if len(user_history) == 0:
        st.warning("⚠️ Người dùng này chưa có đánh giá nào trong dữ liệu.")
        st.stop()
    
    # PHẦN 1: THỐNG KÊ NGƯỜI DÙNG
    st.subheader("📊 Thống Kê Người Dùng")
    
    # Get genre counts
    genre_counts = _get_user_genre_counts(user_history, movies)
    most_genre = max(genre_counts, key=genre_counts.get) if genre_counts else "N/A"
    
    # Get activity level
    n_ratings = len(user_history)
    if n_ratings < 5:
        activity = "thấp"
    elif n_ratings < 20:
        activity = "trung bình"
    else:
        activity = "cao"
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f'<div class="user-metric-card">'
            f'<div class="user-metric-title">Tổng Đánh Giá</div>'
            f'<div class="user-metric-value">{len(user_history)}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f'<div class="user-metric-card">'
            f'<div class="user-metric-title">Đánh Giá Trung Bình</div>'
            f'<div class="user-metric-value">{user_history["rating"].mean():.2f}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f'<div class="user-metric-card">'
            f'<div class="user-metric-title">Thể Loại Yêu Thích</div>'
            f'<div class="user-metric-value">{most_genre}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            f'<div class="user-metric-card">'
            f'<div class="user-metric-title">Mức Hoạt Động</div>'
            f'<div class="user-metric-value">{activity}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    # PHẦN 2: HÀNH VI ĐÁNH GIÁ
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
        genre_data = pd.DataFrame(
            list(genre_counts.items()),
            columns=['Thể Loại', 'Số Lượng']
        )
        genre_data = genre_data.sort_values('Số Lượng', ascending=False).head(10)
        
        fig_genre = px.pie(
            genre_data,
            names='Thể Loại',
            values='Số Lượng',
            title='Top 10 Thể Loại'
        )
        st.plotly_chart(fig_genre, use_container_width=True)
    
    # PHẦN 3: TIMELINE
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
    
    # PHẦN 4: PHIM YÊU THÍCH
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
    
    # PHẦN 5: ĐỘ TƯƠNG ĐỒNG NGƯỜI DÙNG
    st.subheader("🤝 Độ Tương Đồng Người Dùng")
    
    # Build user-movie map
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
    
    # Get top 5 similar users
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


def _get_user_genre_counts(user_history: pd.DataFrame, movies: pd.DataFrame) -> dict:
    """Get genre counts from user's rated movies"""
    genre_counts = {}
    
    for _, row in user_history.iterrows():
        movie = movies[movies['movieId'] == row['movieId']]
        if len(movie) > 0 and pd.notna(movie.iloc[0]['genres']):
            for genre in movie.iloc[0]['genres'].split('|'):
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    return genre_counts