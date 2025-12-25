"""
Tab 2: Khám Phá Phim - Search, discover, and explore movies
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from typing import Any


def render_movie_explorer_tab(
    movies: pd.DataFrame,
    content_rec: Any
):
    """
    Render tab khám phá phim
    
    Args:
        movies: DataFrame phim
        content_rec: Content-based recommender
    """
    st.header("🎞️ Khám Phá Phim")
    
    # PHẦN 1: TÌM KIẾM PHIM
    st.subheader("🔎 Tìm Kiếm Phim")
    search_query = st.text_input("Nhập tên phim để tìm kiếm", "")
    
    filtered_movies = movies[
        movies['title'].str.contains(search_query, case=False, na=False)
    ] if search_query else movies.head(0)
    
    selected_movie = None
    if not filtered_movies.empty:
        movie_titles = filtered_movies['title'].tolist()
        selected_title = st.selectbox("Chọn phim", movie_titles)
        selected_movie = filtered_movies[filtered_movies['title'] == selected_title].iloc[0]
        
        st.markdown(f"**Thể loại:** {selected_movie['genres']}")
        st.markdown(f"**Điểm trung bình:** {selected_movie.get('rating_avg', 0.0):.2f}")
        st.markdown(f"**Tổng lượt đánh giá:** {selected_movie.get('rating_count', 0)}")
    
    # PHẦN 2: PHIM TƯƠNG TỰ
    st.subheader("🎬 Phim Tương Tự (Content-Based)")
    
    if selected_movie is not None:
        with st.spinner("Đang tìm phim tương tự..."):
            try:
                recs_df = content_rec.recommend_multi(
                    [selected_movie['movieId']], 
                    n=10, 
                    verbose=True
                )
                
                if recs_df is not None and not recs_df.empty:
                    recs_df = recs_df[recs_df['movieId'] != selected_movie['movieId']]
                    
                    # Get similarity column
                    similarity_col = 'avg_similarity' if 'avg_similarity' in recs_df.columns else (
                        'similarity_score' if 'similarity_score' in recs_df.columns else None
                    )
                    
                    show_cols = ['title', 'genres', similarity_col, 'rating_avg', 'rating_count'] if similarity_col else [
                        'title', 'genres', 'rating_avg', 'rating_count'
                    ]
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
    
    # PHẦN 3: TOP PHIM ĐƯỢC ĐÁNH GIÁ CAO
    st.subheader("🏆 Top Phim Được Đánh Giá Cao")
    
    if 'rating_count' in movies.columns:
        top_movies = movies[movies['rating_count'] >= 100].sort_values(
            'rating_avg', ascending=False
        ).head(10)
        
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
    
    # PHẦN 4: THỐNG KÊ PHIM
    st.subheader("📊 Thống Kê Phim")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Genre distribution pie chart
        genre_counts = {}
        for genres_str in movies['genres'].dropna():
            for genre in genres_str.split('|'):
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        genre_df = pd.DataFrame(
            list(genre_counts.items()), 
            columns=['Thể Loại', 'Số Lượng']
        )
        genre_df = genre_df.sort_values('Số Lượng', ascending=False)
        
        fig_genre = px.pie(
            genre_df, 
            names='Thể Loại', 
            values='Số Lượng',
            title='Phân Bố Thể Loại Phim',
            hole=0.3
        )
        st.plotly_chart(fig_genre, use_container_width=True)
    
    with col2:
        # Rating distribution histogram
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
    
    # PHẦN 5: VIOLIN PLOT THEO THỂ LOẠI
    if 'genres' in movies.columns and 'rating_avg' in movies.columns:
        genre_ratings = []
        for _, row in movies.iterrows():
            if pd.notna(row['genres']) and not pd.isna(row['rating_avg']):
                for genre in row['genres'].split('|'):
                    genre_ratings.append({
                        'Thể Loại': genre,
                        'Đánh Giá TB': row['rating_avg']
                    })
        
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
    
    # PHẦN 6: SỐ LƯỢNG PHIM THEO THẬP KỶ
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