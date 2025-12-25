"""
Tab 1: Gợi Ý Phim - Generate và hiển thị recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any

from ui.styles import render_metric_card


def render_recommendations_tab(
    settings: dict,
    model_map: dict,
    movies: pd.DataFrame,
    hybrid_rec: Any
):
    """
    Render tab gợi ý phim
    
    Args:
        settings: Dict từ sidebar (user_id, model_name, top_n, etc.)
        model_map: Dict mapping model names to model objects
        movies: DataFrame phim
        hybrid_rec: Hybrid recommender object (for weight calculation)
    """
    st.header(f"Top {settings['top_n']} Phim Được Gợi Ý")
    
    # Select model
    selected_model = model_map[settings['model_name']]
    user_history = settings['user_history']
    user_id = settings['user_id']
    top_n = settings['top_n']
    model_name = settings['model_name']
    
    # Generate recommendations
    with st.spinner(f"Đang tạo gợi ý từ mô hình {model_name}..."):
        try:
            # Get recommendations based on model type
            recommended_movies = _generate_recommendations(
                model_name=model_name,
                selected_model=selected_model,
                user_id=user_id,
                user_history=user_history,
                top_n=top_n
            )
            
            if not recommended_movies:
                st.warning("Không thể tạo gợi ý phim")
                st.stop()
            
            # Filter by genre and rating
            filtered_movies = _filter_recommendations(
                recommended_movies=recommended_movies,
                movies=movies,
                selected_genres=settings['selected_genres'],
                min_rating=settings['min_rating'],
                top_n=top_n
            )
            
            if len(filtered_movies) == 0:
                st.warning("Không có phim nào phù hợp với bộ lọc. Thử điều chỉnh cài đặt.")
                st.stop()
            
            # Build results dataframe
            results_df = _build_results_dataframe(filtered_movies, movies)
            
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
    
    # Hybrid explanation (if using Hybrid model)
    if model_name == "Hybrid-Smooth":
        st.markdown("---")
        _render_hybrid_explanation(
            hybrid_rec=hybrid_rec,
            n_ratings=len(user_history)
        )
    
    # Model comparison
    st.markdown("---")
    st.subheader("🔄 So Sánh Tất Cả Mô Hình")
    
    with st.spinner("Đang tạo gợi ý từ tất cả mô hình..."):
        comparison_results = _generate_all_model_recommendations(
            model_map=model_map,
            user_id=user_id,
            user_history=user_history
        )
        
        # Display comparison
        _render_model_comparison(comparison_results, movies)


def _generate_recommendations(
    model_name: str,
    selected_model: Any,
    user_id: int,
    user_history: pd.DataFrame,
    top_n: int
) -> list:
    """Generate recommendations from a single model"""
    
    if model_name == "Content-Based":
        # Content-based needs seed items
        liked_movies = user_history[user_history['rating'] >= 4.0]['movieId'].values
        if len(liked_movies) == 0:
            st.warning("⚠️ Người dùng chưa có phim nào được đánh giá cao (≥4.0). Không thể tạo gợi ý content-based.")
            return []
        
        seed_items = list(liked_movies)[:5]
        recs_df = selected_model.recommend_multi(seed_items, n=top_n*3, verbose=False)
        
        if recs_df is None or recs_df.empty:
            return []
        
        return recs_df['movieId'].values.tolist()
    
    else:
        # CF models (SVD, NCF, Hybrid)
        recs = selected_model.recommend(user_id, n=top_n*3, exclude_rated=True)
        
        if isinstance(recs, pd.DataFrame):
            return recs['movieId'].values.tolist()
        else:
            recs_array = np.array(recs) if not isinstance(recs, np.ndarray) else recs
            return recs_array.tolist()


def _filter_recommendations(
    recommended_movies: list,
    movies: pd.DataFrame,
    selected_genres: list,
    min_rating: float,
    top_n: int
) -> list:
    """Filter recommendations by genre and rating threshold"""
    
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
    
    return filtered_movies


def _build_results_dataframe(
    filtered_movies: list,
    movies: pd.DataFrame
) -> pd.DataFrame:
    """Build results DataFrame for display"""
    
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
    
    return pd.DataFrame(results)


def _render_hybrid_explanation(hybrid_rec: Any, n_ratings: int):
    """Render hybrid model weight explanation"""
    
    st.subheader("🔍 Giải Thích Mô Hình Kết Hợp")
    
    weights = hybrid_rec.calculate_smooth_weights(n_ratings, method='sigmoid')
    
    col1, col2, col3 = st.columns(3)
    
    # Content-Based gauge
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
    
    # SVD gauge
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
    
    # NCF gauge
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


def _generate_all_model_recommendations(
    model_map: dict,
    user_id: int,
    user_history: pd.DataFrame
) -> dict:
    """Generate recommendations from all models for comparison"""
    
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
                    recs_array = np.array(recs) if not isinstance(recs, np.ndarray) else recs
                    comparison_results[name] = recs_array[:5].tolist()
        
        except:
            comparison_results[name] = []
    
    return comparison_results


def _render_model_comparison(comparison_results: dict, movies: pd.DataFrame):
    """Render side-by-side model comparison"""
    
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
                        
                        # Highlight based on consensus
                        if count >= 3:
                            st.markdown(
                                f'<div style="background: linear-gradient(90deg, #ff6b6b, #ff8787); '
                                f'padding: 8px; border-radius: 5px; margin: 3px 0;">'
                                f'<b>🔥 {title}</b><br/>'
                                f'<small>Được đề xuất bởi {count}/4 mô hình</small></div>',
                                unsafe_allow_html=True
                            )
                        elif count == 2:
                            st.markdown(
                                f'<div style="background: linear-gradient(90deg, #ffd93d, #ffe66d); '
                                f'padding: 8px; border-radius: 5px; margin: 3px 0;">'
                                f'<b>⭐ {title}</b><br/>'
                                f'<small>Được đề xuất bởi {count}/4 mô hình</small></div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f'<div style="background: #e9ecef; padding: 8px; border-radius: 5px; '
                                f'margin: 3px 0; color: #495057;">'
                                f'<b>🔹 {title}</b><br/>'
                                f'<small>Được đề xuất bởi 1/4 mô hình</small></div>',
                                unsafe_allow_html=True
                            )
    
    # Legend
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('🔥 **3+ mô hình** cùng gợi ý (đồng thuận cao)')
    with col2:
        st.markdown('⭐ **2 mô hình** cùng gợi ý (đồng thuận trung bình)')
    with col3:
        st.markdown('🔹 **1 mô hình** gợi ý (đồng thuận thấp)')