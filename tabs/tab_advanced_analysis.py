"""
tabs/tab_advanced_analysis.py
Tab 5: Phân Tích Nâng Cao - Diversity, genre analysis, interactive playground
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from collections import Counter
from typing import Any, Dict


def render_advanced_analysis_tab(
    settings: dict,
    model_map: dict,
    movies: pd.DataFrame,
    content_rec: Any,
    svd_rec: Any,
    ncf_rec: Any,
    hybrid_rec: Any
):
    """
    Render tab phân tích nâng cao
    
    Args:
        settings: Dict từ sidebar
        model_map: Dict mapping model names
        movies: DataFrame phim
        content_rec, svd_rec, ncf_rec, hybrid_rec: Model objects
    """
    st.header("🔍 Phân Tích Nâng Cao")
    
    user_id = settings['user_id']
    user_history = settings['user_history']
    top_n = settings['top_n']
    all_genres = settings['all_genres']
    
    # PHẦN 1: CHỈ SỐ ĐA DẠNG
    st.subheader(f"📊 Chỉ Số Đa Dạng Trong Top-{top_n} Gợi Ý")
    
    with st.spinner("Đang tạo gợi ý để phân tích..."):
        try:
            # Generate recommendations from all models
            model_recs_analysis = _generate_all_recommendations(
                model_map=model_map,
                user_id=user_id,
                user_history=user_history,
                top_n=top_n
            )
            
            # Calculate diversity metrics
            diversity_data = _calculate_diversity_metrics(
                model_recs_analysis=model_recs_analysis,
                movies=movies,
                user_history=user_history,
                all_genres=all_genres
            )
            
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
    
    # PHẦN 2: PHÂN BỐ THỂ LOẠI TRONG GỢI Ý
    st.markdown("---")
    st.subheader(f"🎭 Phân Bố Thể Loại Trong Top-{top_n} Gợi Ý")
    st.caption(
        "Phân tích mang tính mô tả: đo lường mức độ xuất hiện của thể loại "
        "trong danh sách phim được mô hình đề xuất cho một người dùng cụ thể."
    )
    
    selected_genre_analysis = st.selectbox(
        "Chọn thể loại để phân tích",
        options=sorted(all_genres),
        key="genre_analysis"
    )
    
    with st.spinner(f"Đang phân tích phân bố thể loại {selected_genre_analysis}..."):
        try:
            genre_performance = _analyze_genre_performance(
                model_recs_analysis=model_recs_analysis,
                movies=movies,
                selected_genre=selected_genre_analysis
            )
            
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
    
    # PHẦN 3: INTERACTIVE PLAYGROUND
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
        
        # Generate custom recommendations button
        if st.button("🚀 Tạo Gợi Ý Với Trọng Số Tùy Chỉnh", type="primary"):
            _generate_custom_hybrid_recommendations(
                content_rec=content_rec,
                svd_rec=svd_rec,
                ncf_rec=ncf_rec,
                hybrid_rec=hybrid_rec,
                user_id=user_id,
                user_history=user_history,
                movies=movies,
                w_content_norm=w_content_norm,
                w_svd_norm=w_svd_norm,
                w_ncf_norm=w_ncf_norm
            )
    else:
        st.warning("⚠️ Tổng trọng số phải > 0")


def _generate_all_recommendations(
    model_map: dict,
    user_id: int,
    user_history: pd.DataFrame,
    top_n: int
) -> dict:
    """Generate recommendations from all models"""
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
                    recs_array = np.array(recs) if not isinstance(recs, np.ndarray) else recs
                    model_recs_analysis[name] = recs_array.tolist()
        except:
            model_recs_analysis[name] = []
    
    return model_recs_analysis


def _calculate_diversity_metrics(
    model_recs_analysis: dict,
    movies: pd.DataFrame,
    user_history: pd.DataFrame,
    all_genres: list
) -> list:
    """Calculate diversity metrics for each model"""
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
    
    return diversity_data


def _analyze_genre_performance(
    model_recs_analysis: dict,
    movies: pd.DataFrame,
    selected_genre: str
) -> list:
    """Analyze genre performance across models"""
    genre_movies = movies[movies['genres'].str.contains(selected_genre, na=False)]
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
            f'Số Phim {selected_genre}': genre_count,
            '% Trong Gợi Ý': f"{genre_ratio:.1%}",
            'Đánh Giá TB': f"{avg_rating:.2f}"
        })
    
    return genre_performance


def _generate_custom_hybrid_recommendations(
    content_rec, svd_rec, ncf_rec, hybrid_rec,
    user_id: int,
    user_history: pd.DataFrame,
    movies: pd.DataFrame,
    w_content_norm: float,
    w_svd_norm: float,
    w_ncf_norm: float
):
    """Generate recommendations with custom weights"""
    with st.spinner("Đang tạo gợi ý với trọng số tùy chỉnh..."):
        try:
            # Get auto weights
            n_ratings = len(user_history)
            auto_weights = hybrid_rec.calculate_smooth_weights(n_ratings, method='sigmoid')
            
            # Get predictions from each model
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
                                custom_scores[mid]['content'] = float(score) * 5
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