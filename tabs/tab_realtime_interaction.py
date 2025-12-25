"""
tabs/tab_realtime_interaction.py
Tab 7: Real-Time Interaction - Interactive rating and dynamic recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import Any


def render_realtime_interaction_tab(
    settings: dict,
    movies: pd.DataFrame,
    content_rec: Any,
    svd_rec: Any,
    ncf_rec: Any,
    hybrid_rec: Any
):
    """
    Render tab tương tác real-time
    
    Args:
        settings: Dict từ sidebar
        movies: DataFrame phim
        content_rec, svd_rec, ncf_rec, hybrid_rec: Model objects
    """
    st.header("💖 Tương Tác Real-Time")
    st.markdown(
        "**Playground thử nghiệm:** Đánh giá phim → Cập nhật profile → "
        "Hybrid tạo gợi ý → Context filters tinh chỉnh!"
    )
    
    user_id = settings['user_id']
    user_history = settings['user_history']
    top_n = settings['top_n']
    
    # Initialize session state
    user_changed = _init_session_state(user_id)
    
    if user_changed:
        st.info(f"🔄 Chuyển sang User #{user_id} - Session đã được reset")
    
    # START/RESET SESSION
    col_start, col_reset = st.columns([3, 1])
    
    with col_start:
        if not st.session_state.session_started:
            if st.button("🎬 Bắt Đầu Session", type="primary", use_container_width=True):
                _start_session(user_id, hybrid_rec)
        else:
            st.success(
                f"✅ Session hoạt động - {len(st.session_state.session_ratings)} tương tác mới"
            )
    
    with col_reset:
        if st.session_state.session_started:
            if st.button("🔄 Reset", type="secondary", use_container_width=True):
                _reset_session()
    
    if not st.session_state.session_started:
        st.info("👆 Nhấn **Bắt Đầu Session** để thử nghiệm real-time interaction")
        st.stop()
    
    # PHẦN 1: MOVIE RATING INTERFACE
    st.markdown("---")
    st.subheader("🎯 Đánh Giá Phim Mới")
    
    _render_rating_interface(movies, user_history)
    
    # PHẦN 2: CONTEXT-AWARE FILTERS
    st.markdown("---")
    st.subheader("🎭 Context-Aware Filters")
    
    preferred_genres = _render_context_filters()
    
    # PHẦN 3: DYNAMIC RECOMMENDATIONS
    st.markdown("---")
    st.subheader("🔄 Gợi Ý Động (Before vs After)")
    
    if len(st.session_state.session_ratings) == 0:
        st.info("👆 Hãy đánh giá vài phim để thấy thay đổi!")
    else:
        _render_dynamic_recommendations(
            user_id=user_id,
            user_history=user_history,
            content_rec=content_rec,
            svd_rec=svd_rec,
            ncf_rec=ncf_rec,
            hybrid_rec=hybrid_rec,
            movies=movies,
            preferred_genres=preferred_genres
        )
    
    # PHẦN 4: SESSION HISTORY
    st.markdown("---")
    st.subheader("📝 Lịch Sử Session")
    
    _render_session_history(user_id)
    
    # PHẦN 5: INSIGHTS
    if len(st.session_state.session_ratings) >= 2:
        st.markdown("---")
        st.subheader("💡 Insights")
        _render_session_insights(movies)


def _init_session_state(user_id: int) -> bool:
    """Initialize or validate session state"""
    if 'current_user_id' not in st.session_state:
        st.session_state.current_user_id = None
    
    # Check if user changed
    if st.session_state.current_user_id != user_id:
        st.session_state.current_user_id = user_id
        st.session_state.session_ratings = {}
        st.session_state.interaction_history = []
        st.session_state.original_recommendations = None
        st.session_state.session_started = False
        return True
    
    # Ensure all keys exist
    if 'session_ratings' not in st.session_state:
        st.session_state.session_ratings = {}
    if 'interaction_history' not in st.session_state:
        st.session_state.interaction_history = []
    if 'original_recommendations' not in st.session_state:
        st.session_state.original_recommendations = None
    if 'session_started' not in st.session_state:
        st.session_state.session_started = False
    
    return False


def _start_session(user_id: int, hybrid_rec: Any):
    """Start new session and generate baseline recommendations"""
    with st.spinner("Đang tạo baseline recommendations..."):
        try:
            original_recs = hybrid_rec.recommend(
                user_id,
                n=10,
                exclude_rated=True,
                verbose=False
            )
            
            if original_recs is None or original_recs.empty:
                st.error("❌ Không thể tạo gợi ý cho user này")
                st.stop()
            
            st.session_state.original_recommendations = original_recs.copy()
            st.session_state.session_started = True
            st.rerun()
        
        except Exception as e:
            st.error(f"❌ Lỗi khi khởi tạo session: {e}")
            st.stop()


def _reset_session():
    """Reset session for current user"""
    st.session_state.session_ratings = {}
    st.session_state.interaction_history = []
    st.session_state.original_recommendations = None
    st.session_state.session_started = False
    st.rerun()


def _render_rating_interface(movies: pd.DataFrame, user_history: pd.DataFrame):
    """Render movie rating interface"""
    col_search, _ = st.columns([3, 1])
    
    with col_search:
        search_movie = st.text_input(
            "🔍 Tìm phim để đánh giá",
            placeholder="VD: Toy Story, Matrix...",
            key="rt_search"
        )
    
    if search_movie:
        search_movie = search_movie.strip()[:100]
        
        search_results = movies[
            movies['title'].str.contains(search_movie, case=False, na=False, regex=False)
        ].head(10)
        
        if not search_results.empty:
            for idx, movie_row in search_results.iterrows():
                movie_id = int(movie_row['movieId'])
                movie_title = movie_row['title']
                movie_genres = movie_row['genres']
                
                current_rating = st.session_state.session_ratings.get(movie_id, None)
                already_rated_in_history = movie_id in user_history['movieId'].values
                
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    status = ""
                    if current_rating:
                        status = f"✅ Đã đánh giá: {current_rating}⭐"
                    elif already_rated_in_history:
                        status = "⚠️ Đã đánh giá trước đó"
                    
                    st.markdown(f"**{movie_title}**  \n`{movie_genres}` {status}")
                
                with col2:
                    if not already_rated_in_history:
                        rating_cols = st.columns(5)
                        for i, rating_val in enumerate([1, 2, 3, 4, 5]):
                            with rating_cols[i]:
                                if st.button(
                                    f"{rating_val}⭐",
                                    key=f"rate_{movie_id}_{rating_val}",
                                    use_container_width=True
                                ):
                                    st.session_state.session_ratings[movie_id] = rating_val
                                    st.session_state.interaction_history.append({
                                        'timestamp': pd.Timestamp.now(),
                                        'action': 'rate',
                                        'movieId': movie_id,
                                        'title': movie_title,
                                        'rating': rating_val
                                    })
                                    st.rerun()
                    else:
                        st.caption("Đã có trong lịch sử")
                
                with col3:
                    if current_rating:
                        if st.button("🗑️", key=f"remove_{movie_id}", help="Xóa đánh giá"):
                            del st.session_state.session_ratings[movie_id]
                            st.session_state.interaction_history.append({
                                'timestamp': pd.Timestamp.now(),
                                'action': 'remove',
                                'movieId': movie_id,
                                'title': movie_title,
                                'rating': None
                            })
                            st.rerun()
                
                st.markdown("---")
        else:
            st.warning(f"❌ Không tìm thấy: '{search_movie}'")


def _render_context_filters() -> list:
    """Render context-aware filters and return preferred genres"""
    col_mood, col_time, col_context = st.columns(3)
    
    with col_mood:
        mood = st.selectbox(
            "Tâm trạng",
            options=["Không chọn", "Vui vẻ", "Buồn", "Hồi hộp", "Thư giãn"],
            key="mood_filter"
        )
    
    with col_time:
        time_of_day = st.selectbox(
            "Thời gian",
            options=["Không chọn", "Sáng", "Trưa", "Chiều", "Tối"],
            key="time_filter"
        )
    
    with col_context:
        viewing_context = st.selectbox(
            "Ngữ cảnh",
            options=["Không chọn", "Một mình", "Gia đình", "Bạn bè", "Hẹn hò"],
            key="context_filter"
        )
    
    # Map context to genres
    context_genre_map = {
        "Vui vẻ": ["Comedy", "Animation", "Musical"],
        "Buồn": ["Drama", "Romance"],
        "Hồi hộp": ["Action", "Thriller", "Horror"],
        "Thư giãn": ["Comedy", "Romance", "Animation"],
        "Một mình": ["Drama", "Sci-Fi", "Thriller"],
        "Gia đình": ["Animation", "Adventure", "Comedy"],
        "Bạn bè": ["Action", "Comedy", "Adventure"],
        "Hẹn hò": ["Romance", "Comedy", "Drama"]
    }
    
    preferred_genres = []
    if mood != "Không chọn":
        preferred_genres.extend(context_genre_map.get(mood, []))
    if viewing_context != "Không chọn":
        preferred_genres.extend(context_genre_map.get(viewing_context, []))
    
    return list(set(preferred_genres))


def _render_dynamic_recommendations(
    user_id, user_history, content_rec, svd_rec, ncf_rec, hybrid_rec,
    movies, preferred_genres
):
    """Render dynamic before/after recommendations"""
    with st.spinner("Đang tính toán gợi ý mới..."):
        try:
            # Build updated profile
            temp_ratings = user_history.copy()
            for movie_id, rating in st.session_state.session_ratings.items():
                new_rating = pd.DataFrame([{
                    'userId': user_id,
                    'movieId': movie_id,
                    'rating': rating,
                    'timestamp': int(pd.Timestamp.now().timestamp())
                }])
                temp_ratings = pd.concat([temp_ratings, new_rating], ignore_index=True)
            
            # Calculate new weights
            n_ratings_new = len(temp_ratings)
            new_weights = hybrid_rec.calculate_smooth_weights(n_ratings_new, method='sigmoid')
            
            # Get model candidates
            model_candidates = _get_model_candidates(
                content_rec, svd_rec, ncf_rec,
                user_id, temp_ratings, new_weights
            )
            
            if not model_candidates:
                st.error("❌ Không thể tạo gợi ý từ bất kỳ mô hình nào")
                st.stop()
            
            # Blend candidates
            updated_recs = _blend_model_candidates(
                model_candidates, new_weights, temp_ratings, movies, preferred_genres
            )
            
            if updated_recs.empty:
                st.warning("⚠️ Không có gợi ý nào được tạo")
                st.stop()
            
            # Display comparison
            _display_recommendation_comparison(
                original_recs=st.session_state.original_recommendations,
                updated_recs=updated_recs,
                original_weights=hybrid_rec.calculate_smooth_weights(len(user_history), 'sigmoid'),
                new_weights=new_weights,
                preferred_genres=preferred_genres,
                user_history=user_history
            )
        
        except Exception as e:
            st.error(f"❌ Lỗi: {e}")
            import traceback
            with st.expander("🐛 Debug info"):
                st.code(traceback.format_exc())


def _get_model_candidates(content_rec, svd_rec, ncf_rec, user_id, temp_ratings, new_weights):
    """Get candidates from each model"""
    model_candidates = {}
    
    # Content-Based
    if new_weights['content'] > 0.01:
        try:
            liked_movies = temp_ratings[temp_ratings['rating'] >= 4.0]['movieId'].values
            if len(liked_movies) > 0:
                cb_recs = content_rec.recommend_multi(list(liked_movies)[:5], n=50, verbose=False)
                if cb_recs is not None and not cb_recs.empty:
                    model_candidates['content'] = cb_recs
        except:
            pass
    
    # SVD
    if new_weights['svd'] > 0.01:
        try:
            svd_recs = svd_rec.recommend(user_id, n=50, exclude_rated=False, min_rating_count=0)
            if svd_recs is not None and not svd_recs.empty:
                model_candidates['svd'] = svd_recs
        except:
            pass
    
    # NCF
    if new_weights['ncf'] > 0.01:
        try:
            ncf_recs = ncf_rec.recommend(
                user_id, n=50, exclude_rated=False, return_details=True, min_rating_count=0
            )
            if ncf_recs is not None and not ncf_recs.empty:
                model_candidates['ncf'] = ncf_recs
        except:
            pass
    
    return model_candidates


def _blend_model_candidates(model_candidates, new_weights, temp_ratings, movies, preferred_genres):
    """Blend model candidates with weights"""
    all_movie_scores = {}
    temp_rated_movies = set(temp_ratings['movieId'].values)
    
    for model_name, recs_df in model_candidates.items():
        for _, row in recs_df.iterrows():
            movie_id = row['movieId']
            
            if movie_id in temp_rated_movies:
                continue
            
            if movie_id not in all_movie_scores:
                all_movie_scores[movie_id] = {
                    'title_clean': row.get('title_clean', row.get('title', 'N/A')),
                    'genres': row.get('genres', ''),
                    'content_score': 0.0,
                    'svd_score': 0.0,
                    'ncf_score': 0.0
                }
            
            if model_name == 'content':
                score = row.get('content_score', row.get('avg_similarity', row.get('similarity_score', 0)))
                all_movie_scores[movie_id]['content_score'] = float(score) * 5.0
            elif model_name == 'svd':
                score = row.get('svd_score', row.get('predicted_rating', 0))
                all_movie_scores[movie_id]['svd_score'] = float(score)
            elif model_name == 'ncf':
                score = row.get('ncf_score', row.get('predicted_rating', 0))
                all_movie_scores[movie_id]['ncf_score'] = float(score)
    
    # Calculate weighted blend
    results = []
    for movie_id, scores in all_movie_scores.items():
        weighted_score = 0.0
        weight_sum = 0.0
        
        if scores['content_score'] > 0:
            weighted_score += new_weights['content'] * scores['content_score']
            weight_sum += new_weights['content']
        if scores['svd_score'] > 0:
            weighted_score += new_weights['svd'] * scores['svd_score']
            weight_sum += new_weights['svd']
        if scores['ncf_score'] > 0:
            weighted_score += new_weights['ncf'] * scores['ncf_score']
            weight_sum += new_weights['ncf']
        
        if weight_sum > 0:
            final_score = weighted_score / weight_sum
        else:
            continue
        
        results.append({
            'movieId': movie_id,
            'title_clean': scores['title_clean'],
            'genres': scores['genres'],
            'predicted_rating': final_score
        })
    
    updated_recs = pd.DataFrame(results)
    updated_recs = updated_recs.sort_values('predicted_rating', ascending=False).head(50)
    
    # Apply context filters
    if preferred_genres and not updated_recs.empty:
        def calculate_context_boost(row):
            movie_genres = row['genres'].split('|') if pd.notna(row['genres']) else []
            matches = sum(1 for g in movie_genres if g in preferred_genres)
            return min(matches * 0.5, 2.0)
        
        updated_recs['context_boost'] = updated_recs.apply(calculate_context_boost, axis=1)
        updated_recs['predicted_rating'] = updated_recs['predicted_rating'] + updated_recs['context_boost']
        updated_recs['predicted_rating'] = updated_recs['predicted_rating'].clip(1.0, 5.0)
        updated_recs = updated_recs.sort_values('predicted_rating', ascending=False)
    
    updated_recs = updated_recs.head(10).reset_index(drop=True)
    updated_recs['rank'] = range(1, len(updated_recs) + 1)
    
    return updated_recs


def _display_recommendation_comparison(
    original_recs, updated_recs, original_weights, new_weights, preferred_genres, user_history
):
    """Display before/after recommendation comparison"""
    col_before, col_after = st.columns(2)
    
    with col_before:
        st.markdown("### 📋 Gợi Ý Ban Đầu")
        st.caption(f"Profile gốc: {len(user_history)} ratings")
        st.caption(
            f"⚖️ CB={original_weights['content']:.2f} | "
            f"SVD={original_weights['svd']:.2f} | NCF={original_weights['ncf']:.2f}"
        )
        
        if original_recs is not None:
            original_display = []
            for _, row in original_recs.head(10).iterrows():
                title = row.get('title_clean', row.get('title', 'N/A'))
                title = title[:30] + "..." if len(str(title)) > 30 else title
                genres = row.get('genres', 'N/A')
                genres = genres[:25] + "..." if len(str(genres)) > 25 else genres
                
                original_display.append({
                    '#': int(row['rank']),
                    'Phim': title,
                    'Thể Loại': genres,
                    'Điểm': f"{row['predicted_rating']:.2f}"
                })
            
            st.dataframe(pd.DataFrame(original_display), use_container_width=True, hide_index=True)
    
    with col_after:
        st.markdown("### ✨ Gợi Ý Mới")
        n_ratings_new = len(user_history) + len(st.session_state.session_ratings)
        st.caption(f"Profile mới: {n_ratings_new} ratings (+{len(st.session_state.session_ratings)})")
        
        context_note = ""
        if preferred_genres:
            context_note = f" | 🎭 Boost: {', '.join(preferred_genres[:3])}"
        st.caption(
            f"⚖️ CB={new_weights['content']:.2f} | "
            f"SVD={new_weights['svd']:.2f} | NCF={new_weights['ncf']:.2f}{context_note}"
        )
        
        updated_display = []
        for _, row in updated_recs.iterrows():
            is_new = True
            if original_recs is not None:
                original_ids = original_recs['movieId'].values[:10]
                is_new = row['movieId'] not in original_ids
            
            marker = "🆕 " if is_new else ""
            title = row.get('title_clean', row.get('title', 'N/A'))
            title = title[:40] + "..." if len(str(title)) > 40 else title
            genres = row.get('genres', 'N/A')
            genres = genres[:35] + "..." if len(str(genres)) > 35 else genres
            
            updated_display.append({
                '#': int(row['rank']),
                'Phim': marker + title,
                'Thể Loại': genres,
                'Điểm': f"{row['predicted_rating']:.2f}"
            })
        
        st.dataframe(pd.DataFrame(updated_display), use_container_width=True, hide_index=True)
    
    # Calculate changes
    if original_recs is not None:
        original_ids = set(original_recs['movieId'].values[:10])
        updated_ids = set(updated_recs['movieId'].values[:10])
        
        new_movies = updated_ids - original_ids
        removed_movies = original_ids - updated_ids
        overlap = len(original_ids & updated_ids)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            st.metric("Phim Mới", f"{len(new_movies)}/10", delta=f"+{len(new_movies)}")
        with col_m2:
            st.metric("Phim Thay Thế", f"{len(removed_movies)}/10", delta=f"-{len(removed_movies)}")
        with col_m3:
            st.metric("Phim Giữ Nguyên", f"{overlap}/10")


def _render_session_history(user_id: int):
    """Render session interaction history"""
    if st.session_state.interaction_history:
        history_df = pd.DataFrame(st.session_state.interaction_history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        history_df = history_df.sort_values('timestamp', ascending=False)
        
        history_display = []
        for _, row in history_df.iterrows():
            action_icon = "⭐" if row['action'] == 'rate' else "🗑️"
            action_text = f"Đánh giá {row['rating']}⭐" if row['action'] == 'rate' else "Xóa"
            
            history_display.append({
                'Thời gian': row['timestamp'].strftime('%H:%M:%S'),
                'Hành động': f"{action_icon} {action_text}",
                'Phim': row['title']
            })
        
        st.dataframe(pd.DataFrame(history_display), use_container_width=True, hide_index=True)
        
        # Export buttons
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            session_export = pd.DataFrame([
                {'movieId': mid, 'rating': rating}
                for mid, rating in st.session_state.session_ratings.items()
            ])
            if not session_export.empty:
                csv_ratings = session_export.to_csv(index=False)
                st.download_button(
                    "📥 Tải Session Ratings",
                    csv_ratings,
                    f"session_user{user_id}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
        
        with col_e2:
            csv_history = history_df.to_csv(index=False)
            st.download_button(
                "📥 Tải Full History",
                csv_history,
                f"history_user{user_id}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
    else:
        st.info("Chưa có tương tác")


def _render_session_insights(movies: pd.DataFrame):
    """Render session insights"""
    col_i1, col_i2 = st.columns(2)
    
    with col_i1:
        st.markdown("**🎭 Thể Loại Vừa Đánh Giá**")
        
        new_genre_counts = {}
        for movie_id in st.session_state.session_ratings.keys():
            movie = movies[movies['movieId'] == movie_id]
            if len(movie) > 0 and pd.notna(movie.iloc[0]['genres']):
                for genre in movie.iloc[0]['genres'].split('|'):
                    new_genre_counts[genre] = new_genre_counts.get(genre, 0) + 1
        
        if new_genre_counts:
            genre_df = pd.DataFrame(
                list(new_genre_counts.items()),
                columns=['Thể Loại', 'Số Lượng']
            ).sort_values('Số Lượng', ascending=False)
            
            fig = px.bar(
                genre_df.head(5),
                x='Thể Loại',
                y='Số Lượng',
                color='Số Lượng',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col_i2:
        st.markdown("**📊 Phân Bố Rating**")
        
        rating_dist = pd.Series(
            list(st.session_state.session_ratings.values())
        ).value_counts().sort_index()
        
        fig = px.bar(
            x=rating_dist.index,
            y=rating_dist.values,
            labels={'x': 'Rating', 'y': 'Số Lượng'},
            color=rating_dist.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Insight text
    avg_rating = np.mean(list(st.session_state.session_ratings.values()))
    if avg_rating >= 4.0:
        st.success(f"✨ Đánh giá cao ({avg_rating:.1f}⭐ TB) → Hệ thống sẽ gợi ý phim chất lượng cao hơn!")
    elif avg_rating <= 2.5:
        st.warning(f"🤔 Đánh giá thấp ({avg_rating:.1f}⭐ TB) → Hệ thống đang tìm phim phù hợp hơn!")
    else:
        st.info(f"📊 Đánh giá TB ({avg_rating:.1f}⭐) → Hệ thống đang học sở thích!")