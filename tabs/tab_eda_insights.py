"""
tabs/tab_eda_insights.py
Tab 6: EDA Insights - Dataset statistics, visualizations, quality analysis
"""

import streamlit as st
import pandas as pd
import re
from pathlib import Path


def render_eda_insights_tab():
    """Render tab EDA insights - dataset overview và visualizations"""
    
    st.header("📈 Insight Dữ Liệu (EDA)")
    
    # PHẦN 1: TỔNG QUAN DATASET
    st.subheader("📊 Tổng Quan Dataset")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        _render_dataset_stats()
    
    with col2:
        _render_cleaning_report()
    
    # PHẦN 2: THƯ VIỆN BIỂU ĐỒ
    st.markdown("---")
    st.subheader("📸 Thư Viện Biểu Đồ EDA")
    
    _render_figure_gallery()
    
    # PHẦN 3: EDA INSIGHTS TEXT
    st.markdown("---")
    _render_eda_insights_text()
    
    # PHẦN 4: CHẤT LƯỢNG DỮ LIỆU
    st.markdown("---")
    st.subheader("✅ Chất Lượng Dữ Liệu")
    
    _render_data_quality_analysis()


def _render_dataset_stats():
    """Render dataset statistics from raw data"""
    st.markdown("##### 📋 Thống Kê Dataset Gốc")
    
    try:
        dataset_stats = pd.read_csv('data/raw/dataset_stats.csv')
        stats_dict = dataset_stats.iloc[0].to_dict()
        
        # Display metrics
        metric_cols = st.columns(3)
        
        with metric_cols[0]:
            total_movies = int(stats_dict.get('total_movies', 0))
            st.markdown(
                f"<div class='insight-card'>"
                f"<div class='insight-title'>Tổng Số Phim</div>"
                f"<div class='insight-value'>{total_movies:,}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        with metric_cols[1]:
            total_ratings = int(stats_dict.get('total_ratings', 0))
            st.markdown(
                f"<div class='insight-card'>"
                f"<div class='insight-title'>Tổng Đánh Giá</div>"
                f"<div class='insight-value'>{total_ratings:,}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        with metric_cols[2]:
            total_users = int(stats_dict.get('total_users', 0))
            st.markdown(
                f"<div class='insight-card'>"
                f"<div class='insight-title'>Tổng Users</div>"
                f"<div class='insight-value'>{total_users:,}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        # Additional metrics
        metric_cols2 = st.columns(2)
        
        with metric_cols2[0]:
            avg_rating = float(stats_dict.get('avg_rating', 0))
            st.markdown(
                f"<div class='insight-card'>"
                f"<div class='insight-title'>Rating Trung Bình</div>"
                f"<div class='insight-value'>{avg_rating:.2f}⭐</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        with metric_cols2[1]:
            total_features = int(stats_dict.get('total_features', 0))
            st.markdown(
                f"<div class='insight-card'>"
                f"<div class='insight-title'>Số Features</div>"
                f"<div class='insight-value'>{total_features}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        # Sparsity
        if total_users > 0 and total_movies > 0:
            sparsity = 1 - (total_ratings / (total_users * total_movies))
            st.markdown(
                f"<div class='insight-card'>"
                f"<div class='insight-title'>Độ thưa ma trận</div>"
                f"<div class='insight-value'>{sparsity:.2%}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
    
    except FileNotFoundError:
        st.warning("⚠️ Không tìm thấy file `data/raw/dataset_stats.csv`")
    except Exception as e:
        st.error(f"Lỗi khi đọc dataset stats: {e}")


def _render_cleaning_report():
    """Render cleaning report statistics"""
    st.markdown("##### 🧹 Báo Cáo Làm Sạch Dữ Liệu")
    
    try:
        cleaning_report = pd.read_csv('data/cleaned/cleaning_report.csv')
        report_dict = cleaning_report.iloc[0].to_dict()
        
        # Before/After comparison
        col_before, col_after = st.columns(2)
        
        with col_before:
            st.markdown(
                f"<div class='insight-card'>"
                f"<div class='insight-title'>📥 Trước Cleaning</div>"
                f"<div>Phim: {int(report_dict.get('original_movies', 0)):,}</div>"
                f"<div>Ratings: {int(report_dict.get('original_ratings', 0)):,}</div>"
                f"<div>Users: {int(report_dict.get('original_users', 0)):,}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        with col_after:
            st.markdown(
                f"<div class='insight-card'>"
                f"<div class='insight-title'>✅ Sau Cleaning</div>"
                f"<div>Phim: {int(report_dict.get('cleaned_movies', 0)):,}</div>"
                f"<div>Ratings: {int(report_dict.get('cleaned_ratings', 0)):,}</div>"
                f"<div>Users: {int(report_dict.get('cleaned_users', 0)):,}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        # Removed stats
        st.markdown("**🗑️ Đã Loại Bỏ:**")
        removed_cols = st.columns(3)
        
        with removed_cols[0]:
            st.markdown(
                f"<div class='insight-card'>"
                f"<div class='insight-title'>Phim Loại Bỏ</div>"
                f"<div class='insight-value'>{int(report_dict.get('movies_removed', 0)):,}</div>"
                f"<div class='insight-removed-value'>-{float(report_dict.get('movies_removed_pct', 0)):.1f}%</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        with removed_cols[1]:
            st.markdown(
                f"<div class='insight-card'>"
                f"<div class='insight-title'>Ratings Loại Bỏ</div>"
                f"<div class='insight-value'>{int(report_dict.get('ratings_removed', 0)):,}</div>"
                f"<div class='insight-removed-value'>-{float(report_dict.get('ratings_removed_pct', 0)):.1f}%</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        with removed_cols[2]:
            st.markdown(
                f"<div class='insight-card'>"
                f"<div class='insight-title'>Users Loại Bỏ</div>"
                f"<div class='insight-value'>{int(report_dict.get('users_removed', 0)):,}</div>"
                f"<div class='insight-removed-value'>-{float(report_dict.get('users_removed_pct', 0)):.1f}%</div>"
                f"</div>",
                unsafe_allow_html=True
            )
    
    except FileNotFoundError:
        st.warning("⚠️ Không tìm thấy file `data/cleaned/cleaning_report.csv`")
    except Exception as e:
        st.error(f"Lỗi khi đọc cleaning report: {e}")


def _render_figure_gallery():
    """Render figure gallery with tabs"""
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
        # Display in tabs
        figure_tabs = st.tabs([title for _, title in available_figures])
        
        for idx, (filepath, title) in enumerate(available_figures):
            with figure_tabs[idx]:
                try:
                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col2:
                        st.image(filepath, use_container_width=True)
                    
                    # Add captions
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


def _render_eda_insights_text():
    """Render EDA insights from text file"""
    st.markdown("##### 💡 Insights Từ Phân Tích EDA")
    
    try:
        with open('figures/EDA_insights.txt', 'r', encoding='utf-8') as f:
            insights_text = f.read()
        
        # Split by VIZ markers
        viz_pattern = r'(VIZ \d+:.*?)(?=VIZ \d+:|$)'
        viz_sections = re.findall(viz_pattern, insights_text, re.DOTALL)
        
        # Get header
        header_match = re.search(r'^(.*?)(?=VIZ \d+:)', insights_text, re.DOTALL)
        if header_match:
            header_text = header_match.group(1).strip()
            if header_text:
                st.markdown(
                    f"""<div style="border: 2px solid #2196F3; border-radius: 10px; 
                    padding: 15px; background-color: #E3F2FD; margin-bottom: 15px;">
                    {header_text.replace(chr(10), '<br>')}</div>""",
                    unsafe_allow_html=True
                )
        
        # Display each VIZ section
        for viz_section in viz_sections:
            lines = viz_section.strip().split('\n')
            if lines:
                title = lines[0].replace('VIZ', '📊 VIZ').strip()
                content = '\n'.join(lines[1:]).strip()
                
                if content:
                    with st.expander(title, expanded=False):
                        st.markdown(content)
    
    except FileNotFoundError:
        st.warning("⚠️ Không tìm thấy file `figures/EDA_insights.txt`")
    except Exception as e:
        st.error(f"Lỗi khi đọc insights: {e}")


def _render_data_quality_analysis():
    """Render data quality analysis"""
    try:
        cleaning_report = pd.read_csv('data/cleaned/cleaning_report.csv')
        report_dict = cleaning_report.iloc[0].to_dict()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📊 Thống Kê Sau Cleaning")
            
            avg_ratings_movie = float(report_dict.get('avg_ratings_per_movie', 0))
            avg_ratings_user = float(report_dict.get('avg_ratings_per_user', 0))
            
            st.markdown(
                f"<div class='insight-card'>"
                f"<div class='insight-title'>Số lượt rating TB/phim</div>"
                f"<div class='insight-value'>{avg_ratings_movie:.1f}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
            
            st.markdown(
                f"<div class='insight-card'>"
                f"<div class='insight-title'>Số lượt rating TB/user</div>"
                f"<div class='insight-value'>{avg_ratings_user:.1f}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
            
            min_ratings_movie = int(report_dict.get('min_ratings_per_movie', 0))
            max_ratings_movie = int(report_dict.get('max_ratings_per_movie', 0))
            
            st.markdown(
                f"<div class='insight-card'>"
                f"<div class='insight-title'>Range ratings/phim</div>"
                f"<div class='insight-value'>{min_ratings_movie} - {max_ratings_movie:,}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown("##### 🔢 Feature Engineering")
            
            tfidf_features = int(report_dict.get('tfidf_features', 0))
            tfidf_sparsity = float(report_dict.get('tfidf_sparsity', 0))
            
            st.markdown(
                f"<div class='insight-card'>"
                f"<div class='insight-title'>Số TF-IDF Features</div>"
                f"<div class='insight-value'>{tfidf_features:,}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
            
            st.markdown(
                f"<div class='insight-card'>"
                f"<div class='insight-title'>TF-IDF Sparsity</div>"
                f"<div class='insight-value'>{tfidf_sparsity:.2f}%</div>"
                f"</div>",
                unsafe_allow_html=True
            )
            
            quality_score = 100 - float(report_dict.get('movies_removed_pct', 0))
            st.markdown(
                f"<div class='insight-card'>"
                f"<div class='insight-title'>Chất Lượng Dữ Liệu</div>"
                f"<div class='insight-value'>{quality_score:.1f}%</div>"
                f"<div style='font-size:0.9rem; color: {'#08611A' if quality_score > 90 else '#07BF38'};'>"
                f"{'Excellent' if quality_score > 90 else 'Good'}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
    
    except Exception as e:
        st.error(f"Lỗi khi phân tích chất lượng: {e}")