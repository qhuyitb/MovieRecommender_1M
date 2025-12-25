"""
tabs/tab_model_performance.py
Tab 4: Hiệu Suất Mô Hình - Model evaluation metrics and comparison
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


def render_model_performance_tab(eval_results: pd.DataFrame):
    """
    Render tab hiệu suất mô hình
    
    Args:
        eval_results: DataFrame chứa kết quả evaluation
    """
    st.header("📊 Chỉ Số Hiệu Suất Mô Hình")
    
    if eval_results is None:
        st.warning(
            "⚠️ Không tìm thấy kết quả đánh giá. "
            "Vui lòng chạy `run_ranking_evaluation.py` trước."
        )
        st.stop()
    
    # PHẦN 1: OVERALL METRICS TABLE
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
    
    # PHẦN 2: VISUAL COMPARISONS
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
    
    # PHẦN 3: MODEL DETAILS
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
    
    # PHẦN 4: WEIGHT TUNING VISUALIZATION
    st.subheader("⚖️ Weight Tuning")
    
    # Weight curves image
    weight_curves_path = Path('figures/weight_curves.png')
    if weight_curves_path.exists():
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(
                str(weight_curves_path),
                caption="📈 Đường phân bổ trọng số Hybrid (w_content, w_svd, w_ncf)",
                use_container_width=True
            )
    else:
        st.info("Không tìm thấy hình weight_curves.png")
    
    st.subheader("📊 So Sánh Smooth vs Hard")
    
    # Smooth vs Hard comparison image
    comparison_path = Path('figures/smooth_vs_hard_comparison.png')
    if comparison_path.exists():
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(
                str(comparison_path),
                caption="⚡ So sánh hiệu quả khi dùng Smooth weights vs Hard weights",
                use_container_width=True
            )
    else:
        st.info("Không tìm thấy hình smooth_vs_hard_comparison.png")