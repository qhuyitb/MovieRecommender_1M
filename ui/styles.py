"""
Custom CSS styles cho MovieLens Recommender App
"""

import streamlit as st


def load_custom_css():
    """Load tất cả custom CSS cho ứng dụng"""
    st.markdown(get_main_styles(), unsafe_allow_html=True)


def get_main_styles():
    """Trả về CSS chính cho toàn bộ app"""
    return """
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
        
        .user-metric-removed-value {
            color: red;  
            font-size: 1rem;
            font-weight: bold;
        }
        
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
        
       
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
        }

        /* Sidebar: increase font size and control padding for better readability */
        section[data-testid="stSidebar"] {
            font-size: 1.05rem;
        

        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .streamlit-expanderHeader,
        section[data-testid="stSidebar"] .stMarkdown {
            font-size: 1.05rem;
        }

        section[data-testid="stSidebar"] .stButton>button,
        section[data-testid="stSidebar"] .stSelectbox,
        section[data-testid="stSidebar"] .stSlider,
        section[data-testid="stSidebar"] .stCheckbox,
        section[data-testid="stSidebar"] .stRadio {
            font-size: 1rem;
            padding: 6px 8px;
        }

        /* Make tab labels larger and with more padding */
        .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"] > div {
            font-size: 1.3rem;
            font-weight: 600;
            padding: 0.65rem 1.2rem;
            letter-spacing: 0.2px;
        }

        /* Selected tab indicator and color */
        .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"][aria-selected="true"] > div {
            border-bottom: 3px solid #ff6b6b;
            color: #ff6b6b;
        }

        /* Increase icon size inside tabs (if present) */
        .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"] svg,
        .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"] img {
            height: 1.2em;
            width: 1.2em;
            vertical-align: middle;
            margin-right: 0.5rem;
        }
      
        .model-recommendation-high-consensus {
            background: linear-gradient(90deg, #ff6b6b, #ff8787);
            padding: 8px;
            border-radius: 5px;
            margin: 3px 0;
        }
        
        .model-recommendation-medium-consensus {
            background: linear-gradient(90deg, #ffd93d, #ffe66d);
            padding: 8px;
            border-radius: 5px;
            margin: 3px 0;
        }
        
        .model-recommendation-low-consensus {
            background: #e9ecef;
            padding: 8px;
            border-radius: 5px;
            margin: 3px 0;
            color: #495057;
        }
        
     
        .insights-header {
            border: 2px solid #2196F3;
            border-radius: 10px;
            padding: 15px;
            background-color: #E3F2FD;
            margin-bottom: 15px;
        }
        
        
        .center-text {
            text-align: center;
        }
        
        .footer {
            text-align: center;
            color: gray;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #e0e0e0;
        }
    </style>
    """


def render_metric_card(title, value, subtitle=None):
    """
    Render một metric card với styling
    
    Args:
        title (str): Tiêu đề metric
        value (str|int|float): Giá trị hiển thị
        subtitle (str, optional): Phụ đề bổ sung
    """
    subtitle_html = f"<div style='font-size:0.9rem; color: #666;'>{subtitle}</div>" if subtitle else ""
    
    html = f"""
    <div class='user-metric-card'>
        <div class='user-metric-title'>{title}</div>
        <div class='user-metric-value'>{value}</div>
        {subtitle_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_insight_card(title, value, removed_value=None):
    """
    Render insight card với optional removed value
    
    Args:
        title (str): Tiêu đề
        value (str|int|float): Giá trị chính
        removed_value (str, optional): Giá trị đã loại bỏ (hiển thị màu đỏ)
    """
    removed_html = ""
    if removed_value:
        removed_html = f"<div class='insight-removed-value'>{removed_value}</div>"
    
    html = f"""
    <div class='insight-card'>
        <div class='insight-title'>{title}</div>
        <div class='insight-value'>{value}</div>
        {removed_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_movie_recommendation_card(title, count, consensus_level='low'):
    """
    Render movie recommendation card với consensus highlighting
    
    Args:
        title (str): Tên phim
        count (int): Số model đề xuất (1-4)
        consensus_level (str): 'high', 'medium', 'low'
    """
    icons = {
        'high': '🔥',
        'medium': '⭐',
        'low': '🔹'
    }
    
    css_classes = {
        'high': 'model-recommendation-high-consensus',
        'medium': 'model-recommendation-medium-consensus',
        'low': 'model-recommendation-low-consensus'
    }
    
    icon = icons.get(consensus_level, '🔹')
    css_class = css_classes.get(consensus_level, 'model-recommendation-low-consensus')
    
    html = f"""
    <div class='{css_class}'>
        <b>{icon} {title}</b><br/>
        <small>Được đề xuất bởi {count}/4 mô hình</small>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def get_consensus_level(count, total=4):
    """
    Xác định consensus level dựa trên số model đề xuất
    
    Args:
        count (int): Số model đề xuất phim này
        total (int): Tổng số model (default=4)
    
    Returns:
        str: 'high', 'medium', or 'low'
    """
    ratio = count / total
    
    if ratio >= 0.75:  # 3-4 models
        return 'high'
    elif ratio >= 0.5:  # 2 models
        return 'medium'
    else:  # 1 model
        return 'low'


def render_insights_box(text):
    """
    Render insights text box với styling
    
    Args:
        text (str): Nội dung insights (có thể chứa markdown)
    """
    html = f"""
    <div class='insights-header'>
        {text}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_header(text):
    """
    Render main header với styling
    
    Args:
        text (str): Header text
    """
    html = f"<div class='main-header'>{text}</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_footer():
    """Render footer cho app"""
    st.markdown("---")
    st.markdown(
        "<div class='footer'>"
        "Hệ Thống Gợi Ý Phim MovieLens 1M | Xây dựng bằng Streamlit"
        "</div>",
        unsafe_allow_html=True
    )


# Color schemes cho charts
CHART_COLORS = {
    'content': '#FF6B6B',
    'svd': '#4ECDC4',
    'ncf': '#95E1D3',
    'hybrid': '#FFD93D',
    'primary': '#1976d2',
    'secondary': '#ff6b6b',
    'success': '#4caf50',
    'warning': '#ff9800',
    'danger': '#f44336',
    'info': '#2196f3'
}


def get_model_color(model_name):
    """
    Lấy màu cho model
    
    Args:
        model_name (str): Tên model
    
    Returns:
        str: Hex color code
    """
    model_name_lower = model_name.lower()
    
    if 'content' in model_name_lower:
        return CHART_COLORS['content']
    elif 'svd' in model_name_lower:
        return CHART_COLORS['svd']
    elif 'ncf' in model_name_lower:
        return CHART_COLORS['ncf']
    elif 'hybrid' in model_name_lower:
        return CHART_COLORS['hybrid']
    else:
        return CHART_COLORS['primary']