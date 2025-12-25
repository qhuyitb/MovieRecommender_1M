"""
Load và cache models cho MovieLens App
"""

import streamlit as st
import sys
import os
from typing import Tuple, Dict, Any

from config.settings import (
    SRC_DIR,
    RATINGS_PATH,
    MOVIES_PATH,
    AVAILABLE_MODELS
)


# Add src to path
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


@st.cache_resource
def load_content_based_model():
    """
    Load Content-Based Recommender model
    
    Returns:
        ContentBasedRecommender: Loaded model
    """
    try:
        from content_based_recommender import ContentBasedRecommender
        
        with st.spinner("Loading Content-Based model..."):
            model = ContentBasedRecommender()
        
        return model
    
    except ImportError as e:
        st.error(f"Không thể import ContentBasedRecommender: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi khi load Content-Based model: {e}")
        st.stop()


@st.cache_resource
def load_svd_model():
    """
    Load SVD Recommender model
    
    Returns:
        SVDRecommender: Loaded model
    """
    try:
        from svd_recommender import load_recommender
        
        with st.spinner("Loading SVD model..."):
            model = load_recommender()
        
        return model
    
    except ImportError as e:
        st.error(f"Không thể import SVD recommender: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi khi load SVD model: {e}")
        st.stop()


@st.cache_resource
def load_ncf_model():
    """
    Load Neural Collaborative Filtering model
    
    Returns:
        NCFRecommender: Loaded model
    """
    try:
        from neural_cf_recommender import NCFRecommender
        
        with st.spinner("Loading NCF model..."):
            model = NCFRecommender.load()
        
        return model
    
    except ImportError as e:
        st.error(f"Không thể import NCFRecommender: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi khi load NCF model: {e}")
        st.stop()


@st.cache_resource
def load_hybrid_model():
    """
    Load Smooth Hybrid Recommender model
    
    Returns:
        SmoothHybridRecommender: Loaded model
    """
    try:
        from smooth_hybrid_recommender import SmoothHybridRecommender
        
        with st.spinner("Loading Hybrid model..."):
            model = SmoothHybridRecommender.load(
                ratings_path=str(RATINGS_PATH),
                movies_path=str(MOVIES_PATH)
            )
        
        return model
    
    except ImportError as e:
        st.error(f"Không thể import SmoothHybridRecommender: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi khi load Hybrid model: {e}")
        st.stop()


@st.cache_resource
def load_all_models() -> Tuple[Any, Any, Any, Any]:
    """
    Load tất cả models cùng lúc
    
    Returns:
        Tuple: (content_rec, svd_rec, ncf_rec, hybrid_rec)
    """
    with st.spinner("Loading all recommender models..."):
        content_rec = load_content_based_model()
        svd_rec = load_svd_model()
        ncf_rec = load_ncf_model()
        hybrid_rec = load_hybrid_model()
    
    return content_rec, svd_rec, ncf_rec, hybrid_rec


@st.cache_resource
def load_models() -> Tuple[Any, Any, Any, Any]:
    """
    Backwards-compatible wrapper for older code that imports `load_models`.

    Returns:
        Tuple: (content_rec, svd_rec, ncf_rec, hybrid_rec)
    """
    return load_all_models()


def get_model_map(
    content_rec,
    svd_rec,
    ncf_rec,
    hybrid_rec
) -> Dict[str, Any]:
    """
    Tạo dictionary mapping tên model -> model object
    
    Args:
        content_rec: Content-Based model
        svd_rec: SVD model
        ncf_rec: NCF model
        hybrid_rec: Hybrid model
    
    Returns:
        Dict: Model name to model object mapping
    """
    return {
        "Content-Based": content_rec,
        "SVD": svd_rec,
        "NCF": ncf_rec,
        "Hybrid-Smooth": hybrid_rec
    }


def get_model_by_name(model_name: str, model_map: Dict[str, Any]) -> Any:
    """
    Lấy model object theo tên
    
    Args:
        model_name (str): Tên model (phải có trong AVAILABLE_MODELS)
        model_map (Dict): Model mapping dictionary
    
    Returns:
        Model object
    
    Raises:
        ValueError: Nếu model_name không hợp lệ
    """
    if model_name not in model_map:
        raise ValueError(
            f"Model '{model_name}' không hợp lệ. "
            f"Chọn một trong: {list(model_map.keys())}"
        )
    
    return model_map[model_name]


def get_model_description(model_name: str) -> str:
    """
    Lấy mô tả chi tiết của model
    
    Args:
        model_name (str): Tên model
    
    Returns:
        str: Mô tả model
    """
    descriptions = {
        "Content-Based": """
        **Content-Based Filtering**
        - Sử dụng TF-IDF trên thể loại phim
        - Tính độ tương đồng cosine giữa các phim
        - Phù hợp cho cold-start (phim mới)
        - Không cần lịch sử người dùng phong phú
        """,
        
        "SVD": """
        **Singular Value Decomposition (SVD)**
        - Phương pháp phân rá ma trận
        - Học các yếu tố tiềm ẩn (latent factors)
        - Nhanh và hiệu quả cho dữ liệu thưa
        - Cần ít nhất 20+ ratings để hoạt động tốt
        """,
        
        "NCF": """
        **Neural Collaborative Filtering**
        - Sử dụng deep learning (MLP)
        - Nắm bắt tương tác phi tuyến phức tạp
        - Tốt với người dùng có lịch sử phong phú
        - Yêu cầu 50+ ratings để tối ưu
        """,
        
        "Hybrid-Smooth": """
        **Smooth Hybrid Recommender**
        - Kết hợp cả 3 mô hình trên
        - Phân bổ trọng số động theo số lượng ratings:
          * < 5 ratings: Ưu tiên Content-Based
          * 5-20 ratings: Cân bằng Content + SVD
          * 20+ ratings: Ưu tiên NCF
        - Thích ứng tốt với mọi mức độ cold-start
        """
    }
    
    return descriptions.get(model_name, "Không có mô tả")


def validate_model_availability(model_name: str) -> bool:
    """
    Kiểm tra model có khả dụng không
    
    Args:
        model_name (str): Tên model
    
    Returns:
        bool: True nếu model khả dụng
    """
    return model_name in AVAILABLE_MODELS


def get_model_requirements(model_name: str) -> Dict[str, Any]:
    """
    Lấy yêu cầu của model (min ratings, features, etc.)
    
    Args:
        model_name (str): Tên model
    
    Returns:
        Dict: Model requirements
    """
    requirements = {
        "Content-Based": {
            "min_ratings": 1,
            "min_liked_movies": 1,  # Cần ít nhất 1 phim rating >= 4.0
            "requires_user_history": True,
            "requires_movie_features": True,
            "cold_start_friendly": True
        },
        
        "SVD": {
            "min_ratings": 5,
            "min_liked_movies": 0,
            "requires_user_history": True,
            "requires_movie_features": False,
            "cold_start_friendly": False
        },
        
        "NCF": {
            "min_ratings": 10,
            "min_liked_movies": 0,
            "requires_user_history": True,
            "requires_movie_features": False,
            "cold_start_friendly": False
        },
        
        "Hybrid-Smooth": {
            "min_ratings": 1,
            "min_liked_movies": 0,
            "requires_user_history": True,
            "requires_movie_features": True,
            "cold_start_friendly": True
        }
    }
    
    return requirements.get(model_name, {})


def check_model_can_recommend(
    model_name: str,
    n_ratings: int,
    n_liked_movies: int = 0
) -> Tuple[bool, str]:
    """
    Kiểm tra model có đủ điều kiện để tạo gợi ý không
    
    Args:
        model_name (str): Tên model
        n_ratings (int): Số lượng ratings của user
        n_liked_movies (int): Số lượng phim được thích (rating >= 4.0)
    
    Returns:
        Tuple[bool, str]: (can_recommend, reason)
    """
    requirements = get_model_requirements(model_name)
    
    if not requirements:
        return False, f"Model '{model_name}' không được hỗ trợ"
    
    # Check min ratings
    if n_ratings < requirements['min_ratings']:
        return False, (
            f"Cần ít nhất {requirements['min_ratings']} ratings, "
            f"hiện có {n_ratings}"
        )
    
    # Check min liked movies (for Content-Based)
    if model_name == "Content-Based":
        if n_liked_movies < requirements['min_liked_movies']:
            return False, (
                f"Content-Based cần ít nhất {requirements['min_liked_movies']} "
                f"phim được đánh giá cao (≥4.0), hiện có {n_liked_movies}"
            )
    
    return True, "OK"


def get_recommended_model_for_user(n_ratings: int) -> str:
    """
    Gợi ý model phù hợp nhất cho user dựa trên số lượng ratings
    
    Args:
        n_ratings (int): Số lượng ratings của user
    
    Returns:
        str: Tên model được khuyến nghị
    """
    if n_ratings < 5:
        return "Content-Based"
    elif n_ratings < 20:
        return "SVD"
    elif n_ratings < 50:
        return "Hybrid-Smooth"
    else:
        return "NCF"


def get_model_info_summary() -> Dict[str, Dict[str, Any]]:
    """
    Lấy tổng hợp thông tin tất cả models
    
    Returns:
        Dict: Model info summary
    """
    summary = {}
    
    for model_name in AVAILABLE_MODELS.keys():
        summary[model_name] = {
            "full_name": AVAILABLE_MODELS[model_name],
            "description": get_model_description(model_name),
            "requirements": get_model_requirements(model_name),
            "available": validate_model_availability(model_name)
        }
    
    return summary