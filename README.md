---
title: MovieRecommender 1M
emoji: 💻
colorFrom: indigo
colorTo: green
sdk: docker
pinned: false
license: mit
---

<<<<<<< HEAD
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
=======
## 🎯 Kiến Trúc

**3 Mô Hình Độc Lập:**
- **Content-Based:** TF-IDF trên genres, cosine similarity
- **SVD:** Matrix factorization (50 factors) với cold-start fold-in
- **NCF:** Deep learning model (user/movie embeddings → MLP)

**Adaptive Hybrid System (Switching):**
- Trọng số tự động điều chỉnh theo user profile (số ratings)
- Weighted linear combination với dynamic weights
- Normalization: percentile-based (p5/p95) → [0.05, 0.95]
- Adaptive per-item blending: chỉ dùng models có score
- Fallback stubs khi model artifacts thiếu

**Công thức trọng số (sigmoid-based switching):**

```python
# Adaptive weights dựa trên số lượng đánh giá của user (n_ratings)
# Phân loại mức hoạt động:
#   n < 5   → "thấp"
#   5 ≤ n < 20 → "trung bình"
#   n ≥ 20  → "cao"
content_weight = 1 / (1 + np.exp((n_ratings - 5)/2))      # Giảm dần khi user hoạt động nhiều hơn
ncf_weight     = 1 / (1 + np.exp(-(n_ratings - 20)/5))    # Tăng dần khi user hoạt động nhiều
svd_weight    = 1 - content_weight - ncf_weight           # Phần còn lại
```

Trong thực tế, sau bước tính biểu thức thô ở trên hệ thống còn thực hiện xử lý hậu kỳ để đảm bảo trọng số hợp lệ. Cụ thể (xem `calculate_smooth_weights()` trong `src/smooth_hybrid_recommender.py`):

- Nếu `svd_weight` âm thì đặt về 0 (ví dụ `svd_weight = max(0, svd_weight)`).
- Áp dụng sàn `min_weight` (ví dụ 0.02) cho mỗi model để tránh model nào có trọng số 0.
- Chuẩn hóa (renormalize) các trọng số để tổng = 1.0.
- Áp giới hạn `max_content_weight` để tránh `content` chiếm ưu thế quá mức (nếu cần).
- Nếu áp cap làm tổng lệch, chuẩn hóa lại để tổng vẫn = 1.0.

Hàm trả về dict có dạng `{'content': ..., 'svd': ..., 'ncf': ...}` đã được xử lý.
```

**Thuật ngữ:** Đây là **Switching Hybrid** (Burke 2002) với smooth transition thay vì hard switch.

**Mức hoạt động người dùng:**
- **thấp**: <5 đánh giá
- **trung bình**: 5–19 đánh giá
- **cao**: ≥20 đánh giá
→ Quy định này dùng để giải thích UI và điều chỉnh trọng số mô hình.

---

## 🔧 Cài Đặt

```bash
# Clone repo
git clone https://github.com/qhuyitb/MovieRecommender_1M.git
cd MovieRecommender_1M

# Tạo venv & install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt

# Download data MovieLens 1M
# https://grouplens.org/datasets/movielens/1m/
# Giải nén vào data/raw/
```

**Training models (chạy notebooks 01-07):**
```bash
jupyter lab
# Chạy lần lượt: 01_data_collection → 02_cleaning → 03_EDA → 04_content → 05_svd → 06_ncf → 07_hybrid
```

---

## 🚀 Sử Dụng

**1. CLI Evaluation:**
```bash
python scripts/run_ranking_evaluation.py --seed 42
# Kết quả: ranking_results_with_hybrid.csv
```

**2. Demo App (Streamlit):**
```bash
streamlit run app.py
# → http://localhost:8501
```

**3. Code API:**

Lưu ý: cách import phụ thuộc vào thư mục làm việc khi chạy. Nếu bạn chạy từ thư mục gốc của project, dùng import với tiền tố `src.` (hoặc thêm thư mục gốc vào `PYTHONPATH`). Nếu bạn chuyển CWD vào thư mục `src/`, có thể import không cần `src.`.

```python
# Khuyến nghị (chạy từ thư mục gốc của project)
from src.smooth_hybrid_recommender import SmoothHybridRecommender

# Load adaptive hybrid recommender
hybrid = SmoothHybridRecommender.load(
    ratings_path='data/cleaned/ratings_cleaned.csv',
    movies_path='data/cleaned/movies_cleaned.csv'
)

# Existing user (adaptive weight switching)
recs, weights = hybrid.recommend(user_id=1, n=10, method='sigmoid', return_weights=True)

# New user (cold start)
recs = hybrid.recommend_for_new_user(
    favorite_movies=[(1, 5.0), (260, 4.5)],
    n=10
)

# Nếu chạy từ trong `src/` (CWD = src/), import thay thế như sau:
# from smooth_hybrid_recommender import SmoothHybridRecommender
```

---

## 📊 Kết Quả Evaluation

**Metrics:** Precision@K, Recall@K, NDCG@K, MAP@K, ARR

| Model         | K  | Precision@K | Recall@K | NDCG@K | MAP@K | ARR  |
|-------------- |----|-------------|----------|--------|-------|------|
| SVD           | 5  | 0.2162      | 0.0223   | 0.2211 | 0.1538| 4.35 |
| SVD           | 10 | 0.1953      | 0.0404   | 0.2065 | 0.1214| 4.33 |
| SVD           | 20 | 0.1710      | 0.0676   | 0.1911 | 0.0953| 4.30 |
| NCF           | 5  | 0.1826      | 0.0198   | 0.1829 | 0.1189| 4.41 |
| NCF           | 10 | 0.1697      | 0.0355   | 0.1752 | 0.0946| 4.37 |
| NCF           | 20 | 0.1553      | 0.0627   | 0.1681 | 0.0760| 4.33 |
| Content-Based | 5  | 0.0310      | 0.0042   | 0.0324 | 0.0168| 3.26 |
| Content-Based | 10 | 0.0293      | 0.0065   | 0.0311 | 0.0120| 3.26 |
| Content-Based | 20 | 0.0239      | 0.0104   | 0.0279 | 0.0082| 3.26 |
| Hybrid-Smooth | 5  | 0.2148      | 0.0208   | 0.2206 | 0.1569| 4.36 |
| Hybrid-Smooth | 10 | 0.1821      | 0.0351   | 0.1970 | 0.1161| 4.34 |
| Hybrid-Smooth | 20 | 0.1505      | 0.0599   | 0.1742 | 0.0841| 4.33 |

**Nhận xét:**
- SVD vẫn dẫn đầu về Precision@K và NDCG@K ở K=5,10,20
- NCF có ARR cao nhất ở K=5
- Hybrid-Smooth cân bằng tốt, MAP@5 cao nhất, hiệu quả switching rõ rệt
- Content-Based chỉ phù hợp cold-start, kém hơn về các chỉ số tổng thể

---

## 📁 Cấu Trúc

```
MovieRecommender_1M/
├── app.py
├── check_data_comparison.py
├── check_users.py
├── PROJECT_ARCHITECTURE.md
├── README.md
├── requirements.txt
├── data/
│   ├── cleaned/
│   │   ├── cleaning_report.csv
│   │   ├── movies_cleaned.csv
│   │   ├── ratings_cleaned.csv
│   │   └── users_cleaned.csv
│   │── raw/
│   │    ├── dataset_stats.csv
│   │    ├── movies.csv
│   │    ├── movies_with_features.csv
│   │    ├── ratings.csv
│   │    ├── README
│   │    ├── users.csv
│   │    └── users_with_features.csv
│   └── analysis/
│       └── ranking_results_with_hybrid.csv
├── figures/
│   ├── 01_rating_distribution.png
│   ├── 02_top_genres.png
│   ├── 03_user_movie_heatmap.png
│   ├── 04_top_movies.png
│   ├── 05_ratings_over_time.png
│   ├── 06_rating_by_genre.png
│   ├── 07_ncf_training_history.png
│   ├── ARR_by_model.png
│   ├── EDA_insights.txt
│   ├── MAPatK_by_model.png
│   ├── NDCGatK_by_model.png
│   ├── PrecisionatK_by_model.png
│   ├── RecallatK_by_model.png
│   ├── smooth_vs_hard_comparison.png
│   └── weight_curves.png
├── models/
│   ├── comparison_smooth_vs_hard.csv
│   ├── content_based_model.pkl
│   ├── movie_indices.pkl
│   ├── neural_cf_history.csv
│   ├── neural_cf_model.h5
│   ├── neural_cf_movie_encoder.pkl
│   ├── neural_cf_training_info.pkl
│   ├── neural_cf_user_encoder.pkl
│   ├── svd_gridsearch_results.csv
│   ├── svd_model.pkl
│   ├── svd_training_info.pkl
│   ├── tfidf_matrix.pkl
│   └── tfidf_vectorizer.pkl
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_EDA.ipynb
│   ├── 04_content_based.ipynb
│   ├── 05_collaborative_filtering.ipynb
│   ├── 06_neural_cf.ipynb
│   ├── 07_smooth_hybrid_system.ipynb
│   └── 08_evaluation_visualization.ipynb
├── scripts/
│   └── run_ranking_evaluation.py
├── src/
│   ├── content_based_recommender.py
│   ├── neural_cf_recommender.py
│   ├── smooth_hybrid_recommender.py
│   └── svd_recommender.py
```

---

## 🐛 Troubleshooting

**Module not found:**
```bash
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Model files missing:**
- Chạy notebooks 04, 05, 06 để train models

**Data files missing:**
- Download MovieLens 1M từ GroupLens
- Chạy notebooks 01, 02 để clean data

---

## 📝 License & Contact

- **License:** MIT
- **Author:** qhuyitb
- **GitHub:** https://github.com/qhuyitb/MovieRecommender_1M

**⭐ Nếu project hữu ích, hãy cho 1 star!**
>>>>>>> b844581 (docs: update README (structure, API, weights explanation))
