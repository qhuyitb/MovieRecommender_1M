"""
PERFORMANCE-OPTIMIZED Smooth Hybrid Recommender

"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class SmoothHybridRecommender:
    """Performance-optimized Hybrid Recommender"""
    
    def __init__(self, content_based_rec, svd_rec, ncf_rec, ratings_df, movies_df):
        # Validate / normalize inputs early to avoid downstream dtype issues
        # (ratings_df may come from external callers and sometimes be malformed)
        if ratings_df is None:
            ratings_df = pd.DataFrame(columns=['userId', 'movieId', 'rating'])
        else:
            try:
                # ensure DataFrame
                if not isinstance(ratings_df, pd.DataFrame):
                    ratings_df = pd.DataFrame(ratings_df)
            except Exception:
                ratings_df = pd.DataFrame(columns=['userId', 'movieId', 'rating'])

            # coerce numeric id/rating columns
            for c in ['userId', 'movieId', 'rating']:
                if c in ratings_df.columns:
                    ratings_df[c] = pd.to_numeric(ratings_df[c], errors='coerce')

            # drop rows missing essential ids
            if 'userId' in ratings_df.columns and 'movieId' in ratings_df.columns:
                ratings_df = ratings_df[ratings_df['userId'].notna() & ratings_df['movieId'].notna()]
                try:
                    ratings_df['userId'] = ratings_df['userId'].astype(int)
                    ratings_df['movieId'] = ratings_df['movieId'].astype(int)
                except Exception:
                    pass

        if movies_df is None:
            movies_df = pd.DataFrame(columns=['movieId', 'title_clean', 'genres', 'rating_avg', 'rating_count'])
        else:
            try:
                if not isinstance(movies_df, pd.DataFrame):
                    movies_df = pd.DataFrame(movies_df)
            except Exception:
                movies_df = pd.DataFrame(columns=['movieId', 'title_clean', 'genres', 'rating_avg', 'rating_count'])

            if 'movieId' in movies_df.columns:
                movies_df['movieId'] = pd.to_numeric(movies_df['movieId'], errors='coerce')
            if 'rating_avg' in movies_df.columns:
                movies_df['rating_avg'] = pd.to_numeric(movies_df['rating_avg'], errors='coerce')
            if 'rating_count' in movies_df.columns:
                movies_df['rating_count'] = pd.to_numeric(movies_df['rating_count'], errors='coerce')

            # drop invalid movieId rows
            if 'movieId' in movies_df.columns:
                movies_df = movies_df[movies_df['movieId'].notna()]
                try:
                    movies_df['movieId'] = movies_df['movieId'].astype(int)
                except Exception:
                    pass

        # assign validated objects to instance
        self.content_rec = content_based_rec
        self.svd_rec = svd_rec
        self.ncf_rec = ncf_rec
        self.ratings = ratings_df
        self.movies = movies_df
        
        # Build user rated movie sets (fast single-pass using groupby)
        grouped = ratings_df.groupby('userId')['movieId'].apply(lambda s: set(s.values))
        self._user_rated_cache = grouped.to_dict()
        # user_rating_counts derived from the cached sets
        self.user_rating_counts = {uid: len(mids) for uid, mids in self._user_rated_cache.items()}
        # Precompute quick membership sets
        self.user_set = set(self._user_rated_cache.keys())
        self.movie_set = set(self.movies['movieId'].unique())
        
        print("Smooth Hybrid Recommender sẵn sàng!")
        print(f"   - Content-Based: {len(movies_df):,} phim")
        print(f"   - SVD: Phân rã ma trận")
        print(f"   - NCF: Deep Learning")
        print(f"   - Số người dùng: {len(self.user_rating_counts):,}")
    
    @staticmethod
    def calculate_smooth_weights(n_ratings, method='sigmoid', min_weight: float = 0.02, max_content_weight: float = 1.0):
        """Calculate smooth weights with a minimum weight floor.

        Args:
            n_ratings: number of ratings by the user
            method: 'sigmoid'|'exponential'|'polynomial'
            min_weight: minimum per-model weight before renormalization

        Returns a dict of normalized weights summing to 1.0.
        """
        if method == 'sigmoid':
            content_weight = 1 / (1 + np.exp((n_ratings - 5) / 2))
            ncf_weight = 1 / (1 + np.exp(-(n_ratings - 20) / 5))
            svd_weight = 1 - content_weight - ncf_weight
            svd_weight = max(0, svd_weight)
        
        elif method == 'exponential':
            content_weight = np.exp(-n_ratings / 10)
            ncf_weight = 1 - np.exp(-(n_ratings - 15) / 10)
            ncf_weight = max(0, ncf_weight)
            svd_weight = 1 - content_weight - ncf_weight
            svd_weight = max(0, svd_weight)
        
        elif method == 'polynomial':
            if n_ratings <= 5:
                content_weight = 1.0
                svd_weight = 0.0
                ncf_weight = 0.0
            elif n_ratings <= 20:
                t = (n_ratings - 5) / 15
                content_weight = (1 - t) ** 2
                ncf_weight = t ** 2
                svd_weight = 2 * t * (1 - t)
            else:
                content_weight = 0.0
                svd_weight = max(0, 0.3 * np.exp(-(n_ratings - 20) / 20))
                ncf_weight = 1 - svd_weight
        else:
            raise ValueError(f"Unknown method: {method}")

        # Apply minimum weight floor to avoid any model being completely zeroed out
        w = np.array([float(content_weight), float(svd_weight), float(ncf_weight)], dtype=float)
        # if all zeros (defensive), set uniform
        if np.allclose(w, 0.0):
            w = np.array([1.0, 1.0, 1.0], dtype=float)

        # enforce floor
        floor = float(min_weight)
        w = np.maximum(w, floor)

        # renormalize
        total = float(w.sum())
        if total <= 0:
            # fallback to uniform
            w = np.array([1.0, 1.0, 1.0], dtype=float)
            total = float(w.sum())

        content_weight, svd_weight, ncf_weight = (w / total).tolist()

        # Apply a maximum cap to content weight to avoid content dominating (>max_content_weight)
        try:
            cap = float(max_content_weight)
        except Exception:
            cap = 0.2
        if content_weight > cap:
            content_weight = cap
            # renormalize remaining weights proportionally
            rem = svd_weight + ncf_weight
            if rem <= 0:
                # split remaining equally
                svd_weight = (1.0 - content_weight) / 2.0
                ncf_weight = (1.0 - content_weight) / 2.0
            else:
                svd_weight = svd_weight / rem * (1.0 - content_weight)
                ncf_weight = ncf_weight / rem * (1.0 - content_weight)

        return {
            'content': content_weight,
            'svd': svd_weight,
            'ncf': ncf_weight
        }
    
    @staticmethod
    def calculate_hard_weights(n_ratings):
        """Hard switching for comparison"""
        if n_ratings < 5:
            return {'content': 1.0, 'svd': 0.0, 'ncf': 0.0}
        elif n_ratings < 20:
            return {'content': 0.0, 'svd': 1.0, 'ncf': 0.0}
        else:
            return {'content': 0.0, 'svd': 0.0, 'ncf': 1.0}
    
    def get_user_rating_count(self, user_id):
        """Lấy số lượng đánh giá của người dùng"""
        return self.user_rating_counts.get(user_id, 0)
    
    def recommend(self, user_id, n=10, method='sigmoid', 
                  exclude_rated=True, return_weights=False, verbose=False,
                  collect_all_candidates=True, min_weight: float = 0.02,
                  max_content_weight: float = 1.0, content_for_cold_only: bool = True):
        """
        TỐI ƯU HÓA: Gợi ý hybrid với vectorized operations
        
        Tối ưu chính:
        - Không dùng vòng lặp .iterrows()
        - Xử lý batch DataFrame
        - Merge một lần duy nhất
        """
        n_ratings = self.get_user_rating_count(user_id)
        weights = self.calculate_smooth_weights(n_ratings, method=method, min_weight=min_weight, max_content_weight=max_content_weight)

        # If NCF component is not available (stub or None), zero its weight and
        # renormalize remaining weights so hybrid still functions.
        has_ncf = hasattr(self, 'ncf_rec') and (self.ncf_rec is not None)
        if not has_ncf:
            # Zero ncf weight and renormalize content/svd
            weights['ncf'] = 0.0
            total = weights.get('content', 0.0) + weights.get('svd', 0.0)
            if total <= 0:
                # fallback: equal weights between available models
                weights['content'] = 0.5
                weights['svd'] = 0.5
            else:
                weights['content'] = weights['content'] / total
                weights['svd'] = weights['svd'] / total
        
        if verbose:
            
            print(f"GỢI Ý HYBRID - User {user_id}")
            
            print(f"Số đánh giá: {n_ratings}")
            print(f"Trọng số:")
            print(f"   - Content-Based: {weights['content']:.3f}")
            print(f"   - SVD: {weights['svd']:.3f}")
            print(f"   - NCF: {weights['ncf']:.3f}")
           
        
        # TỐI ƯU HÓA: Lấy danh sách phim đã xem từ cache (O(1) thay vì O(n) query)
        rated_movies = self._user_rated_cache.get(user_id, set()) if exclude_rated else set()
        
        # Thu thập gợi ý từ mỗi model
        model_recs = []
        model_counts = {'content': 0, 'svd': 0, 'ncf': 0}
        # candidate multiplier (increase to capture more overlap)
        candidate_mult = 8
        # number of content seeds to use (top rated user movies)
        content_seed_count = 5
        
        # 1. Content-Based: by default only use content for cold-start users (diverse but often irrelevant)
        cb_allowed = (n_ratings == 0) or (not content_for_cold_only and (weights['content'] > 0.01 or collect_all_candidates))
        if cb_allowed:
            try:
                if n_ratings == 0:
                    cb_recs = self.movies.nlargest(n * 3, 'rating_avg')[
                        ['movieId', 'title_clean', 'genres', 'rating_avg']
                    ].copy()
                    cb_recs['predicted_rating'] = cb_recs['rating_avg']
                else:
                    user_movies = self.ratings[
                        self.ratings['userId'] == user_id
                    ].nlargest(5, 'rating')['movieId'].values
                    
                    if len(user_movies) > 0:
                        # Use multiple seeds (top-rated movies) to collect a richer candidate set
                        seeds = list(user_movies[:content_seed_count])
                        cb_parts = []
                        for seed in seeds:
                            try:
                                try:
                                    part = self.content_rec.recommend(movie_id=seed, n=n*candidate_mult, verbose=False)
                                except TypeError:
                                    part = self.content_rec.recommend(seed, n=n*candidate_mult, verbose=False)
                            except Exception:
                                part = pd.DataFrame()
                            if part is not None and not part.empty:
                                # Normalize similarity/predicted columns to content_score
                                if 'similarity_score' in part.columns:
                                    part = part.rename(columns={'similarity_score': 'content_score'})
                                if 'similarity' in part.columns and 'content_score' not in part.columns:
                                    part = part.rename(columns={'similarity': 'content_score'})
                                cb_parts.append(part)
                        if cb_parts:
                            cb_recs = pd.concat(cb_parts, ignore_index=True, sort=False)
                            # drop duplicates keep highest score per movieId
                            if 'content_score' in cb_recs.columns:
                                cb_recs = cb_recs.sort_values('content_score', ascending=False)
                            cb_recs = cb_recs.drop_duplicates(subset='movieId')
                        else:
                            # Fallback: if content model cannot produce candidates for user seeds,
                            # use global popular movies as content candidates so hybrid has content coverage
                            cb_recs = self.movies.nlargest(n * candidate_mult, 'rating_avg')[
                                ['movieId', 'title_clean', 'genres', 'rating_avg']
                            ].copy()
                            cb_recs['content_score'] = cb_recs['rating_avg'].astype(float)
                    else:
                        cb_recs = pd.DataFrame()
                
                if not cb_recs.empty:
                    # TỐI ƯU HÓA: Xử lý vectorized thay vì iterrows()
                    if 'predicted_rating' in cb_recs.columns and 'content_score' not in cb_recs.columns:
                        cb_recs = cb_recs.rename(columns={'predicted_rating': 'content_score'})

                    # Ensure a canonical title column exists and prefer 'title_clean'
                    if 'title_clean' not in cb_recs.columns and 'title' in cb_recs.columns:
                        cb_recs['title_clean'] = cb_recs['title']

                    # Fill missing content_score from rating_avg if present; otherwise leave as NaN
                    if 'content_score' not in cb_recs.columns:
                        if 'rating_avg' in cb_recs.columns:
                            cb_recs['content_score'] = cb_recs['rating_avg'].astype(float)
                        else:
                            cb_recs['content_score'] = pd.NA

                    cb_recs = cb_recs[['movieId', 'title_clean', 'genres', 'content_score']]
                    cb_recs = cb_recs.copy()
                    cb_recs['_source'] = 'content'
                    model_recs.append(cb_recs)
                    model_counts['content'] = model_counts.get('content', 0) + len(cb_recs)
            except Exception as e:
                if verbose:
                    print(f"Content-Based thất bại: {e}")
        
        # 2. SVD
        # 2. SVD: collect candidates even if weight small when collect_all_candidates True
        if weights['svd'] > 0.01 or collect_all_candidates:
            try:
                # ensure we do not over-filter candidates inside SVD: allow low min_rating_count
                svd_recs = self.svd_rec.recommend(user_id, n=n*candidate_mult, exclude_rated=False, min_rating_count=0)
            except ValueError:
                # User unknown to SVD (cold-user). Attempt cold-start SVD routine if available
                try:
                    favs = list(self.ratings[self.ratings['userId'] == user_id][['movieId', 'rating']].itertuples(index=False, name=None))
                    svd_recs = self.svd_rec.recommend_new_user(favorite_movies=favs, n=n*candidate_mult, min_rating_count=0)
                except Exception:
                    # fallback to popular movies
                    svd_recs = self.movies.nlargest(n * candidate_mult, 'rating_avg')[['movieId', 'title_clean', 'genres', 'rating_avg']].copy()
                    svd_recs['predicted_rating'] = svd_recs['rating_avg'].astype(float)

            try:
                if svd_recs is not None and not svd_recs.empty:
                    # TỐI ƯU HÓA: Đổi tên và chọn cột trong một bước
                    svd_recs = svd_recs.rename(columns={
                        'predicted_rating': 'svd_score'
                    })
                    # Normalize title column name to 'title_clean'
                    if 'title_clean' not in svd_recs.columns and 'title' in svd_recs.columns:
                        svd_recs['title_clean'] = svd_recs['title']
                    if 'title_clean' not in svd_recs.columns:
                        svd_recs['title_clean'] = pd.NA
                    svd_recs = svd_recs[['movieId', 'title_clean', 'genres', 'svd_score']]
                    svd_recs = svd_recs.copy()
                    svd_recs['_source'] = 'svd'
                    model_recs.append(svd_recs)
                    model_counts['svd'] = model_counts.get('svd', 0) + len(svd_recs)
            except Exception as e:
                if verbose:
                    print(f"SVD thất bại: {e}")
        
        # 3. NCF
        # 3. NCF: always collect NCF candidates (NCF often strong for active users)
        if weights['ncf'] > 0.01 or collect_all_candidates:
            try:
                # ensure NCF does not over-filter candidates (min_rating_count=0)
                ncf_recs = self.ncf_rec.recommend(
                    user_id, n=n*candidate_mult, exclude_rated=False, return_details=True, min_rating_count=0
                )
                
                if ncf_recs is not None and not ncf_recs.empty:
                    # TỐI ƯU HÓA: Vectorized rename
                    ncf_recs = ncf_recs.rename(columns={'predicted_rating': 'ncf_score'})
                    # Normalize title to 'title_clean' if necessary
                    if 'title_clean' not in ncf_recs.columns and 'title' in ncf_recs.columns:
                        ncf_recs['title_clean'] = ncf_recs['title']
                    if 'title_clean' not in ncf_recs.columns:
                        ncf_recs['title_clean'] = pd.NA
                    ncf_recs = ncf_recs[['movieId', 'title_clean', 'genres', 'ncf_score']]
                    ncf_recs = ncf_recs.copy()
                    ncf_recs['_source'] = 'ncf'
                    model_recs.append(ncf_recs)
                    model_counts['ncf'] = model_counts.get('ncf', 0) + len(ncf_recs)
            except Exception as e:
                if verbose:
                    print(f"NCF thất bại: {e}")
        
        if not model_recs:
            if return_weights:
                return pd.DataFrame(), weights
            return pd.DataFrame()
        
        # MERGE CHANGE: use outer join and impute missing model scores with model-specific means
        score_names = ['content_score', 'svd_score', 'ncf_score']

        # Compute per-model default (mean) score from available model_recs
        # but do NOT fill missing with a constant yet — keep NaN so we can
        # perform adaptive per-item weighting below. Use NaN as a marker.
        default_scores = {k: None for k in score_names}
        for df in model_recs:
            for s in score_names:
                if s in df.columns:
                    try:
                        val = float(df[s].mean())
                        if default_scores[s] is None:
                            default_scores[s] = val
                    except Exception:
                        continue

        # If no mean found for a model, leave as NaN to indicate missing global info
        for s in score_names:
            if default_scores[s] is None:
                default_scores[s] = float('nan')

        # Concatenate all model recommendations and aggregate per movieId (faster than repeated outer-joins)
        # Ensure each model's DataFrame has unique column names (drop duplicate columns keeping first)
        for i, df in enumerate(model_recs):
            try:
                if df.columns.duplicated().any():
                    model_recs[i] = df.loc[:, ~df.columns.duplicated()].copy()
            except Exception:
                continue

        # Ensure each model's DataFrame uses canonical score column names and numeric dtypes
        for i, df in enumerate(model_recs):
            try:
                src = df['_source'] if '_source' in df.columns else None
            except Exception:
                src = None
            # Normalize common predicted_rating -> {svd,ncf}_score
            try:
                if src == 'svd' and 'predicted_rating' in df.columns and 'svd_score' not in df.columns:
                    df = df.rename(columns={'predicted_rating': 'svd_score'})
                if src == 'ncf' and 'predicted_rating' in df.columns and 'ncf_score' not in df.columns:
                    df = df.rename(columns={'predicted_rating': 'ncf_score'})
                if src == 'content':
                    if 'similarity_score' in df.columns and 'content_score' not in df.columns:
                        df = df.rename(columns={'similarity_score': 'content_score'})
                    if 'similarity' in df.columns and 'content_score' not in df.columns:
                        df = df.rename(columns={'similarity': 'content_score'})
            except Exception:
                pass

            # Ensure canonical columns exist
            for c in ['movieId', 'title_clean', 'genres']:
                if c not in df.columns:
                    df[c] = pd.NA

            # Coerce score columns to numeric to avoid later silent type coercion
            for score_col in ['content_score', 'svd_score', 'ncf_score']:
                if score_col in df.columns:
                    df[score_col] = pd.to_numeric(df[score_col], errors='coerce')

            model_recs[i] = df

        # Show a small sample per-source for diagnostics
        if verbose:
            try:
                for df in model_recs:
                    src = df['_source'].iat[0] if ('_source' in df.columns and len(df)>0) else 'unknown'
                    print(f"[Hybrid Debug] sample rows from source={src}:\n", df.head(3))
            except Exception:
                pass

        candidates = pd.concat(model_recs, ignore_index=True, sort=False)

        # Drop candidates whose movieId is not present in our movies DataFrame
        try:
            before_cnt = len(candidates)
            candidates = candidates[candidates['movieId'].isin(self.movie_set)]
            after_cnt = len(candidates)
            if verbose:
                print(f"[Hybrid Debug] Removed {before_cnt-after_cnt} candidates not in movies_df")
        except Exception:
            pass

        # Ensure common columns exist to allow aggregation without KeyError
        if verbose:
            try:
                print(f"[Hybrid Debug] Candidates collected per model (post-dedup lengths): {model_counts}")
            except Exception:
                pass

        for c in ['content_score', 'svd_score', 'ncf_score', 'title_clean', 'title', 'genres', 'rating_avg']:
            if c not in candidates.columns:
                candidates[c] = pd.NA

        agg_dict = {
            'title_clean': 'first',
            'title': 'first',
            'genres': 'first',
            'rating_avg': 'first',
            'content_score': 'mean',
            'svd_score': 'mean',
            'ncf_score': 'mean'
        }

        results_df = candidates.groupby('movieId', as_index=False).agg(agg_dict)

        # Consolidate scores and fill missing with model-specific defaults
        for score_name in score_names:
            score_cols = [col for col in results_df.columns if col.startswith(score_name)]
            if len(score_cols) > 1:
                try:
                    results_df[score_name] = results_df[score_cols].bfill(axis=1).iloc[:, 0]
                except Exception:
                    base = results_df[score_cols[0]]
                    for col in score_cols[1:]:
                        base = base.fillna(results_df[col])
                    results_df[score_name] = base
                for col in score_cols:
                    if col != score_name and col in results_df.columns:
                        results_df = results_df.drop(columns=[col])
            elif len(score_cols) == 1:
                if score_cols[0] != score_name:
                    results_df = results_df.rename(columns={score_cols[0]: score_name})

            # Ensure column exists (leave NaN for missing values)
            if score_name not in results_df.columns:
                results_df[score_name] = pd.Series([float('nan')] * len(results_df))
            else:
                # keep NaNs as-is; do not fill with global defaults here
                results_df[score_name] = results_df[score_name].astype(float)

        # NORMALIZE per-model scores to same scale [0,1]. Use percentile-based strategy:
        # - Use 5th-95th percentiles on candidate set to reduce outlier effects.
        # - If too few non-NaN values, fall back to model-global min/max from model_recs.
        # - If still degenerate, fallback to neutral 0.5.
        eps = 1e-8
        p_low, p_high = 5, 95
        # used_bounds collects percentile info used for normalization per model
        used_bounds = {}
        for score_name in score_names:
            col_data = results_df[score_name]
            # coalesce duplicated cols if necessary (col_data may be Series)
            if isinstance(col_data, pd.DataFrame):
                try:
                    col = col_data.bfill(axis=1).iloc[:, 0].astype(float)
                except Exception:
                    col = col_data.iloc[:, 0].astype(float)
            else:
                col = col_data.astype(float)

            non_na_count = col.notna().sum()
            # compute percentile-based min/max (p5,p95) on candidate set to reduce outlier effect
            p5 = p95 = np.nan
            try:
                if non_na_count >= 5:
                    p5 = float(np.nanpercentile(col.dropna().astype(float).values, p_low))
                    p95 = float(np.nanpercentile(col.dropna().astype(float).values, p_high))
                else:
                    # insufficient candidate coverage: try to aggregate global values from model_recs
                    global_vals = []
                    for df in model_recs:
                        if score_name in df.columns:
                            try:
                                vals = df[score_name].dropna().astype(float).values.tolist()
                                global_vals.extend(vals)
                            except Exception:
                                continue
                    if len(global_vals) >= 5:
                        p5 = float(np.nanpercentile(np.array(global_vals, dtype=float), p_low))
                        p95 = float(np.nanpercentile(np.array(global_vals, dtype=float), p_high))
                    else:
                        p5 = p95 = np.nan
                minv = p5
                maxv = p95
            except Exception:
                # fallback: use simple min/max on aggregated column
                minv = col.min(skipna=True)
                maxv = col.max(skipna=True)
                try:
                    p5 = float(np.nanpercentile(col.dropna().astype(float).values, p_low)) if non_na_count>0 else np.nan
                    p95 = float(np.nanpercentile(col.dropna().astype(float).values, p_high)) if non_na_count>0 else np.nan
                except Exception:
                    p5 = p95 = None

            # Ensure the aggregated raw score column is numeric (coerce any stray types)
            try:
                results_df[score_name] = pd.to_numeric(results_df[score_name], errors='coerce')
            except Exception:
                pass

            # If no variability, set normalized column to neutral 0.5
            if pd.isna(minv) or pd.isna(maxv) or abs(maxv - minv) < eps:
                # degenerate: use neutral 0.5
                results_df[f'{score_name}_norm'] = 0.5
            else:
                # scale to [0,1] based on p5/p95
                results_df[f'{score_name}_norm'] = (results_df[score_name] - minv) / (maxv - minv)
                # clip to [0,1]
                results_df[f'{score_name}_norm'] = results_df[f'{score_name}_norm'].clip(0.0, 1.0)
                # compress extremes slightly to avoid many perfect-1.0 values
                # map [0,1] -> [0.05,0.95] to reserve headroom
                results_df[f'{score_name}_norm'] = results_df[f'{score_name}_norm'] * 0.9 + 0.05
            # record used bounds for diagnostics
            used_bounds[score_name] = {'p5': p5, 'p95': p95, 'non_na': int(non_na_count)}
        
        # Ensure unique column names by coalescing duplicate-named columns (handle duplicates robustly)
        cols_list = list(results_df.columns)
        col_positions = {}
        for idx, cname in enumerate(cols_list):
            col_positions.setdefault(cname, []).append(idx)

        if any(len(v) > 1 for v in col_positions.values()):
            new_df = pd.DataFrame(index=results_df.index)
            for cname, positions in col_positions.items():
                if len(positions) == 1:
                    new_df[cname] = results_df.iloc[:, positions[0]]
                else:
                    # gather the duplicated columns by position and coalesce left-to-right
                    dup_block = results_df.iloc[:, positions]
                    try:
                        new_df[cname] = dup_block.bfill(axis=1).iloc[:, 0]
                    except Exception:
                        base = dup_block.iloc[:, 0]
                        for c in range(1, dup_block.shape[1]):
                            base = base.fillna(dup_block.iloc[:, c])
                        new_df[cname] = base
            results_df = new_df

        # Verbose diagnostics: show candidate counts and per-model score coverage/ranges
        if verbose:
            try:
                print(f"[Hybrid Debug] Candidates after aggregation: {len(results_df)}")
                for s in score_names:
                    non_na = int(results_df[s].notna().sum())
                    vmin = results_df[s].min(skipna=True)
                    vmax = results_df[s].max(skipna=True)
                    ub = used_bounds.get(s, {})
                    p5 = ub.get('p5')
                    p95 = ub.get('p95')
                    nonna = ub.get('non_na')
                    print(f"  - {s}: non-NaN={non_na} (used_non_na={nonna}), range=({vmin}, {vmax}), p5={p5}, p95={p95}, global_mean={default_scores.get(s)}")
                # raw counts from sources (before dedup)
                try:
                    raw_counts = candidates['_source'].value_counts().to_dict()
                except Exception:
                    raw_counts = {}
                print(f"  - raw candidate rows per-model (before dedup): {raw_counts}")
                print(f"  - per-user weights: content={weights['content']:.3f}, svd={weights['svd']:.3f}, ncf={weights['ncf']:.3f}")
            except Exception:
                pass

        # Smooth-only hybrid: no meta-learning/blender. We always use the
        # adaptive per-item weighted blending implemented below.
        blender_used = False

        # Tính trung bình có trọng số nhưng theo cách ADAPTIVE per-item:
        # For each movie row, only include models that have a non-NaN score.
        # denom = sum(weight_model * mask_model). predicted = sum(weight*score_norm*mask)/denom
        final_weights = weights

        # Compute presence masks
        content_mask = results_df['content_score'].notna().astype(float)
        svd_mask = results_df['svd_score'].notna().astype(float)
        ncf_mask = results_df['ncf_score'].notna().astype(float)

        # Numerator using normalized scores (in [0,1])
        num = (
            final_weights['content'] * results_df['content_score_norm'].fillna(0.0) * content_mask +
            final_weights['svd'] * results_df['svd_score_norm'].fillna(0.0) * svd_mask +
            final_weights['ncf'] * results_df['ncf_score_norm'].fillna(0.0) * ncf_mask
        )

        denom = (
            final_weights['content'] * content_mask +
            final_weights['svd'] * svd_mask +
            final_weights['ncf'] * ncf_mask
        )

        # Avoid divide-by-zero: where denom == 0, fall back to mean of available normalized scores;
        # if still NaN, use neutral 0.5 (in normalized [0,1] space)
        with np.errstate(divide='ignore', invalid='ignore'):
            pred = num / denom

        # Rows where denom == 0
        missing_mask = denom == 0
        if missing_mask.any():
            row_mean = results_df[[f'{s}_norm' for s in score_names]].mean(axis=1, skipna=True)
            row_mean = row_mean.fillna(0.5)
            pred = pred.where(~missing_mask, row_mean)

        # scale normalized prediction [0,1] back to rating scale [1,5]
        results_df['predicted_rating'] = 1.0 + 4.0 * pred
        
        # TỐI ƯU HÓA: Lọc vectorized
        if exclude_rated and rated_movies:
            results_df = results_df[~results_df['movieId'].isin(rated_movies)]
        
        # Lấy top N
        if len(results_df) > n:
            results_df = results_df.nlargest(n, 'predicted_rating')
        
        results_df = results_df.reset_index(drop=True)
        results_df['rank'] = range(1, len(results_df) + 1)
        
        # Consolidate title fields into a single `title_clean` column.
        # Some model DataFrames provide `title_clean`, others `title`.
        # Prefer a non-empty value; if both present prefer the longer (more complete) string.
        if 'title_clean' not in results_df.columns:
            results_df['title_clean'] = pd.NA
        if 'title' not in results_df.columns:
            results_df['title'] = pd.NA

        # Make sure working types are strings to allow length checks
        def _safe_str(x):
            if pd.isna(x):
                return ''
            return str(x)

        # Vectorized consolidation: choose the longer non-empty value
        t1 = results_df['title_clean'].fillna('').astype(str)
        t2 = results_df['title'].fillna('').astype(str)

        # If one is empty, take the other; if both non-empty take the longer (likely more complete)
        pick_title = np.where(
            (t1 == '') & (t2 == ''),
            '',
            np.where(
                t1 == '', t2,
                np.where(t2 == '', t1, np.where(t2.str.len() > t1.str.len(), t2, t1))
            )
        )

        results_df['title_clean'] = pick_title

        # Drop the old `title` column to avoid duplicate column names downstream
        if 'title' in results_df.columns:
            try:
                results_df = results_df.drop(columns=['title'])
            except Exception:
                # if drop fails for any reason, ignore and continue
                pass

        # Final column ordering
        cols = ['rank', 'movieId', 'title_clean', 'genres', 'predicted_rating',
                'content_score', 'svd_score', 'ncf_score']
        # Ensure missing cols exist so indexing doesn't KeyError
        for c in cols:
            if c not in results_df.columns:
                results_df[c] = pd.NA
        results_df = results_df[cols]
        # drop temporary _source if present
        if '_source' in results_df.columns:
            try:
                results_df = results_df.drop(columns=['_source'])
            except Exception:
                pass

        if return_weights:
            return results_df, weights
        return results_df
    
    def recommend_for_new_user(self, favorite_movies, n=10, method='sigmoid'):
        """Gợi ý cho người dùng mới (tối ưu hóa)"""
        n_ratings = len(favorite_movies)
        weights = self.calculate_smooth_weights(n_ratings, method=method)
        
        
        print(f"GỢI Ý CHO NGƯỜI DÙNG MỚI")
      
        print(f"Phim yêu thích: {n_ratings}")
        print(f"Trọng số: CB={weights['content']:.2f}, SVD={weights['svd']:.2f}, NCF={weights['ncf']:.2f}")
        
        
        fav_movie_ids = [m for m, r in favorite_movies]
        
        # TỐI ƯU HÓA: Thu thập tất cả gợi ý cùng lúc
        all_cb_recs = []
        
        for movie_id, rating in favorite_movies:
            try:
                cb_recs = self.content_rec.recommend(
                    movie_id=movie_id,  # ← THÊM: chỉ định rõ parameter
                    n=n*2, 
                    verbose=False
                )
                
                if cb_recs is not None and not cb_recs.empty:
                    # ← FIX CHÍNH: Đổi tên TRƯỚC KHI append
                    if 'similarity_score' in cb_recs.columns:
                        cb_recs = cb_recs.rename(columns={'similarity_score': 'similarity'})
                    
                    cb_recs['source_movie'] = movie_id
                    all_cb_recs.append(cb_recs)
                    print(f"Tìm được {len(cb_recs)} phim từ movie {movie_id}")  # ← Debug log
                    
            except Exception as e:
                print(f"Lỗi movie {movie_id}: {e}")  # ← Better error handling
                continue
        
        if not all_cb_recs:
            return pd.DataFrame()
        
        # TỐI ƯU HÓA: Concatenate một lần thay vì vòng lặp dict
        combined_recs = pd.concat(all_cb_recs, ignore_index=True)
        # remove favorites
        combined_recs = combined_recs[~combined_recs['movieId'].isin(fav_movie_ids)]
        # filter out movieIds not present in movies_df to avoid downstream errors
        try:
            combined_recs = combined_recs[combined_recs['movieId'].isin(self.movie_set)]
        except Exception:
            pass
        
        # TỐI ƯU HÓA: Vectorized groupby aggregation
        results_df = combined_recs.groupby('movieId').agg({
            'title_clean': 'first',
            'genres': 'first',
            'similarity': 'mean'
        }).reset_index()
        
        results_df['predicted_rating'] = results_df['similarity'] * 5
        results_df = results_df.nlargest(n, 'predicted_rating')
        results_df['rank'] = range(1, len(results_df) + 1)
        results_df['method'] = 'content-based'
        
        return results_df[['rank', 'movieId', 'title_clean', 'genres', 
                          'predicted_rating', 'method']]
    
    def compare_smooth_vs_hard(self, user_ids, n=10):
        """So sánh smooth vs hard switching"""
        results = []
        
        for user_id in user_ids:
            n_ratings = self.get_user_rating_count(user_id)
            smooth_weights = self.calculate_smooth_weights(n_ratings, 'sigmoid')
            hard_weights = self.calculate_hard_weights(n_ratings)
            
            try:
                smooth_recs = self.recommend(
                    user_id, n=n, method='sigmoid', verbose=False
                )
                smooth_top = smooth_recs['movieId'].tolist() if not smooth_recs.empty else []
            except:
                smooth_top = []
            
            results.append({
                'user_id': user_id,
                'n_ratings': n_ratings,
                'smooth_cb': smooth_weights['content'],
                'smooth_svd': smooth_weights['svd'],
                'smooth_ncf': smooth_weights['ncf'],
                'hard_cb': hard_weights['content'],
                'hard_svd': hard_weights['svd'],
                'hard_ncf': hard_weights['ncf'],
                'smooth_top_movies': smooth_top[:5]
            })
        
        return pd.DataFrame(results)
    # Meta-learning functions removed: system uses smooth switching only.

    
    @classmethod
    def load(cls,
             content_rec_path='../models/content_based_model.pkl',
             ratings_path='../data/cleaned/ratings_cleaned.csv',
             movies_path='../data/cleaned/movies_cleaned.csv'):
        """Load hybrid system từ saved models"""
        print("ĐANG TẢI HỆ THỐNG HYBRID RECOMMENDER")

        # Load data first so fallback stubs can use movies/ratings when models are absent
        print("Đang tải dữ liệu (ratings/movies)...")
        try:
            ratings_df = pd.read_csv(ratings_path)
            # normalize id/rating dtypes
            if 'userId' in ratings_df.columns:
                ratings_df['userId'] = pd.to_numeric(ratings_df['userId'], errors='coerce')
            if 'movieId' in ratings_df.columns:
                ratings_df['movieId'] = pd.to_numeric(ratings_df['movieId'], errors='coerce')
            if 'rating' in ratings_df.columns:
                ratings_df['rating'] = pd.to_numeric(ratings_df['rating'], errors='coerce')
            # drop rows missing essential ids
            if 'userId' in ratings_df.columns and 'movieId' in ratings_df.columns:
                ratings_df = ratings_df[ratings_df['userId'].notna() & ratings_df['movieId'].notna()]
                try:
                    ratings_df['userId'] = ratings_df['userId'].astype(int)
                    ratings_df['movieId'] = ratings_df['movieId'].astype(int)
                except Exception:
                    pass
        except Exception:
            ratings_df = pd.DataFrame(columns=['userId', 'movieId', 'rating'])

        try:
            movies_df = pd.read_csv(movies_path)
            # normalize movieId and rating_avg
            if 'movieId' in movies_df.columns:
                movies_df['movieId'] = pd.to_numeric(movies_df['movieId'], errors='coerce')
            if 'rating_avg' in movies_df.columns:
                movies_df['rating_avg'] = pd.to_numeric(movies_df['rating_avg'], errors='coerce')
            # drop rows with missing movieId
            if 'movieId' in movies_df.columns:
                movies_df = movies_df[movies_df['movieId'].notna()]
                try:
                    movies_df['movieId'] = movies_df['movieId'].astype(int)
                except Exception:
                    pass
        except Exception:
            movies_df = pd.DataFrame(columns=['movieId', 'title_clean', 'genres', 'rating_avg', 'rating_count'])

        # Content-based loader with fallback to popular-movies stub
        print("Đang tải Content-Based...")
        try:
            from content_based_recommender import ContentBasedRecommender
            content_rec = ContentBasedRecommender()
        except Exception as e:
            print(f"[Hybrid Load] Content model load failed: {e}. Using popular-movies fallback.")
            class StubContentRec:
                def __init__(self, movies_df):
                    self.movies = movies_df

                def recommend(self, movie_id=None, movie_title=None, n=10, verbose=False):
                    if self.movies is None or self.movies.empty:
                        return pd.DataFrame()
                    df = self.movies.nlargest(n, 'rating_avg')[['movieId', 'title_clean', 'genres', 'rating_avg']].copy()
                    df['content_score'] = df['rating_avg'].astype(float)
                    df['title'] = df['title_clean']
                    df['rank'] = range(1, len(df) + 1)
                    return df

                def recommend_multi(self, movie_ids, n=10, verbose=False):
                    return self.recommend(n=n, verbose=verbose)

            content_rec = StubContentRec(movies_df)

        # If our movies_df is empty (e.g. failed to load earlier), try to take the
        # movies DataFrame from the content-based recommender which usually
        # loads the same cleaned CSV. This prevents downstream dtype issues
        # where `rating_avg` may be missing or object-typed causing nlargest
        # and other numeric ops to fail.
        try:
            if (movies_df is None or len(movies_df) == 0) and hasattr(content_rec, 'movies'):
                movies_from_cb = content_rec.movies
                if movies_from_cb is not None and len(movies_from_cb) > 0:
                    movies_df = movies_from_cb.copy()
                    # ensure numeric types
                    if 'movieId' in movies_df.columns:
                        movies_df['movieId'] = pd.to_numeric(movies_df['movieId'], errors='coerce')
                    if 'rating_avg' in movies_df.columns:
                        movies_df['rating_avg'] = pd.to_numeric(movies_df['rating_avg'], errors='coerce')
                    if 'rating_count' in movies_df.columns:
                        movies_df['rating_count'] = pd.to_numeric(movies_df['rating_count'], errors='coerce')
                    # drop invalid movieId rows
                    if 'movieId' in movies_df.columns:
                        movies_df = movies_df[movies_df['movieId'].notna()]
                        try:
                            movies_df['movieId'] = movies_df['movieId'].astype(int)
                        except Exception:
                            pass
        except Exception:
            pass

        # SVD loader with fallback to popular-movies stub
        print("Đang tải SVD...")
        try:
            from svd_recommender import load_recommender
            svd_rec = load_recommender()
        except Exception as e:
            print(f"[Hybrid Load] SVD load failed: {e}. Using popular-movies fallback.")
            class StubSVD:
                def __init__(self, movies_df):
                    self.movies = movies_df

                def recommend(self, user_id, n=10, exclude_rated=True, min_rating_count=0):
                    if self.movies is None or self.movies.empty:
                        return pd.DataFrame()
                    df = self.movies.nlargest(n, 'rating_avg')[['movieId', 'title_clean', 'genres', 'rating_avg']].copy()
                    df['predicted_rating'] = df['rating_avg'].astype(float)
                    df['title'] = df['title_clean']
                    df['rank'] = range(1, len(df) + 1)
                    return df

                def recommend_new_user(self, favorite_movies, n=10, min_rating_count=0):
                    return self.recommend(None, n=n, exclude_rated=False)

            svd_rec = StubSVD(movies_df)

        # NCF loader (NCF already has an internal fallback in its class)
        print("Đang tải NCF...")
        try:
            from neural_cf_recommender import NCFRecommender
            ncf_rec = NCFRecommender.load()
        except Exception:
            # NCFRecommender.load already returns a StubNCFRecommender on failure
            try:
                from neural_cf_recommender import NCFRecommender
                ncf_rec = NCFRecommender.load()
            except Exception:
                ncf_rec = None

        print("Hoàn tất tải hybrid (với fallback nếu cần)")

        return cls(content_rec, svd_rec, ncf_rec, ratings_df, movies_df)
    
    def __repr__(self):
        return (f"SmoothHybridRecommender("
                f"users={len(self.user_rating_counts):,}, "
                f"movies={len(self.movies):,})")



def visualize_weight_curves(output_path='../figures/weight_curves.png'):
    """
    Vẽ đồ thị weight curves cho 3 methods
    """
    import matplotlib.pyplot as plt
    
    n_ratings_range = np.arange(0, 51, 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    methods = ['sigmoid', 'exponential', 'polynomial']
    
    for ax, method in zip(axes, methods):
        content_weights = []
        svd_weights = []
        ncf_weights = []
        
        for n in n_ratings_range:
            weights = SmoothHybridRecommender.calculate_smooth_weights(n, method)
            content_weights.append(weights['content'])
            svd_weights.append(weights['svd'])
            ncf_weights.append(weights['ncf'])
        
        ax.plot(n_ratings_range, content_weights, 'b-', linewidth=2.5, 
                label='Content-Based', marker='o', markersize=3)
        ax.plot(n_ratings_range, svd_weights, 'g-', linewidth=2.5,
                label='SVD', marker='s', markersize=3)
        ax.plot(n_ratings_range, ncf_weights, 'r-', linewidth=2.5,
                label='NCF', marker='^', markersize=3)
        
        ax.set_xlabel('Số Ratings của User', fontsize=12, fontweight='bold')
        ax.set_ylabel('Trọng Số', fontsize=12, fontweight='bold')
        ax.set_title(f'Phương pháp: {method.capitalize()}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 1.05)
        
        # Add vertical lines for transition zones
        ax.axvline(x=5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=20, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(5, 1.02, '5', ha='center', fontsize=9, color='gray')
        ax.text(20, 1.02, '20', ha='center', fontsize=9, color='gray')
    
    # Ensure output directory exists and save
    from pathlib import Path
    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(outp), dpi=300, bbox_inches='tight')
    print(f"Đã lưu: {outp}")
    plt.close()


def demo():
    """Demo smooth hybrid system"""
    
    print("SMOOTH HYBRID RECOMMENDER SYSTEM - DEMO")
    
    
    # Load hybrid
    hybrid = SmoothHybridRecommender.load()
    
    # Test with different user types
    test_cases = [
        (1, "User rất tích cực"),
        (100, "User trung bình"),
        (500, "User ít hoạt động")
    ]
    
    for user_id, desc in test_cases:
        print(f"\n{'='*80}")
        print(f"TEST: {desc} (User {user_id})")

        
        recs, weights = hybrid.recommend(
            user_id, n=5, method='sigmoid',
            return_weights=True, verbose=True
        )
        
        if not recs.empty:
            print("\nTop 5 Gợi ý:")
            for _, row in recs.iterrows():
                print(f"#{row['rank']} {row['title_clean'][:40]:<40} "
                      f"→ {row['predicted_rating']:.2f}")
    
    # Visualize
    
    print("TẠO ĐỒ THỊ WEIGHT CURVES...")
    visualize_weight_curves()
    
    print("\nDEMO HOÀN TẤT!")


# ĐÁNH GIÁ HIỆU NĂNG
def benchmark_performance():
    """So sánh hiệu năng cũ vs mới"""
    import time
    
    hybrid = SmoothHybridRecommender.load()
    test_users = [12, 3, 100, 250, 500]
    
    
    print("ĐÁNH GIÁ HIỆU NĂNG")
    
    times = []
    for user_id in test_users:
        start = time.time()
        recs = hybrid.recommend(user_id, n=10, verbose=False)
        elapsed = time.time() - start
        times.append(elapsed)
        
        print(f"User {user_id}: {elapsed*1000:.2f}ms - {len(recs)} gợi ý")
    
    print(f"\nThời gian trung bình: {np.mean(times)*1000:.2f}ms")
    print(f"Tổng thời gian: {sum(times)*1000:.2f}ms")


if __name__ == "__main__":
    demo()