import sys
import os
import numpy as np
import pandas as pd
import argparse

# Ensure src is importable
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from content_based_recommender import ContentBasedRecommender
from svd_recommender import load_recommender as load_svd
from neural_cf_recommender import NCFRecommender
from smooth_hybrid_recommender import SmoothHybridRecommender

print("RUN: Ranking evaluation (including Hybrid-Smooth)")

# CLI args for faster runs
parser = argparse.ArgumentParser(description='Run ranking evaluation')
parser.add_argument('--n_users', type=int, default=300, help='Number of users to sample for ranking evaluation')
parser.add_argument('--weight-mode', type=str, choices=['smooth','blend'], default='smooth', help="DEPRECATED: meta-learning removed. Keep default 'smooth'.")
parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible sampling')
args = parser.parse_args()

N_USERS = args.n_users

# Load data
ratings = pd.read_csv(os.path.join('data', 'cleaned', 'ratings_cleaned.csv'))
movies = pd.read_csv(os.path.join('data', 'cleaned', 'movies_cleaned.csv'))

# Temporal split (same as notebook)
ratings_sorted = ratings.sort_values('timestamp')
split_idx = int(len(ratings_sorted) * 0.8)
train_data = ratings_sorted.iloc[:split_idx].copy()
test_data = ratings_sorted.iloc[split_idx:].copy()

print(f"Train: {len(train_data):,} | Test: {len(test_data):,}")

# Load recommenders
print("Loading recommenders...")
content_rec = ContentBasedRecommender()
svd_rec = load_svd()
ncf_rec = NCFRecommender.load()
hybrid_rec = SmoothHybridRecommender.load(
    ratings_path=os.path.join('data', 'cleaned', 'ratings_cleaned.csv'),
    movies_path=os.path.join('data', 'cleaned', 'movies_cleaned.csv')
)

print("All recommenders loaded. Using smooth switching hybrid by default (no meta-learning).")


def evaluate_ranking_testset_v2(recommender, train_data, test_data, movies_df, model_name,
                                k_values=[5, 10, 20], n_users=300, relevant_threshold=4.0,
                                weight_mode='smooth'):
    print(f"RANKING EVALUATION - {model_name} (Test Set Method v2)")

    # Build ground truth from TEST_DATA only
    test_users_relevant = {}
    for user_id in test_data['userId'].unique():
        user_test = test_data[test_data['userId'] == user_id]
        relevant = set(user_test[user_test['rating'] >= relevant_threshold]['movieId'].values)
        if len(relevant) >= 2:
            test_users_relevant[user_id] = relevant
    
    if len(test_users_relevant) == 0:
        print("No users with relevant items in test set!")
        return []

    # Build user-movie mapping for train data (to exclude when recommending)
    train_user_movies = {}
    for user_id in train_data['userId'].unique():
        user_train = train_data[train_data['userId'] == user_id]
        train_user_movies[user_id] = set(user_train['movieId'].values)

    eval_users = list(test_users_relevant.keys())
    if len(eval_users) > n_users:
        # Use global RNG seed if provided for reproducibility
        try:
            seed = args.seed if 'args' in globals() and hasattr(args, 'seed') else None
        except Exception:
            seed = None
        if seed is not None:
            np.random.seed(int(seed))
        eval_users = np.random.choice(eval_users, size=n_users, replace=False)

    max_k = max(k_values)
    results = {k: {'precision': [], 'recall': [], 'arr': [], 'ndcg': [], 'map': []} for k in k_values}

    failed = 0
    success = 0

    for i, user_id in enumerate(eval_users, 1):
        try:
            test_relevant = test_users_relevant[user_id]
            exclude_items = train_user_movies.get(user_id, set())

            if model_name == "Content-Based":
                user_train = train_data[train_data['userId'] == user_id]
                liked_train = set(user_train[user_train['rating'] >= 4.0]['movieId'].values)
                if len(liked_train) == 0:
                    failed += 1
                    continue
                sample = list(liked_train)[:5]
                recs = recommender.recommend_multi(sample, n=max_k, verbose=False)
                if recs is None or recs.empty:
                    failed += 1
                    continue
                all_recs = recs['movieId'].values
                recommended = np.array([m for m in all_recs if m not in exclude_items])[:max_k]
                if len(recommended) == 0:
                    failed += 1
                    continue

            elif model_name.startswith("Hybrid"):
                try:
                    # Smooth-only hybrid: use recommender.recommend without meta-learning args
                    all_recs_raw = recommender.recommend(user_id, n=max_k*5, exclude_rated=False)
                    if isinstance(all_recs_raw, pd.DataFrame):
                        if all_recs_raw.empty:
                            failed += 1
                            continue
                        all_recs = all_recs_raw['movieId'].values
                    else:
                        if len(all_recs_raw) == 0:
                            failed += 1
                            continue
                        all_recs = np.array(all_recs_raw) if not isinstance(all_recs_raw, np.ndarray) else all_recs_raw

                    recommended = np.array([m for m in all_recs if m not in exclude_items])[:max_k]
                    if len(recommended) == 0:
                        failed += 1
                        continue
                except Exception:
                    failed += 1
                    continue

            else:
                try:
                    recs = recommender.recommend(user_id, n=max_k*3, exclude_rated=False)
                except TypeError:
                    recs = recommender.recommend(user_id, n=max_k*3)

                if isinstance(recs, pd.DataFrame):
                    if recs.empty:
                        failed += 1
                        continue
                    all_recs = recs['movieId'].values
                else:
                    if len(recs) == 0:
                        failed += 1
                        continue
                    all_recs = np.array(recs) if not isinstance(recs, np.ndarray) else recs

                recommended = np.array([m for m in all_recs if m not in exclude_items])[:max_k]
                if len(recommended) == 0:
                    failed += 1
                    continue

            for k in k_values:
                top_k_list = list(recommended[:k])
                top_k = set(top_k_list)
                hits = len(test_relevant & top_k)
                precision = hits / k if k > 0 else 0.0
                recall = hits / len(test_relevant) if len(test_relevant) > 0 else 0.0

                # ARR (average rating of recommended top-k)
                arr = 0.0
                for mid in top_k_list:
                    movie = movies_df[movies_df['movieId'] == mid]
                    if len(movie) > 0:
                        arr += movie.iloc[0].get('rating_avg', 3.0)
                arr = arr / len(top_k_list) if len(top_k_list) > 0 else 0.0

                # NDCG@k (binary relevance)
                # DCG = sum_{i=1..k} rel_i / log2(i+1)
                rels = [1 if mid in test_relevant else 0 for mid in top_k_list]
                if len(rels) == 0:
                    ndcg = 0.0
                else:
                    denom = np.log2(np.arange(2, 2 + len(rels)))
                    dcg = np.sum(np.array(rels, dtype=float) / denom)
                    # ideal DCG: all relevant items at top positions, up to min(n_rel,k)
                    n_rel = min(len(test_relevant), k)
                    if n_rel == 0:
                        idcg = 0.0
                    else:
                        id_denom = np.log2(np.arange(2, 2 + n_rel))
                        idcg = np.sum(np.ones(n_rel, dtype=float) / id_denom)
                    ndcg = float(dcg / idcg) if idcg > 0 else 0.0

                # MAP@k (binary relevance)
                # AP = (1 / min(R, k)) * sum_{i=1..k} precision@i * rel_i
                ap = 0.0
                n_rel = min(len(test_relevant), k)
                if n_rel == 0:
                    ap = 0.0
                else:
                    cum_hits = 0
                    sum_precisions = 0.0
                    for idx, rel in enumerate(rels, start=1):
                        if rel:
                            cum_hits += 1
                            sum_precisions += cum_hits / idx
                    ap = float(sum_precisions / n_rel)

                results[k]['precision'].append(precision)
                results[k]['recall'].append(recall)
                results[k]['arr'].append(arr)
                results[k]['ndcg'].append(ndcg)
                results[k]['map'].append(ap)

            success += 1

        except Exception as e:
            failed += 1
            continue

    print(f"Complete: success={success}, failed={failed}")

    metrics = []
    for k in k_values:
        if len(results[k]['precision']) > 0:
            p = np.mean(results[k]['precision'])
            r = np.mean(results[k]['recall'])
            a = np.mean(results[k]['arr'])
            nd = np.mean(results[k]['ndcg']) if len(results[k]['ndcg'])>0 else 0.0
            mp = np.mean(results[k]['map']) if len(results[k]['map'])>0 else 0.0
            metrics.append({
                'model': model_name,
                'K': k,
                'Precision@K': p,
                'Recall@K': r,
                'ARR': a,
                'NDCG@K': nd,
                'MAP@K': mp,
                'n_users': len(results[k]['precision'])
            })
            print(f"{model_name} K={k}: Precision={p:.4f}, Recall={r:.4f}, ARR={a:.2f}, NDCG={nd:.4f}, MAP={mp:.4f}")

    return metrics


ranking_results = []

ranking_results.extend(evaluate_ranking_testset_v2(svd_rec, train_data, test_data, movies, "SVD", k_values=[5,10,20], n_users=N_USERS))
ranking_results.extend(evaluate_ranking_testset_v2(ncf_rec, train_data, test_data, movies, "NCF", k_values=[5,10,20], n_users=N_USERS))
ranking_results.extend(evaluate_ranking_testset_v2(content_rec, train_data, test_data, movies, "Content-Based", k_values=[5,10,20], n_users=N_USERS))
ranking_results.extend(evaluate_ranking_testset_v2(hybrid_rec, train_data, test_data, movies, "Hybrid-Smooth", k_values=[5,10,20], n_users=N_USERS))

ranking_df = pd.DataFrame(ranking_results)
print("\nFINAL RANKING RESULTS")
print(ranking_df.to_string(index=False))

# Save
ranking_df.to_csv('ranking_results_with_hybrid.csv', index=False)
print('Saved to ranking_results_with_hybrid.csv')