"""
Multi-seed active learning benchmark.

Compares query strategies against random sampling under identical conditions
(same seed set, batch size, rounds and model), repeated over several seeds so
that differences can be judged against run-to-run noise.

Run from the project root:
    python -m src.al_benchmark
Writes al_benchmark_results.pkl, which the Streamlit app reads.
"""
import os
import pickle
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEEDS = [0, 1, 2, 3, 4]
STRATEGIES = ["Random", "Least Confidence", "Margin", "Entropy", "Diverse Margin"]
SEED_FRACTION = 0.1
BATCH_SIZE = 50
N_ROUNDS = 20
DIVERSE_CANDIDATE_FACTOR = 10  # Diverse Margin clusters the 10 * BATCH_SIZE most uncertain docs


def make_model():
    return LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)


def select_batch(strategy, model, X_pool, batch_size, rng):
    """Returns positions (relative to X_pool) of the documents to label next."""
    if strategy == "Random":
        return rng.choice(X_pool.shape[0], size=batch_size, replace=False)

    probas = model.predict_proba(X_pool)
    sorted_probas = np.sort(probas, axis=1)
    margin = sorted_probas[:, -1] - sorted_probas[:, -2]

    if strategy == "Least Confidence":
        scores = 1 - sorted_probas[:, -1]
    elif strategy == "Margin":
        scores = -margin
    elif strategy == "Entropy":
        scores = -np.sum(probas * np.log(probas + 1e-10), axis=1)
    elif strategy == "Diverse Margin":
        # Take the most uncertain candidates, cluster them, then pick the most
        # uncertain document from each cluster so the batch is not redundant.
        n_candidates = min(batch_size * DIVERSE_CANDIDATE_FACTOR, X_pool.shape[0])
        candidates = np.argsort(margin)[:n_candidates]
        kmeans = KMeans(n_clusters=batch_size, n_init=3, random_state=int(rng.integers(1_000_000)))
        clusters = kmeans.fit_predict(X_pool[candidates])
        chosen = []
        for c in range(batch_size):
            members = candidates[clusters == c]
            if len(members) > 0:
                chosen.append(members[np.argmin(margin[members])])
        # Top up with the next most uncertain docs if any cluster came back empty
        for idx in candidates:
            if len(chosen) >= batch_size:
                break
            if idx not in chosen:
                chosen.append(idx)
        return np.array(chosen)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return np.argsort(scores)[-batch_size:][::-1]


def run_single(strategy, seed, X_train, y_train, X_test, y_test):
    rng = np.random.default_rng(seed)
    all_idx = np.arange(X_train.shape[0])
    labeled_idx, _ = train_test_split(
        all_idx, train_size=SEED_FRACTION, random_state=seed, stratify=y_train
    )
    labeled_mask = np.zeros(X_train.shape[0], dtype=bool)
    labeled_mask[labeled_idx] = True

    rows = []
    for rnd in range(N_ROUNDS):
        model = make_model()
        model.fit(X_train[labeled_mask], y_train[labeled_mask])
        y_pred = model.predict(X_test)
        rows.append({
            'strategy': strategy,
            'seed': seed,
            'round': rnd,
            'n_labeled': int(labeled_mask.sum()),
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'macro_f1': f1_score(y_test, y_pred, average='macro'),
        })

        if rnd == N_ROUNDS - 1:
            break
        pool_idx = np.flatnonzero(~labeled_mask)
        chosen = select_batch(strategy, model, X_train[pool_idx], BATCH_SIZE, rng)
        labeled_mask[pool_idx[chosen]] = True

    return rows


def summarize(curves):
    """Final accuracy, area under the learning curve and labels-to-target, as mean ± std over seeds."""
    final = curves[curves['round'] == curves['round'].max()]
    random_final_mean = final[final['strategy'] == 'Random']['accuracy'].mean()

    per_run = []
    for (strategy, seed), g in curves.groupby(['strategy', 'seed']):
        g = g.sort_values('round')
        reached = g[g['accuracy'] >= random_final_mean]
        per_run.append({
            'strategy': strategy,
            'seed': seed,
            'final_accuracy': g['accuracy'].iloc[-1],
            'final_macro_f1': g['macro_f1'].iloc[-1],
            'aulc': g['accuracy'].mean(),
            'labels_to_target': reached['n_labeled'].iloc[0] if len(reached) else np.nan,
        })
    per_run = pd.DataFrame(per_run)

    summary = per_run.groupby('strategy').agg(
        final_acc_mean=('final_accuracy', 'mean'),
        final_acc_std=('final_accuracy', 'std'),
        final_macro_f1_mean=('final_macro_f1', 'mean'),
        aulc_mean=('aulc', 'mean'),
        aulc_std=('aulc', 'std'),
        labels_to_target_mean=('labels_to_target', 'mean'),
        runs_reaching_target=('labels_to_target', lambda s: int(s.notna().sum())),
    ).reindex(STRATEGIES)

    # Paired comparison against random: the same seed means the same starting set
    random_runs = per_run[per_run['strategy'] == 'Random'].set_index('seed')
    diffs = {}
    for strategy in STRATEGIES:
        if strategy == 'Random':
            continue
        s = per_run[per_run['strategy'] == strategy].set_index('seed')
        d = (s['final_accuracy'] - random_runs['final_accuracy']) * 100
        diffs[strategy] = {
            'mean_gain_pp': d.mean(),
            'std_gain_pp': d.std(),
            'seeds_beating_random': int((d > 0).sum()),
        }
    summary = summary.join(pd.DataFrame(diffs).T)
    return summary, per_run, random_final_mean


def main():
    with open(os.path.join(BASE_DIR, 'train_test_data.pkl'), 'rb') as f:
        data = pickle.load(f)
    X_train, X_test = data['X_train'].tocsr(), data['X_test']
    y_train, y_test = np.asarray(data['y_train']), np.asarray(data['y_test'])

    jobs = [(s, seed) for s in STRATEGIES for seed in SEEDS]
    print(f"Running {len(jobs)} runs ({len(STRATEGIES)} strategies x {len(SEEDS)} seeds)...")
    start = time.time()
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(run_single)(s, seed, X_train, y_train, X_test, y_test) for s, seed in jobs
    )
    print(f"Done in {time.time() - start:.0f}s")

    curves = pd.DataFrame([row for run in results for row in run])
    summary, per_run, target = summarize(curves)

    pd.set_option('display.width', 200)
    print(f"\nTarget accuracy (Random's mean final accuracy): {target*100:.2f}%\n")
    print(summary.round(4).to_string())

    best = summary.drop('Random')['mean_gain_pp'].idxmax()
    output = {
        'curves': curves,
        'summary': summary,
        'per_run': per_run,
        'target_accuracy': target,
        'best_strategy': best,
        'best_gain_pp': float(summary.loc[best, 'mean_gain_pp']),
        'config': {
            'seeds': SEEDS,
            'strategies': STRATEGIES,
            'seed_fraction': SEED_FRACTION,
            'batch_size': BATCH_SIZE,
            'n_rounds': N_ROUNDS,
            'model': 'LogisticRegression(C=1.0, lbfgs)',
            'features': 'TF-IDF (5000)',
        },
    }
    with open(os.path.join(BASE_DIR, 'al_benchmark_results.pkl'), 'wb') as f:
        pickle.dump(output, f)
    print("\nSaved al_benchmark_results.pkl")


if __name__ == "__main__":
    main()
