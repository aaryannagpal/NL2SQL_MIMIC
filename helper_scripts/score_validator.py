import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import sys

PROJECT_ROOT = Path().resolve().parent
sys.path.append(str(PROJECT_ROOT))

from utils.query_analyzer import QueryAnalyzer

analyzer = QueryAnalyzer()

from scipy.stats import spearmanr
import json

BASE_WEIGHTS = {
    'results_match':            0.30,
    'table_access_accuracy':    0.20,
    'column_access_accuracy':   0.20,
    'query_similarity':         0.10,
    'plan_similarity':          0.20
}

def compute_model_scores(df, weights):
    agg = df.groupby('model_name').agg({
        k: 'mean' for k in weights.keys()
    }).reset_index()
    for k, w in weights.items():
        agg[k] = agg[k] * w
    agg['composite_score'] = agg[list(weights.keys())].sum(axis=1)
    return agg[['model_name','composite_score']]

def monte_carlo_sensitivity(df, base_weights, n_iter=1000, perturb=0.1):
    base = compute_model_scores(df, base_weights) \
             .sort_values('composite_score', ascending=False) \
             .reset_index(drop=True)
    base['rank'] = base.index + 1
    base_order = base['model_name'].tolist()

    rhos = []
    for _ in range(n_iter):

        pw = {k: v * np.random.uniform(1-perturb,1+perturb)
              for k,v in base_weights.items()}
        total = sum(pw.values())
        pw = {k: v/total for k,v in pw.items()}

        pert = compute_model_scores(df, pw) \
                 .sort_values('composite_score', ascending=False)
        pert_order = pert['model_name'].tolist()

        base_ranks = list(range(1, len(base_order)+1))
        pert_ranks = [pert_order.index(m)+1 for m in base_order]
        rho, _ = spearmanr(base_ranks, pert_ranks)
        rhos.append(rho)
    return np.array(rhos)

def tornado_analysis(df, base_weights, delta=0.2, steps=21):
    models = df['model_name'].unique().tolist()
    metrics = list(base_weights.keys())
    results = {}

    for m in metrics:
        weights_list = np.linspace(1-delta, 1+delta, steps)
        ranks = {mod: [] for mod in models}
        ws = []
        for fac in weights_list:
            w = base_weights.copy()
            w[m] *= fac

            tot = sum(w.values())
            w = {k: v/tot for k,v in w.items()}
            df_sc = compute_model_scores(df, w) \
                       .sort_values('composite_score', ascending=False)
            order = df_sc['model_name'].tolist()
            for mod in models:
                ranks[mod].append(order.index(mod)+1)
            ws.append(w[m])

        results[m] = {'weights': ws, 'ranks': ranks}
    return results

from utils.query_visualizer import analyze_model_results
from config import PROCESSED_RESULT_DIR, STORE_ANALYSIS_DIR, STORE_RESULT_DIR

# testing on a random dataset
test = "zeroshot"
model_type = "m1"


ground_truth_path = str(STORE_RESULT_DIR / "original_query_eval" / "train_exec_results.csv")
model_results_dir = str(PROCESSED_RESULT_DIR / model_type / test)

model_results = {i.split('_results.csv')[0].replace('_', ' ') : os.path.join(model_results_dir, i) for i in os.listdir(model_results_dir)}
output_dir = "./"

_, d = analyze_model_results(ground_truth_path, model_results, output_dir, 'generated_sql', test)

frames = []
for model_name, df in d.items():
    temp = df.copy()
    temp.rename(columns={'plan_plan_similarity':'plan_similarity'}, inplace=True)
    temp['model_name'] = model_name
    frames.append(temp)
merged = pd.concat(frames, ignore_index=True)

rhos = monte_carlo_sensitivity(merged, BASE_WEIGHTS,
                                n_iter=1000, perturb=0.2)

print(f"Sensitivity (Spearman ρ): min={rhos.min():.3f}, "
        f"median={np.median(rhos):.3f}, max={rhos.max():.3f}")


tornado = tornado_analysis(merged, BASE_WEIGHTS,
                            delta=0.5, steps=21)
with open('tornado_results.json', 'w') as f:
    json.dump(tornado, f, indent=2)
print("Tornado analysis saved to tornado_results.json")
