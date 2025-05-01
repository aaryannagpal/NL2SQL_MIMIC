import sys
import os
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.query_visualizer import analyze_model_results
from config import PROCESSED_RESULT_DIR, STORE_ANALYSIS_DIR, STORE_RESULT_DIR

test = "schema_aware"
model_type = "m1"


ground_truth_path = str(STORE_RESULT_DIR / "original_query_eval" / "train_exec_results.csv")
model_results_dir = str(PROCESSED_RESULT_DIR / model_type / test)

model_results = {i.split('_results.csv')[0].replace('_', ' ') : os.path.join(model_results_dir, i) for i in os.listdir(model_results_dir)}
output_dir = "./"

print(model_results)

analyze_model_results(ground_truth_path, model_results, output_dir)
