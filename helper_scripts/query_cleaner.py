import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import sys

PROJECT_ROOT = Path().resolve().parent
sys.path.append(str(PROJECT_ROOT))

from utils.query_evaluator import QueryEvaluator
from config import MYSQL_DB_PATH

test = 'finetune'
model_type = 'm1'
query_col = 'generated_query'

raw_base  = os.path.join(PROJECT_ROOT, f'./results/raw/{model_type}')
semi_base = os.path.join(PROJECT_ROOT, f'./results/semi_processed/{model_type}')
final_base = os.path.join(PROJECT_ROOT, f'./results/processed/{model_type}')

evaluator = QueryEvaluator(db_path = MYSQL_DB_PATH)

raw_dir  = os.path.join(raw_base,  test)
semi_dir = os.path.join(semi_base, test)
final_dir = os.path.join(final_base, test)

for fname in tqdm(os.listdir(raw_dir)):
    print(f"Processing {fname}...")
    raw_path  = os.path.join(raw_dir,  fname)
    semi_path = os.path.join(semi_dir, fname)

    df = pd.read_csv(raw_path)
    df[query_col] = df[query_col].apply(lambda x: evaluator.clean_query(x))
    
    df.to_csv(semi_path, index=False)
