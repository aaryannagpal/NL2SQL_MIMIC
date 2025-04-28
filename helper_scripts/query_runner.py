import pandas as pd
from tqdm import tqdm
from pathlib import Path
import sys, os

PROJECT_ROOT = Path().resolve().parent
sys.path.append(str(PROJECT_ROOT))

from utils.query_evaluator import QueryEvaluator
from config import DATASET_PATH

test = 'finetune'
model_type = 'm1'
query_col = 'generated_query'

raw_base  = os.path.join(PROJECT_ROOT, f'./results/raw/{model_type}')
semi_base = os.path.join(PROJECT_ROOT, f'./results/semi_processed/{model_type}')
final_base = os.path.join(PROJECT_ROOT, f'./results/processed/{model_type}')

evaluator = QueryEvaluator(db_path = '/media/chs.gpu/DATA/nagpal/modified_mimic/data/mimic_iv/mimic_iv.sqlite')

raw_dir  = os.path.join(raw_base,  test)
semi_dir = os.path.join(semi_base, test)
final_dir = os.path.join(final_base, test)

test_df = pd.read_csv(os.path.join(DATASET_PATH, 'test.csv'))

for i in os.listdir(semi_dir):
    print(i)
    df = pd.read_csv(os.path.join(semi_dir, i))

    df = df.merge(test_df[['id', 'result']], on = 'id', how = 'left')

    df.rename(columns={'result':'true_result'}, inplace=True)
    print(len(df), df.columns)
    evaluator.batch_evaluate(df, os.path.join(final_dir, i), query_col, max_workers = 30)