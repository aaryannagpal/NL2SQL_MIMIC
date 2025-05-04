from pathlib import Path
import sys
PROJECT_ROOT = Path().resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.model_utils import M1Tester

model = 'duckdb_nsql'

m2_tester = M2Tester('duckdb_nsql')

m2, m2_tokenizer = m2_tester.load_model()

eval_df = m2_tester.evaluate_model()
eval_df.to_csv(f"{model}.csv", index=False)
print(f"Saved model with {len(eval_df)} rows")