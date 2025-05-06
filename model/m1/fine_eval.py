from pathlib import Path
import sys
PROJECT_ROOT = Path().resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.model_utils import M1Tester

model = 'duckdb_nsql'

m1_tester = M1Tester(model)

m1, m1_tokenizer = m1_tester.load_model()

eval_df = m1_tester.evaluate_model()
eval_df.to_csv(f"{model}.csv", index=False)
print(f"Saved model with {len(eval_df)} rows")