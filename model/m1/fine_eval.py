from pathlib import Path
import sys
PROJECT_ROOT = Path().resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.model_utils import M1Tester

m1_tester = M1Tester()

m1, m1_tokenizer = m1_tester.load_model()

eval_df = m1_tester.evaluate_model()
eval_df.to_csv("test_results_v1_missing.csv", index=False)
print(f"Saved model with {len(eval_df)} rows")