from pathlib import Path
import sys
PROJECT_ROOT = Path().resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.model_utils import M2Tester

model = "qwen_2.5"

m2_tester = M2Tester(model)

_, _ = m2_tester.load_model()

eval_df = m2_tester.evaluate_model('Qwen_2.5_14b_results')
eval_df.to_csv(f"{model}.csv", index=False)
print(f"Saved model with {len(eval_df)} rows")