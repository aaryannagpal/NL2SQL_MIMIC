import pandas as pd
from tqdm import tqdm
from llama_cpp import Llama
import re
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import MODEL_LIST, TRAINING_DATA, MODELS_DIR, RAW_RESULT_DIR

model_list = pd.read_csv(MODEL_LIST, header=0)

if not torch.cuda.is_available():
    raise RuntimeError(
        "GPU not available. Please ensure you have a CUDA-compatible GPU and the necessary drivers installed."
    )
device = "cuda"

df = pd.read_csv(TRAINING_DATA)
df = df.sample(frac=1)
path = MODELS_DIR
models = {
    row["Model Name"].replace(" ", "_"): path + row["Path"]
    for _, row in model_list.iterrows()
}
print(models)

test = "zeroshot"
model_type = "m1"


def generate_sql_batch(llm, prompts):
    sql_queries = []
    responses = []
    for prompt in prompts:
        try:
            response = llm(prompt, max_tokens=256, stop=["\n"], echo=False)
            text = response["choices"][0]["text"].strip()
            match = re.search(r"\[(.*?)\]", text)
            sql_queries.append(match.group(1).strip() if match else text)
            responses.append(response)
        except Exception as e:
            print(f"Error in inference: {str(e)}")
            sql_queries.append(f"ERROR")
            responses.append(None)
    return sql_queries, responses


batch_size = 1
for model_name, model_path in tqdm(models.items()):
    print(f"Processing model: {model_path}")
    try:
        llm = Llama(model_path=model_path, n_ctx=512, n_gpu_layers=-1)
        generated_sqls = []
        actual_response = []

        for i in tqdm(
            range(0, len(df), batch_size), desc=f"Generating SQL for {model_path}"
        ):
            batch_questions = df["question"][i : i + batch_size].tolist()
            batch_prompts = [
                f"Convert this question to a SQL query in [brackets]:\nQuestion: {q}\nSQL:"
                for q in batch_questions
            ]
            batch_sqls, responses = generate_sql_batch(llm, batch_prompts)
            generated_sqls.extend(batch_sqls)
            actual_response.extend(responses)

        df_model = df.copy()
        df_model["generated_sql"] = generated_sqls
        csv_filename = RAW_RESULT_DIR / model_type / test / f"{model_name}_results.csv"
        df_model.to_csv(csv_filename, index=False)
        print(f"Saved results to {csv_filename}")

        del llm
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error with model {model_name}: {str(e)}")
        torch.cuda.empty_cache()
        continue

print("Processing complete!")
