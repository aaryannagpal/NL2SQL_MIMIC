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

test = "fewshot"
model_type = "m1"

few_shot_examples = [
    {
        "question": "Has patient 10014078 undergone a central venous catheter placement with guidance procedure?",
        "sql": "SELECT COUNT(*)>0 FROM procedures_icd WHERE procedures_icd.icd_code = ( SELECT d_icd_procedures.icd_code FROM d_icd_procedures WHERE d_icd_procedures.long_title = 'central venous catheter placement with guidance' ) AND procedures_icd.hadm_id IN ( SELECT admissions.hadm_id FROM admissions WHERE admissions.subject_id = 10014078 )",
    },
    {
        "question": "What precautions should i take after a closed uterine biopsy procedure?",
        "sql": "No SQL query can answer this. This requires medical advice, not database querying.",
    },
    {
        "question": "When did last patient 10005348 have the minimum value of calculated total co2?",
        "sql": "SELECT labevents.charttime FROM labevents WHERE labevents.hadm_id IN ( SELECT admissions.hadm_id FROM admissions WHERE admissions.subject_id = 10005348 ) AND labevents.itemid IN ( SELECT d_labitems.itemid FROM d_labitems WHERE d_labitems.label = 'calculated total co2' ) ORDER BY labevents.valuenum ASC, labevents.charttime DESC LIMIT 1",
    },
    {
        "question": "How much did patient 10026354 weigh when measured for the last time since 14 months ago?",
        "sql": "SELECT chartevents.valuenum FROM chartevents WHERE chartevents.stay_id IN ( SELECT icustays.stay_id FROM icustays WHERE icustays.hadm_id IN ( SELECT admissions.hadm_id FROM admissions WHERE admissions.subject_id = 10026354 ) ) AND chartevents.itemid IN ( SELECT d_items.itemid FROM d_items WHERE d_items.label = 'daily weight' AND d_items.linksto = 'chartevents' ) AND datetime(chartevents.charttime) >= datetime(current_time,'-14 month') ORDER BY chartevents.charttime DESC LIMIT 1",
    },
]


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


def create_few_shot_prompt(question):
    prompt = "Convert these questions to SQL queries in [brackets]. If the question cannot be answered with a SQL query, respond with [No SQL query can answer this]:\n\n"

    for example in few_shot_examples:
        prompt += f"Question: {example['question']}\nSQL: [{example['sql']}]\n\n"

    prompt += f"Question: {question}\nSQL:"

    return prompt


batch_size = 1
for model_name, model_path in tqdm(models.items()):
    print(f"Processing model: {model_path}")
    try:
        llm = Llama(
            model_path=model_path, n_ctx=1024, n_gpu_layers=-1
        )  # Increased context size
        generated_sqls = []
        actual_response = []

        for i in tqdm(
            range(0, len(df), batch_size), desc=f"Generating SQL for {model_path}"
        ):
            batch_questions = df["question"][i : i + batch_size].tolist()

            batch_prompts = [create_few_shot_prompt(q) for q in batch_questions]
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
