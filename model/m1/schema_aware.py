import pandas as pd
from tqdm import tqdm
from llama_cpp import Llama
import re
import torch
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import (
    MODEL_LIST,
    TRAINING_DATA,
    MODELS_DIR,
    RAW_RESULT_DIR,
    MIMIC_SCHEMA_PATH,
    DICTIONARY_MAP_PATH,
)

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
    if row["Chosen"]
}
print("\n\n", models, "\n\n")

test = "schema_aware"
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

with open(MIMIC_SCHEMA_PATH, "r") as f:
    schema = json.load(f)

def create_schema_string():
    """Create a concise schema description for the system prompt"""
    schema_str = "Database Schema:\n"
    
    # Add table descriptions
    for table_file, table_info in schema.items():
        table_name = table_info["name"]
        schema_str += f"- {table_name} table: Contains "
        
        # Add brief description based on column descriptions
        if "patients" in table_name:
            schema_str += "patient demographic information"
        elif "admissions" in table_name:
            schema_str += "hospital admission records"
        elif "icustays" in table_name:
            schema_str += "ICU stay information"
        elif "chart" in table_name:
            schema_str += "patient vital signs and measurements"
        elif "lab" in table_name:
            schema_str += "laboratory test results"
        elif "procedures" in table_name:
            schema_str += "procedure records with ICD codes"
        elif "diagnoses" in table_name:
            schema_str += "diagnosis records with ICD codes"
        elif "d_icd" in table_name:
            schema_str += "ICD code definitions"
        elif "d_item" in table_name:
            schema_str += "definitions for chart items"
        elif "prescription" in table_name:
            schema_str += "medication prescription records"
        elif "micro" in table_name:
            schema_str += "microbiology test results"
        elif "cost" in table_name:
            schema_str += "cost information for healthcare events"
        else:
            schema_str += "related healthcare data"
        
        # Add key columns
        cols = list(table_info["columns"].keys())
        key_cols = [c for c in cols if c in ["subject_id", "hadm_id", "stay_id", "itemid", "icd_code"]]
        if key_cols:
            schema_str += f" (key columns: {', '.join(key_cols)})"
        
        schema_str += "\n"
    
    # Add common joins
    schema_str += "\nCommon table joins:\n"
    schema_str += "- Join patients to admissions using subject_id\n"
    schema_str += "- Join admissions to icustays using hadm_id and subject_id\n"
    schema_str += "- Join icustays to chartevents using stay_id\n"
    schema_str += "- Join chartevents to d_items using itemid\n"
    schema_str += "- Join labevents to d_labitems using itemid\n"
    schema_str += "- Join diagnoses_icd to d_icd_diagnoses using icd_code\n"
    schema_str += "- Join procedures_icd to d_icd_procedures using icd_code\n"
    
    schema_str += "\nIMPORTANT: Not all questions can be answered with SQL queries. If a question asks for medical advice, interpretation of results, or contains information not present in the database, respond with 'No SQL query can answer this question' and briefly explain why."
    
    return schema_str

SCHEMA_STR = create_schema_string()

def create_schema_aware_prompt(question):
    prompt = """Convert these questions to SQL queries in [brackets]. If the question cannot be answered with a SQL query, respond with [No SQL query can answer this].

    Verify joins and filters accordingly.

    Schema:
    """

    prompt += f"{SCHEMA_STR}\n\n"

    prompt += f"Question: {question}\nSQL:"

    return prompt


batch_size = 1
for model_name, model_path in tqdm(models.items()):
    print(f"Processing model: {model_path}")
    try:
        llm = Llama(model_path=model_path, n_ctx=1024, n_gpu_layers=-1)
        generated_sqls = []
        actual_response = []

        for i in tqdm(
            range(0, len(df), batch_size), desc=f"Generating SQL for {model_path}"
        ):
            batch_questions = df["question"][i : i + batch_size].tolist()

            batch_prompts = [create_schema_aware_prompt(q) for q in batch_questions]
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
