import pandas as pd
from tqdm import tqdm
from llama_cpp import Llama
import re
import torch
import sys
import os, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import (
    MODEL_LIST,
    MODELS_DIR,
    RAW_RESULT_DIR,
    PROCESSED_RESULT_DIR,
    MIMIC_SCHEMA_PATH,
    DICTIONARY_MAP_PATH,
)

model_list = pd.read_csv(MODEL_LIST, header=0)

if not torch.cuda.is_available():
    raise RuntimeError(
        "GPU not available. Please ensure you have a CUDA-compatible GPU and the necessary drivers installed."
    )
device = "cuda"

path = MODELS_DIR
models = {
    row["Model Name"].replace(" ", "_"): path + row["Path"]
    for _, row in model_list.iterrows()
    if row["Chosen-2"]
}
print("Available models:", models)


test = "schema_aware"
model_type = "m2"

output_dir = RAW_RESULT_DIR / model_type / test
os.makedirs(output_dir, exist_ok=True)

# Define input file (from M1 results)
input_file_pattern = str(PROCESSED_RESULT_DIR / "m1" / "schema_aware" / "{}_results.csv")

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

def create_schema_aware_fix_prompt(row):
    original_query = row.get("generated_sql", "")
    if pd.isna(original_query) or not original_query:
        return None

    error_info = row.get("error", "")
    if pd.isna(error_info):
        error_info = ""

    question = row.get("question", "")

    prompt = f"""You are an expert at fixing SQL queries specialized for MIMIC-IV Database. Fix the SQL query based on the error message and database schema information.

    Verify joins and filters accordingly.

    Schema: {SCHEMA_STR}

    Question: {question}

    Original SQL query: {original_query}
    """
    
    if error_info:
        prompt += f"Error message: {error_info}\n"
    
    prompt += "Fixed SQL query: "

    return prompt


def generate_fixed_sql_batch(llm, prompts):
    fixed_sql_queries = []
    responses = []

    for prompt in prompts:
        try:
            response = llm(
                prompt,
                max_tokens=512,
                temperature=0.1,
                top_p=0.95,
                repeat_penalty=1.1,
                echo=False,
            )

            raw_text = (
                response["choices"][0]["text"]
                if "choices" in response and response["choices"]
                else ""
            )

            text = raw_text.strip()

            match = re.search(r"\[(.*?)\]", text, re.DOTALL)
            if match:
                fixed_sql = match.group(1).strip()
            else:
                sql_match = re.search(
                    r"Fixed SQL query:\s*(SELECT.*?)(?:$|```|;|\n\n)",
                    text,
                    re.IGNORECASE | re.DOTALL,
                )
                if sql_match:
                    fixed_sql = sql_match.group(1).strip()
                else:
                    select_match = re.search(
                        r"(SELECT.*?)(?:$|```|;|\n\n)", text, re.IGNORECASE | re.DOTALL
                    )
                    if select_match:
                        fixed_sql = select_match.group(1).strip()
                    else:
                        fixed_sql = text

            fixed_sql = re.sub(r"^```sql\s*", "", fixed_sql, flags=re.IGNORECASE)
            fixed_sql = re.sub(r"^```\s*", "", fixed_sql, flags=re.IGNORECASE)
            fixed_sql = re.sub(r"\s*```$", "", fixed_sql)

            fixed_sql = fixed_sql.strip()
            fixed_sql = fixed_sql.replace("\n", " ")

            if fixed_sql and not fixed_sql.endswith(";"):
                fixed_sql = fixed_sql + ";"

            fixed_sql_queries.append(fixed_sql)
            responses.append(response)

        except Exception as e:
            print(f"Error in inference: {str(e)}")
            fixed_sql_queries.append("ERROR")
            responses.append(None)

    return fixed_sql_queries, responses


# Process each input model separately
for input_model in ["DuckDB_NSQL_7b"]:  # Add more models as needed
    input_file = input_file_pattern.format(input_model)
    
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        continue
        
    df = pd.read_csv(input_file)
    print(f"Loaded input file with {len(df)} queries for {input_model}")

    needs_fixing = df[(df["success"] == False)].copy()
    print(f"Found {len(needs_fixing)} queries from {input_model} that need fixing")

    if len(needs_fixing) == 0:
        print(f"No queries need fixing for {input_model}. Skipping.")
        continue

    batch_size = 1
    for model_name, model_path in tqdm(list(models.items())):
        print(f"Processing model: {model_name} for fixing {input_model} queries")

        try:
            llm = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=-1)

            result_df = df.copy()
            result_df.drop(
                columns=[
                    "success",
                    "execution_time",
                    "total_time",
                    "row_count",
                    "error",
                    "results",
                    "result_columns",
                    "execution_plan",
                ],
                inplace=True,
                errors="ignore",
            )

            result_df["valid_sql"] = result_df["generated_sql"]
            result_df["validated"] = 0

            for i in tqdm(
                range(0, len(needs_fixing), batch_size),
                desc=f"Fixing {input_model} queries with {model_name}",
            ):
                batch_rows = needs_fixing.iloc[i : i + batch_size]

                batch_prompts = []
                batch_indices = []
                for idx, row in batch_rows.iterrows():
                    prompt = create_schema_aware_fix_prompt(row)
                    if prompt:
                        batch_prompts.append(prompt)
                        batch_indices.append(idx)
                    else:
                        print(
                            f"Warning: Couldn't create prompt for row id: {row.get('id', 'unknown')}"
                        )

                if not batch_prompts:
                    continue

                batch_fixed, _ = generate_fixed_sql_batch(llm, batch_prompts)

                for j, fixed_query in enumerate(batch_fixed):
                    if j < len(batch_indices):  # Safety check
                        idx = batch_indices[j]
                        if (
                            fixed_query
                            and fixed_query != "ERROR"
                            and fixed_query.strip().upper().startswith("SELECT")
                        ):
                            result_df.at[idx, "valid_sql"] = fixed_query
                            result_df.at[idx, "validated"] = 1
                        else:
                            if fixed_query.strip().upper().startswith("SELECT"):
                                print("Invalid SQL query generated")
                            elif fixed_query == "ERROR":
                                print(f"Error in generating SQL for row {idx}")
                            elif not fixed_query:
                                print(f"Empty SQL query generated for row {idx}")
                            else:
                                print(f"Failed to validate query for row {idx}")

                if (i + batch_size) % 10 == 0 or (i + batch_size) >= len(needs_fixing):
                    csv_filename = output_dir / f"{model_name}_intermediate.csv"
                    result_df.to_csv(csv_filename, index=False)
                    print(f"Saved intermediate results to {csv_filename}")

            validation_count = result_df["validated"].sum()
            validation_rate = (
                validation_count / len(needs_fixing) if len(needs_fixing) > 0 else 0
            )
            print(
                f"Fixed {validation_count} out of {len(needs_fixing)} queries ({validation_rate:.2%})"
            )

            csv_filename = output_dir / f"{model_name}_results.csv"
            result_df.to_csv(csv_filename, index=False)
            print(f"Saved results to {csv_filename}")

            del llm
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error with model {model_name}: {str(e)}")
            torch.cuda.empty_cache()
            continue

print("Processing complete!")