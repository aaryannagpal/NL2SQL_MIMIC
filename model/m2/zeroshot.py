import pandas as pd
from tqdm import tqdm
from llama_cpp import Llama
import re
import torch
import sys
import os
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))


from config import MODEL_LIST, MODELS_DIR, RAW_RESULT_DIR, PROCESSED_RESULT_DIR
from utils.query_handler import QueryHandler
from utils.query_evaluator import QueryEvaluator

if not torch.cuda.is_available():
    raise RuntimeError(
        "GPU not available. Please ensure you have a CUDA-compatible GPU and the necessary drivers installed."
    )
device = "cuda"

model_list = pd.read_csv(MODEL_LIST, header=0)
path = MODELS_DIR
models = {
    row["Model Name"].replace(" ", "_"): path + row["Path"]
    for _, row in model_list.iterrows()
}
print("Available models:", list(models.keys()))
test = "zeroshot"
model_type = "m2"

output_dir = RAW_RESULT_DIR / model_type / test


def generate_fixed_sql_batch(llm, prompts):
    fixed_sql_queries = []
    responses = []

    for prompt in prompts:
        try:
            # print(f"Processing prompt: {prompt}")

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
            # print(f"Raw response from model: {raw_text}")

            text = raw_text.strip()

            match = re.search(r"\[(.*?)\]", text, re.DOTALL)
            if match:
                fixed_sql = match.group(1).strip()
                # print("Found SQL in brackets")
            else:
                sql_match = re.search(
                    r"Fixed SQL query:\s*(SELECT.*?)(?:$|```|;|\n\n)",
                    text,
                    re.IGNORECASE | re.DOTALL,
                )
                if sql_match:
                    fixed_sql = sql_match.group(1).strip()
                    # print("Found SQL after 'Fixed SQL query:'")
                else:
                    select_match = re.search(
                        r"(SELECT.*?)(?:$|```|;|\n\n)", text, re.IGNORECASE | re.DOTALL
                    )
                    if select_match:
                        fixed_sql = select_match.group(1).strip()
                        # print("Found standalone SELECT statement")
                    else:
                        fixed_sql = text
                        # print(
                            # "WARNING: Could not find SQL pattern, using entire response"
                        # )

            fixed_sql = re.sub(r"^```sql\s*", "", fixed_sql, flags=re.IGNORECASE)
            fixed_sql = re.sub(r"^```\s*", "", fixed_sql, flags=re.IGNORECASE)
            fixed_sql = re.sub(r"\s*```$", "", fixed_sql)

            fixed_sql = fixed_sql.strip()
            fixed_sql = fixed_sql.replace("\n", " ")

            if fixed_sql and not fixed_sql.endswith(";"):
                fixed_sql = fixed_sql + ";"

            fixed_sql_queries.append(fixed_sql)
            responses.append(response)

            # print(f"Extracted SQL: {fixed_sql}")

        except Exception as e:
            print(f"Error in inference: {str(e)}")
            fixed_sql_queries.append("ERROR")
            responses.append(None)

    return fixed_sql_queries, responses


def create_fix_query_prompt(row):
    original_query = row.get("generated_sql", "")
    if pd.isna(original_query) or not original_query:
        return None

    error_info = row.get("error", "")
    if pd.isna(error_info):
        error_info = ""

    question = row.get("question", "")

    prompt = (
        f"You are an expert at fixing SQL queries specilized for MIMIC-IV Database. Please fix this SQL query that has errors.\n\n"
        f"Question: {question}\n\n"
        f"Original SQL query: {original_query}\n\n"
    )

    if error_info:
        prompt += f"Error message: {error_info}\n\n"

    prompt += "Fixed SQL query: "

    return prompt


input_file = PROCESSED_RESULT_DIR / "m1" / "schema_aware" / "DuckDB_NSQL_7b_results.csv"

df = pd.read_csv(input_file)
print(f"Loaded input file with {len(df)} queries")

needs_fixing = df[(df["success"] == False)].copy()

print(f"Found {len(needs_fixing)} queries that need fixing")

if len(needs_fixing) == 0:
    print("No queries need fixing. Exiting.")


batch_size = 1
for model_name, model_path in tqdm(
    list(models.items())
):

    print(f"Processing model: {model_name}")

    try:
        llm = Llama(model_path=model_path, n_ctx=1024, n_gpu_layers=-1)

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
            desc=f"Fixing queries with {model_name}",
        ):

            batch_rows = needs_fixing.iloc[i : i + batch_size]

            batch_prompts = []
            batch_indices = []
            for idx, row in batch_rows.iterrows():
                prompt = create_fix_query_prompt(row)
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
            idx = batch_indices[j]
            if (
                fixed_query
                and fixed_query != "ERROR"
                and fixed_query.strip().upper().startswith("SELECT")
            ):
                result_df.at[idx, "valid_sql"] = fixed_query
                result_df.at[idx, "validated"] = 1
                print(f"Successfully validated query for row {idx}")
            else:
                print(f"Failed to validate query for row {idx}: {fixed_query}")

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
