import pandas as pd
from tqdm import tqdm
from llama_cpp import Llama
import re
import torch
import sys
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))


from config import MODEL_LIST, MODELS_DIR, RAW_RESULT_DIR, PROCESSED_RESULT_DIR

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
print("Available models:", models)


test = "fewshot"
model_type = "m2"

output_dir = RAW_RESULT_DIR / model_type / test


few_shot_examples = [
    {
        "question": "Show me the top four diagnoses with the highest 6-month mortality rate.",
        "error": "no such column: d_icd_diagnoses.short_title",
        "original_sql": "SELECT d_icd_diagnoses.short_title FROM d_icd_diagnoses WHERE d_icd_diagnoses.icd_code IN (SELECT t2.icd_code FROM (SELECT t1.icd_code, DENSE_RANK() OVER (ORDER BY t1.c1 DESC) AS c2 FROM (SELECT diagnoses_icd.icd_code, 100 - SUM(CASE WHEN patients.hours_alive > 6 * 24 THEN 1 ELSE 0 END) * 100 / COUNT(*) AS c1 FROM diagnoses_icd INNER JOIN admissions ON diagnoses_icd.hadm_id = admissions.hadm_id WHERE admissions.subject_id = 10026354 GROUP BY diagnoses_icd.icd_code) AS t1) AS t2 WHERE t2.c2 <= 4)",
        "fixed_sql": "SELECT d_icd_diagnoses.long_title FROM d_icd_diagnoses WHERE d_icd_diagnoses.icd_code IN ( SELECT T4.icd_code FROM ( SELECT T3.icd_code, DENSE_RANK() OVER ( ORDER BY T3.C2 ASC ) AS C3 FROM ( SELECT T2.icd_code, AVG(C1) AS C2 FROM ( SELECT T1.icd_code, ( CASE WHEN strftime('%J',patients.dod) - strftime('%J',T1.charttime) < 1 * 365/2 THEN 0 ELSE 1 END ) as C1 FROM ( SELECT admissions.subject_id, diagnoses_icd.icd_code, diagnoses_icd.charttime FROM diagnoses_icd JOIN admissions ON diagnoses_icd.hadm_id = admissions.hadm_id WHERE strftime('%J',current_time) - strftime('%J',diagnoses_icd.charttime) >= 1 * 365/2 GROUP BY admissions.subject_id, diagnoses_icd.icd_code HAVING MIN(diagnoses_icd.charttime) = diagnoses_icd.charttime ) AS T1 JOIN patients ON T1.subject_id = patients.subject_id ) AS T2 GROUP BY T2.icd_code ) AS T3 ) AS T4 WHERE T4.C3 <= 4 );"
    },
    {
        "question": "Since 2100, what are the top five most frequent drugs prescribed to patients within 2 months after the diagnosis of atrial flutter?",
        "error": "incomplete input",
        "original_sql": "SELECT t3.drug FROM (SELECT t2.drug, DENSE_RANK() OVER (ORDER BY COUNT(*) DESC) AS c1 FROM (SELECT admissions.subject_id, diagnoses_icd.charttime FROM diagnoses_icd JOIN admissions ON diagnoses_icd.hadm_id = admissions.hadm_id WHERE diagnoses_icd.icd_code = (SELECT d_icd_diagnoses.icd_code FROM d_icd_diagnoses WHERE d_icd_diagnoses.long_title = 'atrial flutter') AND DATETIME(diagnoses_icd.charttime) >= DATETIME(CURRENT_TIME, '-14 month')) AS t1 JOIN (SELECT admissions.subject_id, prescriptions.drug, prescriptions.charttime FROM prescriptions JOIN admissions ON prescriptions.hadm_id = admissions.hadm_id WHERE DATETIME(prescriptions.charttime) >= DATETIME(CURRENT_TIME, '-14 month')) AS t2 ON t1.subject_id = t2",
        "fixed_sql": "SELECT T3.drug FROM ( SELECT T2.drug, DENSE_RANK() OVER ( ORDER BY COUNT(*) DESC ) AS C1 FROM ( SELECT admissions.subject_id, diagnoses_icd.charttime FROM diagnoses_icd JOIN admissions ON diagnoses_icd.hadm_id = admissions.hadm_id WHERE diagnoses_icd.icd_code = ( SELECT d_icd_diagnoses.icd_code FROM d_icd_diagnoses WHERE d_icd_diagnoses.long_title = 'atrial flutter' ) AND strftime('%Y',diagnoses_icd.charttime) >= '2100' ) AS T1 JOIN ( SELECT admissions.subject_id, prescriptions.drug, prescriptions.starttime FROM prescriptions JOIN admissions ON prescriptions.hadm_id = admissions.hadm_id WHERE strftime('%Y',prescriptions.starttime) >= '2100' ) AS T2 ON T1.subject_id = T2.subject_id WHERE T1.charttime < T2.starttime AND datetime(T2.starttime) BETWEEN datetime(T1.charttime) AND datetime(T1.charttime,'+2 month') GROUP BY T2.drug ) AS T3 WHERE T3.C1 <= 5;"
    },
    {
        "question": "What was the three most common lab test given in 2100?",
        "error": "Query execution timed out",
        "original_sql": "SELECT d_labitems.label FROM d_labitems WHERE d_labitems.itemid IN (SELECT t1.itemid FROM (SELECT labevents.itemid, DENSE_RANK() OVER (ORDER BY COUNT(*) DESC) AS c1 FROM labevents WHERE DATETIME(labevents.charttime) >= DATETIME(CURRENT_TIME, '-14 month') GROUP BY labevents.itemid) AS t1 WHERE t1.c1 <= 3)",
        "fixed_sql": "SELECT d_labitems.label FROM d_labitems WHERE d_labitems.itemid IN ( SELECT T1.itemid FROM ( SELECT labevents.itemid, DENSE_RANK() OVER ( ORDER BY COUNT(*) DESC ) AS C1 FROM labevents WHERE strftime('%Y',labevents.charttime) = '2100' GROUP BY labevents.itemid ) AS T1 WHERE T1.C1 <= 3 );"
    },
    {
        "question": "What was the drug that patient 10025463 was prescribed with during the same day after the catheter based invasive electrophysiologic testing?",
        "error": "incomplete input",
        "original_sql": "SELECT t2.drug_name FROM (SELECT admissions.subject_id, procedures_icd.chartdate FROM procedures_icd JOIN admissions ON procedures_icd.hadm_id = admissions.hadm_id WHERE admissions.subject_id = 10025463 AND procedures_icd.icd_code = ( SELECT d_icd_procedures.icd_code FROM d_icd_procedures WHERE d_icd_procedures.long_title = 'catheter based invasive electrophysiologic testing')) AS t1 JOIN (SELECT admissions.subject_id, prescriptions.drug_name, prescriptions.chartdate FROM prescriptions JOIN admissions ON prescriptions.hadm_id = admissions.hadm_id WHERE admissions.subject_id = 10025463) AS t2 ON t1.subject_id = t2.subject_id WHERE t1.chartdate < t2.chartdate AND DATETIME(t1.chartdate, 'start of day') = DATETIME(t2.chartdate",
        "fixed_sql": "SELECT T2.drug FROM ( SELECT procedures_icd.charttime, procedures_icd.hadm_id FROM procedures_icd WHERE procedures_icd.icd_code = ( SELECT d_icd_procedures.icd_code FROM d_icd_procedures WHERE d_icd_procedures.long_title = 'catheter based invasive electrophysiologic testing' ) AND procedures_icd.hadm_id IN (SELECT admissions.hadm_id FROM admissions WHERE admissions.subject_id = 10025463 ) ) AS T1 JOIN ( SELECT prescriptions.drug, prescriptions.starttime, prescriptions.hadm_id FROM prescriptions WHERE prescriptions.hadm_id IN (SELECT admissions.hadm_id FROM admissions WHERE admissions.subject_id = 10025463 ) ) AS T2 ON T1.hadm_id = T2.hadm_id WHERE T1.charttime < T2.starttime AND datetime(T1.charttime,'start of day') = datetime(T2.starttime,'start of day');"
    },
]


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


def create_fewshot_fix_query_prompt(row):
    original_query = row.get("generated_sql", "")
    if pd.isna(original_query) or not original_query:
        return None

    error_info = row.get("error", "")
    if pd.isna(error_info):
        error_info = ""

    question = row.get("question", "")

    prompt = (
        f"You are an expert at fixing SQL queries specialized for MIMIC-IV Database. "
        f"Fix the SQL query based on the following examples and the error message.\n\n"
    )
    
    for i, example in enumerate(few_shot_examples):
        prompt += f"Example {i+1}:\n"
        prompt += f"Question: {example['question']}\n"
        prompt += f"Original SQL query: {example['original_sql']}\n"
        if example['error']:
            prompt += f"Error message: {example['error']}\n"
        prompt += f"Fixed SQL query: {example['fixed_sql']}\n\n"
    
    prompt += f"Now fix this query:\n"
    prompt += f"Question: {question}\n"
    prompt += f"Original SQL query: {original_query}\n"
    
    if error_info:
        prompt += f"Error message: {error_info}\n"
    
    row_count = row.get("row_count", None)
    if row_count == 0 or row_count == 0.0:
        prompt += f"Note: This query currently returns 0 rows but should return results.\n"
    
    prompt += "Fixed SQL query: "

    return prompt


input_file = PROCESSED_RESULT_DIR / "m1" / "schema_aware" / "DuckDB_NSQL_7b_results.csv"

df = pd.read_csv(input_file)
print(f"Loaded input file with {len(df)} queries")

needs_fixing = df[(df["success"] == False)].copy()

print(f"Found {len(needs_fixing)} queries that need fixing")

if len(needs_fixing) == 0:
    print("No queries need fixing. Exiting.")
    sys.exit(0)

batch_size = 1
for model_name, model_path in tqdm(list(models.items())):
    print(f"Processing model: {model_name}")

    try:
        llm = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=-1)  # Increased context size for few-shot examples

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
                prompt = create_fewshot_fix_query_prompt(row)
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