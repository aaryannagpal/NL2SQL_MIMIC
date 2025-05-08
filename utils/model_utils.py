import torch
import pandas as pd
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import json
from tqdm import tqdm
import re
from pathlib import Path
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MIMIC_SCHEMA_PATH,
    DATASET_PATH,
    SAMPLE_M1_MODEL_DIR,
    SAMPLE_M2_MODEL_DIR,
    PROCESSED_RESULT_DIR,
    RAW_RESULT_DIR
);
def create_schema_string():
    with open(MIMIC_SCHEMA_PATH, "r") as f:
        schema = json.load(f)

    schema_str = "Database Schema:\n"
    
    for table_file, table_info in schema.items():
        table_name = table_info["name"]
        schema_str += f"- {table_name} table: Contains "
        
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
        
        cols = list(table_info["columns"].keys())
        key_cols = [c for c in cols if c in ["subject_id", "hadm_id", "stay_id", "itemid", "icd_code"]]
        if key_cols:
            schema_str += f" (key columns: {', '.join(key_cols)})"
        
        schema_str += "\n"
    
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

class M1Tester:
    def __init__(self, name = 'v1', model_dir = None):
        self.model = None
        self.tokenizer = None
        if model_dir is None:
            self.model_path = os.path.join(SAMPLE_M1_MODEL_DIR, name)
        else:
            self.model_path = os.path.join(model_dir, name)
        
        self.TEST_DATA_PATH = str(DATASET_PATH / "test.csv")
        self.VALIDATION_DATA_PATH = str(DATASET_PATH / "valid.csv")
        self.TRAINING_DATA_PATH = str(DATASET_PATH / "train.csv")


    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        model = AutoPeftModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = tokenizer
        self.model = model
        return model, tokenizer
    
    def generate_sql(self, question):
        SCHEMA_STR = create_schema_string()
        prompt = f"""<|im_start|>system
        You are a SQL assistant specialized in medical database queries. Your task is to convert natural language questions into SQL queries for the MIMIC medical database.

        <|im_end|>
        <|im_start|>user
        {question}<|im_end|>
        <|im_start|>assistant
        """
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.01,
            top_p=0.9,
            do_sample=False,
            num_beams=1,
            early_stopping=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        return response.split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0].strip()

    def evaluate_model(self, test_path = None):
        test_df = pd.read_csv(self.TEST_DATA_PATH) if test_path is None else pd.read_csv(test_path)
        results = []
        
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
            question = row["question"]
            true_query = row.get("true_query", "")

            generated_query = self.generate_sql(question)
            
            results.append({
                "id": row.get("id", idx),
                "question": question,
                "true_query": true_query,
                "generated_query": generated_query,
            })
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1} questions")
        
        results_df = pd.DataFrame(results)
        return results_df

class M2Tester:
    def __init__(self, name='v1', model_dir=None):
        self.name = name
        self.model = None
        self.tokenizer = None
        if model_dir is None:
            self.model_path = os.path.join(SAMPLE_M2_MODEL_DIR, name)
        else:
            self.model_path = os.path.join(model_dir, name)
        
        self.TEST_DATA_PATH = str(DATASET_PATH / "test.csv")
        self.VALIDATION_DATA_PATH = str(DATASET_PATH / "valid.csv")
        self.TRAINING_DATA_PATH = str(DATASET_PATH / "train.csv")

        self.few_shot_examples = [
            {
                "question": "Show me the top four diagnoses with the highest 6-month mortality rate.",
                "error": "no such column: d_icd_diagnoses.short_title",
                "original_sql": "SELECT d_icd_diagnoses.short_title FROM d_icd_diagnoses WHERE d_icd_diagnoses.icd_code IN (SELECT t2.icd_code FROM (SELECT t1.icd_code, DENSE_RANK() OVER (ORDER BY t1.c1 DESC) AS c2 FROM (SELECT diagnoses_icd.icd_code, 100 - SUM(CASE WHEN patients.hours_alive > 6 * 24 THEN 1 ELSE 0 END) * 100 / COUNT(*) AS c1 FROM diagnoses_icd INNER JOIN admissions ON diagnoses_icd.hadm_id = admissions.hadm_id WHERE admissions.subject_id = 10026354 GROUP BY diagnoses_icd.icd_code) AS t1) AS t2 WHERE t2.c2 <= 4)",
                "fixed_sql": "SELECT d_icd_diagnoses.long_title FROM d_icd_diagnoses WHERE d_icd_diagnoses.icd_code IN ( SELECT T4.icd_code FROM ( SELECT T3.icd_code, DENSE_RANK() OVER ( ORDER BY T3.C2 ASC ) AS C3 FROM ( SELECT T2.icd_code, AVG(C1) AS C2 FROM ( SELECT T1.icd_code, ( CASE WHEN strftime('%J',patients.dod) - strftime('%J',T1.charttime) < 1 * 365/2 THEN 0 ELSE 1 END ) as C1 FROM ( SELECT admissions.subject_id, diagnoses_icd.icd_code, diagnoses_icd.charttime FROM diagnoses_icd JOIN admissions ON diagnoses_icd.hadm_id = admissions.hadm_id WHERE strftime('%J',current_time) - strftime('%J',diagnoses_icd.charttime) >= 1 * 365/2 GROUP BY admissions.subject_id, diagnoses_icd.icd_code HAVING MIN(diagnoses_icd.charttime) = diagnoses_icd.charttime ) AS T1 JOIN patients ON T1.subject_id = patients.subject_id ) AS T2 GROUP BY T2.icd_code ) AS T3 ) AS T4 WHERE T4.C3 <= 4 );"
            },
            {
                "question": "What was the three most common lab test given in 2100?",
                "error": "Query execution timed out",
                "original_sql": "SELECT d_labitems.label FROM d_labitems WHERE d_labitems.itemid IN (SELECT t1.itemid FROM (SELECT labevents.itemid, DENSE_RANK() OVER (ORDER BY COUNT(*) DESC) AS c1 FROM labevents WHERE DATETIME(labevents.charttime) >= DATETIME(CURRENT_TIME, '-14 month') GROUP BY labevents.itemid) AS t1 WHERE t1.c1 <= 3)",
                "fixed_sql": "SELECT d_labitems.label FROM d_labitems WHERE d_labitems.itemid IN ( SELECT T1.itemid FROM ( SELECT labevents.itemid, DENSE_RANK() OVER ( ORDER BY COUNT(*) DESC ) AS C1 FROM labevents WHERE strftime('%Y',labevents.charttime) = '2100' GROUP BY labevents.itemid ) AS T1 WHERE T1.C1 <= 3 );"
            }
        ]

    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        model = AutoPeftModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = tokenizer
        self.model = model
        print(f"Loaded model {model}")
        return model, tokenizer
    
    def fix_sql(self, question, original_sql, error=None):
        """Generate a fixed SQL query for a failed query"""
        
        # Create few-shot examples string
        examples_str = ""
        for i, example in enumerate(self.few_shot_examples):
            examples_str += f"Example {i+1}:\n"
            examples_str += f"Question: {example['question']}\n"
            examples_str += f"Original SQL query: {example['original_sql']}\n"
            if example.get('error'):
                examples_str += f"Error message: {example['error']}\n"
            examples_str += f"Fixed SQL query: {example['fixed_sql']}\n\n"
        
        prompt = f"""
                <|im_start|>system
                You are a SQL expert specializing in fixing failed SQL queries for the MIMIC-IV medical database. 
                Your task is to analyze and fix SQL queries that have errors.

                Note these MIMIC-IV schema modifications: (1) Added `charttime` to tables like `diagnoses_icd` (using admittime) and `procedures_icd` (using chartdate); 
                (2) Synthetic `cost` table links to events via `event_type`/`event_id`; 
                (3) Added computed `age` column in admissions; 
                (4) Only selected clinical items retained.
                All timestamps are standardized and may be temporally shifted.
                <|im_end|>

                <|im_start|>user
                I need you to fix a SQL query for the MIMIC database based on these examples:

                {examples_str}
                Now fix this query:

                Question: {question}
                Original SQL query: {original_sql}
                """

        if error:
            prompt += f"Error message: {error}\n"
        
        prompt += "Fixed SQL query:"
        prompt += "<|im_end|>"
        
        prompt += "\n<|im_start|>assistant\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.95,
            do_sample=False,
            num_beams=1,
            early_stopping=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        return self.extract_sql_query(response.split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0].strip())
    
    def extract_sql_query(self, text):
        """Extract the SQL query from the model's response"""
        # Try to find the SQL query within brackets
        match = re.search(r"\[(.*?)\]", text, re.DOTALL)
        if match:
            fixed_sql = match.group(1).strip()
        else:
            # Try to find the SQL query by looking for a SELECT statement
            sql_match = re.search(
                r"Fixed SQL query:\s*(SELECT.*?)(?:$|```|;|\n\n)",
                text,
                re.IGNORECASE | re.DOTALL
            )
            if sql_match:
                fixed_sql = sql_match.group(1).strip()
            else:
                select_match = re.search(
                    r"(SELECT.*?)(?:$|```|;|\n\n)",
                    text,
                    re.IGNORECASE | re.DOTALL
                )
                if select_match:
                    fixed_sql = select_match.group(1).strip()
                else:
                    fixed_sql = text
        
        # Clean up the SQL query
        fixed_sql = re.sub(r"^```sql\s*", "", fixed_sql, flags=re.IGNORECASE)
        fixed_sql = re.sub(r"^```\s*", "", fixed_sql, flags=re.IGNORECASE)
        fixed_sql = re.sub(r"\s*```$", "", fixed_sql)
        fixed_sql = fixed_sql.strip()
        fixed_sql = fixed_sql.replace("\n", " ")
        
        if fixed_sql and not fixed_sql.endswith(";"):
            fixed_sql = fixed_sql + ";"
            
        return fixed_sql
    
    def evaluate_model(self, test_path):
        test_path = PROCESSED_RESULT_DIR / "m1" / "finetune" / f"{test_path}.csv"
        
        output_dir = RAW_RESULT_DIR / "m2" / "finetune"
        os.makedirs(output_dir, exist_ok=True)
        
        df = pd.read_csv(test_path)
        
        needs_fixing = df[(df["success"] == False)].copy()
        print(f"Found {len(needs_fixing)} queries that need fixing")
        
        if len(needs_fixing) == 0:
            print("No queries need fixing. Exiting.")
            return pd.DataFrame()
        
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
        
        result_df["valid_sql"] = result_df["generated_query"]
        result_df["validated"] = 0
        
        for idx, row in tqdm(needs_fixing.iterrows(), total=len(needs_fixing), desc="Fixing queries"):
            try:
                question = row["question"]
                original_sql = row.get("generated_query", "")
                error = row.get("error", "")
                
                fixed_query = self.fix_sql(question, original_sql, error)
                
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
                        
                if (idx + 1) % 10 == 0 or (idx + 1) >= len(needs_fixing):
                    csv_filename = output_dir / f"{self.name}_intermediate.csv"
                    result_df.to_csv(csv_filename, index=False)
                    print(f"Saved intermediate results to {csv_filename}")
                    
            except Exception as e:
                print(f"Error processing query {idx}: {str(e)}")
        
        validation_count = result_df["validated"].sum()
        validation_rate = (
            validation_count / len(needs_fixing) if len(needs_fixing) > 0 else 0
        )
        print(
            f"Fixed {validation_count} out of {len(needs_fixing)} queries ({validation_rate:.2%})"
        )
        
        csv_filename = output_dir / f"{self.name}.csv"
        result_df.to_csv(csv_filename, index=False)
        print(f"Saved results to {csv_filename}")
        
        return result_df