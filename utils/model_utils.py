import torch
import pandas as pd
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import json
from tqdm import tqdm

from pathlib import Path
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MIMIC_SCHEMA_PATH,
    DATASET_PATH,
    SAMPLE_M1_MODEL_DIR
);

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

    def create_schema_string(self):
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
        SCHEMA_STR = self.create_schema_string()
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

