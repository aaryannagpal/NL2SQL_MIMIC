import os
import sys
from pathlib import Path
import json
import pandas as pd
import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import (
    MODEL_LIST,
    TRAINING_DATA,
    MODELS_DIR,
    RAW_RESULT_DIR,
    MIMIC_SCHEMA_PATH,
    DICTIONARY_MAP_PATH,
    DATASET_PATH
)

MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
OUTPUT_DIR = str(PROJECT_ROOT / "model" / "m1" / "finetune" / "output")
DATA_DIR = DATASET_PATH
TRAINING_DATA_PATH = str(TRAINING_DATA)
TESTING_DATA_PATH = str(DATA_DIR / "test.csv")
VALIDATION_DATA_PATH = str(DATA_DIR / "valid.csv")

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
print(SCHEMA_STR)

def prepare_dataset(data_path):
    df = pd.read_csv(data_path)
    # df = df[:5]
    
    formatted_data = []
    for _, row in df.iterrows():
        question = row["question"]
        sql_query = row.get("true_query", None)
        
        if pd.isna(sql_query) or not isinstance(sql_query, str) or sql_query.strip() == "":
            sql_response = "No SQL query can answer this question. This appears to require medical advice or contains information not present in the database."
        else:
            sql_response = sql_query
            
        system_message = f"You are a SQL assistant specialized in medical database queries. Your task is to convert natural language questions into SQL queries for the MIMIC medical database.\n\n{SCHEMA_STR}"
        formatted_text = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{sql_response}<|im_end|>"
        
        formatted_data.append({
            "text": formatted_text  # Use "text" field instead of "messages"
        })
    
    return Dataset.from_list(formatted_data)

train_dataset = prepare_dataset(TRAINING_DATA_PATH)
eval_dataset = prepare_dataset(VALIDATION_DATA_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=4096,
    dtype=torch.float16,
    load_in_4bit=True,
    token=None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"], 
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}% of {total_params:,} total)")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,               # Adjust as needed
    per_device_train_batch_size=1,    # Start small, increase if memory allows
    per_device_eval_batch_size=1, 
    gradient_accumulation_steps=8,    # Adjust based on your GPU memory
    eval_strategy="steps",
    eval_steps=100,                   # Evaluate every 100 steps
    logging_steps=10,
    learning_rate=2e-4,               # Good starting point for LoRA
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,               # Keep only the 3 best checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    bf16=False,                       # Set to True if your GPU supports it
    fp16=True,                        # Use mixed precision
    report_to="none",                 # Change to "wandb" if using Weights & Biases
    optim="adamw_torch",
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    packing=False,                  # Don't pack sequences
    dataset_text_field="text",      # Change to "text" field
    max_seq_length=4096,            # Match model's max_seq_length
)

trainer.train()
trainer.save_model(f"{OUTPUT_DIR}/duckdb_nsql")


def generate_sql(question, model, tokenizer):
    prompt = f"""<|im_start|>system
    You are a SQL assistant specialized in medical database queries. Your task is to convert natural language questions into SQL queries for the MIMIC medical database.

    {SCHEMA_STR}<|im_end|>
    <|im_start|>user
    {question}<|im_end|>
    <|im_start|>assistant
    """
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.01,
        top_p=0.9,
        do_sample=False,
        num_beams=1,
        early_stopping=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return response.split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0].strip()

sql_question = "What was the last recorded heart rate for patient 10014078?"
non_sql_question = "What precautions should I take after a heart surgery?"

print(f"Question requiring SQL: {sql_question}")
print(f"Generated: {generate_sql(sql_question, model, tokenizer)}")
print("\n")
print(f"Question NOT requiring SQL: {non_sql_question}")
print(f"Generated: {generate_sql(non_sql_question, model, tokenizer)}")