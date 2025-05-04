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
    PROCESSED_RESULT_DIR,
    MIMIC_SCHEMA_PATH,
    DICTIONARY_MAP_PATH,
)

# Configuration
MODEL_NAME = "microsoft/phi-4"
OUTPUT_DIR = str(PROJECT_ROOT / "model" / "m2" / "finetune" / "output")
FIX_DATA_PATH = str(PROCESSED_RESULT_DIR / "m1" / "schema_aware")  # Directory containing results to fix

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
    
    return schema_str

SCHEMA_STR = create_schema_string()
print(SCHEMA_STR)

def get_fixed_dataset(input_dir):
    all_data = []
    fixed_data = []
    
    result_files = list(Path(input_dir).glob("*DuckDB_NSQL_7b_results.csv"))
    for file_path in result_files:
        print(f"Processing file: {file_path}")
        try:
            df = pd.read_csv(file_path)
            all_data.append(df)
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
    
    if not all_data:
        raise ValueError(f"No valid data files found in {input_dir}")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined dataframe has {len(combined_df)} rows")
    
    for idx, row in combined_df.iterrows():
        question = row.get("question", "")
        original_query = row.get("generated_sql", "")
        error_info = row.get("error", "")
        success = row.get("success", False)
        
        if pd.isna(question) or pd.isna(original_query):
            continue
            
        true_query = row.get("true_query", "")  
        
        if pd.isna(true_query) or not isinstance(true_query, str) or true_query.strip() == "":
            continue
            
        if not success and not pd.isna(error_info):
            fixed_data.append({
                "question": question,
                "original_query": original_query,
                "error_info": error_info,
                "fixed_query": true_query
            })
            
    print(f"Created {len(fixed_data)} fixing examples")
    return fixed_data

def prepare_dataset(fixed_data):
    """Prepare the dataset for fine-tuning the SQL fixing model"""
    formatted_data = []
    
    for item in fixed_data:
        question = item["question"]
        original_query = item["original_query"]
        error_info = item["error_info"]
        fixed_query = item["fixed_query"]
        
        # Format the prompt
        prompt = f"You are an expert at fixing SQL queries specialized for MIMIC-IV Database. Fix the SQL query based on the error message and database schema information."
        prompt += f"\n\nVerify joins and filters accordingly."
        prompt += f"\n\nSchema: {SCHEMA_STR}"
        prompt += f"\n\nQuestion: {question}"
        prompt += f"\n\nOriginal SQL query: {original_query}"
        
        if error_info:
            prompt += f"\nError message: {error_info}"
            
        prompt += "\nFixed SQL query: "
        
        system_message = "You are an expert at fixing SQL queries specialized for medical databases."
        user_message = prompt
        assistant_message = fixed_query
        
        formatted_text = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n{assistant_message}<|im_end|>"
        
        formatted_data.append({
            "text": formatted_text
        })
    
    return Dataset.from_list(formatted_data)

print("Creating dataset from failed queries...")
fixed_data = get_fixed_dataset(FIX_DATA_PATH)

from sklearn.model_selection import train_test_split
train_data, val_data = train_test_split(fixed_data, test_size=0.1, random_state=42)

print(f"Training set: {len(train_data)} examples")
print(f"Validation set: {len(val_data)} examples")

train_dataset = prepare_dataset(train_data)
eval_dataset = prepare_dataset(val_data)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load model with Unsloth for faster training
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=4096,  # Adjust based on your longest sequence
    dtype=torch.float16,  # Use float16 for efficiency
    load_in_4bit=True,    # Use 4-bit quantization to fit in memory
    token=None,           # Optional: your HF token if needed
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,               # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"], 
    lora_alpha=32,      # LoRA alpha scaling factor
    lora_dropout=0.05,  # Dropout probability for LoRA layers
    bias="none",        # Don't add bias
    use_gradient_checkpointing=True,  # Save memory with gradient checkpointing
)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}% of {total_params:,} total)")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,               # Adjust as needed - more epochs for fixing
    per_device_train_batch_size=1,    # Start small, increase if memory allows
    per_device_eval_batch_size=1, 
    gradient_accumulation_steps=8,    # Adjust based on your GPU memory
    evaluation_strategy="steps",
    eval_steps=50,                    # Evaluate more frequently
    logging_steps=10,
    learning_rate=2e-4,               # Good starting point for LoRA
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    save_strategy="steps",
    save_steps=50,
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
    packing=False,                   # Don't pack sequences
    dataset_text_field="text",       # Field containing formatted text
    max_seq_length=4096,             # Match model's max_seq_length
)

trainer.train()
trainer.save_model(f"{OUTPUT_DIR}/phi4")

def test_fixing(original_query, error_message, question, model, tokenizer):
    prompt = f"""You are an expert at fixing SQL queries specialized for MIMIC-IV Database. Fix the SQL query based on the error message and database schema information.

    Verify joins and filters accordingly.

    Schema: {SCHEMA_STR}

    Question: {question}

    Original SQL query: {original_query}
    """
    
    if error_message:
        prompt += f"Error message: {error_message}\n"
        
    prompt += "Fixed SQL query: "
    
    system_message = "You are an expert at fixing SQL queries specialized for medical databases."
    user_message = prompt
    
    formatted_prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.95,
        do_sample=False,
        num_beams=1
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    try:
        assistant_response = response.split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0].strip()
    except IndexError:
        assistant_response = response.split(formatted_prompt)[1].strip()
        
    return assistant_response

test_query = "SELECT * FROM patients WHERE patientid = 10014078"
test_error = "column patientid does not exist"
test_question = "What was the last recorded heart rate for patient 10014078?"

print("\n--- Testing SQL Fixing Model ---")
print(f"Original Query: {test_query}")
print(f"Error: {test_error}")
print(f"Question: {test_question}")
print(f"Fixed Query: {test_fixing(test_query, test_error, test_question, model, tokenizer)}")