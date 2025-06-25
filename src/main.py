import argparse
import os
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path().resolve().parent
sys.path.append(str(PROJECT_ROOT))

from utils.model_utils import M1Tester, M2Tester
from utils.query_handler import QueryHandler
from utils.query_evaluator import QueryEvaluator

from config import DATASET_PATH, PROCESSED_RESULT_DIR, MYSQL_DB_PATH

class TextToSQLPipeline:
    def __init__(self, m1_model='qwen_2.5', m2_model='phi4'):
        """
        Initialize the two-stage Text-to-SQL pipeline using the best fine-tuned models.
        
        Args:
            m1_model: Name of the generator model
            m2_model: Name of the validator model
        """
        print(f"Initializing Text-to-SQL Pipeline with models: {m1_model} (generator) and {m2_model} (validator)")
        
        self.m1_tester = M1Tester(name=m1_model)
        self.m2_tester = M2Tester(name=m2_model)

        self.m1_name = m1_model
        self.m2_name = m2_model
        
        print("Loading generator model...")
        self.m1, self.m1_tokenizer = self.m1_tester.load_model()
        
        print("Loading validator model...")
        self.m2, self.m2_tokenizer = self.m2_tester.load_model()
        
        self.query_handler = QueryHandler()
        
    def process_query(self, question):
        """
        Process a natural language question through the complete pipeline.
        
        Args:
            question: Natural language question to convert to SQL
            
        Returns:
            Dictionary containing results and processing details
        """
        result = {
            "question": question,
            "initial_query": None,
            "validated_query": None,
            "validation_triggered": False,
            "final_query": None,
            "execution_result": None,
            "success": False,
            "execution_time": 0,
            "error": None,
            "results": None
        }
        
        print("Generating SQL query...")
        generated_sql = self.m1_tester.generate_sql(question)
        result["initial_query"] = generated_sql
        
        print("Executing generated query...")
        execution_result = self.query_handler.execute(generated_sql)
        
        if not execution_result.get("success", False):
            print("Query execution failed. Triggering validation...")
            result["validation_triggered"] = True
            
            validated_sql = self.m2_tester.fix_sql(
                question, 
                generated_sql, 
                error=execution_result.get("error")
            )
            
            result["validated_query"] = validated_sql
            
            print("Executing validated query...")
            execution_result = self.query_handler.execute(validated_sql)
            result["final_query"] = validated_sql
        else:
            print("Query executed successfully.")
            result["final_query"] = generated_sql
        
        result["execution_result"] = execution_result
        result["success"] = execution_result.get("success", False)
        result["execution_time"] = execution_result.get("execution_time", 0)
        result["error"] = execution_result.get("error")
        result["results"] = execution_result.get("results")
        
        return result
    
    def display_results(self, pipeline_result):
        """
        Display the results of the pipeline in a user-friendly format.
        
        Args:
            pipeline_result: Output from process_query method
        """
        print("\n" + "="*80)
        print(f"QUESTION: {pipeline_result['question']}")
        print("-"*80)
        
        print(f"GENERATED SQL: {pipeline_result['initial_query']}")
        
        if pipeline_result["validation_triggered"]:
            print("-"*80)
            print(f"VALIDATION TRIGGERED: Yes")
            print(f"ERROR: {pipeline_result['error']}")
            print(f"VALIDATED SQL: {pipeline_result['validated_query']}")
        
        print("-"*80)
        print(f"EXECUTION SUCCESS: {pipeline_result['success']}")
        print(f"EXECUTION TIME: {pipeline_result['execution_time']:.4f} seconds")
        
        if pipeline_result["success"]:
            print("-"*80)
            print("RESULTS:")
            results = pipeline_result["results"]
            if results and len(results) > 0:
                if isinstance(results[0], dict):
                    headers = list(results[0].keys())
                    header_str = " | ".join(headers)
                    print(header_str)
                    print("-" * len(header_str))
                    
                    for i, row in enumerate(results[:10]):
                        row_values = [str(row.get(h, "")) for h in headers]
                        print(" | ".join(row_values))
                    
                    if len(results) > 10:
                        print(f"... ({len(results)-10} more rows)")
                else:
                    print(results)
            else:
                print("No results returned.")
        else:
            print(f"ERROR: {pipeline_result['error']}")
        
        print("="*80 + "\n")

    def batch_evaluate(self, dataset_path=None, output_dir=None, save_all=False):
        """
        Evaluate the pipeline on a batch of queries from a dataset using the built-in
        evaluation methods from M1Tester and M2Tester.
        
        Args:
            dataset_path: Path to the dataset CSV file (default: test.csv)
            output_dir: Directory to save results (default: processed_results/pipeline)
            
        Returns:
            DataFrame containing evaluation results
        """
        # Set default paths if not provided
        if dataset_path is None:
            dataset_path = self.m1_tester.VALIDATION_DATA_PATH
        
        if output_dir is None:
            output_dir = PROCESSED_RESULT_DIR / "pipeline"
        else:
            output_dir = Path(output_dir)
            
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Phase 1: Evaluating generator model ({self.m1_name})...")
        m1_results = self.m1_tester.evaluate_model(dataset_path)
        
        m1_output_file = output_dir / f"{self.m1_name}_generation.csv"

        if save_all:
            m1_results.to_csv(m1_output_file, index=False)
            print(f"Saved generator results to {m1_output_file}")
        
        print("Phase 2: Executing generated queries...")
        evaluator = QueryEvaluator(db_path = MYSQL_DB_PATH)
        
        if "generated_query" in m1_results.columns and "true_query" in m1_results.columns:
            eval_df = m1_results.copy()
            
            output_path = output_dir / f"{self.m1_name}_execution.csv"
            execution_results = evaluator.batch_evaluate(eval_df, output_path, query_column="generated_query", max_workers=30)
            
            if isinstance(execution_results, str) or isinstance(execution_results, Path):
                execution_results = pd.read_csv(output_path)
            
            print(f"Saved execution results to {output_path}")
            
            success_count = execution_results["success"].sum() if "success" in execution_results.columns else 0
            total_count = len(execution_results)
            print(f"Generator success rate: {success_count}/{total_count} ({success_count/total_count:.2%})")
            
            pipeline_results = execution_results.copy()
            
            pipeline_results["validation_applied"] = False
            pipeline_results["final_query"] = pipeline_results["generated_query"]
            
            failed_queries = execution_results[execution_results["success"] == False].copy() if "success" in execution_results.columns else pd.DataFrame()
            
            if len(failed_queries) > 0:
                print(f"Phase 3: Validating {len(failed_queries)} failed queries...")
                
                tmp_validation_input = output_dir / f"tmp_validation_input.csv"
                
                failed_queries.to_csv(tmp_validation_input, index=False)
                
                validation_results = self.m2_tester.evaluate_model(str(tmp_validation_input).split('.csv')[0])
                
                validation_output_file = output_dir / f"{self.m2_name}_validation.csv"
                
                if save_all:
                    validation_results.to_csv(validation_output_file, index=False)
                    print(f"Saved validation results to {validation_output_file}")
                
                print("Phase 4: Executing validated queries...")
                if "valid_sql" in validation_results.columns:
                    eval_validation_df = validation_results.copy()
                    
                    final_output_path = output_dir / f"pipeline_final_results.csv"
                    validation_execution_results = evaluator.batch_evaluate(
                        eval_validation_df, 
                        final_output_path, 
                        query_column="valid_sql"
                    )
                    
                    if isinstance(validation_execution_results, str) or isinstance(validation_execution_results, Path):
                        validation_execution_results = pd.read_csv(final_output_path)
                    
                    for idx, vrow in validation_execution_results.iterrows():
                        row_id = vrow.get("id", None)
                        if row_id is not None:
                            match_idx = pipeline_results.index[pipeline_results["id"] == row_id].tolist()
                            if match_idx:
                                pipeline_results.loc[match_idx[0], "success"] = vrow.get("success", False)
                                pipeline_results.loc[match_idx[0], "final_query"] = vrow.get("valid_sql", "")
                                pipeline_results.loc[match_idx[0], "validation_applied"] = True
                                pipeline_results.loc[match_idx[0], "error"] = vrow.get("error", "")
                                if "results" in vrow:
                                    pipeline_results.loc[match_idx[0], "results"] = vrow["results"]
                    
                    validation_success = validation_execution_results["success"].sum() if "success" in validation_execution_results.columns else 0
                    improvement = validation_success / len(failed_queries) if len(failed_queries) > 0 else 0
                    
                    pipeline_success_count = pipeline_results["success"].sum() if "success" in pipeline_results.columns else 0
                    
                    print(f"Validation success rate: {validation_success}/{len(failed_queries)} ({improvement:.2%})")
                    print(f"Overall pipeline success rate: {pipeline_success_count}/{total_count} ({pipeline_success_count/total_count:.2%})")
                    
                    pipeline_output_file = output_dir / f"{self.m1_name}-{self.m2_name}_complete_pipeline_results.csv"
                    
                    pipeline_results.to_csv(pipeline_output_file, index=False)
                    print(f"Saved complete pipeline results to {pipeline_output_file}")
                    
                    return pipeline_results
            
            return pipeline_results
        
        return m1_results

pipeline = TextToSQLPipeline()

result = pipeline.process_query("What was the last procedure icd code for patient 10014078")
pipeline.display_results(result)

pipeline.batch_evaluate(DATASET_PATH / "valid_sample.csv")