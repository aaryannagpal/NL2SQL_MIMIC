import pandas as pd
import numpy as np
import multiprocessing
import time
import os, sys
import json
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.query_handler import QueryHandler
from config import MYSQL_DB_PATH

def _process_row_helper(row_dict):
    evaluator = QueryEvaluator()
    return evaluator.evaluate_query(row_dict)

class QueryEvaluator:
    """
    Evaluates SQL queries by executing them and recording their performance and results.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the evaluator with a database path.
        
        Args:
            db_path: Path to the SQLite database file. If None, uses the default from config.
        """

        self.query_handler = QueryHandler()

        if db_path:
            self.query_handler.update_path(db_path)
    
    def evaluate_query(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single query and return the results.
        
        Args:
            row: A dictionary containing at least 'true_query' key with the SQL query to execute
            
        Returns:
            Dictionary with original row data and query execution results
        """

        query = row.get('true_query')
        if not query or query is np.NaN or query is pd.NA:
            return {**row, 'success': False, 'error': 'No query provided'}
        
        try:
            start_time = time.time()
            result = self.query_handler.execute(query)
            total_time = time.time() - start_time
            
            evaluated_row = {**row}
            
            evaluated_row['success'] = result['success']
            evaluated_row['execution_time'] = result['execution_time']
            evaluated_row['total_time'] = total_time
            evaluated_row['row_count'] = result['row_count']
            evaluated_row['error'] = result['error']
            
            if result['results'] and len(result['results']) > 0:
                # num_results = min(5, len(result['results']))
                # evaluated_row['results_sample'] = json.dumps(result['results'][:num_results])
                evaluated_row['results'] = result['results']
                if len(result['results']) > 0:
                    evaluated_row['result_columns'] = list(result['results'][0].keys())
            else:
                # evaluated_row['results_sample'] = None
                evaluated_row['results'] = None
                evaluated_row['result_columns'] = None
            
            if result['execution_plan'] and result['execution_plan'] != "Query plan not available":
                evaluated_row['execution_plan'] = json.dumps(result['execution_plan'])
            else:
                evaluated_row['execution_plan'] = None
                
            return evaluated_row
            
        except Exception as e:
            return {
                **row,
                'success': False,
                'execution_time': 0,
                'total_time': time.time() - start_time,
                'row_count': 0,
                # 'results_sample': None,
                'results': None,
                'result_columns': None,
                'execution_plan': None,
                'error': str(e)
            }
    
    def batch_evaluate(self, 
                       dataset_path: Union[str, Path], 
                       output_path: Union[str, Path], 
                       max_workers: int = None) -> pd.DataFrame:
        """
        Evaluate all queries in a dataset using multiprocessing.
        
        Args:
            dataset_path: Path to the input dataset CSV
            output_path: Path where the output dataset will be saved
            max_workers: Maximum number of worker processes to use. If None, uses cpu_count().
            
        Returns:
            DataFrame containing the evaluation results
        """
        if max_workers is None:
            max_workers = multiprocessing.cpu_count() - 4
        
        try:
            df = pd.read_csv(dataset_path)
            print(f"Loaded dataset with {len(df)} queries")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return pd.DataFrame()
        
        required_columns = ['id', 'question', 'true_query']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Dataset is missing required columns: {missing_columns}")
            return pd.DataFrame()
        
        rows = df.to_dict('records')
        results = []
        
        print(f"Starting evaluation with {max_workers} workers...")
        
        with multiprocessing.Pool(processes=max_workers) as pool:
            for result in tqdm(
                pool.imap(_process_row_helper, rows), 
                total=len(rows), 
                desc="Evaluating queries"
            ):
                results.append(result)
        
        result_df = pd.DataFrame(results)
        
        result_df.to_csv(output_path, index=False)
        print(f"Evaluation complete. Results saved to {output_path}")
        
        return result_df