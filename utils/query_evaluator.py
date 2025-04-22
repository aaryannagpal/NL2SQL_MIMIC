import math
import time
import json
import multiprocessing
from typing import Optional, Dict, Any, Union, Set, List, Tuple
import re
import difflib
from pathlib import Path
from functools import partial
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from utils.query_handler import QueryHandler


def _process_row_helper(row_dict, query_column):
    evaluator = QueryEvaluator()
    return evaluator.evaluate_query(row_dict, query_column)


class QueryEvaluator:
    def __init__(self, db_path: Optional[str] = None):
        self.query_handler = QueryHandler()
        if db_path:
            self.query_handler.update_path(db_path)

    def evaluate_query(
        self, row: Dict[str, Any], query_column: str = "true_query"
    ) -> Dict[str, Any]:
        query = row.get(query_column)

        if not query or not isinstance(query, str):
            if isinstance(query, float) and math.isnan(query):
                return {**row, "success": True}
            return {**row, "success": False, "error": "No query provided"}

        try:
            start_time = time.time()
            result = self.query_handler.execute(query)
            total_time = time.time() - start_time

            evaluated_row = {**row}
            evaluated_row["success"] = result["success"]
            evaluated_row["execution_time"] = result["execution_time"]
            evaluated_row["total_time"] = total_time
            evaluated_row["row_count"] = result["row_count"]
            evaluated_row["error"] = result["error"]

            if result.get("results"):
                evaluated_row["results"] = result["results"]
                evaluated_row["result_columns"] = list(result["results"][0].keys())
            else:
                evaluated_row["results"] = None
                evaluated_row["result_columns"] = None

            plan = result.get("execution_plan")
            if plan and plan != "Query plan not available":
                evaluated_row["execution_plan"] = json.dumps(plan)
            else:
                evaluated_row["execution_plan"] = None

            return evaluated_row

        except Exception as e:
            return {
                **row,
                "success": False,
                "execution_time": 0,
                "total_time": time.time() - start_time,
                "row_count": 0,
                "results": None,
                "result_columns": None,
                "execution_plan": None,
                "error": str(e),
            }

    def batch_evaluate(
        self,
        dataset: Union[pd.DataFrame, str, Path],
        output_path: Union[str, Path],
        query_column: str = "true_query",
        max_workers: int = None,
    ) -> pd.DataFrame:

        if max_workers is None:
            max_workers = multiprocessing.cpu_count() - 4

        try:
            if isinstance(dataset, (str, Path)):
                df = pd.read_csv(dataset)
            else:
                df = dataset
            print(f"Loaded dataset with {len(df)} queries")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return pd.DataFrame()

        required_columns = ["id", "question", query_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Dataset is missing required columns: {missing_columns}")
            return pd.DataFrame()

        rows = df.to_dict("records")
        results = []

        print(f"Starting evaluation with {max_workers} workers...")
        worker = partial(_process_row_helper, query_column=query_column)
        with multiprocessing.Pool(processes=max_workers) as pool:
            for result in tqdm(
                pool.imap(worker, rows), total=len(rows), desc="Evaluating queries"
            ):
                results.append(result)

        result_df = pd.DataFrame(results)
        result_df.to_csv(output_path, index=False)
        print(f"Evaluation complete. Results saved to {output_path}")
        return result_df

    def extract_tables(self, query: str, depth: int = 0) -> Set[str]:
        if not query or query is np.NaN or query is pd.NA or depth > 10:
            return set()
            
        try:
            # Remove string literals and extract basic tables
            query = re.sub(r"'[^']*'", "", query)
            tables = set(re.findall(r'(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)', query, re.IGNORECASE))
            tables.update(re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\.[a-zA-Z_][a-zA-Z0-9_]*', query, re.IGNORECASE))
            
            # Process subqueries
            subqueries = re.findall(r'\(\s*SELECT\s+.+?FROM.+?\)', query, re.IGNORECASE | re.DOTALL)
            for subquery in subqueries:
                if subquery != query and len(subquery) < len(query):
                    tables.update(self.extract_tables(subquery, depth + 1))
        except Exception:
            pass
        
        return tables
    
    def extract_columns(self, query: str, depth: int = 0) -> Set[str]:
        if not query or query is np.NaN or query is pd.NA or depth > 10:
            return set()
            
        try:
            # Remove string literals
            query = re.sub(r"'[^']*'", "", query)
            columns = set(re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)', query))
            
            # Extract from SELECT clause
            select_match = re.search(r'SELECT\s+(.*?)(?:FROM|$)', query, re.IGNORECASE | re.DOTALL)
            if select_match:
                select_clause = select_match.group(1).strip()
                if '*' in select_clause:
                    columns.add('*')
                else:
                    # Parse columns handling functions and parentheses
                    parts = []
                    current = []
                    paren_level = 0
                    for char in select_clause:
                        if char == ',' and paren_level == 0:
                            parts.append(''.join(current).strip())
                            current = []
                        else:
                            if char == '(': paren_level += 1
                            elif char == ')': paren_level -= 1
                            current.append(char)
                    
                    if current:
                        parts.append(''.join(current).strip())
                    
                    for part in parts:
                        col_match = re.search(r'([a-zA-Z0-9_.*]+)(?:\s+AS\s+[a-zA-Z0-9_]+)?', part, re.IGNORECASE)
                        if col_match:
                            columns.add(col_match.group(1).strip())
            
            # Extract from WHERE clause
            where_match = re.search(r'WHERE\s+(.*?)(?:GROUP BY|ORDER BY|LIMIT|$)', query, re.IGNORECASE | re.DOTALL)
            if where_match:
                columns.update(re.findall(r'([a-zA-Z0-9_]+)(?:\s*[=<>])', where_match.group(1)))
            
            # Process subqueries
            subqueries = re.findall(r'\(\s*SELECT\s+.+?FROM.+?\)', query, re.IGNORECASE | re.DOTALL)
            for subquery in subqueries:
                if subquery != query and len(subquery) < len(query):
                    columns.update(self.extract_columns(subquery, depth + 1))
        except Exception:
            pass
        
        return columns
    
    def extract_condition(self, query: str) -> str:
        if not query or query is np.NaN or query is pd.NA:
            return ""
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Find the WHERE clause
        where_match = re.search(r'\bWHERE\b', query, re.IGNORECASE)
        if not where_match:
            return ""
        
        start_pos = where_match.end()
        
        while start_pos < len(query) and query[start_pos].isspace():
            start_pos += 1
        
        paren_level = 0
        in_string = False
        string_char = None
        end_pos = len(query)
        
        for i in range(start_pos, len(query)):
            char = query[i]
            
            if char in ("'", '"') and (i == start_pos or query[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    
            if not in_string:
                if char == '(':
                    paren_level += 1
                elif char == ')':
                    paren_level -= 1
                    
                if paren_level == 0:
                    rest = query[i+1:].lstrip()
                    for keyword in ['GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT', 'OFFSET', 'UNION', 'INTERSECT', 'EXCEPT']:
                        if re.match(r'\b' + keyword + r'\b', rest, re.IGNORECASE):
                            end_pos = i + 1 
                            break
                    
                    if end_pos < len(query):
                        break
        
        # Extract and clean the condition
        condition = query[start_pos:end_pos].strip()
        return condition

    def results_are_equivalent(self, results1: List[Dict], results2: List[Dict]) -> Tuple[bool, Dict[str, Any]]:
        details = {}

        if isinstance(results1, float) and math.isnan(results1):
            results1 = []
        if isinstance(results2, float) and math.isnan(results2):
            results2 = []
        
        # Handle empty results
        if (not results1 or len(results1) == 0) and (not results2 or len(results2) == 0):
            details['reason'] = "Both results are empty"
            return True, details
        if (not results1 or len(results1) == 0) and results2:
            details['reason'] = "First result is empty, second has data"
            details['second_count'] = len(results2)
            return False, details
        if (not results2 or len(results2) == 0) and results1:
            details['reason'] = "Second result is empty, first has data"
            details['first_count'] = len(results1)
            return False, details
        
        # Check row counts and columns
        if len(results1) != len(results2):
            details['reason'] = "Different number of rows"
            details['first_count'] = len(results1)
            details['second_count'] = len(results2)
            return False, details
            
        cols1 = set(results1[0].keys())
        cols2 = set(results2[0].keys())
        if cols1 != cols2:
            details['reason'] = "Different columns"
            details['cols_only_in_first'] = list(cols1 - cols2)
            details['cols_only_in_second'] = list(cols2 - cols1)
            details['common_cols'] = list(cols1 & cols2)
            return False, details
            
        # Compare actual data
        try:
            set1 = {tuple(sorted((k, str(v)) for k, v in row.items())) for row in results1}
            set2 = {tuple(sorted((k, str(v)) for k, v in row.items())) for row in results2}
            
            if set1 == set2:
                details['reason'] = "Results are identical (same rows, possibly different order)"
                return True, details
                
            # Count differences
            diff_count = len(set1.symmetric_difference(set2))
            details['reason'] = f"{diff_count} rows differ between results"
            details['diff_percentage'] = (diff_count / len(results1)) * 100
            
            # Sample difference
            for row1 in results1:
                if all(any(str(row1.get(k)) != str(row2.get(k)) for k in cols1) for row2 in results2):
                    details['example_diff_row'] = row1
                    break
                    
            return False, details
                
        except Exception as e:
            details['reason'] = f"Error comparing results: {str(e)}"
            return False, details