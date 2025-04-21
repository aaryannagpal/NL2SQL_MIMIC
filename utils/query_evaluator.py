import math
import time
import json
import multiprocessing
from typing import Optional, Dict, Any, Union
from pathlib import Path
from functools import partial
import sys
import pandas as pd
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
