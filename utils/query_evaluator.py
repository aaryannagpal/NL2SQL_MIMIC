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
import concurrent.futures

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from utils.query_handler import QueryHandler


def _process_row_helper(row_dict, query_column):
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("Query execution timed out")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(20)  # Hard timeout of 20 seconds for the entire processing

    try:
        evaluator = QueryEvaluator()
        result = evaluator.evaluate_query(row_dict, query_column)
        signal.alarm(0)
        return result
    except TimeoutError:
        return {
            **row_dict,
            "success": False,
            "execution_time": 20.0,
            "total_time": 20.0,
            "row_count": 0,
            "results": None,
            "result_columns": None,
            "execution_plan": None,
            "error": "worker process timeout",
        }
    except Exception as e:
        signal.alarm(0)
        return {
            **row_dict,
            "success": False,
            "execution_time": 0,
            "total_time": 0,
            "row_count": 0,
            "results": None,
            "result_columns": None,
            "execution_plan": None,
            "error": f"worker error: {str(e)}",
        }


class QueryEvaluator:
    QUERY_TIMEOUT = 20
    SAVE_EVERY = 20

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
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

        start_time = time.time()

        handler = QueryHandler()
        if self.db_path:
            handler.update_path(self.db_path)

        try:
            result = handler.execute(query)

        except Exception as e:
            total_time = time.time() - start_time
            return {
                **row,
                "success": False,
                "execution_time": 0,
                "total_time": total_time,
                "row_count": 0,
                "results": None,
                "result_columns": None,
                "execution_plan": None,
                "error": str(e),
            }

        total_time = time.time() - start_time
        evaluated_row = {**row}
        evaluated_row["success"] = result.get("success", False)
        evaluated_row["execution_time"] = result.get("execution_time")
        evaluated_row["total_time"] = total_time
        evaluated_row["row_count"] = result.get("row_count", 0)
        evaluated_row["error"] = result.get("error")

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

    def batch_evaluate(
        self,
        dataset: Union[pd.DataFrame, str, Path],
        output_path: Union[str, Path],
        query_column: str = "true_query",
        max_workers: int = None,
    ):
        if max_workers is None:
            max_workers = multiprocessing.cpu_count() - 1

        if isinstance(dataset, (str, Path)):
            df = pd.read_csv(dataset)
        else:
            df = dataset
        print(f"Loaded dataset with {len(df)} queries")

        required_columns = ["id", "question", query_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Dataset is missing required columns: {missing_columns}")
            return pd.DataFrame()

        rows = df.to_dict("records")
        results = []

        print(f"Starting evaluation with {max_workers} workers...")

        with multiprocessing.Pool(processes=max_workers) as pool:
            worker = partial(_process_row_helper, query_column=query_column)

            iterator = pool.imap_unordered(worker, rows)

            for idx in tqdm(
                range(len(rows)), total=len(rows), desc="Evaluating queries"
            ):
                try:
                    result = next(iterator, None)
                    if result:
                        results.append(result)
                    else:
                        print(f"Warning: No result returned for query {idx}")
                        results.append(
                            {
                                **rows[idx],
                                "success": False,
                                "error": "No result returned",
                            }
                        )

                    if (idx + 1) % self.SAVE_EVERY == 0:
                        temp_df = pd.DataFrame(results)
                        temp_df.to_csv(output_path, index=False)
                        print(f"Periodic save after {idx+1} queries to {output_path}")

                except Exception as e:
                    print(f"Error processing query {idx}: {e}")
                    results.append(
                        {
                            **rows[idx],
                            "success": False,
                            "error": f"Processing error: {str(e)}",
                        }
                    )

        result_df = pd.DataFrame(results)
        result_df.to_csv(output_path, index=False)
        print(f"Evaluation complete. Results saved to {output_path}")
        return result_df

    