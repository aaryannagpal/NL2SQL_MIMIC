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

class QueryAnalyzer:
    def extract_tables(self, query: str, depth: int = 0) -> Set[str]:
        if not query or query is np.NaN or query is pd.NA or depth > 10:
            return set()

        try:
            query = re.sub(r"'[^']*'", "", query)
            tables = set(
                re.findall(
                    r"(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)", query, re.IGNORECASE
                )
            )
            tables.update(
                re.findall(
                    r"([a-zA-Z_][a-zA-Z0-9_]*)\.[a-zA-Z_][a-zA-Z0-9_]*",
                    query,
                    re.IGNORECASE,
                )
            )

            subqueries = re.findall(
                r"\(\s*SELECT\s+.+?FROM.+?\)", query, re.IGNORECASE | re.DOTALL
            )
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
            query = re.sub(r"'[^']*'", "", query)
            columns = set(
                re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)", query)
            )

            select_match = re.search(
                r"SELECT\s+(.*?)(?:FROM|$)", query, re.IGNORECASE | re.DOTALL
            )
            if select_match:
                select_clause = select_match.group(1).strip()
                if "*" in select_clause:
                    columns.add("*")
                else:
                    parts = []
                    current = []
                    paren_level = 0
                    for char in select_clause:
                        if char == "," and paren_level == 0:
                            parts.append("".join(current).strip())
                            current = []
                        else:
                            if char == "(":
                                paren_level += 1
                            elif char == ")":
                                paren_level -= 1
                            current.append(char)

                    if current:
                        parts.append("".join(current).strip())

                    for part in parts:
                        col_match = re.search(
                            r"([a-zA-Z0-9_.*]+)(?:\s+AS\s+[a-zA-Z0-9_]+)?",
                            part,
                            re.IGNORECASE,
                        )
                        if col_match:
                            columns.add(col_match.group(1).strip())

            where_match = re.search(
                r"WHERE\s+(.*?)(?:GROUP BY|ORDER BY|LIMIT|$)",
                query,
                re.IGNORECASE | re.DOTALL,
            )
            if where_match:
                columns.update(
                    re.findall(r"([a-zA-Z0-9_]+)(?:\s*[=<>])", where_match.group(1))
                )

            subqueries = re.findall(
                r"\(\s*SELECT\s+.+?FROM.+?\)", query, re.IGNORECASE | re.DOTALL
            )
            for subquery in subqueries:
                if subquery != query and len(subquery) < len(query):
                    columns.update(self.extract_columns(subquery, depth + 1))
        except Exception:
            pass

        return columns

    def extract_condition(self, query: str) -> str:
        if not query or query is np.NaN or query is pd.NA:
            return ""
        query = re.sub(r"\s+", " ", query.strip())

        where_match = re.search(r"\bWHERE\b", query, re.IGNORECASE)
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

            if char in ("'", '"') and (i == start_pos or query[i - 1] != "\\"):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False

            if not in_string:
                if char == "(":
                    paren_level += 1
                elif char == ")":
                    paren_level -= 1

                if paren_level == 0:
                    rest = query[i + 1 :].lstrip()
                    for keyword in [
                        "GROUP BY",
                        "HAVING",
                        "ORDER BY",
                        "LIMIT",
                        "OFFSET",
                        "UNION",
                        "INTERSECT",
                        "EXCEPT",
                    ]:
                        if re.match(r"\b" + keyword + r"\b", rest, re.IGNORECASE):
                            end_pos = i + 1
                            break

                    if end_pos < len(query):
                        break

        condition = query[start_pos:end_pos].strip()
        return condition

    def results_are_equivalent(
        self, results1: List[Dict], results2: List[Dict]
    ) -> Tuple[bool, Dict[str, Any]]:
        details = {}

        if isinstance(results1, float) and math.isnan(results1):
            results1 = []
        if isinstance(results2, float) and math.isnan(results2):
            results2 = []

        if (not results1 or len(results1) == 0) and (
            not results2 or len(results2) == 0
        ):
            details["reason"] = "Both results are empty"
            return True, details
        if (not results1 or len(results1) == 0) and results2:
            details["reason"] = "First result is empty, second has data"
            details["second_count"] = len(results2)
            return False, details
        if (not results2 or len(results2) == 0) and results1:
            details["reason"] = "Second result is empty, first has data"
            details["first_count"] = len(results1)
            return False, details

        if len(results1) != len(results2):
            details["reason"] = "Different number of rows"
            details["first_count"] = len(results1)
            details["second_count"] = len(results2)
            return False, details

        cols1 = set(results1[0].keys())
        cols2 = set(results2[0].keys())
        if cols1 != cols2:
            details["reason"] = "Different columns"
            details["cols_only_in_first"] = list(cols1 - cols2)
            details["cols_only_in_second"] = list(cols2 - cols1)
            details["common_cols"] = list(cols1 & cols2)
            return False, details

        try:
            set1 = {
                tuple(sorted((k, str(v)) for k, v in row.items())) for row in results1
            }
            set2 = {
                tuple(sorted((k, str(v)) for k, v in row.items())) for row in results2
            }

            if set1 == set2:
                details["reason"] = (
                    "Results are identical (same rows, possibly different order)"
                )
                return True, details

            diff_count = len(set1.symmetric_difference(set2))
            details["reason"] = f"{diff_count} rows differ between results"
            details["diff_percentage"] = (diff_count / len(results1)) * 100

            for row1 in results1:
                if all(
                    any(str(row1.get(k)) != str(row2.get(k)) for k in cols1)
                    for row2 in results2
                ):
                    details["example_diff_row"] = row1
                    break

            return False, details

        except Exception as e:
            details["reason"] = f"Error comparing results: {str(e)}"
            return False, details

    def compare_queries(self, df: pd.DataFrame, to_compare = 'generated_sql') -> pd.DataFrame:
        """
        Compare true and generated queries within a dataframe containing both.

        Args:
            df: DataFrame with columns including 'id', 'true_query', 'true_result', 'generated_sql'

        Returns:
            DataFrame with comparison metrics for each query
        """
        if "true_query" not in df.columns or to_compare not in df.columns:
            print(
                f"Error: DataFrame must contain 'true_query' and '{to_compare}' columns"
            )
            return pd.DataFrame()

        print(f"Comparing queries for {len(df)} rows...")

        results = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Comparing queries"):
            result = {}

            result["id"] = row.get("id", None)
            result["question"] = row.get("question", "")

            true_query = row.get("true_query", "")
            gen_query = row.get(to_compare, "")

            result["true_query"] = true_query
            result["gen_query"] = gen_query

            result["null_query"] = int(pd.isna(true_query))
            if pd.isna(true_query) or true_query is None:
                true_query = ""
            if pd.isna(gen_query) or gen_query is None:
                gen_query = ""

            true_query = str(true_query)
            gen_query = str(gen_query)

            true_tables = self.extract_tables(true_query)
            gen_tables = self.extract_tables(gen_query)

            tables_union = true_tables.union(gen_tables)
            tables_intersection = true_tables.intersection(gen_tables)

            result["true_tables"] = list(true_tables)
            result["gen_tables"] = list(gen_tables)
            result["tables_union"] = list(tables_union)
            result["tables_intersection"] = list(tables_intersection)

            if len(tables_union) > 0:
                result["table_access_accuracy"] = len(tables_intersection) / len(
                    tables_union
                )
            else:
                result["table_access_accuracy"] = 0

            true_columns = self.extract_columns(true_query)
            gen_columns = self.extract_columns(gen_query)

            columns_union = true_columns.union(gen_columns)
            columns_intersection = true_columns.intersection(gen_columns)

            result["true_columns"] = list(true_columns)
            result["gen_columns"] = list(gen_columns)
            result["columns_union"] = list(columns_union)
            result["columns_intersection"] = list(columns_intersection)

            if len(columns_union) > 0:
                result["column_access_accuracy"] = len(columns_intersection) / len(
                    columns_union
                )
            else:
                result["column_access_accuracy"] = 0

            try:
                result["query_similarity"] = difflib.SequenceMatcher(
                    None, true_query, gen_query
                ).ratio()
            except Exception:
                result["query_similarity"] = 0

            if (
                "execution_plan" in row
                and row["execution_plan"]
                and not pd.isna(row["execution_plan"])
            ):
                try:
                    plan = (
                        json.loads(row["execution_plan"])
                        if isinstance(row["execution_plan"], str)
                        else row["execution_plan"]
                    )
                    result["execution_plan_available"] = True
                    result["execution_plan_similarity"] = None
                except:
                    result["execution_plan_available"] = False
                    result["execution_plan_similarity"] = None
            else:
                result["execution_plan_available"] = False
                result["execution_plan_similarity"] = None

            true_result = row.get("true_result", [])
            gen_result = row.get("results", [])

            if pd.isna(true_result):
                true_result = []
            if pd.isna(gen_result):
                gen_result = []

            if isinstance(true_result, str):
                try:
                    true_result = json.loads(true_result)
                except:
                    true_result = []

            if isinstance(gen_result, str):
                try:
                    gen_result = json.loads(gen_result)
                except:
                    gen_result = []
            
            gen_success = row.get('success', False)
            if gen_success:# and true_result and gen_result:
                try:
                    results_equal, comparison_details = self.results_are_equivalent(
                        true_result, gen_result
                    )
                    result["results_match"] = int(results_equal)
                    result["result_comparison"] = json.dumps(
                        comparison_details, default=str
                    )
                except Exception as e:
                    result["results_match"] = 0
                    result["result_comparison"] = json.dumps(
                        {"reason": f"Error comparing results: {str(e)}"}
                    )
            else:
                result["results_match"] = 0
                if not gen_success:
                    details = {"reason": "Generated query failed to execute"}
                else:
                    details = {"reason": "Results not available for comparison"}
                result["result_comparison"] = json.dumps(details)

            results.append(result)

        return pd.DataFrame(results)
