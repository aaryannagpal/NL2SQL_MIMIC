import math
import time
import json
import multiprocessing
from typing import Optional, Dict, Any, Union, Set, List, Tuple
import re
import difflib
from pathlib import Path
from functools import partial
import sys, os
import pandas as pd
import numpy as np
from tqdm import tqdm
import concurrent.futures

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


    def extract_query_components(self, query: str) -> Dict[str, str]:
        """
        Extract and categorize SQL query components with proper handling of nested subqueries.
        
        Args:
            query: SQL query string
            
        Returns:
            Dictionary of query components (select, from, where, etc.)
        """
        if not query or query is np.NaN or query is pd.NA:
            return {
                'select_clause': '',
                'from_clause': '',
                'where_clause': '',
                'group_by_clause': '',
                'having_clause': '',
                'order_by_clause': '',
                'limit_clause': ''
            }
        

        query = re.sub(r'\s+', ' ', query.strip())
        


        subqueries = []
        placeholder_map = {}
        
        def extract_subqueries(sql_text):
            """Recursively extract subqueries and replace with placeholders"""
            processed_text = sql_text
            pattern = r'\(\s*SELECT'
            pos = 0
            
            while True:
                match = re.search(pattern, processed_text[pos:], re.IGNORECASE)
                if not match:
                    break
                    
                start_idx = pos + match.start()
                open_paren_pos = start_idx
                

                paren_count = 1
                in_quote = False
                quote_char = None
                i = start_idx + 1
                
                while i < len(processed_text) and paren_count > 0:
                    char = processed_text[i]
                    

                    if char in ("'", '"') and (i == 0 or processed_text[i-1] != '\\'):
                        if not in_quote:
                            in_quote = True
                            quote_char = char
                        elif char == quote_char:
                            in_quote = False
                    

                    if not in_quote:
                        if char == '(':
                            paren_count += 1
                        elif char == ')':
                            paren_count -= 1
                    
                    i += 1
                
                if paren_count == 0:
                    close_paren_pos = i - 1
                    subquery = processed_text[open_paren_pos+1:close_paren_pos]  # Exclude parentheses
                    

                    placeholder = f"SUBQUERY_{len(subqueries)}"
                    subqueries.append(subquery)
                    

                    processed_text = (
                        processed_text[:open_paren_pos] + 
                        "(" + placeholder + ")" + 
                        processed_text[close_paren_pos+1:]
                    )
                    

                    placeholder_map[placeholder] = subquery
                    

                    pos = open_paren_pos + len(placeholder) + 2  # +2 for parentheses
                else:

                    pos = start_idx + 1
            
            return processed_text
        

        processed_query = extract_subqueries(query)
        

        components = {}
        
        # Extract SELECT
        select_match = re.search(r'SELECT\s+(.*?)(?:\s+FROM\b|$)', processed_query, re.IGNORECASE | re.DOTALL)
        components['select_clause'] = select_match.group(1).strip() if select_match else ""
        
        # Extract FROM
        from_match = re.search(r'FROM\s+(.*?)(?:\s+WHERE\b|\s+GROUP\s+BY\b|\s+HAVING\b|\s+ORDER\s+BY\b|\s+LIMIT\b|$)', 
                            processed_query, re.IGNORECASE | re.DOTALL)
        components['from_clause'] = from_match.group(1).strip() if from_match else ""
        
        # Extract WHERE
        where_match = re.search(r'WHERE\s+(.*?)(?:\s+GROUP\s+BY\b|\s+HAVING\b|\s+ORDER\s+BY\b|\s+LIMIT\b|$)', 
                            processed_query, re.IGNORECASE | re.DOTALL)
        components['where_clause'] = where_match.group(1).strip() if where_match else ""
        
        # Extract GROUP BY
        group_by_match = re.search(r'GROUP\s+BY\s+(.*?)(?:\s+HAVING\b|\s+ORDER\s+BY\b|\s+LIMIT\b|$)', 
                                processed_query, re.IGNORECASE | re.DOTALL)
        components['group_by_clause'] = group_by_match.group(1).strip() if group_by_match else ""
        
        # Extract HAVING
        having_match = re.search(r'HAVING\s+(.*?)(?:\s+ORDER\s+BY\b|\s+LIMIT\b|$)', 
                                processed_query, re.IGNORECASE | re.DOTALL)
        components['having_clause'] = having_match.group(1).strip() if having_match else ""
        
        # Extract ORDER BY
        order_by_match = re.search(r'ORDER\s+BY\s+(.*?)(?:\s+LIMIT\b|$)', 
                                processed_query, re.IGNORECASE | re.DOTALL)
        components['order_by_clause'] = order_by_match.group(1).strip() if order_by_match else ""
        
        # Extract LIMIT
        limit_match = re.search(r'LIMIT\s+(.*?)$', processed_query, re.IGNORECASE | re.DOTALL)
        components['limit_clause'] = limit_match.group(1).strip() if limit_match else ""
        

        if subqueries:
            components['subqueries'] = subqueries

            components['subquery_analysis'] = [self.extract_query_components(sq) for sq in subqueries]
        

        for key in components:
            if key not in ['subqueries', 'subquery_analysis'] and components[key]:
                for placeholder, subquery_text in placeholder_map.items():
                    components[key] = components[key].replace(placeholder, f"SELECT {subquery_text}")
        
        return components

    def compare_execution_plans(self, plan1, plan2) -> Dict[str, Any]:
        """Compare execution plans and calculate similarity metrics"""
        if not plan1 or not plan2:
            return {
                "plan_similarity": 0,
                "operation_match": 0,
                "details": "One or both execution plans missing"
            }
        
        try:

            if isinstance(plan1, str):
                plan1 = json.loads(plan1)
            if isinstance(plan2, str):
                plan2 = json.loads(plan2)
                

            ops1 = [item.get("detail", "") for item in plan1 if "detail" in item]
            ops2 = [item.get("detail", "") for item in plan2 if "detail" in item]
            

            common_ops = set(ops1).intersection(set(ops2))
            total_ops = set(ops1).union(set(ops2))
            op_similarity = len(common_ops) / len(total_ops) if total_ops else 0
            

            sequence_similarity = difflib.SequenceMatcher(None, str(ops1), str(ops2)).ratio()
            
            return {
                "plan_similarity": (op_similarity + sequence_similarity) / 2,
                "operation_match": op_similarity,
                "sequence_similarity": sequence_similarity,
                "common_operations": list(common_ops),
                "operations_only_in_first": list(set(ops1) - set(ops2)),
                "operations_only_in_second": list(set(ops2) - set(ops1))
            }
        except Exception as e:
            return {
                "plan_similarity": 0,
                "operation_match": 0,
                "details": f"Error comparing plans: {str(e)}"
            }
            
    def categorize_errors(self, error_msg) -> str:
        """Categorize SQL errors into types for better analysis"""
        if not error_msg or pd.isna(error_msg):
            return "No error"
            
        error_types = {
            "syntax": ["syntax error", "parse error", "SQL syntax", "unexpected", "missing"],
            "schema": ["no such table", "no such column", "ambiguous", "not recognized"],
            "constraint": ["constraint", "unique", "foreign key", "not null", "check constraint"],
            "type": ["datatype", "type mismatch", "incompatible types", "cannot convert"],
            "function": ["function", "procedure", "aggregate", "window function"],
            "timeout": ["timeout", "query interrupted", "execution time", "too long"],
            "permission": ["permission", "access", "privilege"]
        }
        
        error_msg = str(error_msg).lower()
        
        for category, keywords in error_types.items():
            if any(keyword in error_msg for keyword in keywords):
                return category
                
        return "other"

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
    
    def calculate_composite_score(self, analysis_row) -> float:
        """Calculate a weighted composite score for query evaluation"""
        weights = {
            "results_match": 0.3,
            "table_access_accuracy": 0.2,
            "column_access_accuracy": 0.2,
            "query_similarity": 0.1,
            "plan_similarity": 0.2
        }
        
        score = 0.0
        for metric, weight in weights.items():
            if metric in analysis_row and not pd.isna(analysis_row[metric]):
                score += weight * float(analysis_row[metric])
        
        return score

    def compare_queries(self, df: pd.DataFrame, true_df: Optional[pd.DataFrame] = None, 
                       to_compare='generated_sql', output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Enhanced query comparison with comprehensive metrics.
        
        Args:
            df: DataFrame with generated queries
            true_df: Optional DataFrame with ground truth queries and execution details
            to_compare: Column name of the generated queries to compare
            output_path: Optional path to save the results
            
        Returns:
            DataFrame with comparison metrics
        """
        if "true_query" not in df.columns or to_compare not in df.columns:
            print(
                f"Error: DataFrame must contain 'true_query' and '{to_compare}' columns"
            )
            return pd.DataFrame()


        if true_df is not None:
            print(f"Merging with ground truth dataframe containing {len(true_df)} records")
            required_cols = ["id", "question", "true_query"]
            if not all(col in true_df.columns for col in required_cols):
                print(f"Warning: true_df missing required columns: {required_cols}")
            

            merge_cols = [col for col in ["id", "question", "true_query"] if col in df.columns and col in true_df.columns]
            
            if not merge_cols:
                print("Error: No common columns found for merging")
                df_merged = df.copy()
            else:

                df_merged = df.merge(
                    true_df,
                    on=merge_cols,
                    how="left",
                    suffixes=("", "_true")
                )
                print(f"Merged dataframe has {len(df_merged)} rows and {len(df_merged.columns)} columns")
        else:
            df_merged = df.copy()

        print(f"Comparing queries for {len(df_merged)} rows...")

        results = []

        for _, row in tqdm(df_merged.iterrows(), total=len(df_merged), desc="Comparing queries"):
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


            result["true_success"] = row.get("success_true", row.get("success", False))
            result["gen_success"] = row.get("success", False)
            result["execution_success_match"] = int(result["true_success"] == result["gen_success"])
            

            true_row_count = row.get("row_count_true", row.get("row_count", 0))
            gen_row_count = row.get("row_count", 0)
            result["true_row_count"] = true_row_count
            result["gen_row_count"] = gen_row_count
            result["row_count_match"] = int(true_row_count == gen_row_count)
            
            if true_row_count > 0 and gen_row_count > 0:
                result["row_count_ratio"] = gen_row_count / true_row_count
            else:
                result["row_count_ratio"] = 0
                

            if "execution_time_true" in row or "execution_time" in row:
                true_exec_time = row.get("execution_time_true", row.get("execution_time", 0))
                gen_exec_time = row.get("execution_time", 0)
                result["true_execution_time"] = true_exec_time
                result["gen_execution_time"] = gen_exec_time
                
                if true_exec_time > 0 and gen_exec_time > 0:
                    result["execution_time_ratio"] = gen_exec_time / true_exec_time
                else:
                    result["execution_time_ratio"] = 0
            

            if "error_true" in row or "error" in row:
                true_error = row.get("error_true", row.get("error", ""))
                gen_error = row.get("error", "")
                result["true_error_category"] = self.categorize_errors(true_error)
                result["gen_error_category"] = self.categorize_errors(gen_error)


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
                

            true_components = self.extract_query_components(true_query)
            gen_components = self.extract_query_components(gen_query)


            for component in ['select_clause', 'from_clause', 'where_clause', 
                            'group_by_clause', 'having_clause', 'order_by_clause', 
                            'limit_clause']:
                true_component = true_components.get(component, "")
                gen_component = gen_components.get(component, "")
                

                if isinstance(true_component, str) and isinstance(gen_component, str):
                    component_similarity = difflib.SequenceMatcher(None, true_component, gen_component).ratio()
                    result[f"{component}_similarity"] = component_similarity
                else:
                    result[f"{component}_similarity"] = 0.0


            result["true_components"] = json.dumps(true_components, default=str)
            result["gen_components"] = json.dumps(gen_components, default=str)


            if "execution_plan" in row or "execution_plan_true" in row:
                true_plan = row.get("execution_plan_true", row.get("execution_plan", None))
                gen_plan = row.get("execution_plan", None)
                
                if true_plan or gen_plan:
                    plan_comparison = self.compare_execution_plans(true_plan, gen_plan)
                    for key, value in plan_comparison.items():
                        result[f"plan_{key}"] = value
                else:
                    result["execution_plan_available"] = False


            true_result = row.get("results_true", row.get("true_result", row.get("results", [])))
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
            

            result["true_empty_result"] = len(true_result) == 0 if true_result is not None else True
            result["gen_empty_result"] = len(gen_result) == 0 if gen_result is not None else True
            result["both_empty"] = result["true_empty_result"] and result["gen_empty_result"]
            
            gen_success = row.get('success', False)
            if gen_success:
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
                

            result["composite_score"] = self.calculate_composite_score(result)

            results.append(result)

        analysis_df = pd.DataFrame(results)
        
        try:

            metrics = {
                "total_queries": len(analysis_df),
                "execution_success_rate": analysis_df["gen_success"].mean() if "gen_success" in analysis_df.columns else np.nan,
                "result_match_rate": analysis_df["results_match"].mean() if "results_match" in analysis_df.columns else np.nan,
                "execution_success_match_rate": analysis_df["execution_success_match"].mean() if "execution_success_match" in analysis_df.columns else np.nan,
                "row_count_match_rate": analysis_df["row_count_match"].mean() if "row_count_match" in analysis_df.columns else np.nan,
                "avg_table_access_accuracy": analysis_df["table_access_accuracy"].mean(),
                "avg_column_access_accuracy": analysis_df["column_access_accuracy"].mean(),
                "avg_query_similarity": analysis_df["query_similarity"].mean(),
                "avg_composite_score": analysis_df["composite_score"].mean(),
                "empty_results_both": analysis_df["both_empty"].sum() if "both_empty" in analysis_df.columns else np.nan,
            }
        
            print("\nAggregated Evaluation Metrics:")
            for metric, value in metrics.items():
                if not pd.isna(value):
                    print(f"{metric}: {value:.4f}")
        
        except Exception as e:
            print(e)
            
        if output_path:

            if true_df is not None:
                merged_path = str(output_path).replace('.csv', '_merged.csv')
                df_merged.to_csv(merged_path, index=False)
                print(f"Merged dataframe saved to {merged_path}")
            

            analysis_df.to_csv(output_path, index=False)
            print(f"Analysis dataframe saved to {output_path}")
            

            metrics_path = str(output_path).replace('.csv', '_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Metrics saved to {metrics_path}")

        return analysis_df



def evaluate_model_performance(true_df, generated_df, output_path=None, to_compare='generated_sql'):
    """
    Evaluate model performance by comparing generated SQL queries with ground truth
    
    Args:
        true_data_path: Path to CSV with ground truth data
        generated_data_path: Path to CSV with generated queries
        output_path: Path to save evaluation results
        
    Returns:
        Analysis DataFrame with comparison metrics
    """
    

    analyzer = QueryAnalyzer()
    return analyzer.compare_queries(generated_df, true_df=true_df, output_path=output_path, to_compare=to_compare)