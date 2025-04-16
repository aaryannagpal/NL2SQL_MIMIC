import json
import pandas as pd
import random
import time
import threading
import concurrent.futures
import os
import sys
import sqlite3
from pathlib import Path
import re
from typing import List, Dict, Tuple, Set, Optional, Any, Union
from tqdm import tqdm
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import (
    MIMIC_SCHEMA_PATH,
    DICTIONARY_MAP_PATH,
    MYSQL_DB_PATH,
    MIMIC_SAMPLE_PATH,

    DEFAULT_PATTERN_FOR_LIKE_OPERATION,
)

with open(MIMIC_SCHEMA_PATH) as f:
    MIMIC_SCHEMA = json.load(f)

with open(DICTIONARY_MAP_PATH) as f:
    DICTIONARY_MAP = json.load(f)


class MimicSchema:
    """Class for handling schema information and relationships between tables"""

    def __init__(self):
        self.schema = MIMIC_SCHEMA
        self.dict_mappings = DICTIONARY_MAP
        self.db_path = MYSQL_DB_PATH

        self.max_workers = 35
        self.sample_size = 150

        self.tables = self._extract_tables()
        self.columns = self._extract_columns()
        self.join_paths = self._identify_join_paths()
        self.pk_fk_relationships = self._identify_pk_fk_relationships()

        self.default_sample_values = None

        self.check_status()

    def check_status(self) -> None:
        status = f"""
        Current Status:
            max_workers set: {self.max_workers}
            sample_size set: {self.sample_size}
            sample values stored: {False if self.default_sample_values is None else True}
        """
        print(status)

    def show_mappings(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(DICTIONARY_MAP).T

    def _extract_tables(self) -> List[str]:
        """Extract table names from the schema"""
        return [table.replace(".json", "") for table in self.schema.keys()]

    def _extract_columns(self) -> Dict[str, List[str]]:
        """Extract columns for each table"""
        columns = {}
        for table_json, table_data in self.schema.items():
            table_name = table_json.replace(".json", "")
            columns[table_name] = list(table_data["columns"].keys())
        return columns

    def _identify_join_paths(self) -> Dict[str, Dict[str, List[str]]]:
        """Identify possible join paths between tables"""
        join_paths = {}
        for table1 in self.tables:
            join_paths[table1] = {}
            for table2 in self.tables:
                if table1 != table2:
                    common_columns = set(self.columns[table1]) & set(
                        self.columns[table2]
                    )
                    if common_columns:
                        join_paths[table1][table2] = list(common_columns)
        return join_paths

    def _identify_pk_fk_relationships(
        self,
    ) -> Dict[str, Dict[str, List[Tuple[str, str]]]]:
        """Identify primary key/foreign key relationships"""
        relationships = {}

        # For each table, identify potential FK relationships
        for table in self.tables:
            relationships[table] = {"incoming": [], "outgoing": []}

            # Common identifier columns often used for joins
            id_columns = [col for col in self.columns[table] if col.endswith("_id")]

            for other_table in self.tables:
                if table == other_table:
                    continue

                # Look for matching ID columns
                for id_col in id_columns:
                    if id_col in self.columns[other_table]:
                        # This is a potential relationship
                        if (
                            id_col == f"{table}_id"
                        ):  # This table's PK referenced elsewhere
                            relationships[table]["incoming"].append(
                                (other_table, id_col)
                            )
                        elif (
                            id_col == f"{other_table}_id"
                        ):  # This table references another's PK
                            relationships[table]["outgoing"].append(
                                (other_table, id_col)
                            )

        return relationships

    def generate_sample_values(
        self, save_to: str = "sample_values_mimic.json"
    ) -> Dict[str, Dict[str, List[Any]]]:
        """
        Generate sample values for columns.

        If a database path is provided, this will extract real sample values from the SQLite database.
        Otherwise, it will generate realistic synthetic values based on column types and domain knowledge.
        """
        sample_values = {}

        # Try to load existing sample values file
        try:
            with open(MIMIC_SAMPLE_PATH, "r") as f:
                self.default_sample_values = json.load(f)
                print(f"Default Sample File found at {MIMIC_SAMPLE_PATH}")
                return self.default_sample_values
        except (FileNotFoundError, json.JSONDecodeError):
            print("\n\nSample values file not found or invalid.\n\n")

        # If database path is provided, extract values from database
        if self.db_path and os.path.exists(self.db_path):
            print(f"\n\nExtracting sample values from database: {self.db_path}\n\n")
            self.default_sample_values = self._extract_sample_values_from_sqlite(
                save_path=save_to,
                sample_size=self.sample_size,
                max_workers=self.max_workers,
            )
            return self.default_sample_values

    def _extract_sample_values_from_sqlite(
        self, save_path: str, sample_size: int = 10, max_workers: int = 4
    ) -> Dict[str, Dict[str, List[Any]]]:
        """
        Extract sample values directly from the SQLite database.

        Args:
            sample_size: Number of sample values to extract per column

        Returns:
            Dictionary of sample values for each column in each table
        """

        sample_values = {}
        sample_values_lock = threading.Lock()

        try:
            # Connect to the SQLite database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            existing_tables = [row[0] for row in cursor.fetchall()]
            cursor.close()
            conn.close()

            tables_to_process = [
                table for table in self.tables if table in existing_tables
            ]

            if not tables_to_process:
                print("No tables from schema found in database, using synthetic values")
                return

            print(
                f"Found {len(tables_to_process)} tables to process using {max_workers} parallel workers"
            )

            def process_table(table):
                table_values = {}

                try:
                    time.sleep(random.uniform(0, 0.5))

                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()

                    print(f"Processing table: {table}")

                    total_columns = len(self.columns[table])

                    for i, column in enumerate(self.columns[table]):
                        try:
                            if i % 10 == 0 or i == total_columns - 1:
                                print(
                                    f"  Table {table}: Processing column {i+1}/{total_columns}"
                                )

                            if table in [
                                "chartevents",
                                "labevents",
                                "inputevents",
                                "outputevents",
                            ]:
                                # For very large tables
                                query = f"""
                                SELECT DISTINCT "{column}" 
                                FROM {table}
                                WHERE "{column}" IS NOT NULL
                                ORDER BY RANDOM()
                                LIMIT {sample_size}
                                """
                            else:
                                query = f"""
                                SELECT DISTINCT "{column}" 
                                FROM {table}
                                WHERE "{column}" IS NOT NULL
                                LIMIT {sample_size}
                                """

                            cursor.execute(query)
                            results = cursor.fetchall()

                            values = [row[0] for row in results if row[0] is not None]

                            if values:
                                table_values[column] = values
                            else:
                                table_values[column] = ["<To_Fill>"]

                        except Exception as e:
                            print(
                                f"  Error extracting values for {table}.{column}: {e}"
                            )
                            table_values[column] = ["<To_Fill>"]

                    cursor.close()
                    conn.close()

                    with sample_values_lock:
                        sample_values[table] = table_values

                    return True

                except Exception as e:
                    print(f"Error processing table {table}: {e}")
                    return False

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                futures = [
                    executor.submit(process_table, table) for table in tables_to_process
                ]

                completed = 0
                total = len(futures)

                for future in concurrent.futures.as_completed(futures):
                    completed += 1
                    print(f"Completed {completed}/{total} tables")

            with open(save_path, "w") as f:
                json.dump(sample_values, f, indent=2, default=str)

            print(
                f"Sample values extracted for {len(sample_values)} tables and saved to mimic_sample_values.json"
            )

            return sample_values

        except sqlite3.Error as e:
            print(f"SQLite database error: {e}")
            print("Falling back to synthetic sample values")
            return  # self._generate_synthetic_sample_values(sample_size)

        except Exception as e:
            print(f"Unexpected error during sample value extraction: {e}")
            return

    def get_column_type(self, table: str, column: str) -> str:
        """Get the data type of a column"""
        return self.schema[f"{table}.json"]["columns"][column]["type"]

    def get_column_description(self, table: str, column: str) -> str:
        """Get the description of a column"""
        return self.schema[f"{table}.json"]["columns"][column]["description"]

    def get_joinable_tables(self, table: str) -> List[str]:
        """Get tables that can be joined with the given table"""
        return list(self.join_paths.get(table, {}).keys())

    def get_join_columns(self, table1: str, table2: str) -> List[Tuple[str, str]]:
        """Get columns that can be used to join two tables

        Returns:
            List of tuples (table1_column, table2_column) that can be used for joining
        """
        return self.join_paths.get(table1, {}).get(table2, [])

    def get_dictionary_info(self, table: str) -> Optional[Dict[str, str]]:
        """Get dictionary mapping information for a table if it exists"""
        return self.dict_mappings.get(table)

    def get_random_column(self, table: str, exclude: List[str] = None) -> str:
        """Get a random column from a table, optionally excluding some columns"""
        available_columns = [
            col for col in self.columns[table] if not exclude or col not in exclude
        ]
        if not available_columns:
            return None
        return random.choice(available_columns)

    def get_sample_value(self, table: str, column: str) -> Any:
        """Get a random sample value for a column"""
        if not self.default_sample_values:
            print("Sample values not found or generated.")
            return

        values = self.default_sample_values.get(table, {}).get(column, [])
        if not values:
            return None
        return random.choice(values)

    def get_tables_with_column(self, column_name: str) -> List[str]:
        """Get all tables that have a specific column"""
        return [table for table in self.tables if column_name in self.columns[table]]


class QueryTemplateGenerator:
    """Base class for query template generation"""
    
    def __init__(self, schema: MimicSchema):
        self.schema = schema
        
        self.operators = ["=", ">", "<", ">=", "<=", "<>", "LIKE", "IN", "NOT IN", "IS NULL", "IS NOT NULL", "BETWEEN", "NOT BETWEEN"]
        self.logical_ops = ["AND", "OR"]
        self.aggregations = ["COUNT", "AVG", "MAX", "MIN", "SUM"]
        self.sort_dirs = ["ASC", "DESC"]
        self.limit_values = [3, 5, 10, 15, 20, 25, 50, 100]
        
        self.operator_phrases = {
            "=": ["equal to", "is", "matching"],
            ">": ["greater than", "more than", "exceeding", "above"],
            "<": ["less than", "lower than", "below", "under"],
            ">=": ["at least", "greater than or equal to", "no less than"],
            "<=": ["at most", "less than or equal to", "no more than", "up to"],
            "<>": ["not equal to", "different from", "not", "excluding"],
            "LIKE": ["containing", "including", "with", "like"],
            "IN": ["in", "among", "one of", "included in"],
            "NOT IN": ["not in", "not among", "not one of", "excluded from"],
            "IS NULL": ["is missing", "is not recorded", "is null", "is empty"],
            "IS NOT NULL": ["is recorded", "is available", "is not null", "is not empty"],
            "BETWEEN": ["between {sample_1} and {sample_2}", "in the range from {sample_1} to {sample_2}", "within the range of {sample_1} and {sample_2}", "falling between {sample_1} and {sample_2}", "from {sample_1} to {sample_2}", "in the interval from {sample_1} to {sample_2}", "ranging from {sample_1} to {sample_2}"],
            "NOT BETWEEN": ["not between {sample_1} and {sample_2}", "outside the range from {sample_1} to {sample_2}", "not within the range of {sample_1} and {sample_2}", "not falling between {sample_1} and {sample_2}", "excluding values from {sample_1} to {sample_2}", "not in the interval from {sample_1} to {sample_2}", "not ranging from {sample_1} to {sample_2}", "less than {sample_1} or greater than {sample_2}"]
        }

        self.star_column = [
            "records",
            "all columns",
            "results",
            "all items",
            "rows",
            "complete rows"
        ]

        self.limit_phrases = [
            "show {limit} results",
            "give me {limit} examples",
            "fetch {limit} rows",
            "limit to {limit} results",
            "show the first {limit} results",
            "display up to {limit} records",
            "return a maximum of {limit} items"
        ]

        self.sort_phrases = {
            "ASC": ["smallest first", "in ascending order", "ascendingly", "increasingly", "in increasing order", "least first"],
            "DESC": ["largest first", "in descending order", "descendingly", "decreasingly", "in decreasing order", "greatest first"]
        }

        self.order_phrases = [
            "ordered by {order_col} {nl_dir}",
            "sorted by {order_col} {nl_dir}",
            "arranged by {order_col} {nl_dir}",
            "listed by {order_col} {nl_dir}",
            "ranked by {order_col} {nl_dir}",
        ]
        
        self.aggregation_phrases = {
            "COUNT": ["count", "total number", "quantity"],
            "AVG": ["average", "mean", "typical"],
            "MAX": ["maximum", "highest", "largest", "greatest"],
            "MIN": ["minimum", "lowest", "smallest", "least"],
            "SUM": ["sum", "total", "combined"]
        }
    
    def random_columns(self, table: str, min_cols: int = 1, max_cols: int = 3, all_col = 0.8) -> List[str]:
        """Select random columns from a table"""
        if random.random() > all_col:
            return ["*"]
        columns = self.schema.columns[table]
        num_cols = min(random.randint(min_cols, max_cols), len(columns))
        return random.sample(columns, num_cols)

    def random_filter(self, table: str, column: str) -> Tuple[str, str]:
        """Generate a random filter condition and its NL description"""
        
        op = random.choice(self.operators)
        column_type = self.schema.get_column_type(table, column)
        sample_value = self.schema.get_sample_value(table, column)
        if isinstance(sample_value, str):
            sample_value = sample_value.replace("'", "''")  
        if isinstance(sample_value, list):
            if sample_value:
                sample_value = random.choice(sample_value)
                if isinstance(sample_value, str):
                    sample_value = sample_value.replace("'", "''")  
            else:
                print("Sample Values have NoneType present. Please fix")
                return

        if op == "LIKE":
            if "char" in column_type or "text" in column_type:
                if isinstance(sample_value, str):
                    words = sample_value.split()
                    if len(words) > 1:
                        non_empty_words = [w for w in words if len(w) > 0]
                        if non_empty_words:
                            pattern = random.choice(non_empty_words)
                        else:
                            pattern = sample_value
                    else:
                        pattern = sample_value
                    
                    pattern = re.sub(r'[^\w\s]', '', pattern).strip()
                    
                    if not pattern:
                        pattern = DEFAULT_PATTERN_FOR_LIKE_OPERATION
                    
                    sql_value = f"'%{pattern}%'"
                    display_value = pattern
                else:
                    pattern = str(sample_value)
                    sql_value = f"'%{pattern}%'"
                    display_value = pattern
            else:
                pattern = str(sample_value)
                sql_value = f"'%{pattern}%'"
                display_value = pattern
                
        
        if op in ["BETWEEN", "NOT BETWEEN"]:
            sample_value_2 = self.schema.get_sample_value(table, column)
            
            if "char" in column_type or "text" in column_type:
                if isinstance(sample_value_2, str):
                    sample_value_2 = sample_value_2.replace("'", "''")
                sql_value = f"'{sample_value}' AND '{sample_value_2}'"
                display_value = (str(sample_value), str(sample_value_2))
            
            elif "datetime" in column_type:
                try:
                    if len(sample_value.split(' ')) > 1:    
                        original_timestamp = datetime.strptime(sample_value, '%Y-%m-%d %H:%M:%S')
                        original_timestamp_2 = datetime.strptime(sample_value_2, '%Y-%m-%d %H:%M:%S')
                        
                        date_format_choice = random.choice([
                            "full_datetime",  # "January 15, 2020 at 08:30 AM"
                            "date_only",      # "January 15, 2020"
                            "month_year",     # "January 2020"
                            "date_simple"     # "2020-01-15"
                        ])
                        
                        if date_format_choice == "full_datetime":
                            # Keep original with full date and time
                            sql_value = f"'{sample_value}' AND '{sample_value_2}'"
                            
                            date_str = original_timestamp.strftime('%B %d, %Y')
                            time_str = original_timestamp.strftime('%I:%M %p')

                            date_str_2 = original_timestamp_2.strftime('%B %d, %Y')
                            time_str_2 = original_timestamp_2.strftime('%I:%M %p')

                            display_value = (f"{date_str} at {time_str}", f"{date_str_2} at {time_str_2}")
                            
                        elif date_format_choice == "date_only":
                            # Set time to midnight for date-only comparison
                            date_only = original_timestamp.replace(hour=0, minute=0, second=0)
                            date_only_2 = original_timestamp_2.replace(hour=0, minute=0, second=0)
                            sql_value = f"'{date_only.strftime('%Y-%m-%d %H:%M:%S')}' AND '{date_only_2.strftime('%Y-%m-%d %H:%M:%S')}'"
                            display_value = (date_only.strftime('%B %d, %Y'), date_only_2.strftime('%B %d, %Y'))
                            
                        elif date_format_choice == "month_year":
                            # Set to first day of month at midnight
                            month_year = original_timestamp.replace(day=1, hour=0, minute=0, second=0)
                            month_year_2 = original_timestamp_2.replace(day=1, hour=0, minute=0, second=0)
                            sql_value = f"'{month_year.strftime('%Y-%m-%d %H:%M:%S')}' AND '{month_year_2.strftime('%Y-%m-%d %H:%M:%S')}'"
                            display_value = (month_year.strftime('%B %Y'), month_year_2.strftime('%B %Y'))
                            
                        else:
                            # Date in YYYY-MM-DD format with midnight time
                            date_simple = original_timestamp.replace(hour=0, minute=0, second=0)
                            date_simple_2 = original_timestamp_2.replace(hour=0, minute=0, second=0)
                            sql_value = f"'{date_simple.strftime('%Y-%m-%d %H:%M:%S')}' AND '{date_simple_2.strftime('%Y-%m-%d %H:%M:%S')}'"
                            display_value = (date_simple.strftime('%Y-%m-%d'), date_simple_2.strftime('%Y-%m-%d'))
                    else:
                        original_date = datetime.strptime(sample_value, '%Y-%m-%d')
                        original_date_2 = datetime.strptime(sample_value_2, '%Y-%m-%d')
                        
                        date_format_choice = random.choice([
                            "date_only",      # "January 15, 2020"
                            "month_year",     # "January 2020"
                            "date_simple"     # "2020-01-15"
                        ])
                        
                        if date_format_choice == "date_only":
                            # Set time to midnight for both dates
                            date_only = original_date.replace(hour=0, minute=0, second=0)
                            date_only_2 = original_date_2.replace(hour=0, minute=0, second=0)
                            sql_value = f"'{date_only.strftime('%Y-%m-%d %H:%M:%S')}' AND '{date_only_2.strftime('%Y-%m-%d %H:%M:%S')}'"
                            display_value = (
                                date_only.strftime('%B %d, %Y'), 
                                date_only_2.strftime('%B %d, %Y')
                            )
                            
                        elif date_format_choice == "month_year":
                            # Set to first day of month at midnight for both dates
                            month_year = original_date.replace(day=1, hour=0, minute=0, second=0)
                            month_year_2 = original_date_2.replace(day=1, hour=0, minute=0, second=0)
                            sql_value = f"'{month_year.strftime('%Y-%m-%d %H:%M:%S')}' AND '{month_year_2.strftime('%Y-%m-%d %H:%M:%S')}'"
                            display_value = (
                                month_year.strftime('%B %Y'), 
                                month_year_2.strftime('%B %Y')
                            )
                            
                        else:  # date_simple
                            # Simple YYYY-MM-DD format with midnight time for both dates
                            date_simple = original_date.replace(hour=0, minute=0, second=0)
                            date_simple_2 = original_date_2.replace(hour=0, minute=0, second=0)
                            sql_value = f"'{date_simple.strftime('%Y-%m-%d %H:%M:%S')}' AND '{date_simple_2.strftime('%Y-%m-%d %H:%M:%S')}'"
                            display_value = (
                                date_simple.strftime('%Y-%m-%d'), 
                                date_simple_2.strftime('%Y-%m-%d')
                            )


                except (ValueError, TypeError):
                    sql_value = f"'{sample_value}' AND {sample_value_2}"
                    display_value = (str(sample_value), str(sample_value_2))
            
            else:
                sql_value = f"{sample_value} AND {sample_value_2}"
                display_value = (str(sample_value), str(sample_value_2))


        elif op in ["IN", "NOT IN"]:
            samples_list_size = random.randint(1,10)
            if "int" in column_type or "float" in column_type:
                if isinstance(sample_value, (int, float)):
                    base_value = int(sample_value)
                    all_samples = self.schema.default_sample_values.get(table, {}).get(column, [])
                    if len(all_samples) >= samples_list_size:
                        numeric_samples = []
                        for val in all_samples:
                            try:
                                if isinstance(val, (int, float)):
                                    numeric_samples.append(val)
                                elif isinstance(val, str) and val.replace('.', '', 1).isdigit():
                                    numeric_samples.append(float(val))
                            except:
                                continue
                        
                        if len(numeric_samples) >= samples_list_size:
                            values = random.sample(numeric_samples, samples_list_size)
                        else:
                            values = [base_value, base_value + random.randint(1, 10), base_value + random.randint(11, 20)]
                    else:
                        values = [base_value, base_value + random.randint(1, 10), base_value + random.randint(11, 20)]
                else:
                    values = [random.randint(1, 100) for _ in range(samples_list_size)]
                
                sql_value = f"({', '.join(map(str, values))})"
                display_value = ", ".join(map(str, values))
            
            else:
                all_samples = self.schema.default_sample_values.get(table, {}).get(column, [])
                string_samples = [s.replace("'", "''") for s in all_samples if isinstance(s, str) and s]
                
                if len(string_samples) >= samples_list_size:
                    values = random.sample(string_samples, samples_list_size)
                else:
                    if isinstance(sample_value, str) and sample_value:
                        values = [sample_value]
                        if len(all_samples) > 1:
                            for s in all_samples:
                                if s != sample_value and isinstance(s, str) and s:
                                    values.append(s)
                                    if len(values) >= samples_list_size:
                                        break
                        
                        domain_alternatives = []
                        if column == 'gender':
                            domain_alternatives = ['M', 'F']
                        elif column == 'admission_type':
                            domain_alternatives = ['EMERGENCY', 'ELECTIVE', 'URGENT']
                        elif column == 'ethnicity':
                            domain_alternatives = ['WHITE', 'BLACK', 'HISPANIC', 'ASIAN']
                        elif column == 'insurance':
                            domain_alternatives = ['Medicare', 'Medicaid', 'Private']
                        
                        for alt in domain_alternatives:
                            if alt != sample_value and alt not in values:
                                values.append(alt)
                                if len(values) >= samples_list_size:
                                    break
                    else:
                        values = [f"Sample{i}" for i in range(1, 4)]
                
                # Format for SQL
                sql_value = "(" + ", ".join(f"'{v}'" for v in values) + ")"
                display_value = f"{', '.join(map(str, values))}"

                
        elif op in ["IS NULL", "IS NOT NULL"]:
            sql_value = ""
            display_value = ""
            sql_filter = f"{column} {op}"


        else:
            if "char" in column_type or "text" in column_type:
                sql_value = f"'{sample_value}'"
                display_value = str(sample_value)
            
            elif "datetime" in column_type:
                try:
                    if len(sample_value.split(' ')) > 1:                 
                        original_timestamp = datetime.strptime(sample_value, '%Y-%m-%d %H:%M:%S')
                        
                        date_format_choice = random.choice([
                            "full_datetime",  # "January 15, 2020 at 08:30 AM"
                            "date_only",      # "January 15, 2020"
                            "month_year",     # "January 2020"
                            "date_simple"     # "2020-01-15"
                        ])
                        
                        if date_format_choice == "full_datetime":
                            # Keep original with full date and time
                            sql_value = f"'{sample_value}'"
                            date_str = original_timestamp.strftime('%B %d, %Y')
                            time_str = original_timestamp.strftime('%I:%M %p')
                            display_value = f"{date_str} at {time_str}"
                            
                        elif date_format_choice == "date_only":
                            # Set time to midnight for date-only comparison
                            date_only = original_timestamp.replace(hour=0, minute=0, second=0)
                            sql_value = f"'{date_only.strftime('%Y-%m-%d %H:%M:%S')}'"
                            display_value = date_only.strftime('%B %d, %Y')
                            
                        elif date_format_choice == "month_year":
                            # Set to first day of month at midnight
                            month_year = original_timestamp.replace(day=1, hour=0, minute=0, second=0)
                            sql_value = f"'{month_year.strftime('%Y-%m-%d %H:%M:%S')}'"
                            display_value = month_year.strftime('%B %Y')
                            
                        else:
                            # Date in YYYY-MM-DD format with midnight time
                            date_simple = original_timestamp.replace(hour=0, minute=0, second=0)
                            sql_value = f"'{date_simple.strftime('%Y-%m-%d %H:%M:%S')}'"
                            display_value = date_simple.strftime('%Y-%m-%d')
                    else:
                        original_date = datetime.strptime(sample_value, '%Y-%m-%d')
    
                        date_format_choice = random.choice([
                            "date_only",      # "January 15, 2020"
                            "month_year",     # "January 2020" 
                            "date_simple"     # "2020-01-15"
                        ])
                        
                        if date_format_choice == "date_only":
                            sql_value = f"'{original_date.strftime('%Y-%m-%d')} 00:00:00'"
                            display_value = original_date.strftime('%B %d, %Y')
                            
                        elif date_format_choice == "month_year":
                            month_year = original_date.replace(day=1)
                            sql_value = f"'{month_year.strftime('%Y-%m-%d')} 00:00:00'"
                            display_value = month_year.strftime('%B %Y')
                            
                        else:  # date_simple
                            sql_value = f"'{original_date.strftime('%Y-%m-%d')} 00:00:00'"
                            display_value = original_date.strftime('%Y-%m-%d')
                
                except (ValueError, TypeError) as e:
                    sql_value = f"'{sample_value}'"
                    display_value = str(sample_value)
            
            else:
                sql_value = str(sample_value)
                display_value = str(sample_value)
        
        if op not in ["IS NULL", "IS NOT NULL"]:
            sql_filter = f"{column} {op} {sql_value}"
        
        op_phrase = random.choice(self.operator_phrases[op])
        
        if op in ["IS NULL", "IS NOT NULL"]:
            nl_filter = f"{column.replace('_', ' ')} {op_phrase}"
        elif op in ["BETWEEN", "NOT BETWEEN"]:
            op_phrase = op_phrase.format(sample_1=display_value[0], sample_2=display_value[1])
            nl_filter = f"{column.replace('_', ' ')} {op_phrase}"
        else:
            nl_filter = f"{column.replace('_', ' ')} {op_phrase} {display_value}"
        
        return sql_filter, nl_filter

    def random_join_condition(self, table1: str, table2: str) -> Tuple[str, str]:
        """Generate a join condition between two tables"""
        
        possible_joins = []
        # print(table1, table2)
        if table1 in self.schema.dict_mappings:
            for dict_info in self.schema.dict_mappings[table1]:
                if dict_info["dict_table"] == table2:
                    col1 = dict_info["code_column"]
                    col2 = dict_info["dict_code_column"]
                    dict_sql_join = f"{table1}.{col1} = {table2}.{col2}"
                    desc_column = dict_info["dict_desc_column"]
                    possible_joins.append(("dictionary", dict_sql_join, col1, desc_column))
        
        if table2 in self.schema.dict_mappings:
            for dict_info in self.schema.dict_mappings[table2]:
                if dict_info["dict_table"] == table1:
                    col1 = dict_info["dict_code_column"]
                    col2 = dict_info["code_column"]
                    dict_sql_join = f"{table1}.{col1} = {table2}.{col2}"
                    desc_column = dict_info["dict_desc_column"]
                    possible_joins.append(("dictionary", dict_sql_join, col1, desc_column))
    
        join_columns = self.schema.get_join_columns(table1, table2)
        
        if join_columns:
            for join_column in join_columns:
                regular_sql_join = f"{table1}.{join_column} = {table2}.{join_column}"
                possible_joins.append(("regular", regular_sql_join, join_column, None))
        
        if not possible_joins:
            return None, None
        
        join_info = random.choice(possible_joins)
        join_type, sql_join, join_column, desc_column = join_info
        
        nl_join = self._get_natural_join_phrase(
            table1, table2, join_column, 
            is_dict_join=(join_type == "dictionary"),
            desc_column=desc_column
        )
        
        return sql_join, nl_join

    def _get_natural_join_phrase(self, table1: str, table2: str, join_column: str, is_dict_join: bool = False, desc_column: str = None) -> str:
        """Generate a natural language phrase for a join based on the tables involved"""
        
        if is_dict_join:
            if 'diagnoses' in table1 or 'diagnoses' in table2:
                if desc_column == 'long_title':
                    return "with their complete diagnosis descriptions"
                else:
                    return "with their diagnosis information"
                    
            elif 'procedures' in table1 or 'procedures' in table2:
                if desc_column == 'long_title':
                    return "with their complete procedure descriptions"
                else:
                    return "with their procedure details"
                    
            elif 'lab' in table1 or 'lab' in table2:
                if desc_column == 'label':
                    return "with their lab test names"
                else:
                    return "with their lab test information"
                    
            elif 'chart' in table1 or 'chart' in table2:
                if desc_column == 'label':
                    return "with their chart item descriptions"
                else:
                    return "with their chart details"
                    
            elif 'hcpcs' in table1 or 'hcpcs' in table2:
                if desc_column == 'short_description' or desc_column == 'long_description':
                    return "with their healthcare service descriptions"
                else:
                    return "with their healthcare service details"
                    
            else:
                return "with their associated descriptions"
            
        if ('patients' in table1 and 'admissions' in table2) or ('patients' in table2 and 'admissions' in table1):
            return "during their hospital admissions"
        
        if ('patients' in table1 and 'diagnoses' in table2) or ('patients' in table2 and 'diagnoses' in table1):
            return "who were diagnosed with"
        
        if ('patients' in table1 and 'procedures' in table2) or ('patients' in table2 and 'procedures' in table1):
            return "who underwent"
        
        if ('patients' in table1 and 'prescriptions' in table2) or ('patients' in table2 and 'prescriptions' in table1):
            return "who were prescribed"
        
        if ('patients' in table1 and 'labevents' in table2) or ('patients' in table2 and 'labevents' in table1):
            return "who had lab tests for"
        
        if ('admissions' in table1 and 'transfers' in table2) or ('admissions' in table2 and 'transfers' in table1):
            return "with their transfer details"
        
        if ('admissions' in table1 and 'services' in table2) or ('admissions' in table2 and 'services' in table1):
            return "with their assigned services"
        
        if ('admissions' in table1 and 'icustays' in table2) or ('admissions' in table2 and 'icustays' in table1):
            return "including ICU stay information"
        
        if join_column == 'subject_id':
            return "for each patient"
        
        if join_column == 'hadm_id':
            return "during their hospital stays"
        
        if join_column == 'stay_id':
            return "during their unit stays"
        
        return "along with their related data"

    def operator_to_nl(self, op: str) -> str:
        """Convert SQL operator to natural language"""
        return random.choice(self.operator_phrases.get(op, [op]))
    
    def aggregation_to_nl(self, agg: str) -> str:
        """Convert aggregation function to natural language"""
        return random.choice(self.aggregation_phrases.get(agg, [agg.lower()]))
    
    def generate(self, count: int = 1) -> List[Dict[str, str]]:
        """Generate query templates - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement generate()")


class BasicSelectGenerator(QueryTemplateGenerator):
    """Generate basic SELECT queries with various filters, limits and ordered"""
    
    def generate(self, count: int = 1) -> List[Dict[str, str]]:
        """Generate basic SELECT queries with various WHERE conditions"""
        results = []
        
        for _ in tqdm(range(count), desc = "Generating Queries"):
            table = random.choice(self.schema.tables)            
            columns = self.random_columns(table, min_cols=1, max_cols=3) # input this also
            
            use_where = random.random() > 0.3 #input this value
            where_clause = ""
            nl_filter = ""
            
            if use_where:
                num_conditions = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0] # input
                conditions = []
                nl_conditions = []

                for i in range(num_conditions):
                    filter_col = random.choice(self.schema.columns[table])
                    
                    filter_sql, filter_nl = self.random_filter(table, filter_col)
                    if filter_sql and filter_nl:
                        conditions.append(filter_sql)
                        nl_conditions.append(filter_nl)
                
                if len(conditions) > 1:
                    logical_ops = [random.choice(["AND", "OR"]) for _ in range(len(conditions)-1)]
                    where_clause = "WHERE " + conditions[0]
                    nl_filter = "where " + nl_conditions[0]
                    
                    for i in range(1, len(conditions)):
                        op = logical_ops[i-1]
                        where_clause += f" {op} {conditions[i]}"
                        nl_filter += f" {op.lower()} {nl_conditions[i]}"
                elif len(conditions) == 1:
                    where_clause = f"WHERE {conditions[0]}"
                    nl_filter = f"where {nl_conditions[0]}"

            use_order = random.random() > 0.7 #input
            order_clause = ""
            nl_order = ""
            
            if use_order:
                if len(columns) == 1 and columns[0] == "*":
                    order_col = self.schema.get_random_column(table)
                else:
                    order_col = random.choice(columns)
                order_dir = random.choice(self.sort_dirs)
                order_clause = f"ORDER BY {order_col} {order_dir}"
              
                nl_dir = random.choice(self.sort_phrases[order_dir])
                nl_order = random.choice(self.order_phrases).format(order_col=order_col.replace('_', ' '), nl_dir=nl_dir)    
            
            use_limit = random.random() > 0.25 #input
            limit_clause = ""
            nl_limit = ""
            
            if use_limit:
                limit = random.choice(self.limit_values)
                limit_clause = f"LIMIT {limit}"
                
                nl_limit = random.choice(self.limit_phrases).format(limit=limit)

            if len(columns) == 1 and columns[0] == "*":
                select_clause = f"SELECT * FROM {table}"
            else:
                select_clause = f"SELECT {', '.join(columns)} FROM {table}"
            sql_parts = [
                select_clause,
                where_clause,
                order_clause,
                limit_clause,
                ';'
            ]
            sql_query = " ".join([part for part in sql_parts if part])
            
            entity_type = self._get_entity_type(table)
            
            column_descriptions = []
            if len(columns) == 1 and columns[0] == "*":
                column_descriptions.append(random.choice(self.star_column))
            else:
                for col in columns:
                    readable_col = col.replace('_', ' ')
                    column_descriptions.append(readable_col)
            
            # look over this
            query_templates = [
                f"Show me {', '.join(column_descriptions)} for {entity_type}",
                f"What are the {', '.join(column_descriptions)} of {entity_type}",
                f"List {', '.join(column_descriptions)} from {entity_type}",
                f"Get {', '.join(column_descriptions)} for {entity_type}"
            ]
            template_weights = [0.4, 0.2, 0.2, 0.2]
            base_question = random.choices(query_templates, weights=template_weights)[0]
            
            nl_parts = [
                base_question,
                nl_filter,
                nl_order,
                nl_limit
            ]
            nl_question = " ".join([part for part in nl_parts if part])
            
            if not nl_question.endswith('.'):
                nl_question += '.'
            nl_question = nl_question[0].upper() + nl_question[1:]
            
            results.append({"question": nl_question, "query": sql_query})
        
        return results
    
    def _get_entity_type(self, table: str) -> str:
        """Get a descriptive entity name for a table"""
        entity_mappings = {
            'patients': 'patients',
            'admissions': 'hospital admissions',
            'diagnoses_icd': 'patient diagnoses',
            'procedures_icd': 'patient procedures',
            'prescriptions': 'medications',
            'labevents': 'lab results',
            'chartevents': 'chart entries',
            'icustays': 'ICU stays',
            'transfers': 'patient transfers',
            'services': 'hospital services',
            'microbiologyevents': 'microbiology results',
            'outputevents': 'patient outputs',
            'inputevents': 'patient inputs',
            'emar': 'medication administration records',
            'pharmacy': 'pharmacy orders',
            'poe': 'provider orders',
            'd_icd_diagnoses': 'diagnosis codes',
            'd_icd_procedures': 'procedure codes',
            'd_labitems': 'laboratory tests',
            'd_items': 'charted items'
        }
        
        return entity_mappings.get(table, table.replace('_', ' '))