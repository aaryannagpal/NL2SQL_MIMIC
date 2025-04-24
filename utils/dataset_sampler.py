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
