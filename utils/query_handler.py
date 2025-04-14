import sqlite3
import time
import json
import os
import sys
import re
# Add the parent directory to the path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MYSQL_DB_PATH
import pandas as pd

class QueryHandler:

    def __init__(self):
        self.db_path = MYSQL_DB_PATH
        self.connection = None
        self.connect()

    def update_path(self, new_path):
        """Update path to SQLite DB file"""
        self.db_path = new_path
        if self.connection:
            self.connection.close()
        self.connect()

    def connect(self):
        """Connect to SQLite database"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row  # Return rows as dict-like
        except sqlite3.Error as e:
            print(f"Failed to connect to database: {e}")
            self.connection = None

    def execute(self, sql_query):
        """
        Execute SQL query and return metadata + results.

        Returns:
        - results: list of rows (as dicts)
        - execution_time
        - row_count
        - success flag
        - error message (if any)
        - query plan (if supported)
        """
        # cleaned_query = self.query_cleanup(sql_query)
        response = {
            "query": sql_query,
            # "cleaned_query": cleaned_query,
            "success": False,
            "execution_time": 0,
            "row_count": 0,
            "results": None,
            "execution_plan": None,
            "error": None
        }


        try:
            start_time = time.time()
            cursor = self.connection.cursor()

            # Main query
            cursor.execute(sql_query)
            if cursor.description:
                rows = cursor.fetchall()
                response["results"] = [dict(row) for row in rows]
                response["row_count"] = len(rows)
            else:
                response["row_count"] = cursor.rowcount

            # SQLite-specific EXPLAIN (simple version)
            explain_query = f"EXPLAIN QUERY PLAN {sql_query}"
            try:
                cursor.execute(explain_query)
                plan = cursor.fetchall()
                response["execution_plan"] = [dict(row) for row in plan]
            except:
                response["execution_plan"] = "Query plan not available"

            response["execution_time"] = time.time() - start_time
            response["success"] = True

        except Exception as e:
            response["error"] = str(e)

        return response

    def pretty_execute(self, query, keep_index=False):
        """
        Execute SQL query and return as DataFrame.
        """
        try:
            df = pd.read_sql_query(query, self.connection)
            if 'index' in df.columns and not keep_index:
                df.drop(columns=['index'], inplace=True)
            return df
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None

    def __del__(self):
        if hasattr(self, 'connection') and self.connection:
            self.connection.close()