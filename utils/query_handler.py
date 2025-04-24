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

CURRENT_DATE = "2100-12-31"
CURRENT_TIME = "23:59:00"
NOW = f"{CURRENT_DATE} {CURRENT_TIME}"
PRECOMPUTED_DICT = {
    "temperature": (35.5, 38.1),
    "sao2": (95.0, 100.0),
    "heart rate": (60.0, 100.0),
    "respiration": (12.0, 18.0),
    "systolic bp": (90.0, 120.0),
    "diastolic bp": (60.0, 90.0),
    "mean bp": (60.0, 110.0),
}
TIME_PATTERN = (
    r"(DATE_SUB|DATE_ADD)\((\w+\(\)|'[^']+')[, ]+ INTERVAL (\d+) (MONTH|YEAR|DAY)\)"
)


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
        cleaned_query = self.query_cleanup(sql_query)
        response = {
            "query": sql_query,
            "cleaned_query": cleaned_query,
            "success": False,
            "execution_time": 0,
            "row_count": 0,
            "results": None,
            "execution_plan": None,
            "error": None,
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
        query = self.query_cleanup(sql_query)
        try:
            df = pd.read_sql_query(query, self.connection)
            if "index" in df.columns and not keep_index:
                df.drop(columns=["index"], inplace=True)
            return df
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None

    def convert_date_function(self, match):
        function = match.group(1)
        date = match.group(2)
        number = match.group(3)
        unit = match.group(4).lower()

        # Use singular form when number is 1
        if number == "1":
            unit = unit.rstrip("s")
        else:
            unit += "s" if not unit.endswith("s") else ""

        # Determine the sign based on the function (DATE_SUB or DATE_ADD)
        sign = "-" if function == "DATE_SUB" else "+"

        return f"datetime({date}, '{sign}{number} {unit}')"

    def query_cleanup(self, query):

        query = re.sub("[ ]+", " ", query.replace("\n", " ")).strip()
        query = query.replace("> =", ">=").replace("< =", "<=").replace("! =", "!=")

        query = re.sub(TIME_PATTERN, self.convert_date_function, query)

        if (
            "current_time" in query
        ):  # strftime('%J',current_time) => strftime('%J','2100-12-31 23:59:00')
            query = query.replace("current_time", f"'{NOW}'")
        if (
            "current_date" in query
        ):  # strftime('%J',current_date) => strftime('%J','2100-12-31')
            query = query.replace("current_date", f"'{CURRENT_DATE}'")
        if "'now'" in query:  # 'now' => '2100-12-31 23:59:00'
            query = query.replace("'now'", f"'{NOW}'")
        if "NOW()" in query:  # NOW() => '2100-12-31 23:59:00'
            query = query.replace("NOW()", f"'{NOW}'")
        if "CURDATE()" in query:  # CURDATE() => '2100-12-31'
            query = query.replace("CURDATE()", f"'{CURRENT_DATE}'")
        if "CURTIME()" in query:  # CURTIME() => '23:59:00'
            query = query.replace("CURTIME()", f"'{CURRENT_TIME}'")

        if re.search("[ \n]+([a-zA-Z0-9_]+_lower)", query) and re.search(
            "[ \n]+([a-zA-Z0-9_]+_upper)", query
        ):
            vital_lower_expr = re.findall("[ \n]+([a-zA-Z0-9_]+_lower)", query)[0]
            vital_upper_expr = re.findall("[ \n]+([a-zA-Z0-9_]+_upper)", query)[0]
            vital_name_list = list(
                set(
                    re.findall("([a-zA-Z0-9_]+)_lower", vital_lower_expr)
                    + re.findall("([a-zA-Z0-9_]+)_upper", vital_upper_expr)
                )
            )
            if len(vital_name_list) == 1:
                processed_vital_name = vital_name_list[0].replace("_", " ")
                if processed_vital_name in PRECOMPUTED_DICT:
                    vital_range = PRECOMPUTED_DICT[processed_vital_name]
                    query = query.replace(
                        vital_lower_expr, f"{vital_range[0]}"
                    ).replace(vital_upper_expr, f"{vital_range[1]}")

        query = query.replace("%y", "%Y").replace("%j", "%J")

        return query

    def __del__(self):
        if hasattr(self, "connection") and self.connection:
            self.connection.close()
