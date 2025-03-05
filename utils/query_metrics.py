import time
import sqlparse
import pandas as pd
from Levenshtein import distance as levenshtein_distance
from sqlalchemy import text

class QueryMetrics:
    def __init__(self, db_handler):
        self.db_handler = db_handler

    @staticmethod
    def query_similarity(expected_sql, generated_sql):
        """Computes similarity between expected and generated SQL using Levenshtein distance."""
        return 1 - (levenshtein_distance(expected_sql, generated_sql) / max(len(expected_sql), len(generated_sql)))

    def compare_query_outputs(self, expected_sql, generated_sql):
        """
        Executes both expected and generated SQL queries and compares their outputs.
        Returns True if results match, else False.
        """
        try:
            expected_result = self.db_handler.execute_and_fetch(expected_sql)
            generated_result = self.db_handler.execute_and_fetch(generated_sql)

            # Convert to Pandas DataFrame for structured comparison
            df_expected = pd.DataFrame(expected_result)
            df_generated = pd.DataFrame(generated_result)

            return df_expected.equals(df_generated)  # True if both outputs are the same
        except Exception as e:
            return str(e)  # Return error message for debugging

    @staticmethod
    def canonicalize_sql(sql_query):
        """
        Converts SQL query to a canonical form using sqlparse.
        - Removes extra whitespace, formatting differences, and normalizes
