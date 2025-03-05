import pymysql
import pandas as pd
from sqlalchemy import create_engine

class DBHandler:
    def __init__(self):
        self.engine = create_engine("mysql+pymysql://username:password@localhost/mimic_iv")

    def execute_query(self, sql_query):
        """Executes SQL and returns the number of rows fetched or an error message."""
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(sql_query, conn)
            return df.shape[0]  # Number of rows returned
        except Exception as e:
            return str(e)  # Capture SQL errors

    def execute_and_fetch(self, sql_query):
        """Executes SQL and returns the actual data as a list of tuples."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql_query))
                return result.fetchall()  # Return query result
        except Exception as e:
            return str(e)
