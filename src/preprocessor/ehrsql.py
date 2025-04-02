import pandas as pd
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..query_handler.query_handler import QueryHandler
from tqdm import tqdm

class ResultCollector:
    def __init__(self, sql_column, result_column, max_workers=5):
        self.sql_column = sql_column
        self.result_column = result_column
        self.max_workers = max_workers

    def _run_query(self, sql_query):
        handler = QueryHandler()
        try:
            result = handler.execute(sql_query)
            return result.get("results", None)
        except Exception as e:
            return f"Error: {str(e)}"

    def _run_on_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        results = [None] * len(df)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self._run_query, row[self.sql_column]): idx
                for idx, row in df.iterrows()
            }

            for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Running SQL queries"):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = f"Unhandled error: {str(e)}"

        df[self.result_column] = results
        return df

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the DataFrame by running SQL queries in parallel.

        Args:
            df (pd.DataFrame): Input DataFrame containing SQL queries.

        Returns:
            pd.DataFrame: DataFrame with results of SQL queries.
        """
        df = df.replace("null", np.nan)
        df_null = df[df[self.sql_column].isna()]
        df_not_null = df[~df[self.sql_column].isna()]
        
        df_not_null = df_not_null.reset_index(drop=True)
        df_null = df_null.reset_index(drop=True)

        exec_df = self._run_on_dataframe(df_not_null)
        df_null[self.result_column] = np.nan

        df = pd.concat([df_null, exec_df], ignore_index=True)
        return df



class DataProcessor:
    """
    Base class for data preprocessing. This class provides methods to load, save, and transform data.
    """

    def __init__(self, dir_path: str):
        self.train = None
        self.test = None
        self.validation = None
        self.path = dir_path

    def load_data(self) -> pd.DataFrame:
        """
        Load data into a pandas DataFrame.

        Returns:
            pd.DataFrame: Loaded data as a pandas DataFrame.
        """
        self.train = self._load_split("train")
        self.test = self._load_split("test")
        self.validation = self._load_split("valid")

    def _load_split(self, split: str) -> pd.DataFrame:
        """
        Load a specific data split (train, test, validation) from the directory.

        Args:
            split (str): The data split to load ('train', 'test', 'validation').

        Returns:
            pd.DataFrame: Loaded data as a pandas DataFrame.
        """
        file_path = f"{self.path}/{split}"
        label_json = json.load(open(f"{file_path}/label.json"))
        data_json = json.load(open(f"{file_path}/data.json"))

        label_df = pd.DataFrame.from_dict(label_json, orient='index')
        label_df.reset_index(inplace=True)
        label_df.columns = ['id', 'true_query']

        data_df = pd.DataFrame(data_json["data"])

        data = pd.merge(data_df, label_df, on='id')

        return data