import pymysql
import time
import json

class QueryHandler:
    def __init__(self, db_config):
        """Initialize with database configuration"""
        self.db_config = db_config
        self.connection = pymysql.connect(
            **db_config,
        )

    def execute(self, sql_query):
        """
        Execute query and return JSON with:
        - Results
        - Execution time
        - Row count
        - Execution plan
        - Success status
        """
        response = {
            "query": sql_query,
            "success": False,
            "execution_time": 0,
            "row_count": 0,
            "results": None,
            "execution_plan": None,
            "error": None
        }

        try:
            start_time = time.time()
            
            with self.connection.cursor() as cursor:
                # Execute main query
                cursor.execute(sql_query)
                
                # Get results if available
                if cursor.description:
                    response["results"] = cursor.fetchall()
                    response["row_count"] = len(response["results"])
                else:
                    response["row_count"] = cursor.rowcount
                
                # Get execution plan
                cursor.execute(f"EXPLAIN FORMAT=JSON {sql_query}")
                plan = cursor.fetchone()
                response["execution_plan"] = json.loads(plan['EXPLAIN'])
                
                response["execution_time"] = time.time() - start_time
                response["success"] = True

        except Exception as e:
            response["error"] = str(e)
            
        return response

    def __del__(self):
        """Clean up connection"""
        if hasattr(self, 'connection') and self.connection:
            self.connection.close()


if __name__ == "__main__":
    config = {
        'host': 'localhost',
        'user': 'aaryan',
        'password': os.environ['MIMIC_SQL_PW'],
        'db': 'capstone_mimic',
        'charset': 'utf8mb4',
        'cursorclass': pymysql.cursors.DictCursor
    }

    handler = QueryHandler(config)
    
    # Test query
    result = handler.execute("SELECT * FROM patients WHERE gender = 'F' LIMIT 5")
    print(json.dumps(result, indent=2))