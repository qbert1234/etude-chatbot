from google.cloud import bigquery
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

class BigQueryService:
    def __init__(self):
        self.client = bigquery.Client()
        self.dataset_id = os.environ.get("BIGQUERY_DATASET")
        self.table_id = os.environ.get("BIGQUERY_TABLE")
        self.project_id = os.environ.get("GCP_PROJECT_ID")
        self.full_table_id = f"{self.project_id}.{self.dataset_id}.{self.table_id}"
    
    def query_data(self, query):
        try:
            query_job = self.client.query(query)
            results = query_job.result()
            return [dict(row) for row in results]
        except Exception as e:
            print(f"Error querying BigQuery: {e}")
            return []
    
    def test_connection(self):
        """BigQuery 연결 테스트"""
        try:
            query = f"SELECT * FROM `{self.full_table_id}` LIMIT 1"
            result = self.query_data(query)
            return True, result
        except Exception as e:
            return False, str(e)