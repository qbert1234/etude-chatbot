from google.cloud import aiplatform
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

class VertexAIService:
    def __init__(self):
        self.project_id = os.environ.get("GCP_PROJECT_ID")
        self.location = os.environ.get("GCP_LOCATION")
        # Vertex AI 초기화
        aiplatform.init(project=self.project_id, location=self.location)
    
    def test_connection(self):
        """Vertex AI 연결 테스트"""
        try:
            # 단순한 모델 목록 가져오기 테스트
            models = aiplatform.Model.list(filter="display_name=gemini")
            return True, f"연결 성공. {len(models)} 모델 찾음."
        except Exception as e:
            return False, str(e)