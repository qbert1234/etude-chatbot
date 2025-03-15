# 에뛰드 챗봇 프로젝트 요약

## 프로젝트 구조
/Users/taeseokkim/LangGraph_Chat_0316/
├── .env (환경 변수)
├── app/
│   ├── services/ (BigQuery, Vertex AI, LangGraph 서비스)
│   ├── templates/ (웹 인터페이스)
│   └── main.py (FastAPI 앱)

## 구현된 기능
- BigQuery 연결 및 에뛰드 제품 데이터 쿼리
- Vertex AI Gemini 모델 연결
- LangGraph 기반 챗봇 로직
- 스트리밍 응답 웹 인터페이스

## 환경 설정
- Python 가상환경: venv
- 필요 패키지: langchain, langgraph, fastapi, google-cloud-bigquery, google-cloud-aiplatform
- GCP 서비스 계정 키: glass-proxy-434003-n2-dde922362627.json