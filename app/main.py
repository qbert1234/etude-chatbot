from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os
from typing import Dict, Any, List
from pydantic import BaseModel

# 환경 변수 로드
load_dotenv()

app = FastAPI(title="Etude Chatbot")

# 정적 파일 및 템플릿 설정
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# 채팅 기록 저장 (실제 구현에서는 DB 사용 권장)
chat_sessions = {}

# API 모델
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str

# 간단한 홈페이지 라우트
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# BigQuery 연결 테스트 라우트
@app.get("/test/bigquery")
async def test_bigquery():
    try:
        from app.services.bigquery_service import BigQueryService
        bq_service = BigQueryService()
        success, result = bq_service.test_connection()
        
        if success:
            return JSONResponse(
                status_code=200,
                content={"status": "success", "message": "BigQuery 연결 성공", "data": result}
            )
        else:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": f"BigQuery 연결 실패: {result}"}
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"오류 발생: {str(e)}"}
        )

# Vertex AI 연결 테스트 라우트
@app.get("/test/vertex")
async def test_vertex():
    try:
        from app.services.vertex_service import VertexAIService
        vertex_service = VertexAIService()
        success, result = vertex_service.test_connection()
        
        if success:
            return JSONResponse(
                status_code=200,
                content={"status": "success", "message": result}
            )
        else:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": f"Vertex AI 연결 실패: {result}"}
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"오류 발생: {str(e)}"}
        )

# 일반 챗봇 엔드포인트
@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    try:
        session_id = chat_request.session_id
        
        # 세션 기록 가져오기
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        
        chat_history = chat_sessions[session_id]
        
        # LangGraph 서비스 초기화 및 챗봇 실행
        from app.services.langgraph_service import LangGraphService
        lg_service = LangGraphService()
        response, updated_history = lg_service.run_chatbot(chat_request.message, chat_history)
        
        # 세션 업데이트
        chat_sessions[session_id] = updated_history
        
        return ChatResponse(response=response, session_id=session_id)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"채팅 처리 중 오류 발생: {str(e)}"}
        )

# 스트리밍 챗봇 엔드포인트
@app.get("/chat/stream")
async def chat_stream(message: str, session_id: str = "default"):
    try:
        # 세션 기록 가져오기
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        
        chat_history = chat_sessions[session_id]
        
        # LangGraph 서비스 초기화
        from app.services.langgraph_service import LangGraphService
        lg_service = LangGraphService()
        
        # 스트리밍 응답 생성
        async def generate():
            for text_chunk in lg_service.run_chatbot_stream(message, chat_history):
                yield f"data: {text_chunk}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"status": "error", "message": f"스트리밍 채팅 처리 중 오류 발생: {str(e)}"}
        )

# 서버 실행 코드
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)