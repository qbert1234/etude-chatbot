from typing import TypedDict, List, Dict, Any, Generator
from langgraph.graph import StateGraph
import os
from dotenv import load_dotenv
from app.services.bigquery_service import BigQueryService
from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import ChatPromptTemplate

# 환경 변수 로드
load_dotenv()

# 상태 정의
class ChatState(TypedDict):
    messages: List[Dict[str, Any]]
    user_query: str
    context: List[Dict[str, Any]]
    response: str
    stream: bool

# LangGraph 서비스 클래스
class LangGraphService:
    def __init__(self):
        self.bq_service = BigQueryService()
        self.graph_instance = self._build_graph()
    
    # 노드 함수: 컨텍스트 검색
    def _get_context(self, state: ChatState) -> ChatState:
        """BigQuery에서 관련 정보를 검색하여 컨텍스트로 추가합니다."""
        query = state["user_query"]
        
        # 제품 정보나 관련 데이터 조회
        results = self.bq_service.query_data(f"""
        SELECT p_id, media_short_desc, media_long_desc, media_focus, media_pic_url_300
        FROM `{self.bq_service.full_table_id}`
        WHERE 
            LOWER(media_short_desc) LIKE LOWER('%{query}%') OR
            LOWER(media_long_desc) LIKE LOWER('%{query}%')
        LIMIT 5
        """)
        
        # 컨텍스트가 비어있으면 일반 제품 정보 가져오기
        if not results:
            results = self.bq_service.query_data(f"""
            SELECT p_id, media_short_desc, media_long_desc, media_focus, media_pic_url_300
            FROM `{self.bq_service.full_table_id}`
            LIMIT 5
            """)
        
        state["context"] = results
        return state
    
    # 노드 함수: 응답 생성
    def _generate_response(self, state: ChatState) -> ChatState:
        """Vertex AI Gemini를 사용하여 응답을 생성합니다."""
        # LangChain용 Vertex AI 챗봇 모델 초기화
        llm = ChatVertexAI(
            model_name="gemini-pro",
            temperature=0.2, 
            max_output_tokens=1024,
            top_p=0.8,
            top_k=40,
            project=os.environ.get("GCP_PROJECT_ID"),
            location=os.environ.get("GCP_LOCATION"),
            streaming=False  # 여기서는 스트리밍을 비활성화 (run_chatbot_stream에서 별도로 활성화)
        )
        
        # 컨텍스트 및 프롬프트 설정
        context_str = "\n\n".join([
            f"제품ID: {item.get('p_id', '')}\n"
            f"제품설명: {item.get('media_short_desc', '')}\n"
            f"상세설명: {item.get('media_long_desc', '')}\n"
            f"특징: {item.get('media_focus', '')}"
            for item in state["context"]
        ])
        
        system_template = """
        당신은 에뛰드 화장품 브랜드의 AI 어시스턴트입니다. 
        고객의 질문에 친절하고 전문적으로 답변해주세요.
        주어진 제품 정보를 바탕으로 정확한 정보를 제공하되, 
        없는 정보는 만들어내지 마세요.
        답변은 항상 한국어로 제공하세요.
        """
        
        human_template = """
        다음은 에뛰드 제품 관련 정보입니다:
        {context}
        
        고객 질문: {query}
        """
        
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])
        
        # 메시지 생성
        messages = chat_prompt.format_messages(
            context=context_str,
            query=state["user_query"]
        )
        
        # 응답 생성
        response = llm.invoke(messages)
        response_text = response.content
        
        # 상태 업데이트
        state["response"] = response_text
        
        # 메시지 히스토리에 추가
        state["messages"].append({"role": "user", "content": state["user_query"]})
        state["messages"].append({"role": "assistant", "content": response_text})
        
        return state
    
    # 그래프 구축
    def _build_graph(self):
        # 그래프 정의
        workflow = StateGraph(ChatState)
        
        # 노드 추가
        workflow.add_node("get_context", self._get_context)
        workflow.add_node("generate_response", self._generate_response)
        
        # 엣지 추가
        workflow.add_edge("get_context", "generate_response")
        workflow.set_entry_point("get_context")
        
        # 컴파일
        return workflow.compile()
    
    # 챗봇 실행 함수
    def run_chatbot(self, query: str, chat_history: List[Dict[str, Any]] = None):
        if chat_history is None:
            chat_history = []
        
        # 초기 상태 설정
        state = {
            "messages": chat_history,
            "user_query": query,
            "context": [],
            "response": "",
            "stream": False
        }
        
        # 그래프 실행
        result = self.graph_instance.invoke(state)
        return result["response"], result["messages"]
    
    # 스트리밍 챗봇 함수
    def run_chatbot_stream(self, query: str, chat_history: List[Dict[str, Any]] = None) -> Generator[str, None, None]:
        if chat_history is None:
            chat_history = []
        
        # 먼저 컨텍스트 검색
        state = {
            "messages": chat_history,
            "user_query": query,
            "context": [],
            "response": "",
            "stream": True
        }
        
        # 컨텍스트 검색
        state = self._get_context(state)
        
        # 컨텍스트 및 프롬프트 설정
        context_str = "\n\n".join([
            f"제품ID: {item.get('p_id', '')}\n"
            f"제품설명: {item.get('media_short_desc', '')}\n"
            f"상세설명: {item.get('media_long_desc', '')}\n"
            f"특징: {item.get('media_focus', '')}"
            for item in state["context"]
        ])
        
        system_template = """
        당신은 에뛰드 화장품 브랜드의 AI 어시스턴트입니다. 
        고객의 질문에 친절하고 전문적으로 답변해주세요.
        주어진 제품 정보를 바탕으로 정확한 정보를 제공하되, 
        없는 정보는 만들어내지 마세요.
        답변은 항상 한국어로 제공하세요.
        """
        
        human_template = """
        다음은 에뛰드 제품 관련 정보입니다:
        {context}
        
        고객 질문: {query}
        """
        
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])
        
        # 메시지 생성
        messages = chat_prompt.format_messages(
            context=context_str,
            query=query
        )
        
        # 스트리밍용 LLM 설정
        llm = ChatVertexAI(
            model_name="gemini-pro",
            temperature=0.2, 
            max_output_tokens=1024,
            top_p=0.8,
            top_k=40,
            project=os.environ.get("GCP_PROJECT_ID"),
            location=os.environ.get("GCP_LOCATION"),
            streaming=True
        )
        
        # 스트리밍 응답 생성
        full_response = ""
        for chunk in llm.stream(messages):
            chunk_text = chunk.content
            full_response += chunk_text
            yield chunk_text
        
        # 채팅 히스토리 업데이트 (스트림이 완료된 후)
        if chat_history is not None:
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": full_response})