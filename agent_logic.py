"""
Agent Logic - 노트북의 Cell 4와 Cell 5 로직을 그대로 유지
"""

from config import llm, emb, VECTORSTORE_DIR, LOG_DIR
from datetime import datetime

import os
import fitz  # PyMuPDF
import json
import re
from typing import Annotated, List, Literal, Dict, Any, Set, Optional
from typing_extensions import TypedDict
from copy import deepcopy

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, Send
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# =========================
# 로깅 유틸리티
# =========================
def log_and_print(message: str, log_file: str = None):
    """
    메시지를 콘솔에 출력하고 동시에 로그 파일에 저장
    
    Args:
        message: 출력할 메시지
        log_file: 로그 파일 경로 (None이면 로그 파일에 저장 안함)
    """
    # 콘솔에 출력
    print(message)
    
    # 로그 파일에 저장
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')


def create_log_file(patent_id: str, log_type: str = "preprocessing") -> str:
    """
    특허별 로그 파일 경로 생성
    
    Args:
        patent_id: 특허 ID (예: US8526476)
        log_type: 로그 타입 (preprocessing, chunking 등)
    
    Returns:
        로그 파일 전체 경로
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{patent_id}_{log_type}_{timestamp}.log"
    return os.path.join(LOG_DIR, log_filename)


# =========================
# Vector Store 저장/로드 함수
# =========================
def get_vectorstore_path(patent_id: str) -> str:
    """
    특허 ID에 해당하는 vectorstore 저장 경로 반환
    
    Args:
        patent_id: 특허 ID (예: US8526476)
    
    Returns:
        vectorstore 디렉토리 경로
    """
    return os.path.join(VECTORSTORE_DIR, patent_id)


def vectorstore_exists(patent_id: str) -> bool:
    """
    해당 특허의 vectorstore가 이미 존재하는지 확인
    
    Args:
        patent_id: 특허 ID
    
    Returns:
        존재 여부 (True/False)
    """
    path = get_vectorstore_path(patent_id)
    # FAISS는 index.faiss와 index.pkl 파일이 있어야 함
    return os.path.exists(path) and os.path.exists(os.path.join(path, "index.faiss"))


def save_vectorstore(vectorstore: FAISS, patent_id: str) -> str:
    """
    Vectorstore를 디스크에 저장
    
    Args:
        vectorstore: 저장할 FAISS vectorstore
        patent_id: 특허 ID
    
    Returns:
        저장된 경로
    """
    path = get_vectorstore_path(patent_id)
    os.makedirs(path, exist_ok=True)
    vectorstore.save_local(path)
    print(f"✅ Vector store saved to: {path}")
    return path


def load_vectorstore(patent_id: str, embeddings) -> FAISS:
    """
    디스크에서 Vectorstore 로드
    
    Args:
        patent_id: 특허 ID
        embeddings: Embedding 객체
    
    Returns:
        로드된 FAISS vectorstore
    """
    path = get_vectorstore_path(patent_id)
    if not vectorstore_exists(patent_id):
        raise FileNotFoundError(f"Vectorstore not found for patent {patent_id} at {path}")
    
    vectorstore = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True  # FAISS 로드시 필요
    )
    print(f"✅ Vector store loaded from: {path}")
    return vectorstore


# =========================
# 간단한 RAG 챗봇 함수 (Tool 기반)
# =========================
def simple_rag_chatbot(query: str, patent_id: str, chat_history: List[Dict] = None) -> str:
    """
    Tool 기반 RAG 챗봇 - 특허 문서에 대한 Q&A
    
    3가지 검색 도구를 사용하여 특허 문서를 더 정확하게 검색:
    - get_available_metadata: 사용 가능한 메타데이터 확인
    - search_by_metadata: 특정 섹션/청구항 검색
    - search_by_similarity: 의미 기반 검색
    
    Args:
        query: 사용자 질문
        patent_id: 특허 ID
        chat_history: 이전 대화 히스토리 (선택사항)
    
    Returns:
        챗봇 응답
    """
    try:
        global vectorstore, current_patent_id
        
        # Vectorstore 로드 및 global 변수 설정
        vectorstore = load_vectorstore(patent_id, emb)
        current_patent_id = patent_id
        
        # 기존 툴들을 그대로 사용
        tools = [get_available_metadata, search_by_metadata, search_by_similarity]
        
        # 대화 히스토리를 메시지 형식으로 구성
        history_messages = []
        if chat_history:
            recent_history = chat_history[-6:]  # 최근 6개만 (3턴)
            for msg in recent_history:
                if msg['role'] == 'user':
                    history_messages.append(HumanMessage(content=msg['content']))
                else:
                    history_messages.append(AIMessage(content=msg['content']))
        
        # 강화된 System 프롬프트
        system_prompt = f"""You are an expert patent analysis assistant with access to specialized search tools for Patent {patent_id}.

**YOUR MISSION:** Answer user questions accurately by intelligently using the provided tools.

**AVAILABLE TOOLS (Use them!):**

1. **get_available_metadata** - Check patent structure FIRST
   - Use when: User asks "어떤 섹션이 있어?", "claim이 몇 개야?", "구조가 어떻게 돼?"
   - Returns: Available sections, claim numbers, metadata fields
   - Example: get_available_metadata(metadata_keys=["section", "claim_no"])

2. **search_by_metadata** - Get specific sections/claims (STRUCTURAL search)
   - Use when: User wants specific claim numbers or entire sections
   - Examples:
     * "Claim 1 보여줘" → search_by_metadata(query="", filters="metadata['claim_no'] == 1")
     * "ABSTRACT 섹션 내용" → search_by_metadata(query="", filters="metadata['section'] == 'ABSTRACT'")
     * "모든 독립항" → search_by_metadata(query="", filters="metadata['independent'] == True")
   - ⚠️ Always use empty query "" for structural searches!

3. **search_by_similarity** - Semantic/concept search (MEANING search)
   - Use when: User asks conceptual questions
   - Examples:
     * "혁신 포인트가 뭐야?" → search_by_similarity(query="innovation points advantages", k=10)
     * "기술적 원리 설명해줘" → search_by_similarity(query="technical principle mechanism how it works", k=10)
     * "어떤 문제를 해결하나?" → search_by_similarity(query="problem solved background issues", k=10)
   - Use descriptive English keywords for better search results

**CRITICAL DECISION TREE:**

Question type → Tool to use:
- "Claim X 보여줘" / "청구항 X" → search_by_metadata
- "ABSTRACT 섹션" / "특정 섹션" → search_by_metadata  
- "혁신", "장점", "원리", "방법", "특징" → search_by_similarity
- "어떤 섹션?", "몇 개 claim?" → get_available_metadata
- Complex questions → Use multiple tools sequentially

**ENHANCED GUIDELINES:**

1. **ALWAYS use tools** - Don't try to answer without searching!
2. **For claim questions:**
   - First: get_available_metadata(metadata_keys=["claim_no"]) to see available claims
   - Then: search_by_metadata with appropriate filter
3. **For concept questions:**
   - Use search_by_similarity with descriptive keywords
   - Search in English for better results (e.g., "innovation advantages benefits")
4. **Answer in Korean** - User's language
5. **Be specific** - Cite claim numbers, section names
6. **If not found** - Say "해당 내용이 문서에서 발견되지 않았습니다"
7. **Use multiple tools** - For complex questions, search multiple times

**EXAMPLE WORKFLOW:**

User: "이 특허의 혁신 포인트를 알려줘"
→ Action: search_by_similarity(query="innovation points novel features advantages benefits", k=10)
→ Then synthesize and answer in Korean

User: "Claim 1 내용 알려줘"  
→ Action: search_by_metadata(query="", filters="metadata['claim_no'] == 1")
→ Then present the claim content

**Remember:** Search FIRST, then answer based on results!"""

        # ReAct Agent 생성
        agent = create_react_agent(llm, tools)
        
        # 메시지 구성 (히스토리 포함)
        messages = [SystemMessage(content=system_prompt)] + history_messages + [HumanMessage(content=query)]
        
        # Agent 실행
        state = {"messages": messages}
        result = agent.invoke(state)
        
        # 최종 답변 추출 (더 강력한 추출 로직)
        if result and "messages" in result:
            # 마지막 AI 메시지 찾기
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
                    # Tool 호출 메시지가 아닌 실제 답변만 반환
                    # tool_calls가 없는 일반 텍스트 응답만 반환
                    if not hasattr(msg, 'tool_calls') or not msg.tool_calls:
                        return msg.content
            
            # fallback: 마지막 메시지
            last_message = result["messages"][-1]
            if hasattr(last_message, 'content'):
                return last_message.content
            else:
                return str(last_message)
        else:
            return "죄송합니다. 답변을 생성하는 중 오류가 발생했습니다."
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ Chatbot Error: {error_detail}")
        return f"❌ 오류가 발생했습니다: {str(e)}\n\n기본 검색으로 답변을 시도합니다..."


# =========================
# 0) Helper Functions
# =========================

def load_pages_with_first_page_columns(pdf_path: str):
    """
    0번 페이지만 PyMuPDF로 좌/우 칼럼을 분리해 읽고,
    1페이지 이후는 PyPDFLoader로 읽어 Document 리스트를 반환.
    """
    # 1) 우선 전체 페이지 메타/페이지수 파악을 위해 PyPDFLoader
    loader_pages = PyPDFLoader(pdf_path).load()  # page별 Document
    assert len(loader_pages) >= 1, "PDF에 페이지가 없습니다."

    # 2) 첫 페이지만 PyMuPDF로 2-칼럼 추출
    with fitz.open(pdf_path) as doc:
        p0 = doc[0]
        rect = p0.rect
        mid_x = rect.x0 + rect.width / 2.0

        left_rect  = fitz.Rect(rect.x0, rect.y0, mid_x,  rect.y1)
        right_rect = fitz.Rect(mid_x,  rect.y0, rect.x1, rect.y1)

        left_text  = p0.get_text("text", clip=left_rect) or ""
        right_text = p0.get_text("text", clip=right_rect) or ""

        # 칼럼 순서: 보통 좌 -> 우가 자연스러운 읽기 순서
        first_text = (left_text.strip() + "\n" + right_text.strip()).strip()

    # 3) 첫 페이지 Document 재구성 (metadata 보존 + 보강)
    p0_meta = dict(loader_pages[0].metadata or {})
    p0_meta.update({
        "page": 0,
        "source": pdf_path,
        "column_split": "left|right"  # 후처리/디버깅용 마커
    })
    first_doc = Document(page_content=first_text, metadata=p0_meta)

    # 4) 나머지 페이지는 기존 PyPDFLoader 결과(메타 유지) 활용
    rest_docs = []
    for d in loader_pages[1:]:
        m = dict(d.metadata or {})
        m.setdefault("source", pdf_path)
        rest_docs.append(Document(page_content=d.page_content, metadata=m))

    return [first_doc] + rest_docs


def to_langchain_document(patent_data: Dict[str, Any], source: str = "", log_file: str = None) -> List[Document]:
    """
    특허 JSON 데이터를 LangChain Document 리스트로 변환
    - ABSTRACT: 1개 문서 (분할 안함)
    - 각 섹션 (CLAIMS 제외): 청크로 분할
    - CLAIMS: 각 청구항을 개별 문서로
    
    Args:
        patent_data: 특허 JSON 데이터 (metadata, sections, claims 포함)
        source: 원본 파일 경로
        log_file: 로그 파일 경로 (선택사항)
        
    Returns:
        List[Document]: 처리된 문서 리스트
    """
    metadata = patent_data.get("metadata", {})
    if source:
        metadata["source"] = source
    
    sections = patent_data.get("sections", {})
    claims = patent_data.get("claims", [])
    
    docs = []
    base_meta = deepcopy(metadata)
    
    # 청크 스플리터 설정
    desc_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=200, 
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # 1) ABSTRACT 처리 (분할하지 않음)
    abs_txt = sections.get("ABSTRACT", "") or ""
    if abs_txt.strip():
        meta = deepcopy(base_meta)
        meta.update({"section": "ABSTRACT", "granularity": "full"})
        docs.append(Document(page_content=abs_txt.strip(), metadata=meta))
    
    # 2) CLAIMS 외의 모든 섹션 처리 (desc_splitter 사용)
    for sec_name, txt in sections.items():
        # ABSTRACT와 CLAIMS는 별도 처리
        if sec_name in ["ABSTRACT", "CLAIMS"]:
            continue
        
        if not txt.strip():
            continue
        
        # desc_splitter로 청크 분할
        chunks = desc_splitter.create_documents([txt])
        
        for i, ch in enumerate(chunks):
            ch.metadata.update(deepcopy(base_meta))
            ch.metadata.update({
                "section": sec_name,
                "granularity": "chunk",
                "chunk_id": f"{sec_name}:{i}"
            })
            docs.append(ch)
    
    # 3) CLAIMS 처리 (각 청구항을 개별 문서로)
    if claims:
        for claim_info in claims:
            claim_no = claim_info.get('claim_no')
            claim_text = claim_info.get('claim_text', '').strip()
            is_independent = claim_info.get('independent', False)
            
            if not claim_text:
                continue
            
            meta = deepcopy(base_meta)
            meta.update({
                "section": "CLAIMS",
                "granularity": "claim",
                "claim_no": claim_no,
                "independent": is_independent
            })
            docs.append(Document(page_content=claim_text, metadata=meta))
    

    # ============================================================
    # 디버깅: 청킹 결과 출력 (print + log)
    # ============================================================
    log_and_print("\n" + "="*80, log_file)
    log_and_print("🔪 청킹 결과 - 모든 청크 상세 내용", log_file)
    log_and_print("="*80 + "\n", log_file)
    log_and_print(f"총 청크 개수: {len(docs)}개\n", log_file)
    
    for idx, doc in enumerate(docs, 1):
        section = doc.metadata.get('section', 'Unknown')
        granularity = doc.metadata.get('granularity', 'Unknown')
        chunk_length = len(doc.page_content)
        
        log_and_print("="*80, log_file)
        log_and_print(f"[청크 {idx}/{len(docs)}]", log_file)
        log_and_print("="*80, log_file)
        log_and_print(f"섹션: {section}", log_file)
        log_and_print(f"세분화 수준: {granularity}", log_file)
        log_and_print(f"길이: {chunk_length} 문자", log_file)
        log_and_print(f"\n전체 내용:", log_file)
        log_and_print(doc.page_content, log_file)
        log_and_print("\n" + "="*80 + "\n", log_file)
    
    log_and_print("\n" + "="*80, log_file)
    log_and_print("📊 섹션별 청크 통계", log_file)
    log_and_print("="*80, log_file)
    section_counts = {}
    for doc in docs:
        section = doc.metadata.get('section', 'Unknown')
        section_counts[section] = section_counts.get(section, 0) + 1
    
    for section, count in sorted(section_counts.items()):
        log_and_print(f"  {section}: {count}개 청크", log_file)
    
    log_and_print("\n" + "="*80, log_file)
    log_and_print("✅ 청킹 결과 출력 완료", log_file)
    log_and_print("="*80 + "\n", log_file)


    return docs


# =========================
# 1) LLM & Embeddings Setup
# =========================
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# llm = ChatOpenAI(model="gpt-4o", temperature=0)
# emb = OpenAIEmbeddings()

# Global vectorstore (will be populated in preprocess_node)
vectorstore = None
all_claims = []  # (더 이상 사용하지 않음 - 레거시 코드)
current_patent_id = None  # 현재 처리 중인 특허 ID

# =========================
# 2) Custom Retriever
# =========================
def custom_retrieve(query: str, k: int = 15) -> List[Document]:
    """
    커스텀 retriever: similarity 기반으로 가장 관련성 높은 k개 문서를 검색
    (모든 청구항을 자동으로 포함하지 않음)
    
    Args:
        query: 검색 쿼리
        k: 검색할 문서 수 (기본값: 10)
        
    Returns:
        List[Document]: similarity 기반 상위 k개 문서
    """
    global vectorstore
    
    if vectorstore is None:
        return []
    
    # Similarity search로 k개 검색
    results = vectorstore.similarity_search(query, k=k)
    
    return results


# =========================
# 3) RAG Tool: 두 개의 분리된 검색 Tool
# =========================


@tool
def get_available_metadata(
    metadata_keys: Optional[List[str]] = None
) -> str:
    """
    특허 문서에서 사용 가능한 메타데이터 값들을 조회합니다.
    
    이 툴을 사용하여:
    - 특허에 어떤 섹션들이 있는지 확인
    - 몇 번 claim까지 있는지 확인
    - 기타 메타데이터 필드와 값들 확인
    
    Args:
        metadata_keys: 조회할 메타데이터 키 리스트 (None이면 모든 키 조회)
                       예: ["section", "claim_no", "granularity"]
    
    Returns:
        사용 가능한 메타데이터 정보
    
    Examples:
        - get_available_metadata()  # 모든 메타데이터 조회
        - get_available_metadata(["section"])  # 섹션 목록만 조회
        - get_available_metadata(["claim_no"])  # Claim 번호 목록만 조회
    """
    global vectorstore
    
    if vectorstore is None:
        return "Error: Patent document not yet preprocessed. Please wait for preprocessing to complete."
    
    # 모든 문서 가져오기
    all_docs = vectorstore.similarity_search("", k=10000)
    
    # 메타데이터 수집
    metadata_values = {}
    
    # 기본적으로 조회할 키들
    if metadata_keys is None:
        metadata_keys = ["section", "claim_no", "granularity", "independent"]
    
    for key in metadata_keys:
        values = set()
        for doc in all_docs:
            if key in doc.metadata:
                value = doc.metadata[key]
                if isinstance(value, list):
                    for item in value:
                        values.add(str(item))
                else:
                    values.add(str(value))
        
        if values:
            if key == "claim_no":
                try:
                    metadata_values[key] = sorted([int(v) for v in values if v.isdigit()])
                except:
                    metadata_values[key] = sorted(list(values))
            else:
                metadata_values[key] = sorted(list(values))
    
    # 포맷팅
    output = []
    output.append("=" * 80)
    output.append("📊 AVAILABLE METADATA")
    output.append("=" * 80)
    output.append(f"\n총 문서 수: {len(all_docs)}\n")
    
    for key, values in metadata_values.items():
        output.append(f"\n🔹 {key.upper()}")
        output.append("-" * 40)
        
        if key == "section":
            output.append(f"섹션 수: {len(values)}")
            for v in values:
                count = sum(1 for doc in all_docs if doc.metadata.get('section') == v)
                output.append(f"  • {v} ({count} documents)")
        
        elif key == "claim_no":
            if values:
                output.append(f"Claim 범위: {min(values)} ~ {max(values)} (총 {len(values)}개)")
                independent_count = sum(
                    1 for doc in all_docs 
                    if doc.metadata.get('granularity') == 'claim' 
                    and doc.metadata.get('independent', False)
                )
                output.append(f"  • Independent Claims: {independent_count}개")
                output.append(f"  • Dependent Claims: {len(values) - independent_count}개")
        
        elif key == "granularity":
            for v in values:
                count = sum(1 for doc in all_docs if doc.metadata.get('granularity') == v)
                output.append(f"  • {v}: {count} documents")
        
        else:
            output.append(f"고유 값 수: {len(values)}")
            if len(values) <= 20:
                for v in values:
                    count = sum(1 for doc in all_docs if str(doc.metadata.get(key)) == str(v))
                    output.append(f"  • {v} ({count} documents)")
    
    output.append("\n" + "=" * 80)
    output.append("💡 사용 방법")
    output.append("=" * 80)
    output.append("\n위 메타데이터 값들을 search_by_metadata 함수의 filters 파라미터에 사용할 수 있습니다.")
    output.append("\n혹은 search_by_similarity 함수를 사용하여 개념적 또는 의미 기반 검색을 할 수 있습니다. 동일한 단어가 없어도 의미적으로 유사한 문단을 찾아줍니다.\ㅜ" \
    "질의의 유형에 따라 적절한 도구를 선택하세요.\n")
    # output.append("\n예시:")
    # output.append('  search_by_metadata("검색어", filters={"section": "ABSTRACT"})')
    # output.append('  search_by_metadata("검색어", filters={"claim_no": 1})')
    
    return "\n".join(output)



@tool
def search_by_metadata(
    query: str,
    filters: str = None,
    k: int = 15
) -> str:
    """
    메타데이터 기반 필터링으로 특허 문서를 검색합니다.
    Lambda 표현식을 사용하여 동적으로 메타데이터를 필터링합니다.
    
    ⚠️ 중요 사용 규칙:
    - 이 도구는 "특정 섹션의 전체 내용" 또는 "특정 claim 번호", "전체 claim" 을 가져올 때만 사용합니다.
    - 개념적 질문, 의미 검색, 키워드 검색에는 절대 사용하지 마세요.
    - 개념적 질문(예: "innovation points", "advantages", "problems solved", "how does X work")은 search_by_similarity를 사용하세요.
    
    ⚠️ 사용 전: 먼저 get_available_metadata 툴로 사용 가능한 메타데이터를 확인하세요!
    
    올바른 사용 시기 (ONLY use when):
    - 특정 섹션의 전체 내용을 가져올 때 (예: ABSTRACT 섹션 전체, CLAIMS 섹션 전체)
    - 특정 claim 번호를 정확히 가져올 때 (예: claim 1, claim 2-5)
    - 독립항/종속항을 구분해서 가져올 때
    - 구조적으로 명확한 메타데이터 기반 필터링이 필요한 경우
    
    잘못된 사용 예시 (DO NOT use for):
    - "innovation points", "advantages", "problems solved" 같은 개념적 질문
    - "stepped edge", "etching conditions", "TMAH" 같은 키워드 검색
    - "how does X work", "what is the mechanism" 같은 의미적 질문
    - 특정 섹션 내에서 개념을 검색하려는 경우 (이 경우 search_by_similarity 사용)
    
    Args:
        query: 검색 키워드 (필터링된 결과 내에서 정렬에 사용됨)
        filters: 필터링 조건 (Python 표현식 문자열)
                예: "metadata['section'] == 'ABSTRACT'"
                    "metadata['claim_no'] == 1"
                    "metadata['section'] == 'CLAIMS' and metadata['independent'] == True"
        k: 반환할 최대 문서 수 (기본값: 15)
    
    Returns:
        필터링된 특허 내용
    
    Examples:
        # 1단계: 사용 가능한 메타데이터 확인
        get_available_metadata(["section"])
        
        # 2단계: 구조적 검색 (올바른 사용)
        - search_by_metadata("", filters="metadata['section'] == 'ABSTRACT'")  # ABSTRACT 섹션 전체 가져오기
        - search_by_metadata("", filters="metadata['section'] == 'CLAIMS'")  # CLAIMS 섹션 전체 가져오기
        - search_by_metadata("", filters="metadata['claim_no'] == 1")  # Claim 1만 가져오기
        - search_by_metadata("", filters="metadata['section'] == 'CLAIMS' and metadata['independent'] == True")  # ⚠️독립항만 가져오기⚠️
        - search_by_metadata("", filters="metadata['section'] == 'CLAIMS' and metadata['independent'] == False")  # ⚠️종속항만 가져오기⚠️
    """
    global vectorstore
    
    if vectorstore is None:
        return "Error: Patent document not yet preprocessed. Please wait for preprocessing to complete."
    
    print(f"🔍 [METADATA FILTER] {filters}")
    
    # similarity_search에 lambda 필터 적용
    if filters:
        try:
            # 문자열로 받은 filter를 lambda 함수로 변환
            filter_func = eval(f"lambda metadata: {filters}")
            
            # FAISS는 filter를 지원하지 않으므로 수동 필터링
            # 전체 문서를 가져와서 필터링 (similarity_search가 아닌 전체 문서에서)
            try:
                # FAISS vectorstore의 모든 문서 가져오기
                all_docs = list(vectorstore.docstore._dict.values())
            except:
                # docstore 접근 실패시 fallback: 빈 쿼리로 많은 수의 문서 가져오기
                all_docs = vectorstore.similarity_search("", k=100000)
            
            # 필터링 적용
            filtered_docs = [doc for doc in all_docs if filter_func(doc.metadata)]
            
            print(f"   📊 Total documents: {len(all_docs)}")
            print(f"   ✅ Filtered documents: {len(filtered_docs)}")
            
            # 필터링된 문서가 있으면 상위 k개 반환
            # 쿼리가 있으면 similarity 순으로, 없으면 순서대로
            if filtered_docs and query.strip():
                # 필터링된 문서들 중에서 쿼리와 가장 유사한 문서 찾기
                # 임시 vectorstore 생성하여 검색
                temp_vs = FAISS.from_documents(filtered_docs, vectorstore.embeddings)
                docs = temp_vs.similarity_search(query, k=min(k, len(filtered_docs)))
            else:
                docs = filtered_docs[:k]
            
        except Exception as e:
            print(f"⚠️ Filter evaluation failed: {e}")
            print(f"   Using empty result")
            docs = []
    else:
        docs = vectorstore.similarity_search(query, k=k)
    
    # 결과가 없는 경우
    if not docs:
        output = []
        output.append("=" * 80)
        output.append("⚠️  검색 결과가 없습니다")
        output.append("=" * 80)
        output.append(f"\n검색어: {query}")
        output.append(f"필터: {filters}\n")
        output.append("💡 제안:")
        output.append("1. get_available_metadata() 툴로 사용 가능한 메타데이터를 먼저 확인하세요")
        output.append("2. 필터 조건을 완화하거나 다른 메타데이터를 시도해보세요")
        output.append("3. search_by_similarity() 툴로 의미 기반 검색을 시도해보세요")
        return "\n".join(output)
    
    # 포맷팅 (기존과 동일)
    output = []
    output.append("="*80)
    output.append(f"🔍 METADATA SEARCH: {query}")
    output.append(f"🎯 Filter: {filters}")
    output.append(f"📊 Retrieved {len(docs)} documents")
    output.append("="*80 + "\n")
    
    # Claims 부분
    claims = [d for d in docs if d.metadata.get('granularity') == 'claim']
    if claims:
        independent = [c for c in claims if c.metadata.get('independent', False)]
        dependent = [c for c in claims if not c.metadata.get('independent', False)]
        
        output.append("=" * 80)
        output.append("📋 CLAIMS")
        output.append("=" * 80)
        
        if independent:
            output.append("\n■ Independent Claims:")
            for claim in independent:
                claim_no = claim.metadata.get('claim_no', '?')
                output.append(f"\n[Claim {claim_no}]")
                content = claim.page_content
                content = re.sub(r'^\[CLAIM \d+\]\s+', '', content)
                output.append(content)
        
        if dependent:
            output.append("\n\n■ Dependent Claims:")
            for claim in dependent:
                claim_no = claim.metadata.get('claim_no', '?')
                refs = claim.metadata.get('references', [])
                if refs:
                    ref_str = ', '.join(map(str, refs))
                    output.append(f"\n[Claim {claim_no} - depends on: {ref_str}]")
                else:
                    output.append(f"\n[Claim {claim_no}]")
                content = claim.page_content
                content = re.sub(r'^\[CLAIM \d+\]\s+', '', content)
                output.append(content)
    
    # 기타 섹션
    other = [d for d in docs if d.metadata.get('granularity') != 'claim']
    if other:
        output.append("\n\n" + "=" * 80)
        output.append("📄 RELEVANT SECTIONS")
        output.append("=" * 80)
        
        for doc in other:
            section = doc.metadata.get('section', 'Unknown')
            output.append(f"\n[{section}]")
            content = re.sub(r'^\[.*?\]\s*\n\n', '', doc.page_content)
            output.append(content)
            output.append("-" * 40)
    
    return "\n".join(output)




@tool  
def search_by_similarity(query: str, k: int = 15) -> str:
    """
    벡터 유사도 기반으로 특허 문서를 검색합니다.
    의미적으로 관련된 내용을 찾아 반환합니다.
    
    ⚠️ 중요: 현재 특허(current_patent_id)로 먼저 필터링한 후 similarity 검색을 수행합니다.
    
    올바른 사용 시기 (USE THIS FOR):
    - 개념적/의미적 질문: "innovation points", "problems solved", "advantages", "benefits"
    - 키워드 검색: "stepped edge", "etching conditions", "TMAH", "HCl etching"
    - 기술적 내용 검색: "how does X work", "what is the mechanism", "implementation method"
    - 특정 개념이나 기술에 대한 일반적 질문
    - 섹션을 지정하지 않은 모든 개념적 검색
    
    사용하지 말아야 할 경우:
    - 특정 claim 번호를 정확히 가져올 때 (예: claim 1만) → search_by_metadata 사용
    - 특정 섹션의 전체 내용만 가져올 때 (예: ABSTRACT 전체) → search_by_metadata 사용
    
    Args:
        query: 검색할 질문이나 개념 (예: "innovation points of this patent", "stepped edge advantages", "etching conditions")
        k: 반환할 문서 수 (기본값: 15)
    
    Returns:
        의미적으로 유사한 특허 내용 (자동으로 관련 섹션과 claims 포함)
    
    Examples:
        - search_by_similarity("innovation points of this patent, what problems solved, advantages")
        - search_by_similarity("stepped edge and superlattice, effective mass reduction")
        - search_by_similarity("HCl etching 840-860 C TMAH developer")
        - search_by_similarity("DEPENDENT claims with etching conditions TMAH HCl")
    """
    global vectorstore, current_patent_id
    
    if vectorstore is None:
        return "Error: Patent document not yet preprocessed. Please wait for preprocessing to complete."
    
    print(f"🔍 [SIMILARITY SEARCH]")
    print(f"   Query: {query}")
    print(f"   Current Patent ID: {current_patent_id}")
    
    # 1단계: patent_id로 먼저 필터링 (있는 경우)
    if current_patent_id:
        print(f"   ⚡ Filtering by patent_id: {current_patent_id}")
        # patent_id로 필터링된 문서에서 similarity 검색
        filter_dict = {'patent_id': current_patent_id}
        try:
            docs = vectorstore.similarity_search(
                query,
                k=k,
                filter=filter_dict
            )
            print(f"   ✅ Filtered results: {len(docs)} documents")
        except Exception as e:
            # 필터링 실패시 일반 검색으로 fallback
            print(f"   ⚠️ Filter failed, using regular search: {e}")
            docs = vectorstore.similarity_search(query, k=15)
    else:
        # patent_id가 없으면 일반 similarity 검색
        print(f"   ℹ️ No patent_id filter, using regular similarity search")
        docs = vectorstore.similarity_search(query, k=15)
    
    # 포맷팅
    output = []
    output.append("="*80)
    output.append(f"🔍 SIMILARITY SEARCH: {query}")
    if current_patent_id:
        output.append(f"🎯 Filtered by Patent ID: {current_patent_id}")
    output.append(f"📊 Retrieved {len(docs)} documents")
    output.append("="*80 + "\n")
    
    # Claims 분리
    claims = [d for d in docs if d.metadata.get('granularity') == 'claim']
    if claims:
        output.append("=" * 80)
        output.append("📋 RELEVANT CLAIMS")
        output.append("=" * 80)
        
        for claim in claims:
            claim_no = claim.metadata.get('claim_no', '?')
            claim_type = "Independent" if claim.metadata.get('independent') else "Dependent"
            output.append(f"\n[Claim {claim_no} - {claim_type}]")
            content = re.sub(r'^\[CLAIM \d+\]\s+', '', claim.page_content)
            output.append(content)
    
    # 기타 섹션
    other = [d for d in docs if d.metadata.get('granularity') != 'claim']
    if other:
        output.append("\n\n" + "=" * 80)
        output.append("📄 RELEVANT SECTIONS")
        output.append("=" * 80)
        
        for doc in other:
            section = doc.metadata.get('section', 'Unknown')
            output.append(f"\n[{section}]")
            content = re.sub(r'^\[.*?\]\s*\n\n', '', doc.page_content)
            output.append(content)
    
    return "\n".join(output)


# =========================
# 3.5) New Tools for Horizontal Agent - Patent Comparison
# =========================

@tool
def generate_patent_search_query(abstract: str) -> str:
    """
    특허 abstract를 기반으로 유사한 특허를 검색하기 위한 검색식을 생성합니다.
    
    LLM을 사용하여 abstract에서 핵심 기술 키워드를 추출하고,
    Google Patents에서 사용할 수 있는 최적화된 검색 쿼리를 생성합니다.
    
    Args:
        abstract: 특허의 abstract 전문
    
    Returns:
        Google Patents 검색에 최적화된 검색 쿼리
    
    Example:
        abstract = "A semiconductor device with stepped edge..."
        query = generate_patent_search_query(abstract)
        # Returns: "semiconductor device stepped edge superlattice"
    """
    if not abstract or len(abstract.strip()) < 50:
        return "Error: Abstract is too short or empty. Please provide a valid abstract."


    
    prompt = f"""You are an expert in prior-art search and Google Patents query design.

    GOAL:
    - Your top priority is to FIND many relevant similar patents (high recall), not just a few extremely precise hits.
    - The query must work well in Google Patents and should avoid being too narrow.

    INTERNAL STRATEGY (do NOT include these steps in the output):
    1. From the abstract, identify:
    - (a) the DEVICE/COMPONENT type (e.g., transistor, memory, DRAM, LED)
    - (b) the CORE STRUCTURE or PROCESS (e.g., superlattice, stepped edge, STI recess, gate stack)
    - (c) the FUNCTIONAL EFFECT or PURPOSE (e.g., reduced leakage, improved reliability, phonon isolation).
    2. For each of (a), (b), (c), think of 1–2 common synonyms or alternative phrases used in patents.
    - Example: superlattice ≈ "multiple quantum well" OR MQW
    3. Prefer general, widely-used technical terms over very niche or proprietary words.
    4. Combine these into a single query using:
    - OR between synonyms
    - AND (implicit by space) between main concepts

    OUTPUT FORMAT:
    - Return ONLY ONE final query string.
    - Use 3–6 conceptual elements (words or short phrases), not just 2.
    - Use parentheses and OR for synonyms.
    - Do NOT add any explanation.

    GOOD EXAMPLES:
    Abstract: "A semiconductor device with stepped edge for quantum wells..."
    Query: (stepped edge) (semiconductor OR "quantum well")

    Abstract: "A memory device using superlattice structures with phonon isolation..."
    Query: (superlattice OR "multiple quantum well" OR MQW) (memory OR storage) phonon

    Abstract: "A transistor with non-semiconductor monolayer barriers..."
    Query: (non-semiconductor monolayer OR "insulating monolayer") (transistor OR FET) barrier

    BAD EXAMPLES (too narrow or too many random words):
    ❌ "recessed active region stepped edge isolation superlattice non-semiconductor monolayer"
    ❌ "semiconductor device memory storage quantum well superlattice structure reliability efficiency"

    Abstract:
    {abstract}

    Now generate ONLY the final Google Patents search query string:"""


    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        search_query = response.content.strip()
        
        # Remove quotes if present at the beginning/end
        search_query = search_query.strip('"').strip("'")
        
        # Validate query complexity
        word_count = len(search_query.replace('(', ' ').replace(')', ' ').replace('"', ' ').split())
        
        print(f"✅ Generated search query: {search_query}")
        print(f"   Query complexity: {word_count} keywords")
        
        # If too complex (>7 words excluding operators), warn and suggest simplification
        if word_count > 7:
            print(f"   ⚠️  Query might be too specific. Consider simplifying if no results found.")
            # Extract first 4 meaningful words as fallback
            words = [w for w in search_query.split() if w not in ['OR', 'AND', '(', ')', '"']]
            if len(words) > 4:
                simpler_query = ' '.join(words[:4])
                print(f"   💡 Fallback query: {simpler_query}")
        
        return search_query
    
    except Exception as e:
        return f"Error generating search query: {str(e)}"



@tool
def refine_patent_search_query(
    abstract: str,
    previous_query: str,
    iteration: int = 1
) -> str:
    """
    LLM을 사용하여 특허 검색 쿼리를 점진적으로 고도화합니다.
    함수 내부에서 분석을 수행하고, refined_query를 만들어 반환합니다.
    
    Args:
        abstract: 특허 초록 (쿼리 생성에 사용)
        previous_query: 이전 쿼리 (개선 대상)
        iteration: 현재 iteration 번호 (최대 8)
    
    Returns:
        JSON string:
        {
            "success": true/false,
            "refined_query": "...",  ...
        }
    """
    print(f"\n{'='*80}")
    print(f"🤖 REFINE ITERATION {iteration}/8")
    print(f"{'='*80}")
    print(f"Previous Query: {previous_query}")
    
    if iteration > 8:
        return json.dumps({
            "error": True,
            "message": "Maximum iterations (8) reached",
            "previous_query": previous_query
        }, ensure_ascii=False)
    
    prompt = f"""You are a patent prior-art search expert specializing in Google Patents.

Your role:
- Refine and improve patent search queries to find the most relevant similar patents
- This is NOT just for failed searches - you improve ANY query to make it more sophisticated and effective

Input:
- abstract: The current patent's abstract
- previous_query: The query to be improved (may be initial or already refined)
- iteration: Current refinement iteration number (1-8)

Output:
- A refined query with feedback explaining the improvements made

-------------------------------------------------------------------------------
GENERAL GUIDELINES FOR QUERY REFINEMENT
-------------------------------------------------------------------------------

Your goal: Create progressively better queries that balance precision and recall.

Core Principles:
1. Focus on 3–6 core technical concepts (device type, key structure, process, material, principle)
2. Use standard patent terminology and technical language
3. Use AND to combine distinct concepts, OR for synonyms
4. Keep queries concise (5–12 keywords optimal)
5. Include physical/functional keywords when relevant (bandgap, interface, strain, leakage, quantum, etc.)

What to avoid:
- Too many AND constraints (over-constraining)
- Long literal phrases copied from abstract
- Non-technical stopwords (including, wherein, having, according to, etc.)
- Patent numbers or company names
- Overly specific numbers or measurements

HOW TO USE FEEDBACK FOR REFINEMENT:
- Each iteration builds on the previous query by applying constructive improvements
- Use the GENERAL GUIDELINES above to decide what to improve
- Consider the feedback from previous iterations to avoid repeating unsuccessful approaches
- Progressive refinement: start with core concepts, then expand/adjust based on results
- Balance precision and recall - don't over-constrain too early

REFINEMENT APPROACH:
1. Analyze the previous_query to identify strengths and weaknesses
2. Apply improvements from GENERAL GUIDELINES:
   - Optimize keyword selection (add/remove/replace terms)
   - Adjust synonym coverage (add OR alternatives)
   - Modify specificity level (broader or narrower)
   - Incorporate technical principles when relevant
   - Remove over-constraints if query is too narrow
3. Generate constructive feedback explaining the improvements
4. Output the refined query with clear rationale

Two main approaches to consider:

**Similar Approach**: Find patents with similar device architecture
- Extract: device type + key structure + key process/material
- Example: "(MOSFET OR transistor) AND (buried gate OR embedded gate) AND (high-k dielectric OR HfO2 OR gate oxide)"

**Base Technology Approach**: Find patents sharing underlying principles
- Identify: physical principle or engineering goal
- Generalize specific terms to broader concepts
- Example: "(semiconductor device) AND (bandgap engineering OR energy band modulation) AND (superlattice OR quantum well OR heterostructure)"

Choose or blend approaches based on the abstract and previous_query.

-------------------------------------------------------------------------------
FEEDBACK GUIDELINES
-------------------------------------------------------------------------------

Provide constructive feedback (2-4 sentences) explaining:
- What aspects of previous_query were good
- What specific improvements you're making
- Why these changes will help find more relevant patents

Focus on IMPROVEMENTS, not failures. Be constructive and specific.

Good feedback examples:
- "The previous query had good core concepts but was too narrow. I'm adding synonyms and broadening the structural terms to capture more related patents while maintaining focus."
- "Building on the solid foundation, I'm incorporating the underlying physical principle (bandgap engineering) to find patents that share the same technical approach even with different structures."
- "The query is well-structured. I'm fine-tuning by adding process-related terms and adjusting synonym coverage to improve recall without sacrificing precision."

-------------------------------------------------------------------------------
CONTEXT
-------------------------------------------------------------------------------

- Iteration: {iteration}/8
- Previous query: "{previous_query}"

PATENT ABSTRACT (truncated to ~500 chars):
{abstract[:500]}...

-------------------------------------------------------------------------------
OUTPUT FORMAT (JSON only, no extra text)
-------------------------------------------------------------------------------

{{
  "feedback": "Constructive feedback on improvements being made (2-4 sentences)",
  "refined_query": "The improved Google Patents search query"
}}

IMPORTANT: 
- refined_query MUST be different from previous_query (unless previous_query is already optimal)
- Focus on making queries progressively better, not just reacting to failures
- Each iteration should add sophistication and refinement
"""
    
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        content = resp.content.strip()
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        analysis = json.loads(content)
        
        if "error" in analysis:
            return json.dumps({"error": True, "message": "LLM failed"}, ensure_ascii=False)
        
        refined_query = analysis.get("refined_query", "")
        feedback = analysis.get("feedback", "")        
        if not refined_query:
            return json.dumps({"error": True, "message": "No query generated"}, ensure_ascii=False)
        
        print(f"💡 Feedback: {feedback}")
        
        
        print(f"🔍 Refined Query: {refined_query}\n")
        
        result = {
            "success": True,
            "refined_query": refined_query,
            "feedback": feedback,
            "iteration": iteration,
            "previous_query": previous_query
        }
        
        print(f"✅ Refined: {previous_query} → {refined_query}")
        
        return json.dumps(result, ensure_ascii=False)
        
    except json.JSONDecodeError as e:
        return json.dumps({"error": True, "message": f"JSON error: {str(e)}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": True, "message": f"Error: {str(e)}"}, ensure_ascii=False)
    

@tool
def search_similar_patents_serpapi(search_query: str, num_results: int = 10) -> str:
    """
    SerpAPI를 통해 Google Patents에서 유사한 특허를 검색합니다.
    단순 검색만 수행합니다.
    
    Args:
        search_query: Google Patents 검색 쿼리
        num_results: 반환할 특허 개수 (기본값: 2, 최대 10)
    
    Returns:
        성공: 특허 정보 (formatted string)
        실패: "NO_RESULTS_FOUND"
    """
    try:
        from serpapi import GoogleSearch
    except ImportError:
        return "Error: google-search-results package not installed"
    
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    if not serpapi_key:
        return "Error: SERPAPI_API_KEY not set"
    
    num_results = min(max(1, num_results), 10)
    
    print(f"\n🔍 Searching Google Patents...")
    print(f"   Query: {search_query}")
    print(f"   Results: {num_results}")
    
    try:
        params = {
            "engine": "google_patents",
            "q": search_query,
            "api_key": serpapi_key,
            "num": max(10, num_results)  # SerpAPI 권장 최소값 10
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        if "organic_results" not in results or not results["organic_results"]:
            print(f"   ❌ No results found\n")
            return "NO_RESULTS_FOUND"
        
        print(f"   ✅ Found {len(results['organic_results'])} patents\n")
        
        formatted = []
        formatted.append("="*80)
        formatted.append("✅ SIMILAR PATENTS FOUND")
        formatted.append("="*80)
        formatted.append(f"\nQuery: {search_query}")
        formatted.append(f"Total Results: {len(results['organic_results'])}\n")
        
        for idx, patent in enumerate(results["organic_results"], 1):
            formatted.append(f"\n{'='*80}")
            formatted.append(f"Patent {idx}")
            formatted.append(f"{'='*80}")
            
            title = patent.get("title", "No title")
            patent_id = patent.get("patent_id", "Unknown")
            snippet = patent.get("snippet", "No snippet")
            pdf_link = patent.get("pdf", "")
            filing_date = patent.get("filing_date", "")
            assignee = patent.get("assignee", "")
            
            formatted.append(f"\n📌 Title: {title}")
            formatted.append(f"📄 Patent ID: {patent_id}")
            
            if assignee:
                formatted.append(f"🏢 Assignee: {assignee}")
            if filing_date:
                formatted.append(f"📅 Filing Date: {filing_date}")
            
            formatted.append(f"\n📝 Summary:\n{snippet}")
            
            if pdf_link:
                formatted.append(f"\n🔗 PDF: {pdf_link}")
            
            formatted.append("")
        
        formatted.append("="*80)
        return "\n".join(formatted)
        
    except Exception as e:
        return f"SEARCH_ERROR: {str(e)}"
@tool
def search_similar_patents_serpapi(search_query: str, num_results: int = 2) -> str:
    """
    SerpAPI를 통해 Google Patents에서 유사한 특허를 검색합니다.
    단순 검색만 수행하고 결과를 반환합니다.
    
    Args:
        search_query: Google Patents 검색 쿼리
        num_results: 반환할 특허 개수 (기본값: 2, 최대 10)
    
    Returns:
        검색 성공: 특허 정보 (formatted string)
        검색 실패: "NO_RESULTS_FOUND" 문자열
    
    Example:
        results = search_similar_patents_serpapi("semiconductor device", num_results=2)
    """
    try:
        from serpapi import GoogleSearch
    except ImportError:
        return ("Error: 'google-search-results' package not installed. "
                "Please install it with: pip install google-search-results")
    
    # SerpAPI API 키 확인
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    if not serpapi_key:
        return ("Error: SERPAPI_API_KEY environment variable not set. "
                "Please set it with your SerpAPI key from https://serpapi.com/")
    
    # num_results 제한
    num_results = min(max(1, num_results), 10)
    
    print(f"\n🔍 Searching Google Patents via SerpAPI...")
    print(f"   Query: {search_query}")
    print(f"   Number of results: {num_results}")
    
    try:
        # SerpAPI Google Patents 검색
        params = {
            "engine": "google_patents",
            "q": search_query,
            "api_key": serpapi_key,
            "num": max(10, num_results)  # SerpAPI 권장 최소값 10
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # 결과 확인
        if "organic_results" not in results or not results["organic_results"]:
            print(f"   ❌ No results found")
            return "NO_RESULTS_FOUND"
        
        # 성공적으로 결과를 찾은 경우
        print(f"   ✅ Found {len(results['organic_results'])} results!")
        
        formatted_results = []
        formatted_results.append("=" * 80)
        formatted_results.append("✅ SIMILAR PATENTS FOUND")
        formatted_results.append("=" * 80)
        formatted_results.append(f"\nQuery: {search_query}")
        formatted_results.append(f"Total Results: {len(results['organic_results'])}\n")
        
        for idx, patent in enumerate(results["organic_results"], 1):
            formatted_results.append(f"\n{'='*80}")
            formatted_results.append(f"Patent {idx}")
            formatted_results.append(f"{'='*80}")
            
            title = patent.get("title", "No title")
            patent_id = patent.get("patent_id", "Unknown")
            snippet = patent.get("snippet", "No snippet available")
            pdf_link = patent.get("pdf", "")
            filing_date = patent.get("filing_date", "")
            priority_date = patent.get("priority_date", "")
            grant_date = patent.get("grant_date", "")
            inventor = patent.get("inventor", "")
            assignee = patent.get("assignee", "")
            
            formatted_results.append(f"\n📌 Title: {title}")
            formatted_results.append(f"📄 Patent ID: {patent_id}")
            
            if assignee:
                formatted_results.append(f"🏢 Assignee: {assignee}")
            if inventor:
                formatted_results.append(f"👤 Inventor: {inventor}")
            
            if filing_date:
                formatted_results.append(f"📅 Filing Date: {filing_date}")
            if priority_date:
                formatted_results.append(f"🎯 Priority Date: {priority_date}")
            if grant_date:
                formatted_results.append(f"✅ Grant Date: {grant_date}")
            
            formatted_results.append(f"\n📝 Summary:\n{snippet}")
            
            if pdf_link:
                formatted_results.append(f"\n🔗 PDF: {pdf_link}")
            
            formatted_results.append(f"\n")
        
        formatted_results.append("=" * 80)
        return "\n".join(formatted_results)
        
    except Exception as e:
        error_msg = f"Error during patent search: {str(e)}"
        print(f"   ⚠️  {error_msg}")
        return f"SEARCH_ERROR: {error_msg}"




# =========================
# 4) State Definition
# =========================
# =========================
# Plan Schema (NEW)
# =========================
class Task(TypedDict):
    """단일 Task 스키마"""
    task_id: str  # 예: "T1", "T2"
    description: str  # Task 설명
    agent: str  # 수행할 agent 이름
    depends_on: List[str]  # 선행 task_id 리스트
    parallelizable: bool  # 병렬 실행 가능 여부
    max_retries: int  # 최대 재시도 횟수
    inputs: Dict[str, Any]  # 입력 데이터

class Plan(TypedDict):
    """전체 Plan 스키마"""
    tasks: List[Task]
    goal: str

# =========================
# Enhanced State (기존 + Plan 관련 필드)
# =========================
class State(MessagesState):
    patent_id: str  # PDF 경로 또는 특허 번호
    preprocessed: bool = False  # 전처리 완료 여부
    plan: Optional[Plan] = None  # Planner가 생성한 Plan
    task_results: Dict[str, Any] = {}  # {task_id: result}
    completed_tasks: Set[str] = set()  # 완료된 task_id 집합
    failed_tasks: Dict[str, str] = {}  # {task_id: error_message}
    current_iteration: int = 0  # Supervisor 실행 횟수
    merged_result: str = ""  # Supervisor가 merge한 최종 결과
    next: str = ""  # 다음 노드


# =========================
# 5) Preprocess Node (수정됨 - 저장/로드 로직 추가)
# =========================

def preprocess_node(state: State) -> Command[Literal["planner"]]:
    """
    PDF를 로드하고 LLM으로 전처리한 후 Vector DB에 저장하는 노드
    이미 저장된 특허는 로드만 수행
    """
    global vectorstore, current_patent_id
    
    patent_path = state["patent_id"]
    
    # 특허 ID 추출
    patent_filename = os.path.basename(patent_path)
    current_patent_id = os.path.splitext(patent_filename)[0]
    
    print(f"\n{'='*80}")
    print(f"Starting preprocessing for Patent ID: {current_patent_id}")
    print(f"{'='*80}\n")
    
    # 이미 vectorstore가 존재하는지 확인
    if vectorstore_exists(current_patent_id):
        print(f"✅ Vectorstore already exists for {current_patent_id}")
        print(f"📂 Loading from disk...\n")
        
        # 기존 vectorstore 로드
        vectorstore = load_vectorstore(current_patent_id, emb)
        
        print(f"\n{'='*80}")
        print(f"✅ Loaded existing vectorstore for {current_patent_id}")
        print(f"{'='*80}\n")
        
        return Command(
            update={"preprocessed": True},
            goto="planner"
        )
    
    # 새로운 전처리 시작
    print(f"🔄 No existing vectorstore found. Starting new preprocessing...\n")
    
    # 로그 파일 생성
    preprocessing_log = create_log_file(current_patent_id, "preprocessing")
    chunking_log = create_log_file(current_patent_id, "chunking")
    
    log_and_print(f"{'='*80}", preprocessing_log)
    log_and_print(f"전처리 시작: {current_patent_id}", preprocessing_log)
    log_and_print(f"시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", preprocessing_log)
    log_and_print(f"{'='*80}\n", preprocessing_log)
    
    # 1) PDF 로드 (첫 페이지 칼럼 분리)
    pages = load_pages_with_first_page_columns(patent_path)
    log_and_print(f"Loaded {len(pages)} pages from PDF", preprocessing_log)
    
    # 2) full_text 생성
    full_text = ""
    for page in pages:
        full_text += page.page_content + "\n\n"
    
    log_and_print(f"Generated full_text with {len(full_text)} characters", preprocessing_log)
    
    # 3) LLM으로 전처리 (섹션 분리 및 메타데이터 추출)
    system_prompt = """당신은 미국 특허 문서 분석 전문가입니다. 
주어진 특허 문서를 분석하여 다음 작업을 수행하세요:

1. 메타데이터 추출:
문서의 첫 페이지(Front Page)에서 아래 항목들을 가능한 한 모두 추출하여 JSON 형태로 정리하세요.
존재하지 않는 항목은 빈 문자열("")로 채우세요.

포함해야 하는 주요 메타데이터 필드:
- patent_number: 특허 등록번호 (예: "US 8,526,476 B2")
- publication_number: 공개번호 (공개특허일 경우)
- application_number: 출원번호 (예: "13/113,482")
- filing_date: 출원일 (예: "May 23, 2011")
- publication_date: 공개일 또는 등록일 (예: "Sep. 3, 2013")
- priority_date: 우선권 주장일
- title: 발명의 명칭
- inventor: 발명자 이름 목록
- assignee: 양수인 또는 출원인 (회사명 등)
- examiner: 심사관 이름
- attorney_or_agent: 대리인 또는 법률 사무소
- cpc_class: Cooperative Patent Classification (CPC 코드)
- ipc_class: International Patent Classification (IPC 코드)
- us_class: U.S. Classification (USPC 코드)
- field_of_search: Field of Classification Search
- references_cited: 인용 문헌 또는 특허 목록
- related_applications: 관련 출원 정보 (continuation, divisional 등)
- government_interest: 정부 지원 관련 내용

2. 섹션 구분:
본문을 명확한 섹션 제목(모두 대문자) 기준으로 구분하세요.
섹션 제목은 아래 예시 목록에 포함되지 않아도 됩니다.
문서에 실제 존재하는 섹션을 모두 식별하여 포함하세요.
각 섹션의 내용은 원문 그대로 포함해야 합니다.

[섹션 내용 정제 규칙]
- PDF parsing 중 잘못 인식된 정보(예: "Publication No.", "Publication Date", "Page X of Y", "Line X" 등)는 섹션 본문에서 제거하세요.
- 즉, 'publication_number', 'publication_date', 'page', 'line', 'sheet' 등과 관련된 값이 섹션 내용 안에 있으면 모두 삭제해야 합니다.
- 메타데이터에 해당하는 정보(예: 특허번호, 공개번호, 출원번호, 공개일 등)는 섹션 본문에 포함하지 말고, 오직 metadata 필드에만 포함하세요.
- 본문 중 OCR 노이즈나 표, 번호, 페이지 인덱스 등 문서 구조상 의미 없는 텍스트는 제외하세요.
- SECTION TITLE 은 원문 그대로 유지하되, SECTION 내용은 의미 있는 본문만 포함하세요.

주요 섹션 예시 (참고용):
- ABSTRACT
- FIELD OF THE INVENTION / TECHNICAL FIELD
- BACKGROUND OF THE INVENTION
- SUMMARY OF THE INVENTION
- BRIEF DESCRIPTION OF THE DRAWINGS
- DETAILED DESCRIPTION OF THE INVENTION
- CLAIMS (이 섹션은 별도 처리하지 말고 원문 그대로 포함) (이 섹션은 "THE INVENTION CLAIMED IS"가 아닌 "CLAIMS"로 통일하세요)

3. CLAIMS 분석 (중요!):
CLAIMS 섹션이 존재하는 경우, 각 개별 claim을 식별하고 다음 정보를 추출하세요:
- claim_no: claim 번호 (정수)
- claim_text: claim 전체 텍스트 (원문 그대로)
- independent: 독립항 여부 (true/false)
* 독립항: 다른 claim을 참조하지 않는 claim
* 종속항: "claim X", "claims X-Y", "any of claims", "according to claim" 등의 표현으로 다른 claim을 참조하는 claim

독립항/종속항 판단 기준:
- 다음 패턴이 있으면 종속항으로 판단:
* "the [noun] of claim [number]" (예: "The method of claim 1")
* "the [noun] of any of claims [number]" (예: "The device of any of claims 1-5")
* "according to claim [number]"
* "as recited in claim [number]"
* "dependent on claim [number]"
* "wherein the [noun] of claim [number]"
* "further according to" / "further comprising" (단독으로는 종속항 아님, 하지만 claim 참조와 함께 나오면 종속항)
* "a [noun] as in claim [number]"
* "characterized in that" (유럽식 종속항 표현)
- 위 패턴이 없으면 독립항으로 판단

응답은 반드시 아래 JSON 형식으로만 작성하세요:

{
  "metadata": {
    "patent_number": "",
    "publication_number": "",
    "application_number": "",
    "filing_date": "",
    "publication_date": "",
    "priority_date": "",
    "title": "",
    "inventor": "",
    "assignee": "",
    "examiner": "",
    "attorney_or_agent": "",
    "cpc_class": "",
    "ipc_class": "",
    "us_class": "",
    "field_of_search": "",
    "references_cited": "",
    "related_applications": "",
    "government_interest": ""
  },
  "sections": {
    "SECTION_TITLE_1": "해당 섹션의 원문 내용",
    "SECTION_TITLE_2": "해당 섹션의 원문 내용",
    ...
  },
  "claims": [
    {
      "claim_no": 1,
      "claim_text": "claim 전체 텍스트",
      "independent": true
    },
    {
      "claim_no": 2,
      "claim_text": "claim 전체 텍스트",
      "independent": false
    },
    ...
  ]
}

주의:
- 섹션 제목은 문서 내의 실제 대문자 제목을 그대로 사용하세요.
- 추가적인 섹션이 존재하면 JSON의 "sections"에 새로운 키로 추가하세요.
- CLAIMS 섹션은 "sections"에 포함하되, 동시에 "claims" 배열에 개별 claim을 분리하여 포함하세요.
- claims 배열의 순서는 원문의 claim 번호 순서를 유지하세요.
- 각 claim의 독립항 여부를 정확히 판단하여 "independent" 필드에 true/false로 표시하세요.
- 출력은 반드시 하나의 JSON 객체로만 구성하세요. JSON 이외의 설명, 문장, 텍스트를 포함하지 마세요.
"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=full_text)
    ]
    
    log_and_print("Calling LLM for preprocessing...", preprocessing_log)
    # 긴 특허 문서를 처리하기 위해 max_tokens을 충분히 크게 설정
    response = llm.invoke(messages, config={"max_tokens": 16000})
    result_text = response.content
    log_and_print(f"✓ LLM response received ({len(result_text)} characters)", preprocessing_log)
    
    # JSON 파싱
    try:
        # ```json ... ``` 형태로 감싸진 경우 처리
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]
        
        patent_data = json.loads(result_text)
        log_and_print("✓ Successfully parsed patent data", preprocessing_log)
    except json.JSONDecodeError as e:
        error_msg = f"❌ 오류: {e}"
        log_and_print(error_msg, preprocessing_log)
        
        # 전체 응답을 로그 파일에 저장
        log_and_print("\n" + "="*80, preprocessing_log)
        log_and_print("전체 LLM 응답:", preprocessing_log)
        log_and_print("="*80, preprocessing_log)
        log_and_print(result_text, preprocessing_log)
        log_and_print("="*80 + "\n", preprocessing_log)
        
        # 콘솔에는 요약 출력
        print(f"\n❌ JSON 파싱 실패: {e}")
        print(f"응답 길이: {len(result_text)} 문자")
        print(f"응답 시작: {result_text[:500]}")
        print(f"응답 끝: {result_text[-500:]}")
        print(f"\n💡 해결 방법:")
        print(f"1. config.py에서 llm 정의시 max_tokens을 더 크게 설정 (예: 16000)")
        print(f"2. 또는 agent_logic.py의 preprocess_node에서 llm.invoke() 호출시")
        print(f"   config={{'max_tokens': 16000}}을 명시적으로 전달")
        print(f"\n📝 자세한 내용은 로그 파일을 확인하세요: {preprocessing_log}")
        raise

    # ============================================================
    # 디버깅: 전처리 결과 출력 (print + log)
    # ============================================================
    log_and_print("\n" + "="*80, preprocessing_log)
    log_and_print("📋 전처리 결과 - 섹션 및 Claim 상세 내용", preprocessing_log)
    log_and_print("="*80 + "\n", preprocessing_log)
    
    sections = patent_data.get('sections', {})
    claims = patent_data.get('claims', [])
    
    log_and_print("\n" + "-"*80, preprocessing_log)
    log_and_print("📄 섹션 내용 (전체)", preprocessing_log)
    log_and_print("-"*80, preprocessing_log)
    for idx, (section_name, section_content) in enumerate(sections.items(), 1):
        log_and_print(f"\n{'='*80}", preprocessing_log)
        log_and_print(f"[섹션 {idx}] {section_name}", preprocessing_log)
        log_and_print(f"{'='*80}", preprocessing_log)
        log_and_print(f"길이: {len(section_content)} 문자", preprocessing_log)
        log_and_print(f"\n전체 내용:", preprocessing_log)
        log_and_print(section_content, preprocessing_log)
        log_and_print(f"\n{'='*80}\n", preprocessing_log)
    
    if claims:
        log_and_print("\n" + "-"*80, preprocessing_log)
        log_and_print("📜 Claims 내용 (전체)", preprocessing_log)
        log_and_print("-"*80, preprocessing_log)
        for claim in claims:
            claim_type = "독립항" if claim.get('independent', False) else "종속항"
            log_and_print(f"\n{'='*80}", preprocessing_log)
            log_and_print(f"[Claim {claim['claim_no']}] ({claim_type})", preprocessing_log)
            log_and_print(f"{'='*80}", preprocessing_log)
            log_and_print(f"길이: {len(claim['claim_text'])} 문자", preprocessing_log)
            log_and_print(f"\n전체 내용:", preprocessing_log)
            log_and_print(claim['claim_text'], preprocessing_log)
            log_and_print(f"\n{'='*80}\n", preprocessing_log)
    else:
        log_and_print("\n⚠️ Claims 없음", preprocessing_log)
    
    log_and_print("\n" + "="*80, preprocessing_log)
    log_and_print("✅ 전처리 결과 출력 완료", preprocessing_log)
    log_and_print("="*80 + "\n", preprocessing_log)

    
    # 4) Document로 변환 (청킹 포함) - 로그 파일 전달
    docs = to_langchain_document(patent_data, source=patent_path, log_file=chunking_log)
    log_and_print(f"Created {len(docs)} documents", chunking_log)
    
    # 5) 모든 문서의 metadata에 patent_id 추가 (필터링을 위해)
    for doc in docs:
        doc.metadata['patent_id'] = current_patent_id
    log_and_print(f"✓ Added patent_id to all {len(docs)} documents", chunking_log)
    
    # 6) Claims 정보 출력
    claims_count = len([d for d in docs if d.metadata.get('granularity') == 'claim'])
    log_and_print(f"Found {claims_count} claims (will be retrieved based on relevance)", chunking_log)
    
    # 7) Vector store 생성
    vectorstore = FAISS.from_documents(docs, emb)
    print("✓ Vector store created")
    
    # 8) Vector store 저장
    save_vectorstore(vectorstore, current_patent_id)
    
    log_and_print(f"\n{'='*80}", preprocessing_log)
    log_and_print(f"✅ Preprocessing complete for {current_patent_id}", preprocessing_log)
    log_and_print(f"📊 Total documents: {len(docs)}", preprocessing_log)
    log_and_print(f"📂 Vectorstore saved to: {get_vectorstore_path(current_patent_id)}", preprocessing_log)
    log_and_print(f"📝 Preprocessing log: {preprocessing_log}", preprocessing_log)
    log_and_print(f"📝 Chunking log: {chunking_log}", preprocessing_log)
    log_and_print(f"{'='*80}\n", preprocessing_log)
    
    return Command(
        update={"preprocessed": True},
        goto="planner"
    )


# =========================
# 6) Tools for Each Agent
# =========================
innovation_tools = [get_available_metadata, search_by_metadata, search_by_similarity]
implementation_tools = [get_available_metadata, search_by_metadata, search_by_similarity]
technical_tools = [get_available_metadata, search_by_metadata, search_by_similarity]
horizontal_tools = [
    get_available_metadata, 
    search_by_metadata, 
    search_by_similarity,
    generate_patent_search_query,
    search_similar_patents_serpapi,
    refine_patent_search_query  # NEW: 쿼리 개선 tool
]


# =========================
# 7) Agents with Prompts
# =========================
innovation_agent = create_react_agent(
    llm,
    tools=innovation_tools,
    prompt=(
        "You are an expert skilled in analyzing patents. "
        "Your task is to identify and describe the key innovation points and distinctive features "
        "that differentiate this patent\n\n"
        "Do not search similar patents\n\n"
        "First, abstract is useful so search for abstract\n\n"
        "IMPORTANT: You have THREE search tools available:\n"
        "- 'get_available_metadata': Check available metadata (sections, claims) FIRST\n"
        "- 'search_by_metadata': ONLY use when you need a specific section's full content or a specific claim number (e.g., 'abstract', 'independent claims', 'get all CLAIMS', 'get claim 1', )\n"
        "- 'search_by_similarity': USE THIS for all conceptual questions like 'innovation points', 'advantages', 'problems solved', 'benefits', or any keyword searches\n\n"
        "TOOL SELECTION RULES:\n"
        "- For questions like 'innovation points', 'advantages', 'problems solved', 'benefits', 'features' → ALWAYS use search_by_similarity\n"
        "- For questions asking about specific concepts/keywords (e.g., 'stepped edge', 'etching conditions') → ALWAYS use search_by_similarity\n"
        "- ⚠️ For questions asking for 'all CLAIMS' or 'claim number X' or 'ABSTRACT' → use search_by_metadata with filters\n"
#         "- When in doubt, use search_by_similarity - it works for most questions\n\n"
        "Examples:\n"
        "- 'innovation points' → search_by_similarity('innovation points of this patent')\n"
        "- 'advantages of stepped edge' → search_by_similarity('advantages of stepped edge')\n"
        "- 'get all dependent claims' → search_by_metadata('', filters=\"metadata['section'] == 'CLAIMS' and metadata['independent'] == False\")\n"
    ),
)

implementation_agent = create_react_agent(
    llm,
    tools=implementation_tools,
    prompt=(
        "You are an expert in the fields of semiconductors, "
        "and you are very skilled at interpreting specific implementation methods in patents. "
        "Your task is to summarize and describe the implementation methods in patents.\n\n"
        "IMPORTANT: You have THREE search tools available:\n"
        "- 'get_available_metadata': Check available metadata (sections, claims) FIRST\n"
        "- 'search_by_metadata': ONLY use when you need a specific section's full content or a specific claim number\n"
        "- 'search_by_similarity': USE THIS for all conceptual searches about methods, processes, fabrication, conditions, and keywords\n\n"
        "TOOL SELECTION RULES:\n"
        "- For questions about 'implementation methods', 'processes', 'fabrication steps', 'conditions' → ALWAYS use search_by_similarity\n"
        "- For questions about specific keywords (e.g., 'HCl etching', 'TMAH developer', 'temperature ranges') → ALWAYS use search_by_similarity\n"
        "- For questions asking for 'all of DETAILED DESCRIPTION section' → use search_by_metadata with filters\n"
        "- When searching for concepts/keywords within a section, use search_by_similarity (NOT search_by_metadata)\n\n"
        "Examples:\n"
        "- 'etching conditions' → search_by_similarity('HCl etching 840-860 C TMAH developer')\n"
        "- 'fabrication method' → search_by_similarity('fabrication method implementation process')\n"
        "- 'get all DETAILED DESCRIPTION' → search_by_metadata('', filters=\"metadata['section'] == 'DETAILED DESCRIPTION'\")\n"
    ),
)

technical_agent = create_react_agent(
    llm,
    tools=technical_tools,
    prompt=(
        "You are an expert in the fields of semiconductors, "
        "and you are very skilled at interpreting technical details and principles in patents. "
        "Your task is to summarize and describe the technical details and principles in patents.\n\n"
        "IMPORTANT: You have THREE search tools available:\n"
        "- 'get_available_metadata': Check available metadata (sections, claims) FIRST\n"
        "- 'search_by_metadata': ONLY use when you need a specific section's full content or a specific claim number\n"
        "- 'search_by_similarity': USE THIS for all conceptual searches about technical specs, principles, materials, conditions, and keywords\n\n"
        "TOOL SELECTION RULES:\n"
        "- For questions about 'technical details', 'principles', 'mechanisms', 'specifications', 'materials' → ALWAYS use search_by_similarity\n"
        "- For questions about specific technical keywords or conditions → ALWAYS use search_by_similarity\n"
        "- For questions asking for 'all of a specific section' → use search_by_metadata with filters\n"
        "- When searching for technical concepts/keywords, use search_by_similarity (NOT search_by_metadata)\n\n"
        "Examples:\n"
        "- 'technical details of etching' → search_by_similarity('etching technical details conditions')\n"
        "- 'material properties' → search_by_similarity('material properties specifications')\n"
        "- 'get all DETAILED DESCRIPTION' → search_by_metadata('', filters=\"metadata['section'] == 'DETAILED DESCRIPTION'\")\n"
    ),
)


horizontal_agent = create_react_agent(
    llm,
    tools=horizontal_tools,
    prompt=(
        "You are an expert in horizontal comparison and analysis. "
        "Your task is to compare the CURRENT patent with similar patents (default: 2 patents, or N patents if user specifies) "
        "and provide a structured comparison report.\n\n"
        
        "IMPORTANT: You have SIX search tools available:\n"
        "- 'get_available_metadata': Check available metadata (sections, claims) FIRST\n"
        "- 'search_by_metadata': ONLY use when you need a specific section's full content or a specific claim number\n"
        "- 'search_by_similarity': USE THIS for all conceptual searches about key features, claims content, and keywords\n"
        "- 'generate_patent_search_query': Generate optimized search query from abstract for finding similar patents\n"
        "- 'search_similar_patents_serpapi': Search Google Patents via SerpAPI to find similar patents\n\n"
        "- 'refine_patent_search_query': Progressively refines and improves search queries:\n"
        "  * Uses GENERAL GUIDELINES to enhance query sophistication\n"
        "  * Provides constructive feedback on improvements made\n"
        "  * Returns JSON with refined_query and feedback fields\n"
        
        
        "HOW TO EXTRACT NUMBER OF PATENTS FROM USER QUERY:\n"
        "- Check if user specifies a number: \"3개\", \"5 patents\", \"find 4 similar\", etc.\n"
        "- Extract the number N from patterns like: N개, N patents, N similar, find N, search N\n"
        "- If no number specified, use default N=2\n"
        "- Use this N value in search_similar_patents_serpapi(query, num_results=N)\n"
        
        "PATENT COMPARISON WORKFLOW:\n"
        "When asked to compare patents or find similar patents, follow this EXACT workflow:\n"
        "1. Extract the current patent's ABSTRACT: search_by_metadata('', filters=\"metadata['section'] == 'ABSTRACT'\")\n"
        "2. Generate initial search query: generate_patent_search_query(abstract_text)\n"
        "3. ⚠️⚠️ CRITICAL: Refine query TWO times to improve sophistication:\n"
        "   - First refinement: refine_patent_search_query(abstract, initial_query, iteration=1)\n"
        "   - Extract refined_query from JSON response\n"
        "   - Second refinement: refine_patent_search_query(abstract, first_refined_query, iteration=2)\n"
        "   - Extract refined_query from JSON response\n"
        "   - Progressive refinement ensures high-quality search query\n"
        "4. Search for similar patents: search_similar_patents_serpapi(final_refined_query, num_results=N) where N=2 by default, or N=user_specified_number if mentioned in query\n"
        "   - ⚠️If NO_RESULTS_FOUND⚠️: refine one more time and search again\n"
        "     * refine_patent_search_query(abstract, final_refined_query, iteration=3)\n"
        "     * search_similar_patents_serpapi(new_refined_query, num_results=N)\n"
        "5. Extract key features from current patent: search_by_similarity('key features innovation points claims')\n"
        "6. Create structured comparison report (see OUTPUT FORMAT below)\n\n"
        
        "OUTPUT FORMAT (MUST FOLLOW THIS STRUCTURE):\n"
        "===== PATENT COMPARISON REPORT =====\n\n"
        "## CURRENT PATENT\n"
        "- Patent ID: [ID]\n"
        "- Abstract: [Full abstract from current patent]\n"
        "- Key Innovation Points: [3-5 bullet points]\n\n"
        
        "## SIMILAR PATENT #1\n"
        "- Patent ID: [ID from search results]\n"
        "- Title: [Title from search results]\n"
        "- Assignee: [Assignee]\n"
        "- Abstract/Summary: [Snippet/summary from search results]\n\n"
        
        "## SIMILAR PATENT #2\n"
        "- Patent ID: [ID from search results]\n"
        "- Title: [Title from search results]\n"
        "- Assignee: [Assignee]\n"
        "- Abstract/Summary: [Snippet/summary from search results]\n\n"
        
        "## COMPARATIVE ANALYSIS (Current Patent vs Similar Patents)\n"
        "### 1. Technical Approach\n"
        "- Current Patent: [approach]\n"
        "- Similar Patent #1: [approach]\n"
        "- Similar Patent #2: [approach]\n\n"
        
        "### 2. Key Differences\n"
        "- What makes the current patent unique: [differences]\n\n"
        
        "### 3. Common Elements\n"
        "- Shared technical concepts: [commonalities]\n\n"
        
        "### 4. Advantages of Current Patent\n"
        "- [List advantages over similar patents]\n\n"
        
        "### 5. Potential Disadvantages\n"
        "- [List any limitations]\n\n"
        
        "CRITICAL RULES:\n"
        "- Search for similar patents (num_results=2 by default, or N if user specifies 'N개', 'N patents', etc.)\n"
        "- Always include abstract/summary for BOTH similar patents\n"
        "- Always center the comparison around the CURRENT patent\n"
        "- Use the OUTPUT FORMAT structure above\n"
        "- Be concise but comprehensive\n\n"
        
        "TOOL SELECTION RULES:\n"
        "- For 'key features', 'technical advantages' → search_by_similarity\n"
        "- For specific claim content → search_by_metadata with filters\n"
        "- For 'all independent/dependent claims' → search_by_metadata with filters\n"
        "- For patent comparison → Follow PATENT COMPARISON WORKFLOW\n"
    ),
)


# =========================
# 8) Task Templates
# =========================
innovation_req = """# Requirement:
- You need to read the patent carefully and give the abstract, innovation, strengths and weaknesses, and application prospects. Answer as much as possible from the relevant direction of the user’s question.
- All your outputs must be truthful and rigorous, rejecting fabrications
- Provide detailed descriptions with quantitative figures from the patent
- The final outputs should be rendered in English

# Task Description:
Analyze the patent (ID: {patent_id}) from multiple perspectives, especially the innovative points.
"""

implementation_req = """# Requirement:
- You need to carefully read the patent content and provide specific implementation methods for the patent.
- Please note that you need to describe the implementation process of the patent in as much detail as possible. You are willing to describe it very clearly and output more text.
- Please note that you need to keep the reference to the image number in the original text during the answering process, for example, you need to add "as shown as Fig…" to each of your answers.
- You are very rigorous and serious, never falsifying information. You can provide specific and accurate numbers to enrich the content. You are willing to output any details related to the patent’s process.
- You only need to provide the implementation method, without outputting any other information like abstract or conclusion.
- The final outputs should be rendered in English

# Task Description:
Only tell me the implementation methods of this patent I will give. You should primarily answer based on the patent content, while also using your own knowledge as a supplement. 
Provide detailed implementation methods of the patent (ID: {patent_id})
"""

technical_req = """# Requirement:
- First, you need to carefully read the patent content.
- Then you need to add some technical details and principles based on the content of the patent. For example, what are the special design ideas, what are the preparation methods of materials, what special environmental conditions are required, and what special devices or technologies are needed, etc.
- Search for: technical details, principles, design specifications, materials, conditions
- You are very rigorous and serious, never falsifying information. You are good at discovering any details of patents. You are willing to describe it very clearly and output more text.
- You can provide specific and accurate numbers to enrich technical details. You are willing to output any details related to the patent’s process.
- You only need to provide the technical details, without outputting any other information like abstract or conclusion.
- The final outputs should be rendered in English

# Task Description:
Only tell me the technical details and principles of this patent I will give. You should primarily answer based on the patent content, while also using your own knowledge as a supplement. 
Answer as detailed as possible, pay attention to providing some real numbers to increase reliability. 
Answer in English. The patent is: {patent_id}
"""

horizontal_req = """# Requirement:
- Follow the PATENT COMPARISON WORKFLOW to create a structured comparison report:
  1. Extract the current patent's ABSTRACT using search_by_metadata
  2. Generate an optimized search query using generate_patent_search_query
  3. ⚠️ CRITICAL: Search for similar patents using search_similar_patents_serpapi(query, num_results=N) where N=2 by default, or N=user_specified_number
  4. Extract key features from the current patent using search_by_similarity
  5. Create a structured comparison report following the OUTPUT FORMAT in your instructions

- MUST include in your report:
  1. Current Patent Section:
     * Patent ID and full abstract
     * 3-5 key innovation points
  
  2. Similar Patent #1 Section:
     * Patent ID, title, assignee
     * Abstract or summary from search results
  
  3. Similar Patent #2 Section:
     * Patent ID, title, assignee
     * Abstract or summary from search results
  
  4. Comparative Analysis Section (Current Patent-centered):
     * Technical approach comparison
     * Key differences (what makes current patent unique)
     * Common elements
     * Advantages of current patent
     * Potential disadvantages

- Comparison MUST be centered around the CURRENT patent
- All outputs must be truthful, rigorous, and based on actual patent content
- Use clear sections and bullet points for readability
- The final outputs should be rendered in English

# Task Description:
Compare the patent (ID: {patent_id}) with similar patents found via Google Patents (default: 2 patents, or N if specified by user).
Provide a structured, current-patent-centered comparison report following the OUTPUT FORMAT.
Focus on what makes the current patent unique and innovative compared to the similar patents.
"""

# =========================
# 8.5) Planner Agent (Intent Detection + Plan 생성)
# =========================

# Intent detection patterns (기존 supervisor에서 이동)
KOR_INTENT = {
    "innovation": [
        r"혁신\s*포인트", r"혁신\s*점", r"차별화", r"novel", r"innovation", r"핵심", r"특징"
    ],
    "implementation": [
        r"구현", r"공정", r"절차", r"방법", r"implementation", r"process", r"제조"
    ],
    "technical": [
        r"기술\s*세부", r"원리", r"메커니즘", r"technical", r"principle", r"상세"
    ],
    "horizontal": [
        r"비교", r"수평", r"유사\s*특허", r"similar", r"compare", r"대조", r"검색"
    ],
}

def detect_intents(text: str) -> List[str]:
    """텍스트에서 여러 intent를 감지"""
    detected = []
    
    checks = [
        ("innovation_agent", KOR_INTENT["innovation"]),
        ("implementation_agent", KOR_INTENT["implementation"]),
        ("technical_agent", KOR_INTENT["technical"]),
        ("horizontal_agent", KOR_INTENT["horizontal"]),
    ]
    
    for agent, patterns in checks:
        for p in patterns:
            if re.search(p, text, re.IGNORECASE):
                if agent not in detected:
                    detected.append(agent)
                break
    
    return detected

PLANNER_SYSTEM_PROMPT = """You are an expert task planner for patent analysis workflows.

Your role:
1. Analyze the user's query to understand what they want to know about the patent
2. Decompose complex queries into minimal independent sub-tasks
3. Create a DAG (Directed Acyclic Graph) of tasks with proper dependencies
4. Assign each task to the appropriate expert agent

Available Agents:
- innovation_agent: Identifies innovation points, key features, advantages, novelty
- implementation_agent: Explains implementation methods, processes, fabrication steps, procedures
- technical_agent: Describes technical details, principles, specifications, mechanisms
- horizontal_agent: Compares with other similar patents using Google Patents search, identifies unique aspects and differences

Task Planning Guidelines:
1. **Intent Detection**: Carefully analyze what the user is asking for
   - Keywords like "혁신", "innovation", "핵심", "특징" → innovation_agent
   - Keywords like "구현", "방법", "공정", "제조" → implementation_agent  
   - Keywords like "기술", "원리", "메커니즘", "상세" → technical_agent
   - Keywords like "비교", "compare", "유사", "다른 특허", "similar patents" → horizontal_agent

2. **Single Agent for Simple Queries**: If the query asks for only ONE thing, create only ONE task
   - "이 특허의 혁신 포인트를 알려줘" → Only innovation_agent (1 task)
   - "구현 방법을 설명해줘" → Only implementation_agent (1 task)

3. **Multiple Agents for Complex Queries**: If query asks for MULTIPLE things, create multiple tasks
   - "혁신 포인트와 구현 방법을 알려줘" → innovation_agent + implementation_agent (2 tasks)
   - Task order matters: innovation usually comes before implementation

4. **Dependencies**: 
   - If one task needs results from another, use depends_on
   - If tasks are independent, mark parallelizable=true

5. **Clear Inputs**:
   - First task: Use {"query": "specific question for this agent"}
   - Dependent tasks: Use {"use_result_from": "T1"} or {"query": "..."}

Output Format (JSON only):
{{
  "goal": "Clear description of what user wants to know",
  "tasks": [
    {{
      "task_id": "T1",
      "description": "Brief task description",
      "agent": "innovation_agent",
      "depends_on": [],
      "parallelizable": true,
      "max_retries": 2,
      "inputs": {{"query": "Specific question"}}
    }}
  ]
}}

CRITICAL RULES:
- Output ONLY valid JSON, no other text
- For simple queries, create ONLY ONE task
- Match agent to user's actual question
- Task IDs: T1, T2, T3... (sequential)
- Validate dependencies (no cycles)
"""


class PlanOutput(TypedDict):
    """Planner의 구조화된 출력"""
    goal: str
    tasks: List[Task]

def planner_node(state: State) -> Command[Literal["supervisor"]]:
    """
    Planner Agent: Query 분석 + Plan 생성
    종합 분석 요청 시 모든 agent를 순차적으로 실행
    """
    print("\n" + "="*80)
    print("🧠 PLANNER - Plan Creation")
    print("="*80)
    
    # 사용자 쿼리 추출
    user_query = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break
    
    print(f"\n📝 User Query: {user_query}")
    print(f"📄 Patent ID: {state['patent_id']}")
    
    # ============================================
    # 🆕 종합 분석 패턴 감지
    # ============================================
    comprehensive_patterns = [
        r"종합\s*분석",
        r"전체\s*분석", 
        r"모든\s*것",
        r"모두\s*분석",
        r"완전한\s*분석",
        r"comprehensive\s+analys",
        r"complete\s+analys",
        r"full\s+analys",
    ]
    
    # 일반적인 "분석해줘" 패턴 (다른 키워드가 없을 때만)
    simple_analysis_patterns = [
        r"^분석해줘?\.?$",
        r"^분석\s*해\s*줘?\.?$",
        r"^analyz",
        r"^tell\s+me\s+about",
        r"^explain\s+this\s+patent"
    ]
    
    is_comprehensive = False
    
    # 종합 분석 패턴 체크
    for pattern in comprehensive_patterns:
        if re.search(pattern, user_query, re.IGNORECASE):
            is_comprehensive = True
            print("\n🎯 Comprehensive analysis request detected!")
            break
    
    # 일반 분석 패턴 체크 (특정 키워드가 없을 때만)
    if not is_comprehensive:
        for pattern in simple_analysis_patterns:
            if re.search(pattern, user_query, re.IGNORECASE):
                # 다른 특정 키워드가 없는지 확인
                specific_keywords = ["혁신", "구현", "기술", "비교", "innovation", "implementation", "technical", "compare", "horizontal"]
                has_specific = any(kw in user_query.lower() for kw in specific_keywords)
                
                if not has_specific:
                    is_comprehensive = True
                    print("\n🎯 General analysis request detected (no specific agent mentioned)!")
                    break
    
    # ============================================
    # 종합 분석 Plan 자동 생성
    # ============================================
    if is_comprehensive:
        print("\n📋 Creating comprehensive analysis plan with all agents...")
        
        comprehensive_plan = Plan(
            goal="Comprehensive analysis of the patent covering all aspects: innovation points, implementation methods, technical details, and comparison with similar patents",
            tasks=[
                Task(
                    task_id="T1",
                    description="Analyze innovation points and key features",
                    agent="innovation_agent",
                    depends_on=[],
                    parallelizable=False,
                    max_retries=2,
                    inputs={"query": "Analyze the innovation points, key features, advantages, and distinctive aspects of this patent in detail."}
                ),
                Task(
                    task_id="T2",
                    description="Explain implementation methods and processes",
                    agent="implementation_agent",
                    depends_on=["T1"],
                    parallelizable=False,
                    max_retries=2,
                    inputs={"query": "Describe the implementation methods, fabrication processes, and procedural steps of this patent in detail. Include figure references where applicable."}
                ),
                Task(
                    task_id="T3",
                    description="Describe technical details and principles",
                    agent="technical_agent",
                    depends_on=["T2"],
                    parallelizable=False,
                    max_retries=2,
                    inputs={"query": "Explain the technical details, principles, mechanisms, material specifications, and design considerations of this patent in detail."}
                ),
                Task(
                    task_id="T4",
                    description="Compare with similar patents",
                    agent="horizontal_agent",
                    depends_on=["T3"],
                    parallelizable=False,
                    max_retries=2,
                    inputs={"query": "Find similar patents (2 patents) and create a comprehensive comparison report highlighting what makes the current patent unique and innovative."}
                )
            ]
        )
        
        print("\n✅ Comprehensive Plan Created!")
        print("\n" + "="*80)
        print("📋 COMPREHENSIVE EXECUTION PLAN")
        print("="*80)
        print(f"🎯 Goal: {comprehensive_plan['goal']}")
        print(f"📊 Total Tasks: {len(comprehensive_plan['tasks'])}")
        print("\n" + "-"*80)
        
        for i, task in enumerate(comprehensive_plan['tasks'], 1):
            deps = ", ".join(task['depends_on']) if task['depends_on'] else "None"
            
            print(f"\n[Task {i}] {task['task_id']}: {task['description']}")
            print(f"  Agent: {task['agent']}")
            print(f"  Dependencies: {deps}")
            print(f"  Input: {task['inputs']['query'][:80]}...")
            print("  " + "-"*40)
        
        print("\n" + "="*80)
        print("🚀 Starting comprehensive analysis...")
        print("="*80)
        
        return Command(
            update={
                "plan": comprehensive_plan,
                "task_results": {},
                "completed_tasks": set(),
                "failed_tasks": {},
                "current_iteration": 0
            },
            goto="supervisor"
        )
    
    # ============================================
    # 일반 Plan 생성 (기존 로직)
    # ============================================
    
    # Intent detection
    detected_intents = detect_intents(user_query)
    if detected_intents:
        print(f"\n🔍 Detected Intents: {', '.join(detected_intents)}")
    else:
        print("\n🔍 No specific intents detected, will use LLM judgment")
    
    # Plan 생성
    planner_messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"""
User Query: {user_query}
Patent ID: {state['patent_id']}

Detected intents: {detected_intents if detected_intents else 'None - use your judgment'}

Analyze the query carefully and create an appropriate execution plan.
For simple queries asking for ONE thing, create only ONE task.
Output ONLY the JSON plan.
""")
    ]
    
    # LLM 호출
    try:
        response = llm.with_structured_output(PlanOutput).invoke(planner_messages)
        plan = Plan(goal=response["goal"], tasks=response["tasks"])
        
        print("\n✅ Plan Created Successfully!")
        print("\n" + "="*80)
        print("📋 EXECUTION PLAN")
        print("="*80)
        print(f"🎯 Goal: {plan['goal']}")
        print(f"📊 Total Tasks: {len(plan['tasks'])}")
        print("\n" + "-"*80)
        
        for i, task in enumerate(plan['tasks'], 1):
            deps = ", ".join(task['depends_on']) if task['depends_on'] else "None"
            parallel = "✓ Yes" if task['parallelizable'] else "✗ No"
            
            print(f"\n[Task {i}] {task['task_id']}: {task['description']}")
            print(f"  Agent: {task['agent']}")
            print(f"  Dependencies: {deps}")
            print(f"  Parallelizable: {parallel}")
            
            # Inputs 출력
            if 'use_result_from' in task['inputs']:
                print(f"  Input: Uses result from {task['inputs']['use_result_from']}")
            elif 'query' in task['inputs']:
                query_preview = task['inputs']['query'][:60] + "..." if len(task['inputs']['query']) > 60 else task['inputs']['query']
                print(f"  Input: {query_preview}")
            
            print("  " + "-"*40)
        
        print("\n" + "="*80)
        print("🚀 Starting execution...")
        print("="*80)
        
        return Command(
            update={
                "plan": plan,
                "task_results": {},
                "completed_tasks": set(),
                "failed_tasks": {},
                "current_iteration": 0
            },
            goto="supervisor"
        )
        
    except Exception as e:
        print(f"\n❌ Error creating plan: {e}")
        
        # Fallback: 가장 기본적인 플랜
        # Intent detection 결과 활용
        if detected_intents:
            agent = detected_intents[0]  # 첫 번째 intent 사용
        else:
            agent = "innovation_agent"  # 기본값
        
        fallback_plan = Plan(
            goal=user_query,
            tasks=[
                Task(
                    task_id="T1",
                    description=user_query,
                    agent=agent,
                    depends_on=[],
                    parallelizable=True,
                    max_retries=2,
                    inputs={"query": user_query}
                )
            ]
        )
        
        print(f"\n⚠️ Using fallback plan: Single task with {agent}")
        
        return Command(
            update={
                "plan": fallback_plan,
                "task_results": {},
                "completed_tasks": set(),
                "failed_tasks": {},
                "current_iteration": 0
            },
            goto="supervisor"
        )

# =========================
# 9) Node Runner (기존 + Task 기반 확장)
# =========================
def _run_agent(agent, task_template: str, state: State, name: str) -> Command[Literal["supervisor"]]:
    """기존 방식의 agent 실행 (legacy, 호환성 유지)"""
    task = task_template.format(patent_id=state["patent_id"])
    user_msg = HumanMessage(content=task, name=name)
    
    result = agent.invoke({"messages": state["messages"] + [user_msg]})
    
    completed = state.get("completed_agents", [])
    return Command(
        update={
            "messages": result["messages"],
            "completed_agents": completed + [name] if name not in completed else completed
        },
        goto="supervisor"
    )

def _run_task(agent, task: Task, task_template: str, state: State, name: str) -> Command[Literal["supervisor"]]:
    """NEW: Task 기반 agent 실행 (템플릿 사용) - 이전 결과를 참고"""
    task_id = task['task_id']
    task_results = state.get("task_results", {})
    previous_context = state.get("previous_context", "")
    
    print(f"\n{'='*80}")
    print(f"🤖 [{name}] Executing Task: {task_id}")
    print(f"{'='*80}")
    print(f"📝 Description: {task['description']}")
    
    # 1. 기존 템플릿의 기본 요구사항 적용
    base_requirements = task_template.format(patent_id=state["patent_id"])
    
    # 2. Task의 구체적인 입력 구성
    task_inputs = task['inputs']
    
    if 'use_result_from' in task_inputs:
        # 이전 task 결과 사용
        ref_task_id = task_inputs['use_result_from']
        if ref_task_id in task_results:
            context = task_results[ref_task_id]
            specific_query = f"{task['description']}\n\nContext from previous task [{ref_task_id}]:\n{context}"
            print(f"\n📎 Using result from: {ref_task_id}")
        else:
            specific_query = task['description']
            print(f"\n⚠️ Warning: Referenced task {ref_task_id} result not found")
    elif 'query' in task_inputs:
        specific_query = task_inputs['query']
    else:
        specific_query = task['description']
    
    # 3. 이전 task들의 context 추가 (있으면)
    if previous_context:
        specific_query = f"{specific_query}{previous_context}"
        print(f"\n📚 Including context from previous tasks")
    
    # 4. 최종 쿼리 = 기본 템플릿 + 구체적인 task 내용 + 이전 context
    final_query = f"{base_requirements}\n\n# Specific Task for this execution:\n{specific_query}"
    
    print(f"\n💬 Using template: {name}")
    print(f"📋 Task-specific query: {specific_query[:100]}..." if len(specific_query) > 100 else f"📋 Task-specific query: {specific_query}")
    
    # Agent 실행
    try:
        user_msg = HumanMessage(content=final_query, name=name)
        
        result = agent.invoke({"messages": state["messages"] + [user_msg]})
        
        # 결과 추출
        if result and "messages" in result and result["messages"]:
            last_msg = result["messages"][-1]
            output = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
        else:
            output = "No response from agent"
        
        print(f"\n✅ Task {task_id} completed successfully")
        print(f"📊 Output length: {len(output)} characters")
        
        # Update completed tasks
        new_completed = state.get("completed_tasks", set()) | {task_id}
        print(f"✓ Marking task as completed: {task_id}")
        print(f"✓ Total completed tasks: {len(new_completed)} - {sorted(new_completed)}")
        print(f"{'='*80}\n")
        
        return Command(
            update={
                "task_results": {**task_results, task_id: output},
                "completed_tasks": new_completed,
                "messages": result["messages"]
            },
            goto="supervisor"
        )
        
    except Exception as e:
        error_msg = f"Error in {name}: {str(e)}"
        print(f"\n❌ {error_msg}")
        print(f"{'='*80}\n")
        
        failed = state.get("failed_tasks", {})
        failed[task_id] = error_msg
        
        return Command(
            update={"failed_tasks": failed},
            goto="supervisor"
        )


# =========================
# 10) Agent Nodes
# =========================
def innovation_node(state: State) -> Command[Literal["supervisor"]]:
    # Task 기반 실행인지 확인
    if "task" in state:
        return _run_task(innovation_agent, state["task"], innovation_req, state, "innovation_agent")
    # Legacy 방식
    return _run_agent(innovation_agent, innovation_req, state, "innovation_agent")

def implementation_node(state: State) -> Command[Literal["supervisor"]]:
    # Task 기반 실행인지 확인
    if "task" in state:
        return _run_task(implementation_agent, state["task"], implementation_req, state, "implementation_agent")
    # Legacy 방식
    return _run_agent(implementation_agent, implementation_req, state, "implementation_agent")

def technical_node(state: State) -> Command[Literal["supervisor"]]:
    # Task 기반 실행인지 확인
    if "task" in state:
        return _run_task(technical_agent, state["task"], technical_req, state, "technical_agent")
    # Legacy 방식
    return _run_agent(technical_agent, technical_req, state, "technical_agent")

def horizontal_node(state: State) -> Command[Literal["supervisor"]]:
    # Task 기반 실행인지 확인
    if "task" in state:
        return _run_task(horizontal_agent, state["task"], horizontal_req, state, "horizontal_agent")
    # Legacy 방식
    return _run_agent(horizontal_agent, horizontal_req, state, "horizontal_agent")


# =========================
# 11) Supervisor (Plan 전용 - Merge + 결과 요약)
# =========================

MAX_SUPERVISOR_ITERATIONS = 10  # 무한루프 방지

def get_ready_tasks(plan: Plan, completed: Set[str], failed: Set[str]) -> List[Task]:
    """실행 가능한 task들을 반환 (의존성이 모두 완료된 task)"""
    ready = []
    
    for task in plan['tasks']:
        task_id = task['task_id']
        
        # 이미 완료되었거나 실패한 task는 제외
        if task_id in completed or task_id in failed:
            continue
        
        # 의존성 확인
        deps = task['depends_on']
        
        # 의존성이 없거나, 모든 의존성이 완료된 경우
        if not deps or all(dep in completed for dep in deps):
            ready.append(task)
    
    return ready

def merge_task_results(plan: Plan, task_results: Dict[str, Any], completed: Set[str]) -> str:
    """완료된 task들의 결과를 통합"""
    
    print("\n" + "-"*80)
    print("📝 Merging task results...")
    print("-"*80)
    
    merged = f"# Analysis of Patent\n\n"
    merged += f"**Goal**: {plan['goal']}\n\n"
    merged += "---\n\n"
    
    # 결과 추가
    if task_results:
        print(f"  Including {len(completed)} results from execution")
        for task_id in sorted(completed):
            task = next((t for t in plan['tasks'] if t['task_id'] == task_id), None)
            if task and task_id in task_results:
                merged += f"### {task['description']}\n"
                merged += f"*(Analyzed by {task['agent']})*\n\n"
                merged += task_results[task_id]
                merged += "\n\n"
    
    print(f"✓ Total sections merged: {len(completed)}")
    print(f"✓ Total length: {len(merged)} characters")
    
    return merged

FINAL_SUMMARIZATION_PROMPT = """You are finalizing the analysis results for the user.

You have completed multiple tasks analyzing a patent, and the results have been merged together.
However, the merged result may contain:
- Redundant information from multiple executions
- Poor structure or organization
- Difficult-to-read formatting

Your task:
Transform the merged results into a clear, well-structured, easy-to-read final report.

Guidelines:
1. **Use Plan Structure**: Organize the report following the execution plan's task descriptions
   - Use numbered sections (1., 2., 3., etc.) for each task
   - Use the task description as the section title
   - Clearly indicate which agent analyzed each section

2. **Remove Redundancy**: If the same information appears multiple times, keep only the best version

3. **Enhance Readability**: 
   - Use proper markdown formatting
   - Add clear headers and subheaders
   - Use bullet points for lists
   - Add transitions between sections

4. **Preserve All Key Information**: Don't omit important details, just reorganize them

5. **Maintain Accuracy**: Don't change technical facts or add information not in the source

Original User Query:
{user_query}

Execution Plan:
{plan_info}

Merged Results (may contain redundancy):
{merged_results}

Output a polished, professional final report with the following structure:
- Brief introduction referencing the user's query
- Numbered sections (1., 2., 3., etc.) following the execution plan's task descriptions
- Each section should clearly show which agent performed the analysis
- Clear, well-organized content with proper markdown formatting
- Brief conclusion or summary if appropriate

Use markdown formatting with clear headers and structure.
"""

def supervisor_node(state: State):
    """
    Supervisor: Plan 실행 → 결과 Merge → 최종 요약
    """
    # 전처리 확인
    if not state.get("preprocessed", False):
        print("\n❌ ERROR: Preprocessing not complete")
        return Command(goto=END)
    
    plan = state.get('plan')
    
    if not plan:
        print("\n❌ ERROR: No plan available")
        return Command(goto=END)
    
    print("\n" + "="*80)
    print(f"🎮 SUPERVISOR - Plan Execution (Iteration {state.get('current_iteration', 0) + 1})")
    print("="*80)
    
    # 무한루프 방지
    iteration = state.get('current_iteration', 0)
    if iteration >= MAX_SUPERVISOR_ITERATIONS:
        print("\n⚠️ Max iterations reached. Stopping execution.")
        return Command(goto=END)
    
    completed = state.get('completed_tasks', set())
    failed = state.get('failed_tasks', {})
    task_results = state.get('task_results', {})
    
    total_tasks = len(plan['tasks'])
    print(f"\n📊 Progress: {len(completed)}/{total_tasks} completed, {len(failed)} failed")
    
    if completed:
        print(f"   ✅ Completed: {', '.join(sorted(completed))}")
    if failed:
        print(f"   ❌ Failed: {', '.join(failed.keys())}")
    
    # 실행 가능한 task 찾기
    ready_tasks = get_ready_tasks(plan, completed, failed)
    
    # ===================================
    # Case 1: 아직 실행할 task가 남음
    # ===================================
    if ready_tasks:
        print(f"\n🚀 Ready to execute: {len(ready_tasks)} task(s)")
        for task in ready_tasks:
            deps_str = ", ".join(task['depends_on']) if task['depends_on'] else "None"
            print(f"   → [{task['task_id']}] {task['agent']} (deps: {deps_str})")
        
        # 순차 실행: 첫 번째 task만 실행
        task = ready_tasks[0]
        print(f"\n📌 Executing task sequentially: {task['task_id']}")
        
        # 이전 완료된 task들의 결과를 context로 전달
        previous_context = ""
        if completed:
            previous_context = "\n\n# Context from Previously Completed Tasks:\n"
            for prev_task_id in sorted(completed):
                if prev_task_id in task_results:
                    prev_task = next((t for t in plan['tasks'] if t['task_id'] == prev_task_id), None)
                    if prev_task:
                        previous_context += f"\n## [{prev_task_id}] {prev_task['description']}\n"
                        previous_context += task_results[prev_task_id] + "...\n"  # Summary for context
        
        task_state = {
            "messages": state["messages"],
            "patent_id": state["patent_id"],
            "preprocessed": state["preprocessed"],
            "plan": state["plan"],
            "task": task,
            "task_results": state.get("task_results", {}),
            "completed_tasks": state.get("completed_tasks", set()),
            "failed_tasks": state.get("failed_tasks", {}),
            "current_iteration": state.get("current_iteration", 0),
            "previous_context": previous_context  # 이전 결과 context
        }
        
        print(f"\n📤 Dispatching task {task['task_id']} to {task['agent']}...")
        if completed:
            print(f"   📎 Including context from {len(completed)} previous task(s)")
        print("="*80)
        
        return Command(
            update={"current_iteration": iteration + 1},
            goto=Send(node=task['agent'], arg=task_state)
        )
    
    # ===================================
    # Case 2: 모든 task 완료 - Merge & 최종 요약
    # ===================================
    if len(completed) + len(failed) == total_tasks:
        print("\n✅ All tasks processed!")
        
        # 실패한 task가 너무 많으면 그냥 종료
        if len(failed) > len(completed):
            print("\n⚠️ More failed than completed. Ending.")
            return Command(goto=END)
        
        # 결과 Merge
        merged_result = merge_task_results(plan, task_results, completed)
        
        # 최종 요약 수행
        print("\n" + "-"*80)
        print("✅ Performing final summarization...")
        print("-"*80)
        
        user_query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break
        
        # Plan 정보를 포맷팅
        plan_info = f"Goal: {plan['goal']}\n\nTasks:\n"
        for i, task in enumerate(plan['tasks'], 1):
            plan_info += f"{i}. [{task['task_id']}] {task['description']} (Agent: {task['agent']})\n"
        
        summarization_messages = [
            SystemMessage(content=FINAL_SUMMARIZATION_PROMPT.format(
                user_query=user_query,
                plan_info=plan_info,
                merged_results=merged_result
            )),
            HumanMessage(content="Please create a clear, well-structured final report following the execution plan structure.")
        ]
        
        try:
            final_response = llm.invoke(summarization_messages)
            final_result = final_response.content if hasattr(final_response, 'content') else str(final_response)
            
            print("\n✓ Final summarization complete")
            print(f"✓ Final result length: {len(final_result)} characters")
            
        except Exception as e:
            print(f"\n⚠️ Error in final summarization: {e}")
            print("   Using merged result as-is")
            final_result = merged_result
        
        return Command(
            update={
                "messages": [AIMessage(content=final_result)],
                "merged_result": final_result
            },
            goto=END
        )




# =========================
# 12) Graph (기존 + Planner 추가)
# =========================
graph_builder = StateGraph(State)

# 노드 추가
graph_builder.add_node("preprocess", preprocess_node)
graph_builder.add_node("planner", planner_node)  # NEW
graph_builder.add_node("supervisor", supervisor_node)
graph_builder.add_node("innovation_agent", innovation_node)
graph_builder.add_node("implementation_agent", implementation_node)
graph_builder.add_node("technical_agent", technical_node)
graph_builder.add_node("horizontal_agent", horizontal_node)

# 엣지 설정: START -> preprocess -> planner -> supervisor -> agents -> supervisor -> ...
graph_builder.add_edge(START, "preprocess")
# preprocess에서 planner로 (preprocess_node에서 Command로 지정)
# planner에서 supervisor로 (planner_node에서 Command로 지정)
# supervisor에서 agents로 (Send 사용) 또는 END로
# agents에서 supervisor로 (_run_task에서 Command로 지정)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

print("✅ Agent 및 Graph 구성 완료 (Planner 포함)")


# =========================
# Query Runner from Cell 5
# =========================
# =========================
# 13) Test Runner
# =========================
import nest_asyncio
import asyncio
nest_asyncio.apply()

CONFIG = {"configurable": {"thread_id": "1"}}

async def run_query(user_input: str, patent_id: str):
    """
    사용자 질문을 처리합니다.
    
    Args:
        user_input: 사용자 질문 (한국어/영어)
        patent_id: 특허 ID (예: US8526476)
    """
    init_state = {
        "messages": [HumanMessage(content=user_input)],
        "patent_id": patent_id,
        "preprocessed": False,
        "plan": None,
        "task_results": {},
        "completed_tasks": set(),
        "failed_tasks": {},
        "current_iteration": 0,
        "merged_result": "",
        "replan_feedback": "",
        "plan_iteration": 0,
        "next": ""
    }

    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#  🚀 고도화된 LangGraph 멀티에이전트 시스템 실행" + " "*26 + "#")
    print("#" + " "*78 + "#")
    print("#"*80)
    print(f"\n💬 User Query: {user_input}")
    print(f"📄 Patent: {patent_id}")
    print("\n" + "="*80 + "\n")
    async for namespace, chunk in graph.astream(
        init_state,
        stream_mode="updates",
        subgraphs=True,
        config=CONFIG,
    ):
        for node_name, node_chunk in chunk.items():
            print(f"\n--- [{node_name}] ---")
            # node_chunk가 None이거나 dict가 아닌 경우 스킵
            if node_chunk is None or not isinstance(node_chunk, dict):
                print(f"(No update data)")
                continue
            
            if "messages" in node_chunk and node_chunk["messages"]:
                # 🔥 수정: 모든 메시지 출력
                messages = node_chunk["messages"]
                
                # ToolMessage들을 찾아서 모두 출력
                from langchain_core.messages import ToolMessage
                tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
                
                if tool_messages:
                    print(f"\n📊 총 {len(tool_messages)}개의 Tool 호출 결과:")
                    for i, tool_msg in enumerate(tool_messages, 1):
                        print(f"\n{'='*80}")
                        print(f"Tool Result #{i}")
                        print(f"{'='*80}")
                        try:
                            tool_msg.pretty_print()
                        except Exception:
                            print(tool_msg.content)
                else:
                    # Tool 메시지가 아닌 경우 마지막 메시지만 출력 (기존 로직)
                    try:
                        messages[-1].pretty_print()
                    except Exception:
                        print(getattr(messages[-1], "content", messages[-1]))
            else:
                print(node_chunk)
    
    # 최종 상태 반환
    final_state = await graph.aget_state(config=CONFIG)
    return final_state.values if final_state else init_state