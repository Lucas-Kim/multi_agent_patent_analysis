"""
Patent Analysis System - Streamlit App (Enhanced UI)
"""
import streamlit as st
import os
import asyncio
from pathlib import Path
import tempfile
from datetime import datetime
import json

# agent_logic에서 필요한 함수들 import
from agent_logic import (
    vectorstore_exists,
    get_vectorstore_path,
    load_vectorstore,
    save_vectorstore,
    run_query,
    load_pages_with_first_page_columns,
    to_langchain_document,
    create_log_file,
    log_and_print,
    simple_rag_chatbot  # 챗봇 함수 추가
)
from config import VECTORSTORE_DIR, emb, llm
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage

# 페이지 설정
st.set_page_config(
    page_title="Patent Analysis System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS로 세련된 UI 적용
st.markdown("""
<style>
    /* 전체 배경 */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* 제목 스타일 */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1e40af 0%, #0f172a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .sub-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2d3748;
        margin: 1.5rem 0 0.5rem 0;
    }
    
    /* 카드 스타일 */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        margin: 1rem 0;
        border-left: 4px solid #1e40af;
    }
    
    .success-card {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        margin: 1rem 0;
        color: #1a4d2e;
        font-weight: 500;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        margin: 1rem 0;
        color: #7c2d12;
        font-weight: 500;
    }
    
    /* 버튼 스타일 개선 */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    /* Primary 버튼 (실행 버튼) */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1e40af 0%, #0f172a 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(30, 64, 175, 0.4);
    }
    
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 16px rgba(30, 64, 175, 0.6);
        transform: translateY(-2px);
    }
    
    /* Secondary 버튼 */
    .stButton > button[kind="secondary"] {
        background: white;
        color: #1e40af;
        border: 2px solid #1e40af;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: #1e40af;
        color: white;
        transform: translateY(-2px);
    }
    
    /* 텍스트 영역 스타일 */
    .stTextArea textarea {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #1e40af;
        box-shadow: 0 0 0 3px rgba(30, 64, 175, 0.1);
    }
    
    /* 사이드바 스타일 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e40af 0%, #0f172a 100%);
    }
    
    [data-testid="stSidebar"] .element-container {
        color: white;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: white !important;
    }
    
    /* Expander 스타일 */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 8px;
        font-weight: 600;
        color: #2d3748;
    }
    
    /* Select box 스타일 */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
    }
    
    /* 파일 업로더 스타일 */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 1rem;
        border: 2px dashed rgba(255, 255, 255, 0.5);
    }
    
    /* Browse files 버튼만 검정색 텍스트 */
    [data-testid="stFileUploader"] button {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    
    /* 메트릭 스타일 */
    [data-testid="stMetric"] {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Popover 스타일 */
    [data-testid="stPopover"] button {
        background: linear-gradient(135deg, #1e40af 0%, #0f172a 100%);
        color: white;
        border: none;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        font-size: 1.2rem;
        box-shadow: 0 2px 8px rgba(30, 64, 175, 0.3);
        transition: all 0.3s ease;
    }
    
    [data-testid="stPopover"] button:hover {
        transform: scale(1.1);
        box-shadow: 0 4px 12px rgba(30, 64, 175, 0.5);
    }
    
    /* Popover 내용 스타일 */
    [data-testid="stPopover"] > div > div {
        background: white;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
        border: none;
    }
    
    /* Footer 스타일 */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #718096;
        font-size: 0.9rem;
    }
    
    /* 호버 효과 */
    .hover-scale {
        transition: transform 0.3s ease;
    }
    
    .hover-scale:hover {
        transform: scale(1.02);
    }
    
    /* 로딩 애니메이션 */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
</style>
""", unsafe_allow_html=True)

# 세션 state 초기화
# 특허별로 히스토리를 독립적으로 유지하기 위해 딕셔너리 사용
if 'query_history_by_patent' not in st.session_state:
    st.session_state.query_history_by_patent = {}  # {patent_id: [history_items]}
if 'chatbot_history_by_patent' not in st.session_state:
    st.session_state.chatbot_history_by_patent = {}  # {patent_id: [chat_messages]}
if 'current_patent_id' not in st.session_state:
    st.session_state.current_patent_id = None
if 'preprocessed' not in st.session_state:
    st.session_state.preprocessed = False
# 입력창 초기화를 위한 카운터
if 'query_input_counter' not in st.session_state:
    st.session_state.query_input_counter = 0
if 'chatbot_input_counter' not in st.session_state:
    st.session_state.chatbot_input_counter = 0

# 현재 특허의 히스토리를 가져오는 헬퍼 함수
def get_current_query_history():
    """현재 선택된 특허의 query history 반환"""
    if st.session_state.current_patent_id is None:
        return []
    if st.session_state.current_patent_id not in st.session_state.query_history_by_patent:
        st.session_state.query_history_by_patent[st.session_state.current_patent_id] = []
    return st.session_state.query_history_by_patent[st.session_state.current_patent_id]

def get_current_chatbot_history():
    """현재 선택된 특허의 chatbot history 반환"""
    if st.session_state.current_patent_id is None:
        return []
    if st.session_state.current_patent_id not in st.session_state.chatbot_history_by_patent:
        st.session_state.chatbot_history_by_patent[st.session_state.current_patent_id] = []
    return st.session_state.chatbot_history_by_patent[st.session_state.current_patent_id]

def get_patent_list():
    """Vector DB에 저장된 특허 목록 반환"""
    if not os.path.exists(VECTORSTORE_DIR):
        return []
    
    patents = []
    for item in os.listdir(VECTORSTORE_DIR):
        item_path = os.path.join(VECTORSTORE_DIR, item)
        if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "index.faiss")):
            patents.append(item)
    return sorted(patents)

def extract_patent_id_from_filename(filename):
    """파일명에서 특허 ID 추출 (예: US8526476.pdf -> US8526476)"""
    return Path(filename).stem

def save_uploaded_file(uploaded_file):
    """업로드된 파일을 임시 디렉토리에 저장"""
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

async def process_preprocess(pdf_path, patent_id):
    """전처리 실행"""
    try:
        # 로그 파일 생성
        preprocessing_log = create_log_file(patent_id, "preprocessing")
        chunking_log = create_log_file(patent_id, "chunking")
        
        log_and_print(f"{'='*80}", preprocessing_log)
        log_and_print(f"전처리 시작: {patent_id}", preprocessing_log)
        log_and_print(f"시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", preprocessing_log)
        log_and_print(f"{'='*80}\n", preprocessing_log)
        
        # 1) PDF 로드 (첫 페이지 칼럼 분리)
        pages = load_pages_with_first_page_columns(pdf_path)
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

예시 섹션 목록 (하지만 이에 국한되지 않음):
- ABSTRACT
- BACKGROUND OF THE INVENTION
- BRIEF SUMMARY OF THE INVENTION
- DETAILED DESCRIPTION
- BRIEF DESCRIPTION OF THE DRAWINGS
- DESCRIPTION OF THE PREFERRED EMBODIMENTS
등

3. Claims 추출:
Claims 섹션에서 각 청구항을 개별 항목으로 추출하세요.
각 청구항(claim)은 다음 정보를 포함해야 합니다:
- claim_no: 청구항 번호 (예: "1", "2", ...)
- claim_text: 청구항의 전체 텍스트 (원문 그대로)
- independent: 독립항 여부 (true/false). "comprising" 또는 "consisting"을 포함하면서 다른 청구항을 참조하지 않으면 독립항.

반환 형식:
반드시 JSON 형식으로 반환하세요:
{
  "metadata": {
    "patent_number": "...",
    "publication_number": "...",
    ...
  },
  "sections": {
    "ABSTRACT": "...",
    "BACKGROUND OF THE INVENTION": "...",
    ...
  },
  "claims": [
    {
      "claim_no": "1",
      "claim_text": "...",
      "independent": true
    },
    ...
  ]
}
"""
        
        user_content = f"특허 문서 전문:\n\n{full_text}"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ]
        
        log_and_print("Calling LLM for preprocessing...", preprocessing_log)
        response = await llm.ainvoke(messages)
        result_text = response.content
        log_and_print(f"✓ LLM response received ({len(result_text)} characters)", preprocessing_log)
        
        # 4) JSON 파싱
        try:
            # ```json ... ``` 형태로 감싸진 경우 처리
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            patent_data = json.loads(result_text)
            log_and_print("✓ Successfully parsed patent data", preprocessing_log)
        except Exception as e:
            error_msg = f"Error parsing JSON: {e}\nRaw response: {result_text[:500]}"
            log_and_print(error_msg, preprocessing_log)
            raise
        
        # 5) Document로 변환 (청킹 포함)
        docs = to_langchain_document(patent_data, source=pdf_path, log_file=chunking_log)
        log_and_print(f"Created {len(docs)} documents", chunking_log)
        
        # 6) 모든 문서의 metadata에 patent_id 추가
        for doc in docs:
            doc.metadata['patent_id'] = patent_id
        log_and_print(f"✓ Added patent_id to all {len(docs)} documents", chunking_log)
        
        # 7) Vector store 생성
        vectorstore = FAISS.from_documents(docs, emb)
        print("✓ Vector store created")
        
        # 8) Vector store 저장
        save_vectorstore(vectorstore, patent_id)
        
        log_and_print(f"\n{'='*80}", preprocessing_log)
        log_and_print(f"✅ Preprocessing complete for {patent_id}", preprocessing_log)
        log_and_print(f"📊 Total documents: {len(docs)}", preprocessing_log)
        log_and_print(f"📂 Vectorstore saved to: {get_vectorstore_path(patent_id)}", preprocessing_log)
        log_and_print(f"{'='*80}\n", preprocessing_log)
        
        return {"success": True, "message": "전처리 완료!", "docs_count": len(docs)}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

async def process_query(query, patent_id):
    """쿼리 실행"""
    try:
        result = await run_query(query, patent_id)
        return result
    except Exception as e:
        return {"error": str(e)}

# ==========================
# 사이드바
# ==========================
with st.sidebar:
    st.markdown('<h1 style="color: white; text-align: center;">📄 Patent Manager</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 파일 업로드
    st.markdown('<h3 style="color: white;">📤 Upload Patent</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "PDF 파일을 선택하세요", 
        type=['pdf'],
        help="특허 PDF 파일을 업로드하세요",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        patent_id = extract_patent_id_from_filename(uploaded_file.name)
        st.markdown(f'<div class="info-card" style="background: rgba(255,255,255,0.15); color: white; border-left: 4px solid #fbbf24;">📋 <strong>특허 ID:</strong> {patent_id}</div>', unsafe_allow_html=True)
        
        # Vector DB 존재 여부 확인
        if vectorstore_exists(patent_id):
            st.markdown('<div class="success-card" style="background: rgba(16, 185, 129, 0.2); color: white;">✅ 전처리 완료됨</div>', unsafe_allow_html=True)
            st.session_state.current_patent_id = patent_id
            st.session_state.preprocessed = True
        else:
            st.markdown('<div class="warning-card" style="background: rgba(251, 191, 36, 0.2); color: white;">⚠️ 전처리 필요</div>', unsafe_allow_html=True)
            
            # Preprocess 버튼
            if st.button("🔄 전처리 시작", type="primary", use_container_width=True):
                with st.spinner("🔄 전처리 중... 잠시만 기다려주세요."):
                    # 파일 저장
                    pdf_path = save_uploaded_file(uploaded_file)
                    
                    # 전처리 실행
                    result = asyncio.run(process_preprocess(pdf_path, patent_id))
                    
                    if result.get("success"):
                        st.success(f"✅ {result['message']} (총 {result['docs_count']} 문서)")
                        st.session_state.current_patent_id = patent_id
                        st.session_state.preprocessed = True
                        st.rerun()
                    else:
                        st.error(f"❌ 오류: {result.get('error', 'Unknown error')}")
    
    st.markdown("---")
    
    # 저장된 특허 목록
    st.markdown('<h3 style="color: white;">💾 Saved Patents</h3>', unsafe_allow_html=True)
    patent_list = get_patent_list()
    
    if patent_list:
        selected_patent = st.selectbox(
            "특허 선택",
            patent_list,
            index=patent_list.index(st.session_state.current_patent_id) 
                  if st.session_state.current_patent_id in patent_list else 0,
            label_visibility="collapsed"
        )
        
        if st.button("📂 선택한 특허 분석", type="secondary", use_container_width=True):
            st.session_state.current_patent_id = selected_patent
            st.session_state.preprocessed = True
            st.success(f"✅ {selected_patent} 선택됨")
    else:
        st.markdown('<div style="color: rgba(255,255,255,0.7); text-align: center; padding: 1rem;">저장된 특허가 없습니다</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 통계 정보
    if patent_list:
        st.markdown('<h3 style="color: white;">📊 Statistics</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("총 특허", len(patent_list), label_visibility="visible")
        with col2:
            # 현재 특허의 분석 수만 표시
            current_analysis_count = len(get_current_query_history()) if st.session_state.current_patent_id else 0
            st.metric("현재 분석 수", current_analysis_count, label_visibility="visible")

# ==========================
# 메인 영역
# ==========================
st.markdown('<div class="main-title">🔬 Patent Analysis System</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #718096; font-size: 1.1rem; margin-bottom: 2rem;">AI-Powered Multi-Agent Patent Analyzer</p>', unsafe_allow_html=True)

# 현재 선택된 특허 표시
if st.session_state.current_patent_id:
    st.markdown(f'''
    <div class="info-card">
        <h3 style="margin: 0; color: #1e40af;">📋 현재 분석 중인 특허</h3>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; font-weight: 600; color: #2d3748;">{st.session_state.current_patent_id}</p>
    </div>
    ''', unsafe_allow_html=True)
else:
    st.markdown('''
    <div class="warning-card">
        <h3 style="margin: 0;">⚠️ 특허를 선택해주세요</h3>
        <p style="margin: 0.5rem 0 0 0;">왼쪽 사이드바에서 특허를 업로드하거나 선택하세요.</p>
    </div>
    ''', unsafe_allow_html=True)

# ==========================
# 탭으로 기능 분리
# ==========================
tab1, tab2 = st.tabs(["🤖 멀티에이전트 분석", "💬 챗봇 Q&A"])

# ==========================
# 탭 1: 멀티에이전트 분석
# ==========================
with tab1:
    # 쿼리 입력 영역
    col_title, col_help, col_rest = st.columns([1.5, 0.4, 10])
    with col_title:
        st.markdown('<div class="sub-title">💬 질문하기</div>', unsafe_allow_html=True)
    with col_help:
        st.markdown('<div style="margin-top: 1.8rem;"></div>', unsafe_allow_html=True)  # 수직 정렬
        with st.popover("❓", help="시스템 사용 가이드"):
            st.markdown("""
            ### 🤖 AI 멀티에이전트 특허 분석 시스템
            
            이 시스템은 **4개의 전문 AI 에이전트**가 협력하여 특허를 종합적으로 분석합니다.
            
            ---
            
            #### 📌 핵심 기능
            
            **1. 🔬 혁신 포인트 분석 (Innovation Agent)**
            - 특허의 핵심 혁신과 차별화 요소 파악
            - 기존 기술 대비 개선점 분석
            - 특허의 독창성과 기술적 우위 평가
            
            ***예시 질문:***
            - *"이 특허의 혁신 포인트를 알려줘"*
            - *"핵심 차별화 요소는 무엇인가?"*
            - *"기존 기술 대비 어떤 개선이 있는지 설명해줘"*
            
            **2. 🏗️ 구현 방법 분석 (Implementation Agent)**
            - 구체적인 제조 및 구현 방법 설명
            - 공정 단계별 상세 프로세스
            - 실제 적용 가능한 실시예 분석
            
            ***예시 질문:***
            - *"구현 방법을 상세히 설명해줘"*
            - *"제조 공정을 단계별로 알려줘"*
            - *"실시예를 분석해줘"*
            
            **3. ⚙️ 기술적 원리 분석 (Technical Agent)**
            - 핵심 기술의 동작 원리 및 메커니즘
            - 기술적 세부사항 및 스펙
            - 물리적/화학적 원리 설명
            
            ***예시 질문:***
            - *"기술적 원리를 분석해줘"*
            - *"동작 메커니즘을 설명해줘"*
            - *"기술적 세부사항을 알려줘"*
            
            **4. 🔍 유사 특허 비교 (Horizontal Agent)**
            - Google Patents에서 유사 특허 자동 검색
            - 현재 특허와 유사 특허 간 비교 분석
            - 기술적 차이점 및 공통점 도출
            
            ***예시 질문:***
            - *"유사 특허를 찾아서 비교해줘"*
            - *"유사 특허 3개와 비교 분석해줘"*
            - *"경쟁 특허와의 차이점을 알려줘"*
            
            ---
            
            #### 💡 단일 & 복합 질문 모두 가능!
            
            **단일 질문 (하나의 에이전트):**
            - "혁신 포인트를 알려줘"
            - "구현 방법을 설명해줘"
            - "유사 특허 5개를 찾아줘"
            
            **복합 질문 (여러 에이전트 협업):**
            - "혁신 포인트와 구현 방법을 모두 분석해줘"
            - "기술적 원리를 분석하고 유사 특허와 비교해줘"
            - "구현 방법과 기술적 원리를 설명하고, 유사 특허 3개와 비교해줘"
            
            ---
            
            #### 🎯 종합 분석 기능 (All-in-One)
            
            **"종합적으로 분석해줘"** 같은 키워드를 입력하면,
            위의 **모든 핵심 기능(4개)이 자동으로 실행**됩니다!
            
            - 혁신 포인트 분석
            - 구현 방법 분석  
            - 기술적 원리 분석
            - 유사 특허 비교 (2개 기본)
            
            **종합 분석 키워드:**
            - "종합 분석해줘"
            - "전체 분석해줘"
            - "모든 것을 분석해줘"
            - "완전한 분석 리포트를 작성해줘"
            - "comprehensive analysis"
            
            ⚠️ **참고:** 종합 분석은 모든 에이전트가 순차적으로 실행되므로 
            **4~6분 정도 시간이 소요**될 수 있습니다.
            
            ---
            
            #### 📝 사용 팁
            
            1. **구체적인 질문**일수록 더 정확한 답변을 받을 수 있습니다
            2. **여러 분석을 원하면** 하나의 질문에 모두 포함시키세요
            3. **유사 특허 개수를 지정**하려면 숫자를 명시하세요 (예: "유사 특허 5개")
            4. **시간이 부족하다면** 단일 질문을 사용하고, **종합적인 분석이 필요하면** 종합 분석을 사용하세요
            5. **이전 분석 결과**는 히스토리에서 언제든 다시 확인할 수 있습니다
            6. **PDF 전처리**는 한 번만 하면 되며, 이후에는 즉시 분석이 가능합니다
            """)

    query_input = st.text_area(
        "질문을 입력하세요",
        height=120,
        placeholder="💡 예시:\n• 이 특허의 핵심 혁신 포인트를 분석해줘\n• 기술적 구현 방법을 상세히 설명해줘\n• 유사 특허를 찾아서 비교 분석해줘\n• 종합적으로 분석해줘 (모든 기능 실행)",
        disabled=not st.session_state.preprocessed,
        label_visibility="collapsed",
        key=f"query_input_{st.session_state.query_input_counter}"
    )

    col1, col2, col3 = st.columns([2, 2, 6])

    with col1:
        run_button = st.button(
            "🚀 분석 시작", 
            type="primary",
            disabled=not st.session_state.preprocessed,
            use_container_width=True
        )

    with col2:
        if st.button("🗑️ 히스토리 초기화", use_container_width=True):
            # 현재 특허의 히스토리만 초기화
            if st.session_state.current_patent_id:
                st.session_state.query_history_by_patent[st.session_state.current_patent_id] = []
            # 입력창도 초기화
            st.session_state.query_input_counter += 1
            st.rerun()

    # 쿼리 실행
    if run_button:
        if not query_input.strip():
            st.warning("⚠️ 질문을 입력해주세요.")
        else:
            with st.spinner("🤖 AI 에이전트들이 분석 중입니다... 잠시만 기다려주세요."):
                # 쿼리 실행
                result = asyncio.run(process_query(
                    query_input.strip(), 
                    st.session_state.current_patent_id
                ))
                
                if result and not result.get("error"):
                    # 현재 특허의 History에 추가
                    query_history = get_current_query_history()
                    query_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "query": query_input.strip(),
                        "result": result,
                        "patent_id": st.session_state.current_patent_id
                    })
                    
                    # 입력창 초기화 (카운터 증가로 새로운 위젯 생성)
                    st.session_state.query_input_counter += 1
                    
                    st.rerun()
                else:
                    st.error(f"❌ 오류: {result.get('error', 'Unknown error')}")

    # History 표시
    st.markdown("---")
    st.markdown('<div class="sub-title">📜 분석 히스토리</div>', unsafe_allow_html=True)

    # 현재 특허의 히스토리 가져오기
    query_history = get_current_query_history()
    
    if query_history:
        # 최신 항목부터 표시
        for idx, item in enumerate(reversed(query_history)):
            with st.expander(
                f"🕐 {item['timestamp']} | 📋 {item['patent_id']}", 
                expanded=(idx == 0)  # 최신 항목만 펼쳐놓기
            ):
                # 질문 표시
                st.markdown(f"""
                <div style="background: #f7fafc; padding: 1rem; border-radius: 8px; border-left: 4px solid #1e40af; margin-bottom: 1rem;">
                    <strong style="color: #1e40af;">💬 질문:</strong>
                    <p style="margin: 0.5rem 0 0 0; color: #2d3748;">{item['query']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # 결과 표시
                st.markdown('<strong style="color: #2d3748;">📝 분석 결과:</strong>', unsafe_allow_html=True)
                
                result = item['result']
                if isinstance(result, dict):
                    # merged_result가 있는 경우 (supervisor의 최종 결과)
                    if 'merged_result' in result and result['merged_result']:
                        st.markdown(f"""
                        <div style="background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                            {result['merged_result']}
                        </div>
                        """, unsafe_allow_html=True)
                    # messages가 있는 경우
                    elif 'messages' in result and result['messages']:
                        from langchain_core.messages import AIMessage
                        ai_messages = [msg for msg in result['messages'] if isinstance(msg, AIMessage)]
                        if ai_messages:
                            st.markdown(f"""
                            <div style="background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                                {ai_messages[-1].content}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            last_msg = result['messages'][-1]
                            if hasattr(last_msg, 'content'):
                                st.markdown(last_msg.content)
                            else:
                                st.write(last_msg)
                    else:
                        st.json(result)
                elif isinstance(result, str):
                    st.markdown(f"""
                    <div style="background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                        {result}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.write(result)
    else:
        st.markdown('''
        <div style="text-align: center; padding: 3rem; background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
            <p style="font-size: 1.2rem; color: #718096; margin: 0;">📭 아직 분석 히스토리가 없습니다</p>
            <p style="color: #a0aec0; margin: 0.5rem 0 0 0;">위에서 질문을 입력하고 분석을 시작해보세요!</p>
        </div>
        ''', unsafe_allow_html=True)

# ==========================
# 탭 2: 챗봇 Q&A
# ==========================
with tab2:
    st.markdown('<div class="sub-title">💬 간편한 특허 Q&A 챗봇</div>', unsafe_allow_html=True)
    
    # 설명 카드
    st.markdown("""
    <div class="info-card">
        <h4 style="margin: 0 0 0.5rem 0; color: #1e40af;">💡 챗봇 Q&A 모드</h4>
        <p style="margin: 0; color: #4a5568; font-size: 0.95rem;">
            간단하고 빠른 질문-답변을 위한 모드입니다. 복잡한 분석 대신 특허 내용에 대한 
            직접적인 질문에 빠르게 답변을 받을 수 있습니다.
        </p>
        <ul style="margin: 0.5rem 0 0 1.5rem; color: #4a5568; font-size: 0.9rem;">
            <li>특정 섹션이나 청구항 내용 확인</li>
            <li>용어나 개념에 대한 설명</li>
            <li>간단한 비교나 요약</li>
            <li>대화 이력이 유지되어 연속적인 질문 가능</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.preprocessed:
        st.warning("⚠️ 특허를 먼저 선택하고 전처리를 완료해주세요.")
    else:
        # 대화 히스토리 표시
        st.markdown("---")
        st.markdown('<div class="sub-title">📝 대화 내용</div>', unsafe_allow_html=True)
        
        # 현재 특허의 챗봇 히스토리 가져오기
        chatbot_history = get_current_chatbot_history()
        
        # 대화 히스토리를 위한 컨테이너
        chat_container = st.container()
        
        with chat_container:
            if chatbot_history:
                for msg in chatbot_history:
                    if msg["role"] == "user":
                        st.markdown(f"""
                        <div style="background: #e3f2fd; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #1e40af;">
                            <strong style="color: #1e40af;">👤 You:</strong>
                            <p style="margin: 0.3rem 0 0 0; color: #2d3748;">{msg['content']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background: #f7fafc; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #48bb78;">
                            <strong style="color: #48bb78;">🤖 Assistant:</strong>
                            <p style="margin: 0.3rem 0 0 0; color: #2d3748;">{msg['content']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown('''
                <div style="text-align: center; padding: 2rem; background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                    <p style="font-size: 1.1rem; color: #718096; margin: 0;">💬 대화를 시작해보세요!</p>
                    <p style="color: #a0aec0; margin: 0.5rem 0 0 0; font-size: 0.9rem;">특허에 대해 궁금한 것을 물어보세요.</p>
                </div>
                ''', unsafe_allow_html=True)
        
        # 입력 영역 (하단에 고정)
        st.markdown("---")
        
        col_input, col_btn1, col_btn2 = st.columns([6, 1, 1])
        
        with col_input:
            chat_input = st.text_input(
                "메시지 입력",
                placeholder="예: Claim 1의 내용을 요약해줘 / 이 특허의 주요 기술은 무엇인가요?",
                key=f"chatbot_input_{st.session_state.chatbot_input_counter}",
                label_visibility="collapsed"
            )
        
        with col_btn1:
            send_button = st.button("📤 전송", type="primary", use_container_width=True)
        
        with col_btn2:
            clear_button = st.button("🗑️ 초기화", use_container_width=True)
        
        # 전송 버튼 클릭 처리
        if send_button:
            if not chat_input.strip():
                st.warning("⚠️ 메시지를 입력해주세요.")
            else:
                # 현재 특허의 챗봇 히스토리 가져오기
                chatbot_history = get_current_chatbot_history()
                
                # 사용자 메시지 추가
                chatbot_history.append({
                    "role": "user",
                    "content": chat_input.strip()
                })
                
                # 챗봇 응답 생성
                with st.spinner("🤖 답변 생성 중..."):
                    response = simple_rag_chatbot(
                        query=chat_input.strip(),
                        patent_id=st.session_state.current_patent_id,
                        chat_history=chatbot_history,
                    )
                
                # 챗봇 응답 추가
                chatbot_history.append({
                    "role": "assistant",
                    "content": response
                })
                
                # 입력창 초기화 (카운터 증가로 새로운 위젯 생성)
                st.session_state.chatbot_input_counter += 1
                
                # 페이지 새로고침
                st.rerun()
        
        # 초기화 버튼 클릭 처리
        if clear_button:
            # 현재 특허의 챗봇 히스토리만 초기화
            if st.session_state.current_patent_id:
                st.session_state.chatbot_history_by_patent[st.session_state.current_patent_id] = []
            # 입력창도 초기화
            st.session_state.chatbot_input_counter += 1
            st.rerun()


# Footer
st.markdown("---")
st.markdown('''
<div class="footer">
    <p><strong>Patent Analysis System v2.1</strong></p>
    <p>Powered by LangGraph & Multi-Agent AI Technology 🤖</p>
    <p style="font-size: 0.8rem; color: #a0aec0; margin-top: 0.5rem;">© 2024 All Rights Reserved</p>
</div>
''', unsafe_allow_html=True)
