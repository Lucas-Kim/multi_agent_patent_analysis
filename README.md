# Patent Analysis Multi-Agent System

LangGraph 기반 반도체 특허 분석을 위한 다중 에이전트 시스템입니다.

## 📋 프로젝트 개요

이 프로젝트는 4개의 전문 AI 에이전트를 활용하여 반도체 특허를 다각도로 분석하는 시스템입니다. RAG(Retrieval-Augmented Generation) 기술과 자체 개선 검색 메커니즘을 통해 정확하고 포괄적인 특허 분석을 제공합니다.

## ✨ 주요 기능

### 4가지 전문 에이전트
- **Innovation Agent**: 특허의 혁신성과 독창성 분석
- **Implementation Agent**: 구현 가능성 및 실용성 평가
- **Technical Detail Agent**: 기술적 세부사항 심층 분석
- **Horizontal Comparison Agent**: 유사 특허 검색 및 비교 분석

### 핵심 기술
- 🔄 LangGraph 기반 워크플로우 관리
- 📚 RAG를 활용한 문서 기반 답변 생성
- 🔍 자체 개선 검색 시스템 (최대 2회 반복)
- 🎨 Streamlit 기반 사용자 친화적 UI
- ⚡ 증분적 결과 병합으로 빠른 응답

## 🏗️ 시스템 아키텍처

```
특허 문서 입력
    ↓
PDF 처리 및 청킹
    ↓
Vector DB 저장 (FAISS)
    ↓
LangGraph 워크플로우
    ├─ Innovation Agent
    ├─ Implementation Agent
    ├─ Technical Detail Agent
    └─ Horizontal Agent (검색 반복)
    ↓
결과 통합 및 출력
```

## 🚀 설치 방법

### 1. 저장소 클론

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. 가상환경 생성 (권장)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

## ⚙️ 환경 설정

### 1. 환경 변수 파일 생성

```bash
# Windows
copy .env.example.txt .env

# Linux/Mac
cp .env.example.txt .env
```

### 2. `.env` 파일 수정

```env
# Azure OpenAI 설정
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Azure Embedding 설정
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# LangSmith 설정 (선택사항)
LANGSMITH_API_KEY=your-langsmith-api-key-here
LANGSMITH_PROJECT=patent-analysis
```

### 3. 특허 문서 준비

`data` 폴더에 분석할 PDF 파일을 저장하세요.

```
data/
  └── US8526476.pdf  # 예제 파일 포함
```

## 💻 사용 방법

### Streamlit 앱 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속

### 기본 워크플로우

1. **PDF 업로드**: 분석할 특허 문서 업로드
2. **모드 선택**: 
   - Multi-Agent Analysis: 4개 에이전트 종합 분석
   - Simple Q&A: 간단한 질의응답
3. **질문 입력**: 분석하고자 하는 내용 입력
4. **결과 확인**: 각 에이전트의 분석 결과 확인

## 📁 프로젝트 구조

```
.
├── app.py                  # Streamlit 메인 앱
├── agent_logic.py          # 에이전트 로직 및 LangGraph 워크플로우
├── config.py              # 설정 및 LLM 초기화
├── requirements.txt       # Python 패키지 의존성
├── .env.example.txt       # 환경 변수 템플릿
├── .gitignore            # Git 제외 파일 목록
├── data/                 # 특허 PDF 파일
│   └── US8526476.pdf
├── vectorstore/          # Vector DB 저장소 (자동 생성)
└── log/                  # 로그 파일 (자동 생성)
```

## 🛠️ 기술 스택

- **Framework**: LangGraph, LangChain
- **LLM**: Azure OpenAI (GPT-4)
- **Embedding**: Azure OpenAI Embeddings
- **Vector DB**: FAISS (via ChromaDB)
- **UI**: Streamlit
- **PDF Processing**: PyPDF, PDFPlumber
- **Monitoring**: LangSmith (선택사항)

## 📊 주요 특징

### 지능형 검색
- 실패한 쿼리에 대한 자동 개선 및 재시도
- 다중 검색 전략 (하이브리드 검색)
- 최대 5회 반복 개선

### 문서 처리
- 섹션별 청킹 전략
- PDF 하이브리드 로딩
- 메타데이터 보존

### 평가 시스템
- GPT-4 기반 정량적 점수화
- 임계값 기반 품질 관리

## ⚠️ 주의사항

- `.env` 파일은 절대 Git에 커밋하지 마세요
- Azure OpenAI API 키가 필요합니다
- 대용량 PDF 처리 시 시간이 소요될 수 있습니다

## 🔧 문제 해결

### Vector DB 오류
- `vectorstore` 폴더를 삭제하고 다시 PDF를 업로드하세요

### API 키 오류
- `.env` 파일의 API 키와 엔드포인트를 확인하세요
- Azure Portal에서 배포 이름이 정확한지 확인하세요

### PDF 로딩 실패
- PDF 파일이 손상되지 않았는지 확인하세요
- 텍스트 인식이 가능한 PDF를 업로드해야 합니다. 
- 파일 크기가 너무 크면 처리 시간이 오래 걸릴 수 있습니다

## 📝 라이센스

이 프로젝트는 개인/연구 목적으로 사용 가능합니다.

## 📧 연락처

질문이나 제안사항이 있으시면 이슈를 등록해주세요.

---

