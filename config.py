"""
설정 파일 - LLM, 경로 등 설정
"""
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# 환경 변수 로드
load_dotenv()

# LangSmith 설정 (선택사항)
# .env 파일에 LANGSMITH_API_KEY와 LANGSMITH_PROJECT를 설정하면 자동으로 추적이 활성화됩니다.
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "patent-analysis")
    print("✅ LangSmith 추적 활성화")

# LLM 설정
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

# Embedding 설정
emb = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# 경로 설정
VECTORSTORE_DIR = "vectorstore"
LOG_DIR = "log"

# 디렉토리 생성
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print("✅ 설정 완료")
print(f"✅ Vector DB 저장 경로: {VECTORSTORE_DIR}")
print(f"✅ 로그 저장 경로: {LOG_DIR}")
