#10ad pdf
#split into chunks
#create the embeddings I
#store into chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_google_genai._common import GoogleGenerativeAIError

load_dotenv()

data = PyPDFLoader("document_loader/deeplearning.pdf") 
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

try:
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db_pdf",
    )
except GoogleGenerativeAIError as error:
    if "RESOURCE_EXHAUSTED" in str(error):
        raise RuntimeError(
            "Gemini embedding quota is exhausted. Wait for the quota reset "
            "or use a project with available billing/quota, then rerun this script."
        ) from error
    raise


