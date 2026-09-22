from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


pdf_path = Path(__file__).with_name("GRU.pdf")
data = PyPDFLoader(str(pdf_path))

docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap=10
)

chunks = splitter.split_documents(docs)

print(chunks[0].page_content)