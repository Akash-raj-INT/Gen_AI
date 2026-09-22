from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document

docs = [
    Document(page_content="Python is widely used in Ai", metadata={"source": "Ai_book"}),
    Document(page_content="Numpy is used for data analysis in python.", metadata={"source": "Data_analysis_book"}),
    Document(page_content="Neural Networking used in deep learning", metadata={"source": "DL_book"}),
] 

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="chroma_db",
    ) 

result = vector_store.similarity_search("What is used in deep learning?", k=2)
for i in result:
    print(i.page_content)
    print(i.metadata)

retriver = vector_store.as_retriever()

docs =  retriver.invoke("Explain Deep learning")

for d in docs:
    print(d.page_content)