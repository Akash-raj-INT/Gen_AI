from langchain_community.document_loaders import WebBaseLoader

url = "https://github.com/Akash-raj-INT"

data = WebBaseLoader(url)

docs = data.load()

print(docs[0].page_content)