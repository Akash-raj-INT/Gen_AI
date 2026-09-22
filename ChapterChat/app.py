import hashlib
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(
    page_title="ChapterChat",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: #f7f5f0;
    }
    [data-testid="stSidebar"] {
        background: #202d2b;
    }
    [data-testid="stSidebar"] * {
        color: #f4f1e8;
    }
    .hero {
        padding: 2.5rem 0 1.5rem;
        border-bottom: 1px solid #d9d4c8;
        margin-bottom: 1.5rem;
    }
    .eyebrow {
        color: #b15b38;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .hero h1 {
        color: #202d2b;
        font-family: Georgia, serif;
        font-size: clamp(2.2rem, 5vw, 4.4rem);
        line-height: 1;
        margin: 0.4rem 0 0.7rem;
    }
    .hero p {
        color: #56615c;
        font-size: 1.05rem;
        max-width: 680px;
    }
    .book-status {
        background: #e6eee9;
        border-left: 4px solid #527c68;
        color: #27493d;
        padding: 0.8rem 1rem;
        margin: 1rem 0 1.5rem;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] span {
        color: #202d2b !important;
    }
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] textarea::placeholder {
        color: #202d2b !important;
        background: #ffffff !important;
        opacity: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def get_llm():
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is missing from the .env file.")
    return ChatGoogleGenerativeAI(model="gemini-3.6-flash")


def get_answer(response):
    if isinstance(response.content, str):
        return response.content.strip()
    answer = "\n".join(
        item["text"]
        for item in response.content
        if isinstance(item, dict) and item.get("type") == "text"
    )
    return answer.strip()


def build_uploaded_vectorstore(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()[:12]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        documents = PyPDFLoader(temp_path).load()
    finally:
        os.unlink(temp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=f"book_{file_hash}",
    )
    return vectorstore, len(documents), len(chunks)


def get_default_vectorstore():
    if not os.path.isdir("chroma_db_pdf"):
        return None
    return Chroma(
        persist_directory="chroma_db_pdf",
        embedding_function=get_embeddings(),
    )


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant answering questions about a book.
Use ONLY the provided context to answer.
If the answer is not present in the context, say: "I could not find the answer in the document."
Keep the answer clear and concise.
""",
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
""",
        ),
    ]
)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = get_default_vectorstore()
if "book_name" not in st.session_state:
    st.session_state.book_name = "Deep Learning PDF"

with st.sidebar:
    st.markdown("## Your library")
    uploaded_file = st.file_uploader(
        "Upload a book",
        type=["pdf"],
        help="Upload a PDF book to create a searchable private index.",
    )

    if uploaded_file is not None:
        upload_signature = f"{uploaded_file.name}:{uploaded_file.size}"
        if st.session_state.get("upload_signature") != upload_signature:
            with st.spinner("Reading and indexing your book..."):
                try:
                    vectorstore, page_count, chunk_count = build_uploaded_vectorstore(
                        uploaded_file
                    )
                    st.session_state.vectorstore = vectorstore
                    st.session_state.book_name = uploaded_file.name
                    st.session_state.upload_signature = upload_signature
                    st.session_state.page_count = page_count
                    st.session_state.chunk_count = chunk_count
                    st.session_state.messages = []
                    st.rerun()
                except Exception as error:
                    st.error(f"Could not index this PDF: {error}")

    if st.session_state.vectorstore is not None:
        st.success(f"Ready: {st.session_state.book_name}")
        if "page_count" in st.session_state:
            st.caption(
                f"{st.session_state.page_count} pages · "
                f"{st.session_state.chunk_count} searchable sections"
            )
    else:
        st.warning("Upload a PDF or create the default database first.")

    st.caption("Your uploaded book is indexed locally for this session.")

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">A quiet reading companion</div>
        <h1>ChapterChat</h1>
        <p>Upload a PDF, then explore its ideas with grounded answers drawn from the text.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.vectorstore is None:
    st.info("Upload a PDF from the sidebar to begin.")
    st.stop()

st.markdown(
    f'<div class="book-status"><strong>Reading:</strong> {st.session_state.book_name}</div>',
    unsafe_allow_html=True,
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("View sources"):
                for source in message["sources"]:
                    st.caption(source)

query = st.chat_input("Ask about this book...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching the book..."):
            try:
                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
                )
                documents = retriever.invoke(query)
                context = "\n\n".join(doc.page_content for doc in documents)
                response = get_llm().invoke(
                    prompt.invoke({"context": context, "question": query})
                )
                answer = get_answer(response) or (
                    "I received an empty response from the language model. "
                    "Please try the question again."
                )
                sources = [
                    f"Page {doc.metadata.get('page', 0) + 1}"
                    for doc in documents
                    if "page" in doc.metadata
                ]
                st.markdown(answer)
                if sources:
                    with st.expander("View sources"):
                        st.caption(" · ".join(dict.fromkeys(sources)))
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as error:
                st.error(f"Something went wrong: {error}")
