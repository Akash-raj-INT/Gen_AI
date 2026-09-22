# ChapterChat

ChapterChat is a Streamlit PDF question-answering app. Upload a book, build a local vector index, and ask questions with answers grounded in the document. It uses Google Gemini for responses, Hugging Face embeddings for semantic search, and Chroma for vector storage.

## Live demo

Try ChapterChat on Streamlit Community Cloud:

<https://rrvtdh73jqf8n2kmwhhabn.streamlit.app/>

## Features

- Upload and index PDF books from the Streamlit sidebar
- Retrieve relevant passages with maximal marginal relevance (MMR)
- Show source page references for answers
- Keep uploaded document indexes local to the running session
- Use an existing Chroma index from `chroma_db_pdf` when available

## Requirements

- Python 3.10 or newer
- A Google Gemini API key

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

Never commit `.env` or API keys to source control.

## Run the Streamlit app

```powershell
streamlit run app.py
```

Open the local URL shown by Streamlit, upload a PDF, and start asking questions.

## Run the command-line chat

The CLI expects a Chroma index in `chroma_db_pdf`:

```powershell
python main.py
```

Enter `0` to exit.

## Project layout

```text
app.py                  Streamlit user interface and PDF indexing
main.py                 Command-line RAG chat
create_database.py      Create the default Chroma database
document_loader/        Document loading utilities
retriver/               Retrieval strategies
vector_store/           Vector-store helpers
chroma_db_pdf/          Local Chroma database, when generated
requirements.txt        Python dependencies
```

## Notes

The first run may download the `sentence-transformers/all-MiniLM-L6-v2` embedding model. Generated indexes, caches, virtual environments, and secrets are ignored by Git and should be recreated locally as needed.