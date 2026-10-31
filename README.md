# Company Information RAG Chatbot

A PDF-grounded question-answering app built with Python, LangChain, ChromaDB,
OpenAI embeddings, GPT-4.1 mini, and Streamlit.

## Features

- Upload a company PDF (including a 30-page profile).
- Extract PDF text and retain page metadata.
- Split text into overlapping chunks.
- Create embeddings with `text-embedding-3-small`.
- Persist vectors locally in ChromaDB.
- Retrieve relevant passages and generate grounded answers.
- Show retrieved source passages and page numbers.
- Maintain conversation context for follow-up questions.
- Ask the assistant to say when the document does not contain an answer.

## Requirements

- Python 3.10 or newer (Python 3.11 recommended).
- An OpenAI API key with API access and billing configured.
- An extractable-text PDF. Scanned/image-only PDFs require OCR first.

## Setup (Windows PowerShell)

```powershell
cd company_rag_chatbot
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and replace the placeholder with your OpenAI API key.

If PowerShell blocks activation, run this once in the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run

```powershell
streamlit run main.py
```

Open the local URL printed by Streamlit, upload the company PDF in the sidebar,
and wait for indexing to finish. Then ask questions in the chat box.

## How it works

### Ingestion
1. `PyPDFLoader` extracts text page by page.
2. `RecursiveCharacterTextSplitter` creates 1,000-character chunks with
   200-character overlap.
3. `OpenAIEmbeddings` converts chunks into vectors.
4. ChromaDB stores vectors, text, and page metadata under `chroma_db/`.

### Retrieval and generation
1. The user's question is embedded by the Chroma integration.
2. ChromaDB returns the top-k similar chunks.
3. The application sends those chunks, the question, and recent chat turns
   to GPT-4.1 mini.
4. The system prompt instructs the model to answer only from the PDF context.
5. Retrieved page numbers and excerpts are shown as sources.

## Important notes

- OpenAI API usage is billed separately from a ChatGPT subscription.
- Never commit `.env` or expose your API key.
- ChromaDB is local persistent storage in this starter project.
- The app reuses the vector store for the same PDF hash. A changed PDF creates
  a separate local store.
- Similarity retrieval is not a guarantee of factual correctness. Review the
  cited pages for important decisions.
- Scanned PDFs need OCR (for example, a preprocessing step using an OCR engine).
- The current version supports one active PDF per browser session. Multi-document
  collections, authentication, streaming, hybrid search, reranking, and formal
  RAG evaluation can be added as extensions.

## Suggested next improvements

1. Add OCR for scanned documents.
2. Add hybrid keyword + vector retrieval.
3. Add a reranker and tune chunk size/top-k using a test question set.
4. Add automated evaluation for retrieval relevance and answer faithfulness.
5. Add document upload management and multi-PDF collections.
6. Add tests and deploy the app securely.
