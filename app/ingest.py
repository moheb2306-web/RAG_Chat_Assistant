"""PDF ingestion: load pages, split text, embed chunks, and persist them in Chroma."""
from pathlib import Path
import hashlib
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


def _file_sha256(file_bytes: bytes) -> str:
    """Create a stable ID so the same PDF can reuse its existing vector store."""
    return hashlib.sha256(file_bytes).hexdigest()


def ingest_pdf(file_bytes: bytes, filename: str, api_key: str, persist_root: str = "chroma_db"):
    """Ingest a PDF and return (vectorstore, document_id, chunk_count)."""
    if not file_bytes:
        raise ValueError("The uploaded PDF is empty.")

    # PyPDFLoader works with a local file path, so temporarily save the upload.
    suffix = Path(filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        pages = PyPDFLoader(temp_path).load()
    finally:
        Path(temp_path).unlink(missing_ok=True)

    # Keep the original page number in metadata for source citations.
    for page in pages:
        page.metadata["source_file"] = filename
        # PyPDFLoader page metadata is zero-based; display pages as one-based.
        page.metadata["page_number"] = int(page.metadata.get("page", 0)) + 1

    # Character-based chunking is a sensible starting point for a company profile.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise ValueError(
            "No text could be extracted. The PDF may be scanned; run OCR first."
        )

    document_id = _file_sha256(file_bytes)
    persist_dir = str(Path(persist_root) / document_id[:16])

    # Reuse an existing persisted collection for this exact PDF.
    if Path(persist_dir).exists() and any(Path(persist_dir).iterdir()):
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=api_key,
        )
        vectorstore = Chroma(
            collection_name="company_documents",
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        return vectorstore, document_id, len(chunks)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="company_documents",
        persist_directory=persist_dir,
    )
    return vectorstore, document_id, len(chunks)
