import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from app.ingest import ingest_pdf
from app.chatbot import answer_question

load_dotenv()

st.set_page_config(
    page_title="Company RAG Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Company Information Assistant")
st.caption("Ask questions about a company using information from its PDF document.")

# Allow either a local .env key or a key entered in the sidebar.
with st.sidebar:
    st.header("Configuration")
    env_key = os.getenv("OPENAI_API_KEY", "")
    api_key = st.text_input(
        "OpenAI API key",
        value=env_key,
        type="password",
        help="Your key is used to create embeddings and generate answers.",
    )
    st.divider()
    st.subheader("Document")
    uploaded_pdf = st.file_uploader("Upload company PDF", type=["pdf"])
    top_k = st.slider("Retrieved chunks (top-k)", min_value=2, max_value=8, value=4)
    st.caption("The vector database is stored locally in the chroma_db/ folder.")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "doc_signature" not in st.session_state:
    st.session_state.doc_signature = None
if "messages" not in st.session_state:
    st.session_state.messages = []

if not api_key:
    st.info("Add your OpenAI API key in the sidebar or in a .env file to begin.")
    st.stop()

if uploaded_pdf is not None:
    pdf_bytes = uploaded_pdf.getvalue()
    # Only ingest when the uploaded document changes.
    import hashlib
    signature = hashlib.sha256(pdf_bytes).hexdigest()

    if signature != st.session_state.doc_signature:
        with st.spinner("Reading PDF and creating searchable embeddings..."):
            try:
                vectorstore, document_id, chunk_count = ingest_pdf(
                    pdf_bytes, uploaded_pdf.name, api_key
                )
                st.session_state.vectorstore = vectorstore
                st.session_state.doc_signature = signature
                st.session_state.messages = []
                st.success(
                    f"Indexed **{uploaded_pdf.name}** — "
                    f"{chunk_count} text chunks are ready."
                )
            except Exception as exc:
                st.error(f"Could not process the PDF: {exc}")
                st.stop()

if st.session_state.vectorstore is None:
    st.info("Upload a company PDF (for example, a 30-page company profile) to start.")
    st.markdown(
        "**What this app does:** extracts PDF text → chunks it → creates embeddings "
        "→ stores them in ChromaDB → retrieves relevant passages → generates an "
        "answer with OpenAI."
    )
    st.stop()

# Display conversation history.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("View retrieved PDF sources"):
                for src in message["sources"]:
                    st.markdown(f"**{src['file']} — page {src['page']}**")
                    st.caption(src["content"][:700])

question = st.chat_input("Ask something about the company...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Convert previous turns to LangChain message objects for follow-up questions.
    history = []
    for item in st.session_state.messages[:-1]:
        if item["role"] == "user":
            history.append(HumanMessage(content=item["content"]))
        elif item["role"] == "assistant":
            history.append(AIMessage(content=item["content"]))

    with st.chat_message("assistant"):
        with st.spinner("Searching the PDF and generating an answer..."):
            try:
                answer, sources = answer_question(
                    st.session_state.vectorstore,
                    question,
                    history,
                    api_key,
                    top_k=top_k,
                )
                st.markdown(answer)
                if sources:
                    with st.expander("View retrieved PDF sources"):
                        for src in sources:
                            st.markdown(f"**{src['file']} — page {src['page']}**")
                            st.caption(src["content"][:700])
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")

if st.sidebar.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()
