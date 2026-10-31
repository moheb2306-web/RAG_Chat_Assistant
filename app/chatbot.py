"""Retrieval and answer generation for the company-document assistant."""
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


SYSTEM_PROMPT = """You are a careful company-information assistant.
Answer the user's question using ONLY the supplied PDF context and conversation
when it helps resolve a follow-up question. Do not use outside knowledge to fill
gaps. If the PDF does not contain enough information, say:
"I couldn't find sufficient information about this in the uploaded document."
Do not invent names, dates, numbers, products, or policies.
Be clear and concise. If the context contains conflicting details, explain that."""


def answer_question(vectorstore, question: str, chat_history: list, api_key: str,
                    top_k: int = 4):
    """Retrieve relevant chunks, ask the LLM, and return answer plus source pages."""
    if not question.strip():
        raise ValueError("Please enter a question.")

    # Retrieve the most relevant chunks from ChromaDB.
    retrieved_docs = vectorstore.similarity_search(question, k=top_k)
    context_parts = []
    sources = []
    for doc in retrieved_docs:
        page = doc.metadata.get("page_number", doc.metadata.get("page", "Unknown"))
        source = doc.metadata.get("source_file", "Uploaded PDF")
        context_parts.append(f"[Source: {source}, page {page}]\n{doc.page_content}")
        sources.append({"file": source, "page": page, "content": doc.page_content})

    context = "\n\n---\n\n".join(context_parts)
    if not context.strip():
        context = "No relevant context was retrieved."

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        api_key=api_key,
    )

    # Include recent turns so follow-up questions such as "explain the first one"
    # have conversational context. The answer must still be grounded in PDF context.
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(chat_history[-8:])
    messages.append(
        HumanMessage(
            content=(
                f"PDF context:\n{context}\n\n"
                f"User question: {question}\n\n"
                "Answer using the PDF context. Cite page numbers in your answer "
                "where useful, for example (page 4)."
            )
        )
    )

    response = llm.invoke(messages)
    return response.content, sources
