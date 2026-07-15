"""
rag.py

Shared retrieval-augmented generation logic: builds the stock knowledge
base and answers questions using it. Used by both build_knowledge_base.py
(CLI build step) and app.py (the web app).
"""

import chromadb
import ollama

DB_PATH = "./chroma_db"
COLLECTION_NAME = "stock_knowledge"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "ledger"
TOP_K = 5

SYSTEM_PROMPT = (
    "You are a careful investment research assistant. Use the CONTEXT "
    "provided to answer the user's question. If the context doesn't "
    "contain enough information, say so instead of guessing. Always "
    "remind the user that this is not licensed financial advice and "
    "that markets are unpredictable."
)


def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(COLLECTION_NAME)


def get_tracked_tickers() -> list[str]:
    """Return the distinct tickers currently stored in the knowledge base."""
    collection = get_collection()
    if collection.count() == 0:
        return []
    data = collection.get(include=["metadatas"])
    return sorted({m["ticker"] for m in data["metadatas"] if "ticker" in m})


def retrieve_context(question: str) -> list[str]:
    client = chromadb.PersistentClient(path=DB_PATH)
    q_embedding = ollama.embeddings(model=EMBED_MODEL, prompt=question)["embedding"]

    all_chunks = []
    for name in ["stock_knowledge", "personal_thesis", "market_news"]:
        try:
            collection = client.get_collection(name)
            if collection.count() == 0:
                continue
            results = collection.query(query_embeddings=[q_embedding], n_results=TOP_K)
            all_chunks.extend(results["documents"][0])
        except Exception:
            continue
    return all_chunks



def ask(question: str, history: list[dict] | None = None) -> dict:
    """
    Answer a question using RAG.
    `history` is a list of {"role": "user"|"assistant", "content": str}
    from earlier in the conversation (optional).
    Returns {"answer": str, "sources": list[str]}.
    """
    chunks = retrieve_context(question)
    context = "\n\n".join(chunks) if chunks else "No matching data found in the knowledge base."

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({
        "role": "user",
        "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}",
    })

    response = ollama.chat(model=CHAT_MODEL, messages=messages)
    return {"answer": response["message"]["content"], "sources": chunks}

