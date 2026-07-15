"""
app.py

Local web app for the Ollama investment chatbot. Serves a chat UI and
two API endpoints:
  POST /api/chat   -> ask a question, get an answer grounded in the
                       knowledge base (RAG)
  POST /api/build   -> (re)build the knowledge base for a list of tickers
  GET  /api/tickers -> list tickers currently in the knowledge base

Run with: python3 app.py
Then open http://127.0.0.1:5000
"""

from flask import Flask, request, jsonify, render_template



import os
from werkzeug.utils import secure_filename
from pypdf import PdfReader
import chromadb
import ollama

import rag
from build_knowledge_base import build_knowledge_base




app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/tickers")
def tickers():
    try:
        return jsonify({"tickers": rag.get_tracked_tickers()})
    except Exception as e:
        return jsonify({"tickers": [], "error": str(e)})


@app.route("/api/build", methods=["POST"])
def build():
    data = request.get_json(force=True)
    raw = data.get("tickers", "")
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    if not tickers:
        return jsonify({"error": "No tickers provided."}), 400
    try:
        count = build_knowledge_base(tickers)
        return jsonify({"status": "ok", "tickers": tickers, "documents": count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat", methods=["POST"])
def chat():

    data = request.get_json(force=True)

    model = data.get("model", "llama3.1:8b")
    system = data.get("system", "You are a helpful AI assistant.")
    question = (data.get("question") or "").strip()
    history = data.get("history") or []

    if not question:
        return jsonify({"error": "No question provided."}), 400

    try:
        result = rag.ask(
            question,
            history=history
        )

        print(result)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]

def extract_text(filepath):
    if filepath.lower().endswith(".pdf"):
        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        with open(filepath, "r", errors="ignore") as f:
            return f.read()

@app.route("/api/knowledge/upload", methods=["POST"])
def upload_knowledge():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    ticker = request.form.get("ticker", "GENERAL").upper()
    collection_name = request.form.get("collection", "personal_thesis")

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    try:
        text = extract_text(filepath)
        chunks = chunk_text(text)

        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection(collection_name)

        for i, chunk in enumerate(chunks):
            embedding = ollama.embeddings(model="nomic-embed-text", prompt=chunk)["embedding"]
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{"ticker": ticker, "source": filename}],
                ids=[f"{filename}-{i}"]
            )

        return jsonify({
            "status": "ok",
            "filename": filename,
            "chunks_added": len(chunks),
            "collection": collection_name
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500





if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

