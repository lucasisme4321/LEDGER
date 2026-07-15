"""
build_knowledge_base.py

Pulls stock data with yfinance, turns it into readable text "documents",
embeds them with Ollama's nomic-embed-text model, and stores them in the
local Chroma vector database that the app reads from.

Run directly to build from the command line:
    python3 build_knowledge_base.py AAPL GOOG MSFT AMZN

Or import build_knowledge_base(tickers) from the Flask app.
"""

import sys
import yfinance as yf
import chromadb
import ollama

DB_PATH = "./chroma_db"
COLLECTION_NAME = "stock_knowledge"
EMBED_MODEL = "nomic-embed-text"
PERIOD = "1y"


def make_documents_for_ticker(ticker: str) -> list[str]:
    stock = yf.Ticker(ticker)
    hist = stock.history(period=PERIOD)
    info = stock.info

    if hist.empty:
        return [f"No price data found for ticker {ticker}."]

    docs = []

    docs.append(
        f"{ticker} ({info.get('longName', ticker)}) overview: "
        f"Sector: {info.get('sector', 'N/A')}. "
        f"Industry: {info.get('industry', 'N/A')}. "
        f"Market cap: {info.get('marketCap', 'N/A')}. "
        f"Trailing P/E: {info.get('trailingPE', 'N/A')}. "
        f"Forward P/E: {info.get('forwardPE', 'N/A')}. "
        f"Dividend yield: {info.get('dividendYield', 'N/A')}. "
        f"52-week high: {info.get('fiftyTwoWeekHigh', 'N/A')}. "
        f"52-week low: {info.get('fiftyTwoWeekLow', 'N/A')}. "
        f"Business summary: {info.get('longBusinessSummary', 'N/A')}"
    )

    monthly = hist["Close"].resample("ME").agg(["first", "last", "min", "max"])
    for date, row in monthly.iterrows():
        pct_change = (row["last"] - row["first"]) / row["first"] * 100
        docs.append(
            f"{ticker} performance for {date.strftime('%B %Y')}: "
            f"opened around {row['first']:.2f}, closed around {row['last']:.2f}, "
            f"low {row['min']:.2f}, high {row['max']:.2f}, "
            f"change of {pct_change:.2f}% for the month."
        )

    returns = hist["Close"].pct_change().dropna()
    docs.append(
        f"{ticker} risk profile over the last {PERIOD}: "
        f"average daily return {returns.mean()*100:.4f}%, "
        f"daily volatility (std dev) {returns.std()*100:.4f}%, "
        f"best day {returns.max()*100:.2f}%, worst day {returns.min()*100:.2f}%."
    )

    return docs


def build_knowledge_base(tickers: list[str], progress_callback=None) -> int:
    """Rebuild the knowledge base for the given tickers. Returns doc count."""
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    doc_id = 0
    for ticker in tickers:
        if progress_callback:
            progress_callback(ticker)
        for doc in make_documents_for_ticker(ticker):
            embedding = ollama.embeddings(model=EMBED_MODEL, prompt=doc)["embedding"]
            collection.add(
                ids=[str(doc_id)],
                embeddings=[embedding],
                documents=[doc],
                metadatas=[{"ticker": ticker}],
            )
            doc_id += 1

    return doc_id


if __name__ == "__main__":
    tickers = [t.upper() for t in sys.argv[1:]] or ["AAPL", "GOOG", "MSFT", "AMZN"]
    print(f"Building knowledge base for: {', '.join(tickers)}")
    count = build_knowledge_base(tickers, progress_callback=lambda t: print(f"  processing {t}..."))
    print(f"Done. Stored {count} documents in '{DB_PATH}'.")

