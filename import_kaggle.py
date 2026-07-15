"""
import_kaggle.py

Import a Kaggle CSV into the ChromaDB knowledge base.
Handles column-name guessing, batching, and time-series summarization
so large datasets don't turn into thousands of near-duplicate chunks.

Usage:
  python3 import_kaggle.py path/to/file.csv --collection personal_thesis
"""

import sys
import argparse
import pandas as pd
import chromadb
import ollama
from datetime import datetime
import os



DB_PATH = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"
BATCH_SIZE = 20  # embed in small batches so you get progress feedback


def guess_column(df, candidates):
    """Find the first matching column name, case-insensitive."""
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_cols:
            return lower_cols[cand.lower()]
    return None


def is_time_series(df, ticker_col, date_col):
    """If a dataset has repeated tickers across many dates, treat as time-series."""
    if not ticker_col or not date_col:
        return False
    return df[ticker_col].value_counts().max() > 20


def summarize_time_series(df, ticker_col, date_col, price_col):
    """
    Collapse daily rows per ticker into one summary sentence instead of
    hundreds of near-identical rows. Keeps retrieval useful.
    """
    summaries = []
    for ticker, group in df.groupby(ticker_col):
        group = group.sort_values(date_col)
        start_date = group[date_col].iloc[0]
        end_date = group[date_col].iloc[-1]
        start_price = group[price_col].iloc[0]
        end_price = group[price_col].iloc[-1]
        high = group[price_col].max()
        low = group[price_col].min()
        pct_change = ((end_price - start_price) / start_price) * 100 if start_price else 0

        sentence = (
            f"{ticker} price history from {start_date} to {end_date}: "
            f"started at {start_price}, ended at {end_price}, "
            f"high of {high}, low of {low}, "
            f"overall change of {pct_change:.1f}% over the period."
        )
        summaries.append((ticker, sentence))
    return summaries


def row_to_sentence(row, columns):
    parts = [f"{col} is {row[col]}" for col in columns if pd.notna(row[col])]
    return ", ".join(parts)










def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Folder containing per-ticker files, e.g. kaggle_data/Stocks")
    parser.add_argument("--tickers", required=True, help="Comma-separated tickers, e.g. AAPL,NVDA,TSLA")
    parser.add_argument("--collection", default="personal_thesis")
    args = parser.parse_args()

    ticker_list = [t.strip().upper() for t in args.tickers.split(",")]
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(args.collection)
    existing_ids = set(collection.get(include=[])["ids"]) if collection.count() > 0 else set()

    for ticker in ticker_list:
        filename = f"{ticker.lower()}.us.txt"
        filepath = os.path.join(args.data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Skipping {ticker}: file not found ({filepath})")
            continue

        print(f"Processing {ticker} ...")
        df = pd.read_csv(filepath)
        df.columns = [c.strip() for c in df.columns]  # this dataset uses "Date","Open","High","Low","Close","Volume","OpenInt"


        # Since this file has no ticker column, inject it manually
        if "Date" in df.columns and "Close" in df.columns:
            df_sorted = df.sort_values("Date")
            start_date, end_date = df_sorted["Date"].iloc[0], df_sorted["Date"].iloc[-1]
            start_price, end_price = df_sorted["Close"].iloc[0], df_sorted["Close"].iloc[-1]
            high, low = df_sorted["Close"].max(), df_sorted["Close"].min()
            pct_change = ((end_price - start_price) / start_price) * 100 if start_price else 0

            sentence = (
                f"{ticker} price history from {start_date} to {end_date}: "
                f"started at {start_price}, ended at {end_price}, "
                f"high of {high}, low of {low}, "
                f"overall change of {pct_change:.1f}% over the period."
            )
            doc_id = f"{ticker}-kaggle-summary"
            if doc_id not in existing_ids:
                embedding = ollama.embeddings(model=EMBED_MODEL, prompt=sentence)["embedding"]
                collection.add(
                    documents=[sentence],
                    embeddings=[embedding],
                    metadatas=[{"ticker": ticker, "source": filename}],
                    ids=[doc_id]
                )
                print(f"  Added summary for {ticker}")
            else:
                print(f"  {ticker} already in knowledge base, skipping")
        else:
            print(f"  Unexpected column format for {ticker}, skipping")

    print(f"Collection '{args.collection}' now has {collection.count()} total documents.")


 







if __name__ == "__main__":
    main()
