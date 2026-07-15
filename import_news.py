"""
import_news.py

Import DJIA news headlines into the knowledge base, grouped into
weekly summaries (daily would be way too many near-duplicate chunks).
"""

import os
import pandas as pd
import chromadb
import ollama

DB_PATH = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"

def main():
    filepath = os.path.expanduser("~/invest_app/kaggle_data/news/Combined_News_DJIA.csv")
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Week"] = df["Date"].dt.to_period("W").astype(str)

    headline_cols = [c for c in df.columns if c.startswith("Top")]

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection("market_news")
    existing_ids = set(collection.get(include=[])["ids"]) if collection.count() > 0 else set()

    for week, group in df.groupby("Week"):
        doc_id = f"news-week-{week}"
        if doc_id in existing_ids:
            continue

        up_days = (group["Label"] == 1).sum()
        down_days = (group["Label"] == 0).sum()

        # Grab a handful of representative headlines instead of all of them
        sample_headlines = []
        for _, row in group.head(3).iterrows():
            for col in headline_cols[:3]:
                val = row[col]
                if isinstance(val, str):
                    clean = val.lstrip("b").strip("'\"")
                    sample_headlines.append(clean[:150])

        sentence = (
            f"Week of {week}: DJIA rose on {up_days} day(s) and fell on {down_days} day(s). "
            f"Notable headlines: " + " | ".join(sample_headlines[:6])
        )

        embedding = ollama.embeddings(model=EMBED_MODEL, prompt=sentence)["embedding"]
        collection.add(
            documents=[sentence],
            embeddings=[embedding],
            metadatas=[{"source": "Combined_News_DJIA.csv", "week": week}],
            ids=[doc_id]
        )
        print(f"Added summary for week {week}")

    print(f"market_news collection now has {collection.count()} documents.")

if __name__ == "__main__":
    main()
