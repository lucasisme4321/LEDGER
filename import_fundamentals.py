"""
import_fundamentals.py

Import S&P 500 company fundamentals (financials.csv) into the
knowledge base as one natural-language summary per company.
"""

import os
import pandas as pd
import chromadb
import ollama

DB_PATH = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"

def main():
    filepath = os.path.expanduser("~/invest_app/kaggle_data/fundamentals/financials.csv")
    df = pd.read_csv(filepath)

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection("personal_thesis")
    existing_ids = set(collection.get(include=[])["ids"]) if collection.count() > 0 else set()

    added = 0
    for _, row in df.iterrows():
        ticker = str(row["Symbol"]).upper()
        doc_id = f"fundamentals-{ticker}"
        if doc_id in existing_ids:
            continue

        sentence = (
            f"{row['Name']} ({ticker}) operates in the {row['Sector']} sector. "
            f"Current price: ${row['Price']:.2f}, P/E ratio: {row['Price/Earnings']:.2f}, "
            f"dividend yield: {row['Dividend Yield']:.2f}%, EPS: ${row['Earnings/Share']:.2f}. "
            f"52-week range: ${row['52 Week Low']:.2f} to ${row['52 Week High']:.2f}. "
            f"Market cap: ${row['Market Cap']:,.0f}, EBITDA: ${row['EBITDA']:,.0f}. "
            f"Price/Sales: {row['Price/Sales']:.2f}, Price/Book: {row['Price/Book']:.2f}."
        )

        embedding = ollama.embeddings(model=EMBED_MODEL, prompt=sentence)["embedding"]
        collection.add(
            documents=[sentence],
            embeddings=[embedding],
            metadatas=[{"ticker": ticker, "source": "financials.csv"}],
            ids=[doc_id]
        )
        added += 1
        if added % 50 == 0:
            print(f"  ...{added} companies imported")

    print(f"Done. Added {added} company fundamentals.")
    print(f"personal_thesis collection now has {collection.count()} total documents.")

if __name__ == "__main__":
    main()
