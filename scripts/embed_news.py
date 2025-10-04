#!/usr/bin/env python3
import os, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = os.path.dirname(os.path.dirname(__file__))
IN = os.path.join(ROOT, "data", "news_headlines.csv")
OUT_EMB = os.path.join(ROOT, "data", "news_embeddings.npz")

def main():
    if not os.path.exists(IN):
        raise SystemExit(f"{IN} not found. Run news/fetch_news.py first.")
    df = pd.read_csv(IN)
    if df.empty:
        print("No headlines to embed.")
        return
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = df['title'].astype(str).tolist()
    print("Computing embeddings for", len(texts), "headlines...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    np.savez_compressed(OUT_EMB, embeddings=embeddings, titles=np.array(texts), publishedAt=np.array(df['publishedAt'].astype(str)))
    print("Saved embeddings to", OUT_EMB)

if __name__ == "__main__":
    main()
