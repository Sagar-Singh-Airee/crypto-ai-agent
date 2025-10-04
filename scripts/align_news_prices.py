#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(__file__))
# use the actual price file you have
PRICE_CSV = os.path.join(ROOT, "data", "BTC_USDT_1h.csv")
NEWS_EMB = os.path.join(ROOT, "data", "news_embeddings.npz")
OUT = os.path.join(ROOT, "data", "news_price_aligned.csv")

def load_price_df():
    # Read price CSV and force timestamps to UTC (tz-aware)
    df = pd.read_csv(PRICE_CSV, parse_dates=['timestamp'], infer_datetime_format=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)  # force tz-aware UTC
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.set_index('timestamp')
    return df

def main():
    if not os.path.exists(PRICE_CSV):
        raise SystemExit(f"{PRICE_CSV} not found. Run scripts/download_prices.py first")
    if not os.path.exists(NEWS_EMB):
        raise SystemExit(f"{NEWS_EMB} not found. Run scripts/embed_news.py first")

    price = load_price_df()
    print("Price rows:", len(price))

    npz = np.load(NEWS_EMB, allow_pickle=True)
    # handle different possible key names (titles/publishedAt/embeddings)
    keys = list(npz.keys())
    # expected: 'embeddings', 'titles', 'publishedAt'
    if 'embeddings' not in keys:
        raise SystemExit("news_embeddings.npz missing 'embeddings' array")
    embeddings = npz['embeddings']
    if 'publishedAt' in keys:
        published_raw = npz['publishedAt']
    elif 'timestamps' in keys:
        published_raw = npz['timestamps']
    else:
        # fallback: try to load titles length and set published to now (rare)
        published_raw = np.array([datetime.utcnow().isoformat()] * embeddings.shape[0])

    # parse published timestamps and force UTC
    published = pd.to_datetime(published_raw, utc=True, errors='coerce')

    titles = npz.get('titles', np.array([str(i) for i in range(len(published))], dtype=object)).astype(str)

    rows = []
    for i, (t, pub) in enumerate(zip(titles, published)):
        if pd.isna(pub):
            # if parsing failed, skip
            continue
        # find the next price index >= published time
        idx = price.index.searchsorted(pub)
        if idx >= len(price):
            # published after last price timestamp — skip
            continue
        entry_price = price.iloc[idx]['close']
        def ret_ahead(hours):
            idx2 = idx + hours
            if idx2 >= len(price):
                return np.nan
            future = price.iloc[idx2]['close']
            return (future - entry_price) / entry_price

        r1 = ret_ahead(1)
        r4 = ret_ahead(4)
        r24 = ret_ahead(24)
        rows.append({
            "title": t,
            "publishedAt": str(pub),
            "r1": r1, "r4": r4, "r24": r24,
            "emb_index": i
        })

    outdf = pd.DataFrame(rows)
    outdf.to_csv(OUT, index=False)
    print("Saved aligned dataset:", OUT, "rows:", len(outdf))

if __name__ == "__main__":
    main()