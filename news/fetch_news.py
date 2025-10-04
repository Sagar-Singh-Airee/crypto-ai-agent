#!/usr/bin/env python3
"""
Robust crypto news fetcher:
- Uses NewsAPI if NEWS_API_KEY is present
- Otherwise scrapes cryptonews.com, cointelegraph.com, coindesk.com with fallback selectors
Saves results to ../data/news_headlines.csv (title + publishedAt + source)
"""
import os
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
NEWS_API_KEY = os.getenv("NEWS_API_KEY", None)

ROOT = os.path.dirname(os.path.dirname(__file__))
OUT = os.path.join(ROOT, "data", "news_headlines.csv")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CryptoNewsBot/1.0; +https://github.com/)"}

def fetch_with_newsapi(query="crypto OR bitcoin OR ethereum", page=1, page_size=100):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "pageSize": page_size,
        "page": page,
        "sortBy": "publishedAt",
        "apiKey": NEWS_API_KEY
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    js = resp.json()
    articles = js.get("articles", [])
    out = []
    for a in articles:
        out.append({
            "title": a.get("title"),
            "publishedAt": a.get("publishedAt") or datetime.now(timezone.utc).isoformat(),
            "source": a.get("source", {}).get("name", "newsapi")
        })
    return out

def scrape_url(url, selectors):
    """Generic scraper: try list of selectors and return list of (title)"""
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return []
    soup = BeautifulSoup(r.text, "lxml")
    results = []
    for sel in selectors:
        for node in soup.select(sel):
            text = node.get_text(separator=" ", strip=True)
            if text:
                results.append(text)
        if results:
            break
    return results

def scrape_sites():
    items = []
    now = datetime.now(timezone.utc).isoformat()
    # cryptonews
    cn_selectors = ["h4.article__title", "h3.title", "a.title"]
    cn = scrape_url("https://cryptonews.com/", cn_selectors)
    for t in cn:
        items.append({"title": t, "publishedAt": now, "source": "cryptonews.com"})
    # cointelegraph
    ct_selectors = ["h2.post-card-inline__title", "a.post-card-inline__title", "h3.post-card-inline__title"]
    ct = scrape_url("https://cointelegraph.com/", ct_selectors)
    for t in ct:
        items.append({"title": t, "publishedAt": now, "source": "cointelegraph.com"})
    # coindesk
    cd_selectors = ["h3.card-title", "a.card-title", "h2.heading"]
    cd = scrape_url("https://www.coindesk.com/", cd_selectors)
    for t in cd:
        items.append({"title": t, "publishedAt": now, "source": "coindesk.com"})
    # fallback: crypto news aggregator (cryptoslate)
    cs_selectors = ["h2.article-title", "a.article-title", "h3.card-title"]
    cs = scrape_url("https://cryptoslate.com/", cs_selectors)
    for t in cs:
        items.append({"title": t, "publishedAt": now, "source": "cryptoslate.com"})
    return items

def dedupe_keep_first(items):
    seen = set()
    out = []
    for it in items:
        txt = (it.get("title") or "").strip()
        if not txt:
            continue
        key = txt.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out

def main():
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    all_items = []
    if NEWS_API_KEY:
        try:
            print("Using NewsAPI...")
            all_items.extend(fetch_with_newsapi(page_size=100))
        except Exception as e:
            print("NewsAPI error:", e)
    if not all_items:
        print("No NewsAPI results or key not provided — scraping sites (cryptonews, cointelegraph, coindesk, cryptoslate)...")
        scraped = scrape_sites()
        all_items.extend(scraped)
    all_items = dedupe_keep_first(all_items)
    if not all_items:
        print("No articles found.")
        return
    df = pd.DataFrame(all_items)
    # ensure publishedAt parseable datetimes
    try:
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], utc=True)
    except Exception:
        df["publishedAt"] = datetime.now(timezone.utc)
    df.to_csv(OUT, index=False)
    print("Saved", OUT, "rows:", len(df))

if __name__ == "__main__":
    main()
