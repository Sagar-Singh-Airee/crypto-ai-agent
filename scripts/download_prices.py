#!/usr/bin/env python3
import ccxt, pandas as pd, time, os
from datetime import datetime, timedelta, timezone

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(OUTDIR, exist_ok=True)

exchange = ccxt.binance({'enableRateLimit': True})

symbol_list = ["BTC/USDT", "ETH/USDT"]
timeframe = "1h"
limit = 1000

def fetch_all_ohlcv(symbol, timeframe='1h', since=None):
    all_ohl = []
    now = exchange.milliseconds()
    if since is None:
        since = int((datetime.now(tz=timezone.utc) - timedelta(days=365)).timestamp() * 1000)
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        except Exception as e:
            print("Fetch error:", e)
            time.sleep(5)
            continue
        if not ohlcv:
            break
        all_ohl.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        print(f"Fetched {len(all_ohl)} candles so far for {symbol}...")
        time.sleep(exchange.rateLimit / 1000.0 + 0.1)
        if ohlcv[-1][0] >= now - 60_000:
            break
    df = pd.DataFrame(all_ohl, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    return df

def main():
    for sym in symbol_list:
        print("Downloading:", sym)
        df = fetch_all_ohlcv(sym, timeframe=timeframe)
        fname = os.path.join(OUTDIR, f"{sym.replace('/','_')}_{timeframe}.csv")
        df.to_csv(fname, index=False)
        print("Saved", fname, "rows:", len(df))

if __name__ == "__main__":
    main()
