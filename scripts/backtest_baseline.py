#!/usr/bin/env python3
"""
Robust backtest + classification evaluation for baseline model.

- Loads: models/baseline_lr.pkl
- Uses: data/news_price_aligned.csv + data/news_embeddings.npz + data/BTC_USDT_1h.csv
- Outputs: logs/backtest_baseline_trades.csv
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(ROOT, "models", "baseline_lr.pkl")
ALIGNED_CSV = os.path.join(ROOT, "data", "news_price_aligned.csv")
EMB_NPZ = os.path.join(ROOT, "data", "news_embeddings.npz")
PRICE_CSV = os.path.join(ROOT, "data", "BTC_USDT_1h.csv")
OUT_TRADES = os.path.join(ROOT, "logs", "backtest_baseline_trades.csv")

THRESH = 0.45  # FIXED: lowered from 0.55 to 0.45
PREDICTION_HOURS = 24  # FIXED: added this to match training timeframe

def load_model(path):
    obj = joblib.load(path)
    # handle either saved model or dict with {"model": model}
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj

def build_X_y_from_aligned(aligned_df, emb_npz, target_col='r24'):
    """
    FIXED: Added target_col parameter to choose prediction timeframe
    """
    if "emb_index" not in aligned_df.columns:
        raise SystemExit("aligned csv missing 'emb_index' column")
    emb = np.load(emb_npz, allow_pickle=True)
    if "embeddings" not in emb:
        raise SystemExit("embeddings npz missing 'embeddings' key")
    embeddings = emb["embeddings"]
    
    # build X by indexing embeddings with emb_index
    idxs = aligned_df["emb_index"].astype(int).values
    X = np.vstack([embeddings[i] for i in idxs])
    
    # FIXED: Use specified target column (r24 instead of r1)
    if target_col not in aligned_df.columns:
        raise SystemExit(f"aligned csv missing '{target_col}' column (target)")
    y = (aligned_df[target_col].fillna(0).values > 0).astype(int)
    
    return X, y

def eval_classification(model, X, y, split_frac=0.8):
    n = len(X)
    split = int(n * split_frac)
    X_test = X[split:]
    y_test = y[split:]
    if len(y_test) == 0:
        print("No test samples (aligned dataset too small).")
        return None
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    auc = None
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:,1]
            auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else None
    except Exception:
        auc = None
    print(f"Classification (test) | samples={len(y_test)} acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} auc={(auc if auc is not None else 'N/A')}")
    return {
        "X_test": X_test, "y_test": y_test, "y_pred": y_pred,
        "probs": (probs if 'probs' in locals() else None),
        "test_idx_start": split
    }

def run_trade_sim(model, aligned_df, eval_info, price_csv, threshold=THRESH, hold_hours=PREDICTION_HOURS):
    """
    FIXED: Added hold_hours parameter to match prediction timeframe
    """
    # prepare price series
    price = pd.read_csv(price_csv, parse_dates=['timestamp'])
    price['timestamp'] = pd.to_datetime(price['timestamp'], utc=True)
    price = price.sort_values('timestamp').reset_index(drop=True).set_index('timestamp')

    split_start = eval_info["test_idx_start"]
    X_test = eval_info["X_test"]
    probs = eval_info.get("probs")
    if probs is None:
        # get probabilities via predict_proba if available, otherwise use predicted labels
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:,1]
        else:
            probs = np.array(eval_info["y_pred"])

    trades = []
    # iterate over test rows (use the aligned dataframe indices for timing)
    test_rows = aligned_df.iloc[split_start:].reset_index(drop=True)
    for i, row in test_rows.iterrows():
        # publishedAt should be present
        pub = row.get("publishedAt", None)
        if pd.isna(pub):
            continue
        try:
            pub_ts = pd.to_datetime(pub, utc=True)
        except Exception:
            # skip if cannot parse
            continue
        # locate next price candle >= pub
        pos = price.index.searchsorted(pub_ts)
        
        # FIXED: Hold for hold_hours instead of just 1 hour
        if pos >= len(price) - hold_hours:
            continue
        
        entry_price = price.iloc[pos]['close']
        exit_price = price.iloc[pos + hold_hours]['close']  # FIXED: hold_hours later
        rtn = (exit_price / entry_price) - 1.0
        prob = float(probs[i]) if i < len(probs) else 0.0
        signal = int(prob >= threshold)
        trades.append({
            "publishedAt": str(pub_ts), 
            "entry_ts": str(price.index[pos]), 
            "entry": float(entry_price),
            "exit_ts": str(price.index[pos + hold_hours]), 
            "exit": float(exit_price),
            "pred_proba": prob, 
            "signal": signal, 
            "rtn": float(rtn),
            "hold_hours": hold_hours
        })
    
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        print(f"No trades triggered at threshold {threshold}")
        return trades_df
    
    # filter only signaled buys
    buys = trades_df[trades_df["signal"]==1].copy()
    if buys.empty:
        print(f"No buy trades at threshold {threshold}")
        return buys
    
    # compute simple stats
    avg = buys["rtn"].mean()
    winrate = (buys["rtn"] > 0).mean()
    total = (1+buys["rtn"]).prod() - 1.0
    buys["cum"] = (1+buys["rtn"]).cumprod()
    
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS (hold for {hold_hours} hours)")
    print(f"{'='*60}")
    print(f"Total trades: {len(buys)}")
    print(f"Avg return per trade: {avg:.4%}")
    print(f"Win rate: {winrate:.2%}")
    print(f"Total compounded return: {total:.4%}")
    print(f"{'='*60}\n")
    
    return buys

def main():
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        raise SystemExit("Model not found at " + MODEL_PATH)
    if not os.path.exists(ALIGNED_CSV):
        raise SystemExit("Aligned CSV not found at " + ALIGNED_CSV)
    if not os.path.exists(EMB_NPZ):
        raise SystemExit("Embeddings npz not found at " + EMB_NPZ)
    if not os.path.exists(PRICE_CSV):
        raise SystemExit("Price CSV not found at " + PRICE_CSV)

    model = load_model(MODEL_PATH)
    aligned = pd.read_csv(ALIGNED_CSV)
    print("Aligned columns:", aligned.columns.tolist(), "rows:", len(aligned))

    # FIXED: Use r24 target to match training
    X, y = build_X_y_from_aligned(aligned, EMB_NPZ, target_col='r24')
    eval_info = eval_classification(model, X, y, split_frac=0.8)
    if eval_info is None:
        print("Not enough data to evaluate.")
        return

    # FIXED: Pass hold_hours parameter
    trades_df = run_trade_sim(model, aligned, eval_info, PRICE_CSV, 
                              threshold=THRESH, hold_hours=PREDICTION_HOURS)
    
    # save trades
    if not trades_df.empty:
        trades_df.to_csv(OUT_TRADES, index=False)
        print(f"✅ Saved {len(trades_df)} trades to {OUT_TRADES}")
    else:
        print("No trades saved.")

if __name__ == "__main__":
    main()