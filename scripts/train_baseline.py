import os
import sys
import logging
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, roc_auc_score, precision_score, recall_score
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline logistic regression model on news embeddings.")
    parser.add_argument('--root', type=str, default=str(Path(__file__).resolve().parent.parent),
                        help="Project root directory")
    parser.add_argument('--emb_file', type=str, default="data/news_embeddings.npz",
                        help="Relative path to embeddings NPZ file")
    parser.add_argument('--aligned_file', type=str, default="data/news_price_aligned.csv",
                        help="Relative path to aligned CSV file")
    parser.add_argument('--out_model', type=str, default="models/baseline_lr.pkl",
                        help="Relative path to save the model")
    parser.add_argument('--n_splits', type=int, default=5,
                        help="Number of splits for TimeSeriesSplit")
    parser.add_argument('--max_iter', type=int, default=1000,
                        help="Max iterations for LogisticRegression")
    parser.add_argument('--random_state', type=int, default=42,
                        help="Random state for reproducibility")
    parser.add_argument('--class_weight', type=str, default='balanced',
                        choices=['balanced', 'none'],
                        help="Class weight for LogisticRegression: 'balanced' or 'none'")
    parser.add_argument('--n_jobs', type=int, default=1,
                        help="Number of jobs for cross_validate (-1 for all cores)")
    # NEW: Choose prediction timeframe
    parser.add_argument('--timeframe', type=str, default='r24',
                        choices=['r1', 'r4', 'r24'],
                        help="Prediction timeframe: r1 (1h), r4 (4h), or r24 (24h)")
    # NEW: Minimum return threshold (helps filter noise)
    parser.add_argument('--min_return', type=float, default=0.0,
                        help="Minimum return to classify as positive (e.g., 0.02 for 2%)")
    return parser.parse_args()


def safe_roc_auc_score(y_true, y_proba):
    """
    Safe ROC AUC scorer that returns NaN if only one class is present or other issues.
    Expects y_proba to be the predicted probability of the positive class.
    """
    try:
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            # not enough classes to compute AUC
            return np.nan
        return roc_auc_score(y_true, y_proba)
    except Exception:
        return np.nan


def main():
    args = parse_args()
    ROOT = Path(args.root)
    EMB_FILE = ROOT / args.emb_file
    ALIGNED = ROOT / args.aligned_file
    OUT_MODEL = ROOT / args.out_model
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading aligned data: {ALIGNED}")
    try:
        df = pd.read_csv(ALIGNED)
    except FileNotFoundError:
        logger.error(f"Aligned file not found: {ALIGNED}. Ensure alignment script has run.")
        sys.exit(1)

    logger.info(f"Loading embeddings: {EMB_FILE}")
    try:
        npz = np.load(EMB_FILE, allow_pickle=True)
        embeddings = npz["embeddings"]
    except FileNotFoundError:
        logger.error(f"Embeddings file not found: {EMB_FILE}. Ensure embedding script has run.")
        sys.exit(1)

    # Filter rows with non-NaN returns for chosen timeframe and valid emb_index
    timeframe_col = args.timeframe
    if timeframe_col not in df.columns or 'emb_index' not in df.columns:
        logger.error(f"Aligned CSV must contain '{timeframe_col}' and 'emb_index' columns.")
        sys.exit(1)

    # FIXED: Filter by chosen timeframe instead of r1
    df = df[~df[timeframe_col].isna() & df["emb_index"].notnull()].copy()
    df["emb_index"] = df["emb_index"].astype(int)

    # Extract features and labels with error handling
    try:
        X = np.vstack([embeddings[i] for i in df["emb_index"].values])
        # FIXED: Use chosen timeframe and minimum return threshold
        y = (df[timeframe_col].values > args.min_return).astype(int)
    except IndexError:
        logger.error("Embedding indices out of bounds. Check data alignment.")
        sys.exit(1)

    logger.info(f"Timeframe: {timeframe_col}, Min return threshold: {args.min_return}")
    logger.info(f"Examples: {X.shape}, Positive rate: {y.mean():.4f}")
    
    # Warn if data is too imbalanced
    if y.mean() < 0.1 or y.mean() > 0.9:
        logger.warning(f"⚠️  Data is very imbalanced ({y.mean():.1%} positive). Consider adjusting --min_return threshold.")

    # Resolve class_weight argument ('none' -> None)
    class_weight = None if args.class_weight == 'none' else args.class_weight

    # Choose solver heuristically
    solver = 'saga' if X.shape[0] * X.shape[1] > 1e6 else 'lbfgs'

    # Build pipeline
    model = LogisticRegression(
        max_iter=args.max_iter,
        class_weight=class_weight,
        random_state=args.random_state,
        solver=solver,
        tol=1e-4,
        n_jobs=None
    )
    pipeline = Pipeline([
        ('scaler', StandardScaler(with_mean=True, with_std=True)),
        ('classifier', model)
    ])

    # Define scorers
    scoring = {
        'accuracy': 'accuracy',
        'roc_auc': make_scorer(safe_roc_auc_score, needs_proba=True),
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0)
    }

    # Time-aware cross-validation
    tscv = TimeSeriesSplit(n_splits=args.n_splits)
    logger.info("Running cross_validate with TimeSeriesSplit...")
    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=tscv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=args.n_jobs,
    )

    # Print and log averaged results
    logger.info("Cross-validation results (means):")
    avg_acc = np.nanmean(cv_results.get('test_accuracy'))
    avg_auc = np.nanmean(cv_results.get('test_roc_auc'))
    avg_prec = np.nanmean(cv_results.get('test_precision'))
    avg_rec = np.nanmean(cv_results.get('test_recall'))

    print(f"\n{'='*50}")
    print(f"RESULTS (predicting {timeframe_col} > {args.min_return:.2%})")
    print(f"{'='*50}")
    print(f"Accuracy  (cv mean): {avg_acc:.4f} ({avg_acc*100:.1f}%)")
    print(f"ROC AUC   (cv mean): {avg_auc:.4f}")
    print(f"Precision (cv mean): {avg_prec:.4f} ({avg_prec*100:.1f}%)")
    print(f"Recall    (cv mean): {avg_rec:.4f} ({avg_rec*100:.1f}%)")
    print(f"{'='*50}\n")

    logger.info(f"Average: acc={avg_acc:.4f}, auc={avg_auc:.4f}, prec={avg_prec:.4f}, rec={avg_rec:.4f}")

    # Train on full data
    logger.info("Training on full dataset")
    pipeline.fit(X, y)

    # Save the pipeline
    joblib.dump(pipeline, OUT_MODEL)
    logger.info(f"✅ Saved model to {OUT_MODEL}")


if __name__ == "__main__":
    main()