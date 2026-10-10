"""
Phases 3-4 - Train both models, evaluate them, and predict the MVP per season.

Usage:
    python src/run_experiment.py --test-start 2017
    (train = seasons < test-start, test = seasons >= test-start; split is by
     season, never random, so no future data leaks into training)

Metrics reported per model:
  threshold acc : classify every finalist as MVP / not-MVP (posterior >= 0.5)
  pure acc      : "pure" mode - pick the top-w players over the whole test set,
                  w = number of test seasons (paper's logistic-regression method)
  seasonal acc  : "seasonal" mode - pick 1 of 3 finalists per season;
                  reported as (a) winner-pick accuracy, (b) per-player label accuracy
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler as SkScaler
from sklearn.pipeline import make_pipeline

from models import GaussianNaiveBayes, LogisticRegressionNewton

FEATURES = ["G", "MP", "FG%", "3P%", "FT%", "TRB", "AST", "STL", "BLK", "PTS"]
DATA_PATH = Path("data/processed/finalists.csv")
RESULTS_DIR = Path("results")


# ---------- evaluation helpers ----------
def seasonal_labels(df, scores):
    """1 for the highest-scoring finalist of each season, else 0."""
    tmp = df[["season"]].copy()
    tmp["score"] = scores
    best = tmp.groupby("season")["score"].transform("max")
    return (tmp["score"] == best).astype(int).to_numpy()


def pure_labels(scores, w):
    """1 for the top-w scores over the whole set, else 0."""
    labels = np.zeros(len(scores), dtype=int)
    labels[np.argsort(-scores)[:w]] = 1
    return labels


def evaluate(name, df, y, proba):
    n_seasons = df["season"].nunique()
    thr = (proba >= 0.5).astype(int)
    pure = pure_labels(proba, n_seasons)
    seas = seasonal_labels(df, proba)
    winner_hits = int(((seas == 1) & (y == 1)).sum())
    return {
        "model": name,
        "threshold_acc": (thr == y).mean(),
        "mvps_predicted_by_threshold": int(thr.sum()),
        "pure_acc": (pure == y).mean(),
        "seasonal_winner_acc": winner_hits / n_seasons,
        "seasonal_label_acc": (seas == y).mean(),
    }


def baseline_row(name, df, y, scores):
    r = evaluate(name, df, y, scores)
    return {k: r[k] for k in ("model", "pure_acc", "seasonal_winner_acc", "seasonal_label_acc")}


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-start", type=int, default=2017)
    ap.add_argument("--l2", type=float, default=1e-2, help="L2 strength for logistic regression")
    args = ap.parse_args()

    data = pd.read_csv(DATA_PATH)
    train = data[data["season"] < args.test_start].reset_index(drop=True)
    test = data[data["season"] >= args.test_start].reset_index(drop=True)
    Xtr, ytr = train[FEATURES].to_numpy(float), train["winner"].to_numpy()
    Xte, yte = test[FEATURES].to_numpy(float), test["winner"].to_numpy()

    # --- verification of the split ---
    print(f"Train: {len(train)} rows / {train['season'].nunique()} seasons "
          f"({train['season'].min()}-{train['season'].max()})")
    print(f"Test : {len(test)} rows / {test['season'].nunique()} seasons "
          f"({test['season'].min()}-{test['season'].max()})")
    assert ytr.sum() == train["season"].nunique(), "each season needs exactly 1 winner"
    assert yte.sum() == test["season"].nunique(), "each season needs exactly 1 winner"

    # --- fit models ---
    nb = GaussianNaiveBayes().fit(Xtr, ytr)
    lr = LogisticRegressionNewton(l2=args.l2).fit(Xtr, ytr)
    p_nb = nb.predict_proba(Xte)[:, 1]
    p_lr = lr.predict_proba(Xte)[:, 1]
    print(f"Logistic regression converged in {lr.n_iter_} Newton iterations")

    # --- metrics ---
    res = pd.DataFrame([evaluate("Naive Bayes", test, yte, p_nb),
                        evaluate("Logistic Regression (Newton)", test, yte, p_lr)])
    pd.options.display.float_format = "{:.3f}".format
    print("\n=== Model comparison (test set) ===")
    print(res.to_string(index=False))

    # --- trivial baselines (models must beat these) ---
    rng = np.random.default_rng(0)
    base = pd.DataFrame([
        baseline_row("Baseline: highest PPG", test, yte, test["PTS"].to_numpy()),
        baseline_row("Baseline: random (seed 0)", test, yte, rng.random(len(test))),
    ])
    print("\n=== Baselines ===")
    print(base.to_string(index=False))

    # --- naive bayes: most indicative features ---
    imp = pd.Series(nb.feature_importance(), index=FEATURES).sort_values(ascending=False)
    print("\nTop 5 indicative features (Naive Bayes):", imp.head(5).index.tolist())

    # --- per-season predictions ---
    rows = []
    for season, g in test.groupby("season"):
        idx = g.index.to_numpy()
        rows.append({
            "season": season,
            "actual": g.loc[g["winner"] == 1, "player"].iloc[0],
            "naive_bayes_pick": g.loc[g.index[np.argmax(p_nb[idx])], "player"],
            "logreg_pick": g.loc[g.index[np.argmax(p_lr[idx])], "player"],
        })
    pred = pd.DataFrame(rows)
    pred["nb_correct"] = pred["actual"] == pred["naive_bayes_pick"]
    pred["lr_correct"] = pred["actual"] == pred["logreg_pick"]
    print("\n=== Predicted MVP per season ===")
    print(pred.to_string(index=False))

    # --- cross-check against scikit-learn (verifies the from-scratch code) ---
    sk_nb = GaussianNB().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    sk_lr = make_pipeline(SkScaler(), LogisticRegression(C=1 / (args.l2 * len(Xtr)), max_iter=1000)
                          ).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    print("\n=== Sanity check vs scikit-learn ===")
    print(f"NB  max |prob diff|: {np.abs(p_nb - sk_nb).max():.4f}")
    print(f"LR  max |prob diff|: {np.abs(p_lr - sk_lr).max():.4f}")

    # --- save ---
    RESULTS_DIR.mkdir(exist_ok=True)
    res.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    pred.to_csv(RESULTS_DIR / "season_predictions.csv", index=False)
    out = test[["season", "player", "winner"]].copy()
    out["p_naive_bayes"], out["p_logreg"] = p_nb, p_lr
    out.to_csv(RESULTS_DIR / "player_probabilities.csv", index=False)
    print(f"\nSaved results to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
