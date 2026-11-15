"""
ROC curve and AUC for all four classification models.

Uses the same split as run_experiment.py (train before --test-start, test from --test-start).
Run from the project folder:
    python src/roc_auc.py --test-start 2016

Output: results/roc_curve.png and results/auc.csv
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

from models import (
    GaussianNaiveBayes, LogisticRegressionNewton,
    RandomForestModel, GradientBoostingModel,
)

FEATURES = ["G", "MP", "FG%", "3P%", "FT%", "TRB", "AST", "STL", "BLK", "PTS"]
DATA_PATH = Path("data/processed/finalists.csv")
RESULTS_DIR = Path("results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-start", type=int, default=2016)
    ap.add_argument("--l2", type=float, default=1e-2)
    args = ap.parse_args()

    data = pd.read_csv(DATA_PATH)
    train = data[data["season"] < args.test_start]
    test = data[data["season"] >= args.test_start]
    Xtr, ytr = train[FEATURES].to_numpy(float), train["winner"].to_numpy()
    Xte, yte = test[FEATURES].to_numpy(float), test["winner"].to_numpy()

    nb = GaussianNaiveBayes().fit(Xtr, ytr)
    lr = LogisticRegressionNewton(l2=args.l2).fit(Xtr, ytr)
    rf = RandomForestModel().fit(Xtr, ytr)
    gb = GradientBoostingModel().fit(Xtr, ytr)
    scores = {
        "Naive Bayes": nb.predict_proba(Xte)[:, 1],
        "Logistic Regression": lr.predict_proba(Xte)[:, 1],
        "Random Forest": rf.predict_proba(Xte)[:, 1],
        "Gradient Boosting": gb.predict_proba(Xte)[:, 1],
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(5.5, 5), dpi=150)
    styles = {
        "Naive Bayes": ("#7a8699", "--"),
        "Logistic Regression": ("#1F3A5F", "-"),
        "Random Forest": ("#2a9d8f", "-."),
        "Gradient Boosting": ("#e76f51", ":"),
    }
    rows = []
    for name, p in scores.items():
        fpr, tpr, _ = roc_curve(yte, p)
        auc = roc_auc_score(yte, p)
        rows.append({"model": name, "auc": auc})
        color, ls = styles[name]
        plt.plot(fpr, tpr, color=color, linestyle=ls, linewidth=2, label=f"{name} (AUC = {auc:.3f})")
        print(f"{name}: AUC = {auc:.3f}")

    plt.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1, label="Random guess (AUC = 0.500)")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC curve, test seasons {test['season'].min()}-{test['season'].max()}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curve.png")
    plt.close()

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "auc.csv", index=False)
    print(f"Test set: {len(yte)} rows, {int(yte.sum())} MVPs")
    print("Saved results/roc_curve.png and results/auc.csv")


if __name__ == "__main__":
    main()
