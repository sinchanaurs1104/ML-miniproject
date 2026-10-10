"""
Live demo - predict the MVP with both models.

Past season (trains only on EARLIER seasons, so there is no leakage):
    python src/demo.py --season 2021

New / unlabeled season (trains on every season in finalists.csv):
    python src/demo.py --candidates data/candidates.csv

candidates.csv columns: player,G,MP,FG%,3P%,FT%,TRB,AST,STL,BLK,PTS
(one row per candidate, per-game stats for that season)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from models import GaussianNaiveBayes, LogisticRegressionNewton

FEATURES = ["G", "MP", "FG%", "3P%", "FT%", "TRB", "AST", "STL", "BLK", "PTS"]
DATA_PATH = Path("data/processed/finalists.csv")


def show(players, p_nb, p_lr, actual=None):
    out = pd.DataFrame({"player": players,
                        "P(MVP) Naive Bayes": p_nb,
                        "P(MVP) Logistic Reg.": p_lr})
    pd.options.display.float_format = "{:.3f}".format
    print(out.to_string(index=False))
    print()
    print(f"Naive Bayes pick        : {players[int(np.argmax(p_nb))]}")
    print(f"Logistic Regression pick: {players[int(np.argmax(p_lr))]}")
    if actual:
        print(f"Actual MVP              : {actual}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, help="past season to predict, e.g. 2021")
    ap.add_argument("--candidates", type=str, help="CSV of candidates for an unlabeled season")
    args = ap.parse_args()
    if bool(args.season) == bool(args.candidates):
        ap.error("give exactly one of --season or --candidates")

    data = pd.read_csv(DATA_PATH)

    if args.season:
        train = data[data["season"] < args.season]
        test = data[data["season"] == args.season].reset_index(drop=True)
        if test.empty or train.empty:
            raise SystemExit(f"Season {args.season} not available (or nothing earlier to train on).")
        actual = test.loc[test["winner"] == 1, "player"].iloc[0]
        header = f"MVP prediction for {args.season - 1}-{str(args.season)[2:]}"
    else:
        train = data
        test = pd.read_csv(args.candidates)
        actual, header = None, "MVP prediction for candidates in " + args.candidates

    Xtr, ytr = train[FEATURES].to_numpy(float), train["winner"].to_numpy()
    nb = GaussianNaiveBayes().fit(Xtr, ytr)
    lr = LogisticRegressionNewton().fit(Xtr, ytr)

    X = test[FEATURES].to_numpy(float)
    print(f"{header}  (trained on {train['season'].nunique()} seasons: "
          f"{train['season'].min()}-{train['season'].max()})\n")
    show(test["player"].tolist(), nb.predict_proba(X)[:, 1], lr.predict_proba(X)[:, 1], actual)


if __name__ == "__main__":
    main()
