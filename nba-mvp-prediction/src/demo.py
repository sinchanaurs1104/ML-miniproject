
"""
Live demo - predict the MVP with all four models.

Past season:
    Trains only on earlier seasons to prevent data leakage.

Example:
    python src/demo.py --season 2021

New / unlabeled season:
    python src/demo.py --candidates data/candidates.csv

Candidates CSV columns:
    player,G,MP,FG%,3P%,FT%,TRB,AST,STL,BLK,PTS
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from models import (
    GaussianNaiveBayes,
    LogisticRegressionNewton,
    RandomForestModel,
    GradientBoostingModel,
)

FEATURES = [
    "G", "MP", "FG%", "3P%", "FT%",
    "TRB", "AST", "STL", "BLK", "PTS"
]

DATA_PATH = Path("data/processed/finalists.csv")


def show(players, scores, actual=None):
    """Display probabilities and the predicted MVP for each model."""

    out = pd.DataFrame({"player": players})

    for name, probabilities in scores.items():
        out[f"P(MVP) {name}"] = probabilities

    pd.options.display.float_format = "{:.3f}".format

    print("\n=== MVP Probability Scores ===")
    print(out.to_string(index=False))
    print()

    print("=== Predicted MVP by Model ===")

    for name, probabilities in scores.items():
        best_index = int(np.argmax(probabilities))
        print(f"{name}: {players[best_index]}")

    if actual is not None:
        print(f"\nActual MVP: {actual}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict NBA MVP using four classification models."
    )

    parser.add_argument(
        "--season",
        type=int,
        help="Past season year to predict, e.g. 2021"
    )

    parser.add_argument(
        "--candidates",
        type=str,
        help="CSV containing candidates for an unlabeled season"
    )

    args = parser.parse_args()

    if bool(args.season) == bool(args.candidates):
        parser.error("Provide exactly one of --season or --candidates")

    if not DATA_PATH.exists():
        raise SystemExit(
            f"Dataset not found: {DATA_PATH}\n"
            "Run src/prepare_data.py first."
        )

    data = pd.read_csv(DATA_PATH)

    if args.season:
        # Prevent future-season data leakage.
        train = data[data["season"] < args.season].copy()
        test = data[data["season"] == args.season].copy()
        test = test.reset_index(drop=True)

        if train.empty:
            raise SystemExit(
                f"No earlier seasons available to train for {args.season}."
            )

        if test.empty:
            raise SystemExit(
                f"Season {args.season} is not available in the dataset."
            )

        winners = test.loc[test["winner"] == 1, "player"]

        actual = winners.iloc[0] if not winners.empty else None

        header = (
            f"MVP prediction for "
            f"{args.season - 1}-{str(args.season)[2:]}"
        )

    else:
        train = data.copy()
        test = pd.read_csv(args.candidates)
        actual = None
        header = f"MVP prediction for candidates in {args.candidates}"

        required_columns = ["player"] + FEATURES
        missing = [
            col for col in required_columns
            if col not in test.columns
        ]

        if missing:
            raise SystemExit(
                f"Candidates CSV is missing columns: {missing}"
            )

    if train["winner"].nunique() < 2:
        raise SystemExit(
            "Training data must contain both winner and non-winner rows."
        )

    # Validate feature data.
    missing_features = [
        col for col in FEATURES
        if col not in train.columns
    ]

    if missing_features:
        raise SystemExit(
            f"Training dataset is missing features: {missing_features}"
        )

    if train[FEATURES].isna().any().any():
        raise SystemExit(
            "Training data contains missing feature values. "
            "Check data/processed/finalists.csv."
        )

    if test[FEATURES].isna().any().any():
        raise SystemExit(
            "Test/candidate data contains missing feature values."
        )

    Xtr = train[FEATURES].to_numpy(dtype=float)
    ytr = train["winner"].to_numpy(dtype=int)
    X = test[FEATURES].to_numpy(dtype=float)

    print(f"\n{header}")
    print(
        f"Training seasons: {train['season'].nunique()} "
        f"({train['season'].min()}-{train['season'].max()})"
    )
    print(f"Training rows: {len(train)}")
    print(f"Candidates evaluated: {len(test)}")

    # Train all four models using the same training data.
    models = {
        "Naive Bayes": GaussianNaiveBayes(),
        "Logistic Regression": LogisticRegressionNewton(),
        "Random Forest": RandomForestModel(),
        "Gradient Boosting": GradientBoostingModel(),
    }

    scores = {}

    for name, model in models.items():
        model.fit(Xtr, ytr)
        probabilities = model.predict_proba(X)[:, 1]
        scores[name] = probabilities

    # Display probabilities and model predictions.
    show(
        test["player"].tolist(),
        scores,
        actual
    )


if __name__ == "__main__":
    main()
