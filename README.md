# NBA MVP Prediction using Machine Learning

UE24CS352A Machine Learning mini-project.
Team member 1: Sinchana Urs (PES1UG24CS626)
Team member 2: Bindu I (PES1UG24CS659)

## What this project does

We follow the approach of the reference paper and try to predict the NBA MVP from player statistics.
Every season we look at the 3 MVP finalists and use 10 per-game stats (games, minutes, FG%, 3P%, FT%,
rebounds, assists, steals, blocks, points). We train two models, both written from scratch with NumPy:

1. Gaussian Naive Bayes
2. Logistic Regression trained with Newton's method

Each model gives a probability of being MVP for every finalist. In "seasonal" mode we pick the finalist
with the highest probability each season. In "pure" mode we pick the top scores over the whole test set.

## Data

We used the Kaggle dataset "NBA Stats (1947-present)". Only two of its files are needed and they are
already in data/raw/:

- Player Per Game.csv
- Player Award Shares.csv

The finalists are the top 3 players by MVP vote share each season. We use seasons 1981 to 2025.
We start at 1981 because MVPs before that were voted by players, and the 3-point line only started
in 1980. The processed file is data/processed/finalists.csv (45 seasons, 135 rows).

## How to run

Python 3 is needed. From the project folder:

    python -m venv venv
    venv\Scripts\activate          (on Mac/Linux: source venv/bin/activate)
    pip install -r requirements.txt

    python src/prepare_data.py --start-year 1981 --end-year 2025
    python src/run_experiment.py --test-start 2016

prepare_data.py builds data/processed/finalists.csv. run_experiment.py trains on seasons 1981-2015,
tests on 2016-2025, prints the results and saves them in the results folder.

    python src/roc_auc.py --test-start 2016

roc_auc.py draws the ROC curve and computes the AUC of both models (results/roc_curve.png, results/auc.csv).

Optional: python src/make_figures.py makes two charts in results/figures.

## Live demo

    python src/demo.py --season 2021

This trains both models only on seasons before 2021 and prints the MVP probability of each 2021 finalist,
the pick of each model and the actual MVP. Any season from 1982 to 2025 works.

To predict a new season, put the stats of three players in a csv file and run:

    python src/demo.py --candidates data/candidates.csv

The csv needs these columns: player,G,MP,FG%,3P%,FT%,TRB,AST,STL,BLK,PTS

Note: the demo trains on all earlier seasons, but run_experiment.py uses one fixed split (train up to 2015),
so a few picks can differ between the two.

## Results (test seasons 2016-2025)

Seasonal winner accuracy (how many of the 10 MVPs were picked correctly):

    Naive Bayes            40%  (4 of 10)
    Logistic Regression    70%  (7 of 10)
    Highest points/game    50%  (simple baseline)
    Random guess           33%

AUC: Naive Bayes 0.54, Logistic Regression 0.65 (random guess is 0.50).

Both models missed 2019 and 2024. More discussion is in docs/report.pdf.

## Files

    src/prepare_data.py     builds finalists.csv from the Kaggle files
    src/models.py           Naive Bayes and logistic regression (NumPy)
    src/run_experiment.py   split, training, evaluation, baselines, check against scikit-learn
    src/roc_auc.py          ROC curve and AUC
    src/make_figures.py     optional charts
    src/demo.py             live demo
    docs/report.pdf         two page write-up
    docs/presentation.pptx  slides
    docs/demo_notes.txt     how we run the demo
    docs/qa_prep.txt        notes for the methodology and code questions
    results/                output of run_experiment.py

## Individual contributions

Sinchana Urs: data preparation, Naive Bayes, evaluation.
Bindu I: logistic regression, report, slides.
Both of us worked on testing the code and preparing the presentation.

## Reference

The paper given by faculty on predicting the NBA MVP with Naive Bayes and logistic regression.
