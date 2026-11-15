# NBA MVP Prediction using Machine Learning

UE24CS352A Machine Learning mini-project.
Team member 1: Sinchana Urs (PES1UG24CS626)
Team member 2: Bindu I (PES1UG24CS659)

## What this project does

We follow the approach of the reference paper and try to predict the NBA MVP from player statistics.
Every season we look at the 3 MVP finalists and use 10 per-game stats (games, minutes, FG%, 3P%, FT%,
rebounds, assists, steals, blocks, points). We train four models:

1. Gaussian Naive Bayes (written from scratch with NumPy)
2. Logistic Regression trained with Newton's method (written from scratch with NumPy)
3. Random Forest (scikit-learn)
4. Gradient Boosting (scikit-learn)

The paper only used the first two. We added the other two to see if they predict the MVP better.

Each model gives a probability of being MVP for every finalist. In "seasonal" mode we pick the finalist
with the highest probability each season. In "pure" mode we pick the top scores over the whole test set.
Our main metric is the seasonal winner accuracy: in how many test seasons the real MVP was picked.

## Data

We used the Kaggle dataset "NBA Stats (1947-present)". Only two of its files are needed and they are
already in nba-mvp-prediction/data/raw/:

- Player Per Game.csv
- Player Award Shares.csv

The finalists are the top 3 players by MVP vote share each season. We use seasons 1981 to 2025.
We start at 1981 because MVPs before that were voted by players, and the 3-point line only started
in 1980. The processed file is nba-mvp-prediction/data/processed/finalists.csv (45 seasons, 135 rows).

The split is by season, not random: seasons 1981-2015 are used for training (35 seasons, 105 rows)
and seasons 2016-2025 are used for testing (10 seasons, 30 rows).

## How to run

Python 3 is needed. After cloning the repo, go into the project folder:

    cd nba-mvp-prediction
    python -m venv venv
    venv\Scripts\activate          (on Mac/Linux: source venv/bin/activate)
    pip install -r requirements.txt

Then run these in order:

    python src/prepare_data.py --start-year 1981 --end-year 2025
    python src/run_experiment.py --test-start 2016
    python src/roc_auc.py --test-start 2016

prepare_data.py builds data/processed/finalists.csv. run_experiment.py trains the four models on
seasons 1981-2015, tests them on 2016-2025, prints the results and saves them in the results folder.
roc_auc.py draws the ROC curve and computes the AUC of every model (results/roc_curve.png and
results/auc.csv).

Please always use --test-start 2016, because this is the split used for all the results below.
Nothing in the code is random (the random forest and gradient boosting use a fixed seed), so running
it again gives the same numbers.

## Live demo

    python src/demo.py --season 2021

This trains all four models only on the seasons before 2021 and prints the MVP probability of each
2021 finalist, the pick of each model and the actual MVP. Any season from 1982 to 2025 works.
For 2021 the actual MVP is Nikola Jokić: Logistic Regression picks him, Naive Bayes picks Joel Embiid,
and Random Forest and Gradient Boosting pick Stephen Curry.

To predict a new season, put the stats of the candidates in a csv file and run:

    python src/demo.py --candidates data/candidates.csv

The csv needs these columns: player,G,MP,FG%,3P%,FT%,TRB,AST,STL,BLK,PTS

Note: the demo trains on all earlier seasons, but run_experiment.py uses one fixed split (train up to 2015),
so a few picks can differ between the two.

## Results (test seasons 2016-2025)

Seasonal winner accuracy (how many of the 10 MVPs were picked correctly) and AUC:

    Naive Bayes            40%  (4 of 10)    AUC 0.54
    Logistic Regression    70%  (7 of 10)    AUC 0.65
    Random Forest          40%  (4 of 10)    AUC 0.625
    Gradient Boosting      30%  (3 of 10)    AUC 0.60
    Highest points/game    50%  (simple baseline)
    Random guess           33%

Logistic Regression is the best model in our experiment. Random Forest and Gradient Boosting did not
improve on it. All four models missed 2019 and 2024. The test set is small (10 seasons), so one season
changes the accuracy by 10 points and the results should be seen as indicative, not final.

## Files

    README.md                             this file
    Report.pdf                            two page write-up
    Presentation.pptx                     slides
    nba-mvp-prediction/
        requirements.txt                  libraries needed
        .gitignore                        files git should ignore
        data/raw/                         the two Kaggle csv files
        data/processed/finalists.csv      3 finalists per season with their stats
        src/prepare_data.py               builds finalists.csv from the Kaggle files
        src/models.py                     the four models
        src/run_experiment.py             split, training, evaluation, baselines, check against scikit-learn
        src/roc_auc.py                    ROC curve and AUC
        src/demo.py                       live demo
        results/metrics.csv               accuracy of each model
        results/season_predictions.csv    pick of each model for every test season
        results/player_probabilities.csv  probability of each finalist from each model
        results/auc.csv, roc_curve.png    AUC values and the ROC curve

## Individual contributions

Sinchana Urs: data preparation, Naive Bayes, evaluation.
Bindu I: logistic regression, report, slides.
Both of us worked on testing the code and preparing the presentation.

## Reference

The paper given by faculty on predicting the NBA MVP with Naive Bayes and logistic regression.
