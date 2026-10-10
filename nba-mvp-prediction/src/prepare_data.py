"""
Phase 1 (Kaggle version) - Build data/processed/finalists.csv from the Kaggle
"NBA Stats (1947-present)" CSVs. No scraping needed.

Inputs (put these two files in data/raw/):
    data/raw/Player Per Game.csv
    data/raw/Player Award Shares.csv

Usage:
    python src/prepare_data.py --start-year 1981 --end-year 2025
    python src/prepare_data.py --start-year 2001 --end-year 2021   # paper-like range

"Season year" = the year the season ends (2021 -> 2020-21 season).
Finalists = the top 3 players by MVP vote share each season; winner = 1 for the MVP.
"""
import argparse
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/finalists.csv")

# Kaggle column -> project column (the paper's 10 features)
FEATURE_MAP = {
    "g": "G", "mp_per_game": "MP", "fg_percent": "FG%", "x3p_percent": "3P%",
    "ft_percent": "FT%", "trb_per_game": "TRB", "ast_per_game": "AST",
    "stl_per_game": "STL", "blk_per_game": "BLK", "pts_per_game": "PTS",
}
FEATURES = list(FEATURE_MAP.values())
COLUMNS = ["season", "rank", "player"] + FEATURES + ["winner"]


def one_row_per_player_season(pg: pd.DataFrame) -> pd.DataFrame:
    """Traded players have a combined row (TOT / 2TM / 3TM ...) plus one row per team.
    Keep the combined row when it exists, otherwise the only row."""
    team = pg["team"].astype(str)
    pg = pg.assign(_combined=(team.eq("TOT") | team.str.match(r"^\d+TM$")).astype(int))
    pg = pg.sort_values("_combined", ascending=False)
    return pg.drop_duplicates(["player_id", "season"], keep="first").drop(columns="_combined")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=1981)
    ap.add_argument("--end-year", type=int, default=2025)
    args = ap.parse_args()

    pg = pd.read_csv(RAW_DIR / "Player Per Game.csv")
    aw = pd.read_csv(RAW_DIR / "Player Award Shares.csv")

    pg = one_row_per_player_season(pg[pg["lg"] == "NBA"])
    mvp = aw[(aw["award"] == "nba mvp")
             & aw["season"].between(args.start_year, args.end_year)].copy()
    mvp = (mvp.sort_values(["season", "share"], ascending=[True, False])
              .groupby("season").head(3).copy())
    mvp["rank"] = mvp.groupby("season").cumcount() + 1

    df = mvp.merge(pg[["player_id", "season"] + list(FEATURE_MAP)],
                   on=["player_id", "season"], how="left", validate="one_to_one")
    df = df.rename(columns=FEATURE_MAP)
    df["3P%"] = df["3P%"].fillna(0.0)  # no 3-point attempts -> 0, same as the paper
    df["winner"] = df["winner"].astype(int)
    df = df[COLUMNS].sort_values(["season", "rank"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    # Verification
    print(f"Saved {len(df)} rows, {df['season'].nunique()} seasons "
          f"({df['season'].min()}-{df['season'].max()}) -> {OUT_PATH}")
    print("Rows per season (expect 3):", df.groupby("season").size().unique().tolist())
    print("Winners per season (expect 1):", df.groupby("season")["winner"].sum().unique().tolist())
    print("Missing values:", int(df[FEATURES].isna().sum().sum()))
    print("Rank-1 equals winner everywhere:", bool((df[df['rank'] == 1]['winner'] == 1).all()))


if __name__ == "__main__":
    main()
