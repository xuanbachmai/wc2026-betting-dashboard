"""
data.py — Download and clean historical international football results.

Source: github.com/martj42/international-results
Columns: date, home_team, away_team, home_score, away_score, tournament, neutral
"""

import requests
import pandas as pd
from pathlib import Path
from config import DATA_URL, MIN_YEAR

CACHE_PATH = Path("data/results.csv")


def download_data(force: bool = False) -> pd.DataFrame:
    """Download match data (cached locally after first download)."""
    CACHE_PATH.parent.mkdir(exist_ok=True)

    if CACHE_PATH.exists() and not force:
        print(f"[data] Loading cached data from {CACHE_PATH}")
        df = pd.read_csv(CACHE_PATH, parse_dates=["date"])
    else:
        print(f"[data] Downloading from {DATA_URL} ...")
        response = requests.get(DATA_URL, timeout=30)
        response.raise_for_status()
        CACHE_PATH.write_bytes(response.content)
        df = pd.read_csv(CACHE_PATH, parse_dates=["date"])
        print(f"[data] Saved to {CACHE_PATH}")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic cleaning and filtering."""
    df = df.copy()

    # Drop rows with missing scores
    df = df.dropna(subset=["home_score", "away_score"])

    # Cast scores to int
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Keep only matches from MIN_YEAR onwards
    df = df[df["date"].dt.year >= MIN_YEAR].copy()

    # Add result column: H = home win, D = draw, A = away win
    df["result"] = "D"
    df.loc[df["home_score"] > df["away_score"], "result"] = "H"
    df.loc[df["home_score"] < df["away_score"], "result"] = "A"

    # Flag whether match is a World Cup match (higher stakes)
    df["is_wc"] = df["tournament"].str.contains("FIFA World Cup", na=False)

    # Sort chronologically (critical for no-leakage feature engineering)
    df = df.sort_values("date").reset_index(drop=True)

    print(f"[data] {len(df):,} matches loaded ({df['date'].min().year}–{df['date'].max().year})")
    return df


def get_all_teams(df: pd.DataFrame) -> list[str]:
    """Return sorted list of all team names in the dataset."""
    teams = set(df["home_team"]) | set(df["away_team"])
    return sorted(teams)


def load() -> pd.DataFrame:
    """Convenience: download + clean in one call."""
    raw = download_data()
    return clean_data(raw)
