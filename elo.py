"""
elo.py — Time-aware ELO rating system for international football.

Key design: ratings are updated match-by-match in chronological order.
The ELO value stored for a team is always its rating BEFORE the next match,
so features built from ELO never use future information (no data leakage).
"""

import pandas as pd
from config import ELO_INITIAL, ELO_K_NORMAL, ELO_K_WC


def _expected_score(rating_a: float, rating_b: float) -> float:
    """ELO expected score for team A against team B."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def _k_factor(is_wc: bool) -> float:
    return ELO_K_WC if is_wc else ELO_K_NORMAL


def compute_elo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Walk through all matches chronologically and compute ELO ratings.

    Returns the input dataframe with four new columns:
        home_elo_before  — home team ELO at kick-off
        away_elo_before  — away team ELO at kick-off
        home_elo_after   — home team ELO after result
        away_elo_after   — away team ELO after result
    """
    df = df.sort_values("date").copy()

    ratings: dict[str, float] = {}
    last_played: dict[str, str] = {}   # team → ISO date of most recent match

    home_before, away_before = [], []
    home_after,  away_after  = [], []

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        date_str = str(row["date"])[:10]

        r_h = ratings.get(h, ELO_INITIAL)
        r_a = ratings.get(a, ELO_INITIAL)

        home_before.append(r_h)
        away_before.append(r_a)

        # Actual scores (1 = win, 0.5 = draw, 0 = loss)
        if row["result"] == "H":
            s_h, s_a = 1.0, 0.0
        elif row["result"] == "A":
            s_h, s_a = 0.0, 1.0
        else:
            s_h, s_a = 0.5, 0.5

        k = _k_factor(row["is_wc"])
        e_h = _expected_score(r_h, r_a)
        e_a = 1 - e_h

        new_r_h = r_h + k * (s_h - e_h)
        new_r_a = r_a + k * (s_a - e_a)

        ratings[h] = new_r_h
        ratings[a] = new_r_a
        last_played[h] = date_str
        last_played[a] = date_str

        home_after.append(new_r_h)
        away_after.append(new_r_a)

    df["home_elo_before"] = home_before
    df["away_elo_before"] = away_before
    df["home_elo_after"]  = home_after
    df["away_elo_after"]  = away_after

    print(f"[elo] ELO computed for {len(ratings)} teams")
    return df, ratings, last_played


def get_current_ratings(df: pd.DataFrame) -> dict[str, float]:
    """
    Return the latest ELO rating per team (after all matches in df).
    Uses home_elo_after / away_elo_after from the last match each team played.
    """
    _, ratings, _ = compute_elo(df)
    return ratings


