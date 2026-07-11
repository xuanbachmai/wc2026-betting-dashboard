"""
features.py — Build the feature matrix for match outcome prediction.

All features use only data available BEFORE each match (no leakage).

Features per match:
    elo_diff          — home ELO minus away ELO at kick-off
    home_form         — home team pts/game in last FORM_GAMES matches
    away_form         — same for away team
    form_diff         — home_form - away_form
    home_goals_scored — rolling avg goals scored (home perspective)
    home_goals_conceded
    away_goals_scored
    away_goals_conceded
    h2h_home_winrate  — home team win rate vs this opponent (last 10 h2h)
    is_neutral        — 1 if neutral ground
    target_result     — 0=away win, 1=draw, 2=home win
    target_home_goals
    target_away_goals
"""

import pandas as pd
import numpy as np
from config import FORM_GAMES, GOALS_WINDOW


# ── Vectorised per-team rolling stats ─────────────────────────────────────────

def _build_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-team rolling form/goals pooled across ALL matches regardless
    of home/away role.  This removes the fake venue asymmetry: at a neutral-
    ground tournament (WC 2026) it makes no sense that team A has different
    "form" depending on which column they appear in.
    Returns a lookup keyed by (match_id, team_role) for backward compatibility.
    """
    home = df[["date", "home_team", "home_score", "away_score", "result"]].copy()
    home.columns = ["date", "team", "goals_scored", "goals_conceded", "raw_result"]
    home["match_id"]  = df.index
    home["is_home"]   = True
    home["points"]    = home["raw_result"].map({"H": 3, "D": 1, "A": 0})

    away = df[["date", "away_team", "away_score", "home_score", "result"]].copy()
    away.columns = ["date", "team", "goals_scored", "goals_conceded", "raw_result"]
    away["match_id"]  = df.index
    away["is_home"]   = False
    away["points"]    = away["raw_result"].map({"A": 3, "D": 1, "H": 0})

    # Pool home + away rows together — form is about the TEAM, not the venue role
    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["team", "date"]).reset_index(drop=True)

    # Shift(1) so we only use PAST games (no data leakage).
    # Plain rolling mean — no manual decay constants.
    long["form"] = (
        long.groupby("team")["points"]
        .transform(lambda s: s.shift(1).rolling(FORM_GAMES, min_periods=1).mean())
    )
    long["avg_scored"] = (
        long.groupby("team")["goals_scored"]
        .transform(lambda s: s.shift(1).rolling(GOALS_WINDOW, min_periods=1).mean())
    )
    long["avg_conceded"] = (
        long.groupby("team")["goals_conceded"]
        .transform(lambda s: s.shift(1).rolling(GOALS_WINDOW, min_periods=1).mean())
    )

    return long[["match_id", "team", "is_home", "form", "avg_scored", "avg_conceded"]]


# ── Vectorised H2H win rate ────────────────────────────────────────────────────

def _build_h2h_winrate(df: pd.DataFrame) -> pd.Series:
    """
    For each match, compute team1 win rate (regardless of home/away role)
    in the 10 most recent head-to-head meetings.

    "team1" is whichever team is listed in the home_team column — but we
    count wins from BOTH home and away appearances to remove venue bias.
    At WC prediction time this is overridden with 0.5 (neutral prior)
    via is_neutral=1 and the simulate.py hardcode.

    Returns a Series aligned with df.index.
    """
    df = df.reset_index(drop=True)

    # Create a canonical pair key (alphabetical so both sides hash the same)
    df["_pair"] = df.apply(
        lambda r: tuple(sorted([r["home_team"], r["away_team"]])), axis=1
    )

    h2h_rates = []

    for idx, row in df.iterrows():
        pair = row["_pair"]
        home = row["home_team"]
        date = row["date"]

        # Past meetings between this pair (both home AND away appearances)
        hist = df[(df["_pair"] == pair) & (df["date"] < date)].tail(10)
        if len(hist) == 0:
            h2h_rates.append(0.5)
            continue

        # Count wins regardless of which column the team appeared in
        wins = (
            ((hist["home_team"] == home) & (hist["result"] == "H")).sum() +
            ((hist["away_team"] == home) & (hist["result"] == "A")).sum()
        )
        h2h_rates.append(wins / len(hist))

    df.drop(columns=["_pair"], inplace=True)
    return pd.Series(h2h_rates, index=df.index)


# ── Main feature builder ───────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all features onto the match dataframe.
    Returns a clean feature matrix ready for model training.
    """
    df = df.sort_values("date").reset_index(drop=True)

    # 1. Rolling team stats
    stats = _build_team_stats(df)
    home_s = stats[stats["is_home"]].rename(columns={
        "form": "home_form", "avg_scored": "home_avg_scored",
        "avg_conceded": "home_avg_conceded"
    })
    away_s = stats[~stats["is_home"]].rename(columns={
        "form": "away_form", "avg_scored": "away_avg_scored",
        "avg_conceded": "away_avg_conceded"
    })

    feat = df.copy()
    feat = feat.merge(
        home_s[["match_id", "home_form", "home_avg_scored", "home_avg_conceded"]],
        left_index=True, right_on="match_id", how="left"
    ).set_index("match_id")
    feat = feat.merge(
        away_s[["match_id", "away_form", "away_avg_scored", "away_avg_conceded"]],
        left_index=True, right_on="match_id", how="left"
    ).set_index("match_id")

    # 2. ELO diff
    feat["elo_diff"] = feat["home_elo_before"] - feat["away_elo_before"]
    feat["form_diff"] = feat["home_form"] - feat["away_form"]

    # 3. H2H
    print("[features] Computing head-to-head stats...")
    feat["h2h_home_winrate"] = _build_h2h_winrate(feat.reset_index(drop=True))

    # 4. Neutral
    feat["is_neutral"] = feat["neutral"].astype(int)

    # NOTE: team_intelligence (star_rating, squad_depth, etc.) are manually
    # assigned subjective scores — NOT used in model training to avoid
    # contaminating the ML with fake data. They remain available in
    # factor_engine.py for qualitative factor cards only.

    # 5. Targets
    feat["target_result"]     = feat["result"].map({"A": 0, "D": 1, "H": 2})
    feat["target_home_goals"] = feat["home_score"]
    feat["target_away_goals"] = feat["away_score"]

    feature_cols = [
        "date", "home_team", "away_team",
        # ELO (derived from actual match results)
        "elo_diff", "home_elo_before", "away_elo_before",
        # Form & goals (rolling averages from real match data)
        "home_form", "away_form", "form_diff",
        "home_avg_scored", "home_avg_conceded",
        "away_avg_scored", "away_avg_conceded",
        # H2H + venue (from actual historical meetings)
        "h2h_home_winrate", "is_neutral",
        # Targets
        "target_result", "target_home_goals", "target_away_goals",
    ]
    feat = feat[feature_cols].dropna().reset_index(drop=True)
    print(f"[features] Feature matrix: {len(feat):,} rows × {len(feature_cols)} columns")
    return feat


FEATURE_COLS = [
    # ELO (derived from actual match results)
    "elo_diff", "home_elo_before", "away_elo_before",
    # Form & goals (rolling averages from real match data)
    "home_form", "away_form", "form_diff",
    "home_avg_scored", "home_avg_conceded",
    "away_avg_scored", "away_avg_conceded",
    # H2H + venue (from actual historical meetings)
    "h2h_home_winrate", "is_neutral",
]
