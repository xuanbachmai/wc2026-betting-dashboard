"""
simulate.py — Fast Monte Carlo simulation of the 2026 FIFA World Cup.

Optimisation: all pairwise (home_goals_lambda, away_goals_lambda) and
outcome probabilities are pre-computed once before any simulation loop.
Each of the N_SIMULATIONS runs is then pure-numpy Poisson sampling.

2026 format:
  • 12 groups of 4 → round-robin (6 matches/group, 72 total)
  • Top 2 + 8 best 3rd-place teams → Round of 32 (32 teams)
  • Round of 32 → R16 → QF → SF → Final
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations, permutations
from config import WC_2026_GROUPS, N_SIMULATIONS
from features import FEATURE_COLS


# ── Build match feature vector (numpy) ────────────────────────────────────────
# Only real features derived from actual match data are used here.
# Subjective scouting scores (star_rating, squad_depth, etc.) are excluded
# from the model — they are fake data and corrupt predictions.

def _features_np(home: str, away: str, elo: dict, stats: dict) -> np.ndarray:
    h_elo = elo.get(home, 1500)
    a_elo = elo.get(away, 1500)
    h = stats.get(home, {})
    a = stats.get(away, {})

    h_form      = h.get("form", 1.5)
    a_form      = a.get("form", 1.5)
    h_scored    = h.get("avg_scored",   1.2)
    h_conceded  = h.get("avg_conceded", 1.2)
    a_scored    = a.get("avg_scored",   1.2)
    a_conceded  = a.get("avg_conceded", 1.2)

    return np.array([
        # ELO (from actual match history)
        h_elo - a_elo,
        h_elo,
        a_elo,
        # Form & goals (rolling averages from real results)
        h_form,
        a_form,
        h_form - a_form,
        h_scored,
        h_conceded,
        a_scored,
        a_conceded,
        # H2H + venue (from actual historical meetings)
        0.5,   # h2h_home_winrate (neutral prior for future matches)
        1.0,   # is_neutral (all WC on neutral ground)
    ], dtype=np.float32)


def _build_match_features(home: str, away: str, elo: dict, stats: dict) -> pd.DataFrame:
    """Build a single-row DataFrame for predict_proba / predict_lambda calls."""
    return pd.DataFrame(
        [_features_np(home, away, elo, stats)],
        columns=FEATURE_COLS
    )


# ── Pre-compute all pairwise lambdas ──────────────────────────────────────────

def precompute_lambdas(goals_model, elo: dict, stats: dict,
                        all_teams: list[str]) -> dict:
    """
    Build a dict: (home, away) → (lambda_home, lambda_away)
    for every ordered pair of WC teams.
    """
    import pandas as pd
    pairs = list(permutations(all_teams, 2))
    X = pd.DataFrame(
        [_features_np(h, a, elo, stats) for h, a in pairs],
        columns=FEATURE_COLS
    )
    lam_h, lam_a = goals_model.predict_lambda(X)
    lam_h = np.clip(lam_h, 0.2, 5.0)
    lam_a = np.clip(lam_a, 0.2, 5.0)
    return {pair: (lam_h[i], lam_a[i]) for i, pair in enumerate(pairs)}


# ── Single match: sample Poisson score ────────────────────────────────────────

def _play(home: str, away: str, lambdas: dict,
           allow_draw: bool, elo: dict) -> str:
    lh, la = lambdas[(home, away)]
    hg = np.random.poisson(lh)
    ag = np.random.poisson(la)
    if hg > ag:
        return home
    elif ag > hg:
        return away
    else:
        if allow_draw:
            return "DRAW"
        # Penalty shootout: ELO-weighted 50/50
        p = 1 / (1 + 10 ** ((elo.get(away, 1500) - elo.get(home, 1500)) / 800))
        return home if np.random.random() < p else away


# ── Group stage ───────────────────────────────────────────────────────────────

def _simulate_group(teams: list[str], lambdas: dict, elo: dict) -> list[dict]:
    pts = {t: 0 for t in teams}
    gf  = {t: 0 for t in teams}
    ga  = {t: 0 for t in teams}

    for home, away in combinations(teams, 2):
        lh, la = lambdas[(home, away)]
        hg = np.random.poisson(lh)
        ag = np.random.poisson(la)
        gf[home] += hg; ga[home] += ag
        gf[away] += ag; ga[away] += hg
        if hg > ag:   pts[home] += 3
        elif ag > hg: pts[away] += 3
        else:         pts[home] += 1; pts[away] += 1

    standings = sorted(
        teams,
        key=lambda t: (pts[t], gf[t]-ga[t], gf[t]),
        reverse=True
    )
    return [{"team": t, "pos": i+1,
             "pts": pts[t], "gd": gf[t]-ga[t], "gf": gf[t]}
            for i, t in enumerate(standings)]


# ── Full tournament ───────────────────────────────────────────────────────────

def _simulate_one(lambdas: dict, elo: dict) -> dict[str, str]:
    reached = {}

    # Group stage
    all_thirds = []
    firsts, seconds = [], []

    for grp_teams in WC_2026_GROUPS.values():
        standings = _simulate_group(grp_teams, lambdas, elo)
        firsts.append(standings[0]["team"])
        seconds.append(standings[1]["team"])
        all_thirds.append(standings[2])

    # Best 8 third-place teams
    best_thirds = sorted(all_thirds, key=lambda x: (x["pts"], x["gd"], x["gf"]),
                          reverse=True)[:8]
    best_thirds = [x["team"] for x in best_thirds]

    r32 = firsts + seconds + best_thirds   # 12 + 12 + 8 = 32 teams
    for t in r32:
        reached[t] = "R32"

    # Knock out rounds
    round_names = ["R16", "QF", "SF", "Final"]
    remaining = list(r32)
    np.random.shuffle(remaining)

    for rnd in round_names:
        if len(remaining) <= 1:
            break
        winners = []
        for i in range(0, len(remaining) - 1, 2):
            w = _play(remaining[i], remaining[i+1], lambdas,
                      allow_draw=False, elo=elo)
            winners.append(w)
        remaining = winners
        for t in remaining:
            reached[t] = rnd

    if remaining:
        reached[remaining[0]] = "Winner"

    return reached


# ── Run N simulations ─────────────────────────────────────────────────────────

def run_simulations(goals_model, outcome_model,
                    elo_ratings: dict, team_stats: dict) -> pd.DataFrame:
    all_teams = [t for grp in WC_2026_GROUPS.values() for t in grp]

    print("[simulate] Pre-computing pairwise goal lambdas...")
    lambdas = precompute_lambdas(goals_model, elo_ratings, team_stats, all_teams)

    round_order = ["R32", "R16", "QF", "SF", "Final", "Winner"]
    counters = {t: {r: 0 for r in round_order} for t in all_teams}

    print(f"[simulate] Running {N_SIMULATIONS:,} simulations...")
    for _ in tqdm(range(N_SIMULATIONS)):
        result = _simulate_one(lambdas, elo_ratings)
        for team, rnd in result.items():
            if team not in counters:
                continue
            idx = round_order.index(rnd)
            for r in round_order[idx:]:
                counters[team][r] += 1

    rows = []
    for team in all_teams:
        c = counters[team]
        rows.append({
            "team":           team,
            "elo":            round(elo_ratings.get(team, 1500), 1),
            "win_%":          round(100 * c["Winner"] / N_SIMULATIONS, 2),
            "finalist_%":     round(100 * c["Final"]  / N_SIMULATIONS, 2),
            "semifinal_%":    round(100 * c["SF"]     / N_SIMULATIONS, 2),
            "quarterfinal_%": round(100 * c["QF"]     / N_SIMULATIONS, 2),
        })

    return (pd.DataFrame(rows)
              .sort_values("win_%", ascending=False)
              .reset_index(drop=True))
