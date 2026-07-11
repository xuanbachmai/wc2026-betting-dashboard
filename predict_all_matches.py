"""
predict_all_matches.py — Full match-by-match prediction for WC 2026.

What this produces
──────────────────
  GROUP STAGE  : All 72 matches with P(home win / draw / away win),
                 expected goals (xG), most likely scoreline.
  EXPECTED STANDINGS: Most likely 1st/2nd/3rd/4th finish per group.
  KNOCKOUT ROUNDS: Most probable matchup for every slot in R32 → Final,
                   with P(these two teams actually meet) and outcome probs.

Method
──────
  • Group matches are predicted directly (models are deterministic).
  • Knockout bracket is inferred from 10,000 Monte Carlo simulations.
    Each sim records the full match log.  The greedy bracket algorithm
    picks the most common unseen pair for each round slot in order.

Usage
─────
    python predict_all_matches.py
    python predict_all_matches.py --simulations 20000

Output
──────
    output/all_group_matches.csv
    output/all_knockout_matches.csv
    Printed report to stdout
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from collections import defaultdict, Counter
from scipy.stats import poisson
from tqdm import tqdm

import data
import elo as elo_module
import features as feat_module
import model as model_module
from features import FEATURE_COLS
from simulate import _build_match_features, _features_np, precompute_lambdas
from config import WC_2026_GROUPS


# ── Match prediction helper ───────────────────────────────────────────────────

def predict_match(home: str, away: str,
                  outcome_model, goals_model,
                  elo: dict, stats: dict) -> dict:
    """
    Return a full prediction dict for one match.
    Includes: win/draw/loss probabilities, xG, most likely scoreline.
    """
    X = _build_match_features(home, away, elo, stats)
    proba = outcome_model.predict_proba(X)[0]   # [P(away), P(draw), P(home)]
    lam_h_arr, lam_a_arr = goals_model.predict_lambda(X)
    lam_h = float(lam_h_arr[0])
    lam_a = float(lam_a_arr[0])

    # Most likely integer scoreline (up to 6 goals per side)
    best_score = (0, 0)
    best_p     = 0.0
    for hg in range(7):
        for ag in range(7):
            p = poisson.pmf(hg, lam_h) * poisson.pmf(ag, lam_a)
            if p > best_p:
                best_p = p
                best_score = (hg, ag)

    return {
        "home_team":      home,
        "away_team":      away,
        "p_home_win":     round(proba[2] * 100, 1),
        "p_draw":         round(proba[1] * 100, 1),
        "p_away_win":     round(proba[0] * 100, 1),
        "xg_home":        round(lam_h, 2),
        "xg_away":        round(lam_a, 2),
        "likely_score":   f"{best_score[0]}-{best_score[1]}",
        "likely_score_%": round(best_p * 100, 1),
        "favourite":      home if proba[2] > proba[0] else (away if proba[0] > proba[2] else "Draw"),
        "favourite_p":    round(max(proba[0], proba[2]) * 100, 1),
    }


# ── Group stage: predict all 72 matches ──────────────────────────────────────

def predict_group_stage(outcome_model, goals_model, elo, stats) -> pd.DataFrame:
    rows = []
    for grp_name, teams in WC_2026_GROUPS.items():
        for home, away in combinations(teams, 2):
            row = predict_match(home, away, outcome_model, goals_model, elo, stats)
            row["group"]    = grp_name
            row["round"]    = "Group Stage"
            rows.append(row)
    return pd.DataFrame(rows)


# ── Full simulation tracking every match played ───────────────────────────────

def _simulate_one_full(lambdas: dict, elo: dict) -> tuple[dict, list]:
    """
    Run one tournament simulation.
    Returns:
        reached   — {team: best_round}
        match_log — list of (round, team1, team2, winner, goals1, goals2)
    """
    match_log = []
    reached   = {}

    # ── Group stage ──────────────────────────────────────────────────────────
    all_thirds = []
    firsts, seconds = [], []

    for grp_name, grp_teams in WC_2026_GROUPS.items():
        pts = {t: 0 for t in grp_teams}
        gf  = {t: 0 for t in grp_teams}
        ga  = {t: 0 for t in grp_teams}

        for home, away in combinations(grp_teams, 2):
            lh, la = lambdas[(home, away)]
            hg = int(np.random.poisson(lh))
            ag = int(np.random.poisson(la))
            gf[home] += hg; ga[home] += ag
            gf[away] += ag; ga[away] += hg
            if hg > ag:
                pts[home] += 3
                result = home
            elif ag > hg:
                pts[away] += 3
                result = away
            else:
                pts[home] += 1
                pts[away] += 1
                result = "Draw"
            match_log.append((f"Group {grp_name}", home, away, result, hg, ag))

        standings = sorted(
            grp_teams,
            key=lambda t: (pts[t], gf[t] - ga[t], gf[t]),
            reverse=True
        )
        firsts.append(standings[0])
        seconds.append(standings[1])
        all_thirds.append({
            "team": standings[2],
            "pts":  pts[standings[2]],
            "gd":   gf[standings[2]] - ga[standings[2]],
            "gf":   gf[standings[2]],
        })

    # ── Best 8 third-place teams ──────────────────────────────────────────────
    best_thirds = sorted(all_thirds, key=lambda x: (x["pts"], x["gd"], x["gf"]),
                         reverse=True)[:8]
    best_thirds_teams = [x["team"] for x in best_thirds]

    r32 = firsts + seconds + best_thirds_teams   # 12 + 12 + 8 = 32
    for t in r32:
        reached[t] = "R32"

    # ── Knockout rounds ───────────────────────────────────────────────────────
    round_names = ["R32", "R16", "QF", "SF", "Final"]
    remaining   = list(r32)
    np.random.shuffle(remaining)

    for rnd in round_names:
        if len(remaining) <= 1:
            break
        winners = []
        for i in range(0, len(remaining) - 1, 2):
            t1, t2 = remaining[i], remaining[i + 1]
            lh, la = lambdas[(t1, t2)]
            hg = int(np.random.poisson(lh))
            ag = int(np.random.poisson(la))
            if hg > ag:
                winner = t1
            elif ag > hg:
                winner = t2
            else:
                # Penalty shootout — ELO-weighted
                p = 1 / (1 + 10 ** ((elo.get(t2, 1500) - elo.get(t1, 1500)) / 800))
                winner = t1 if np.random.random() < p else t2
            match_log.append((rnd, t1, t2, winner, hg, ag))
            winners.append(winner)
        remaining = winners
        for t in remaining:
            reached[t] = rnd

    if remaining:
        reached[remaining[0]] = "Winner"

    return reached, match_log


def run_full_simulations(goals_model, elo: dict, stats: dict,
                          n: int = 10_000) -> tuple[dict, dict, dict]:
    """
    Run n full simulations.

    Returns:
        team_round_counts  — {team: {round: count}}
        ko_pair_counts     — {round: Counter({frozenset(t1,t2): count})}
        group_pos_counts   — {group: {team: {1:count, 2:count, 3:count, 4:count}}}
    """
    all_teams = [t for grp in WC_2026_GROUPS.values() for t in grp]

    print("[sim] Pre-computing pairwise lambdas...")
    lambdas = precompute_lambdas(goals_model, elo, stats, all_teams)

    round_order = ["R32", "R16", "QF", "SF", "Final", "Winner"]
    team_round_counts = {t: {r: 0 for r in round_order} for t in all_teams}

    ko_rounds     = ["R32", "R16", "QF", "SF", "Final"]
    ko_pair_counts = {rnd: Counter() for rnd in ko_rounds}

    # Track group positions per team per group
    group_pos_counts: dict[str, dict[str, Counter]] = {}
    for grp_name, teams in WC_2026_GROUPS.items():
        group_pos_counts[grp_name] = {t: Counter() for t in teams}

    print(f"[sim] Running {n:,} full-tournament simulations...")
    for _ in tqdm(range(n)):
        reached, match_log = _simulate_one_full(lambdas, elo)

        # Update team round counters
        for team, rnd in reached.items():
            if team not in team_round_counts:
                continue
            idx = round_order.index(rnd)
            for r in round_order[idx:]:
                team_round_counts[team][r] += 1

        # Update KO pair counters & group position counters
        group_pts: dict[str, dict[str, int]] = {g: {t: 0 for t in teams}
                                                  for g, teams in WC_2026_GROUPS.items()}
        group_gf:  dict[str, dict[str, int]] = {g: {t: 0 for t in teams}
                                                  for g, teams in WC_2026_GROUPS.items()}
        group_ga:  dict[str, dict[str, int]] = {g: {t: 0 for t in teams}
                                                  for g, teams in WC_2026_GROUPS.items()}

        for rnd, t1, t2, winner, g1, g2 in match_log:
            if rnd in ko_pair_counts:
                ko_pair_counts[rnd][frozenset([t1, t2])] += 1

            # Reconstruct group standings
            if rnd.startswith("Group "):
                grp = rnd.split(" ")[1]
                if grp in group_pts and t1 in group_pts[grp]:
                    group_pts[grp][t1] += (3 if winner == t1 else (1 if winner == "Draw" else 0))
                    group_pts[grp][t2] += (3 if winner == t2 else (1 if winner == "Draw" else 0))
                    group_gf[grp][t1] += g1; group_ga[grp][t1] += g2
                    group_gf[grp][t2] += g2; group_ga[grp][t2] += g1

        for grp, teams in WC_2026_GROUPS.items():
            standings = sorted(
                teams,
                key=lambda t: (group_pts[grp][t],
                               group_gf[grp][t] - group_ga[grp][t],
                               group_gf[grp][t]),
                reverse=True
            )
            for pos, team in enumerate(standings, 1):
                group_pos_counts[grp][team][pos] += 1

    return team_round_counts, ko_pair_counts, group_pos_counts


# ── Build the knockout bracket from simulation pair counts ────────────────────

def build_ko_bracket(ko_pair_counts: dict, total_sims: int,
                     outcome_model, goals_model, elo: dict, stats: dict) -> pd.DataFrame:
    """
    For each KO round, greedily pick the most-common unseen pair to fill bracket slots.
    Returns a DataFrame of all KO match predictions.
    """
    ko_rounds   = ["R32", "R16", "QF", "SF", "Final"]
    slots_per   = [16, 8, 4, 2, 1]
    rows        = []

    for rnd, n_slots in zip(ko_rounds, slots_per):
        counter    = ko_pair_counts[rnd]
        used_teams: set[str] = set()
        slot       = 1

        # Sort pairs by frequency desc
        sorted_pairs = counter.most_common()

        for pair_frozen, count in sorted_pairs:
            if slot > n_slots:
                break
            pair = list(pair_frozen)
            if pair[0] in used_teams or pair[1] in used_teams:
                continue

            t1, t2       = pair[0], pair[1]
            meet_prob    = round(100 * count / total_sims, 1)

            pred = predict_match(t1, t2, outcome_model, goals_model, elo, stats)
            rows.append({
                "round":        rnd,
                "slot":         slot,
                "team1":        t1,
                "team2":        t2,
                "meet_%":       meet_prob,
                "p_team1_win":  pred["p_home_win"],
                "p_draw":       pred["p_draw"],
                "p_team2_win":  pred["p_away_win"],
                "xg_team1":     pred["xg_home"],
                "xg_team2":     pred["xg_away"],
                "likely_score": pred["likely_score"],
                "likely_score_%": pred["likely_score_%"],
                "predicted_winner": pred["favourite"],
                "predicted_winner_p": pred["favourite_p"],
            })
            used_teams.add(t1)
            used_teams.add(t2)
            slot += 1

    return pd.DataFrame(rows)


# ── Print reports ─────────────────────────────────────────────────────────────

def _bar(p: float, width: int = 20) -> str:
    filled = int(round(p / 100 * width))
    return "█" * filled + "░" * (width - filled)


def print_group_stage(df: pd.DataFrame):
    print("\n" + "═" * 90)
    print("  GROUP STAGE — ALL 72 MATCHES")
    print("═" * 90)

    for grp in sorted(df["group"].unique()):
        grp_df = df[df["group"] == grp]
        print(f"\n  ┌─ GROUP {grp} {'─'*70}")
        for _, r in grp_df.iterrows():
            fav_arrow = "◄" if r["p_home_win"] > r["p_away_win"] else ("►" if r["p_away_win"] > r["p_home_win"] else "●")
            print(
                f"  │  {r['home_team']:<22} {fav_arrow}  {r['away_team']:<22}"
                f"  {r['p_home_win']:>5.1f}% / {r['p_draw']:>4.1f}% / {r['p_away_win']:>5.1f}%"
                f"   xG {r['xg_home']:.1f}–{r['xg_away']:.1f}"
                f"   score: {r['likely_score']} ({r['likely_score_%']:.0f}%)"
            )


def print_group_standings(group_pos_counts: dict, total_sims: int):
    print("\n" + "═" * 90)
    print("  EXPECTED GROUP STANDINGS  (% chance to finish in each position)")
    print("═" * 90)

    for grp_name in sorted(group_pos_counts.keys()):
        teams_data = group_pos_counts[grp_name]
        team_list  = list(WC_2026_GROUPS[grp_name])

        print(f"\n  GROUP {grp_name}")
        print(f"  {'Team':<24} {'1st':>6} {'2nd':>6} {'3rd':>6} {'4th':>6}   Qualify%")
        print(f"  {'─'*55}")

        sorted_teams = sorted(
            team_list,
            key=lambda t: teams_data[t].get(1, 0),
            reverse=True
        )
        for team in sorted_teams:
            c = teams_data[team]
            p1 = 100 * c.get(1, 0) / total_sims
            p2 = 100 * c.get(2, 0) / total_sims
            p3 = 100 * c.get(3, 0) / total_sims
            p4 = 100 * c.get(4, 0) / total_sims
            qualify = p1 + p2   # top 2 guaranteed qualify
            print(f"  {team:<24} {p1:>5.1f}% {p2:>5.1f}% {p3:>5.1f}% {p4:>5.1f}%   {qualify:>5.1f}%")


def print_knockout_bracket(ko_df: pd.DataFrame):
    ROUND_LABELS = {
        "R32":   "ROUND OF 32  (16 matches)",
        "R16":   "ROUND OF 16  (8 matches)",
        "QF":    "QUARTER-FINALS  (4 matches)",
        "SF":    "SEMI-FINALS  (2 matches)",
        "Final": "THE FINAL",
    }

    for rnd in ["R32", "R16", "QF", "SF", "Final"]:
        rnd_df = ko_df[ko_df["round"] == rnd].sort_values("slot")
        if rnd_df.empty:
            continue

        print(f"\n{'═'*90}")
        print(f"  {ROUND_LABELS[rnd]}")
        print(f"{'═'*90}")
        print(f"  {'#':<3} {'Team 1':<22} vs  {'Team 2':<22}  {'Meet%':>5}  "
              f"{'T1 win':>7} {'Draw':>5} {'T2 win':>7}  "
              f"{'xG':>7}  {'Likely score':<10} {'Winner'}")
        print(f"  {'─'*87}")

        for _, r in rnd_df.iterrows():
            winner_str = (f"► {r['predicted_winner']}" if r["predicted_winner"] not in ("Draw", "")
                          else "Draw")
            t1_bold = "▶ " if r["predicted_winner"] == r["team1"] else "  "
            t2_bold = " ◀" if r["predicted_winner"] == r["team2"] else "  "
            print(
                f"  {int(r['slot']):<3} "
                f"{t1_bold}{r['team1']:<20} vs  "
                f"{r['team2']:<20}{t2_bold}  "
                f"{r['meet_%']:>5.1f}%  "
                f"{r['p_team1_win']:>6.1f}%"
                f" {r['p_draw']:>5.1f}%"
                f" {r['p_team2_win']:>6.1f}%  "
                f"{r['xg_team1']:.1f}–{r['xg_team2']:.1f}  "
                f"{r['likely_score']:<8}  "
                f"{winner_str} ({r['predicted_winner_p']:.0f}%)"
            )


# ── Save to CSV ───────────────────────────────────────────────────────────────

def save_outputs(group_df: pd.DataFrame, ko_df: pd.DataFrame):
    Path("output").mkdir(exist_ok=True)

    g_path = "output/all_group_matches.csv"
    group_df.to_csv(g_path, index=False)
    print(f"\n[output] Group matches → {g_path}")

    k_path = "output/all_knockout_matches.csv"
    ko_df.to_csv(k_path, index=False)
    print(f"[output] Knockout matches → {k_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Predict every WC 2026 match")
    parser.add_argument("--simulations", type=int, default=10_000,
                        help="Number of Monte Carlo simulations (default 10,000)")
    args = parser.parse_args()

    np.random.seed(42)

    # ── Pipeline ──────────────────────────────────────────────────────────────
    print("\n=== Loading & training ===")
    df = data.load()
    df_elo, elo_ratings, elo_last_played = elo_module.compute_elo(df)
    feat_df = feat_module.build_features(df_elo)
    outcome_model, goals_model = model_module.train_all(feat_df)

    # Team stats for feature building
    from main import build_team_stats_lookup
    team_stats = build_team_stats_lookup(feat_df)

    # ── Group stage predictions (deterministic) ────────────────────────────────
    print("\n=== Predicting 72 group stage matches ===")
    group_df = predict_group_stage(outcome_model, goals_model, elo_ratings, team_stats)

    # ── Full simulations ───────────────────────────────────────────────────────
    print(f"\n=== Running {args.simulations:,} full-tournament simulations ===")
    team_round_counts, ko_pair_counts, group_pos_counts = run_full_simulations(
        goals_model, elo_ratings, team_stats, n=args.simulations
    )

    # ── Knockout bracket ───────────────────────────────────────────────────────
    print("\n=== Building most-likely knockout bracket ===")
    ko_df = build_ko_bracket(
        ko_pair_counts, args.simulations,
        outcome_model, goals_model, elo_ratings, team_stats
    )

    # ── Print ──────────────────────────────────────────────────────────────────
    print_group_stage(group_df)
    print_group_standings(group_pos_counts, args.simulations)
    print_knockout_bracket(ko_df)
    save_outputs(group_df, ko_df)

    # ── Summary stats ──────────────────────────────────────────────────────────
    print("\n" + "═" * 90)
    print("  TOTAL MATCHES PREDICTED")
    print("═" * 90)
    print(f"  Group stage  : {len(group_df):>3} matches")
    print(f"  Knockout     : {len(ko_df):>3} matches")
    print(f"  Total        : {len(group_df) + len(ko_df):>3} matches")
    print(f"\n  Simulation base: {args.simulations:,} runs")
    print("═" * 90)


if __name__ == "__main__":
    main()
