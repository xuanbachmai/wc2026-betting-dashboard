"""
main.py — World Cup 2026 Prediction System
==========================================

Usage:
    python main.py                  # Full pipeline: train + simulate
    python main.py --list-teams     # Print all team names in the dataset
    python main.py --match "Brazil" "Argentina"  # Predict a specific match

Pipeline:
    1. Download historical match data (cached)
    2. Compute ELO ratings (time-aware)
    3. Build feature matrix (no data leakage)
    4. Train XGBoost outcome model + Poisson goals model
    5. Evaluate on 2022 WC period
    6. Run 10,000 Monte Carlo simulations of 2026 WC
    7. Print & save results
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import data
import elo as elo_module
import features as feat_module
import model as model_module
import simulate as sim_module
from features import FEATURE_COLS


# ── Utilities ─────────────────────────────────────────────────────────────────

def build_team_stats_lookup(feat_df: pd.DataFrame) -> dict:
    """
    Extract the most recent form/goals stats per team from the feature matrix.
    Used to provide team_stats to the simulator.
    """
    stats = {}
    for team in set(feat_df["home_team"]) | set(feat_df["away_team"]):
        home_rows = feat_df[feat_df["home_team"] == team].tail(1)
        away_rows = feat_df[feat_df["away_team"] == team].tail(1)

        if len(home_rows) > 0:
            r = home_rows.iloc[-1]
            stats[team] = {
                "form":         r["home_form"],
                "avg_scored":   r["home_avg_scored"],
                "avg_conceded": r["home_avg_conceded"],
            }
        elif len(away_rows) > 0:
            r = away_rows.iloc[-1]
            stats[team] = {
                "form":         r["away_form"],
                "avg_scored":   r["away_avg_scored"],
                "avg_conceded": r["away_avg_conceded"],
            }
    return stats


def predict_match(home: str, away: str, outcome_model, goals_model,
                  elo_ratings: dict, team_stats: dict):
    """Print detailed prediction for one match."""
    from simulate import _build_match_features

    X = _build_match_features(home, away, elo_ratings, team_stats)
    proba = outcome_model.predict_proba(X)[0]  # [P(away), P(draw), P(home)]
    lam_h, lam_a = goals_model.predict_lambda(X)

    print(f"\n{'─'*50}")
    print(f"  {home}  vs  {away}")
    print(f"{'─'*50}")
    print(f"  ELO:  {elo_ratings.get(home, 1500):.0f}  vs  {elo_ratings.get(away, 1500):.0f}")
    print(f"  Expected goals:  {lam_h[0]:.2f}  –  {lam_a[0]:.2f}")
    print(f"  P(home win):  {proba[2]*100:.1f}%")
    print(f"  P(draw):      {proba[1]*100:.1f}%")
    print(f"  P(away win):  {proba[0]*100:.1f}%")
    print(f"{'─'*50}\n")


def save_results(results: pd.DataFrame):
    """Save simulation results to CSV and a bar chart."""
    Path("output").mkdir(exist_ok=True)
    csv_path = "output/wc2026_predictions.csv"
    results.to_csv(csv_path, index=False)
    print(f"\n[output] Results saved to {csv_path}")

    # Bar chart — top 16 teams by win probability
    top16 = results.head(16)
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(top16["team"][::-1], top16["win_%"][::-1], color="steelblue")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_xlabel("Win Probability (%)")
    ax.set_title("2026 FIFA World Cup — Win Probabilities (Monte Carlo)")
    ax.set_xlim(0, top16["win_%"].max() * 1.2)
    plt.tight_layout()
    chart_path = "output/wc2026_chart.png"
    plt.savefig(chart_path, dpi=150)
    print(f"[output] Chart saved to {chart_path}")

    # Full table print
    print("\n" + "═"*65)
    print(f"{'Rank':<5} {'Team':<25} {'Win%':>6} {'Final%':>8} {'Semi%':>7} {'QF%':>6}")
    print("─"*65)
    for i, row in results.iterrows():
        print(f"{i+1:<5} {row['team']:<25} {row['win_%']:>6.2f} "
              f"{row['finalist_%']:>8.2f} {row['semifinal_%']:>7.2f} "
              f"{row['quarterfinal_%']:>6.2f}")
    print("═"*65)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="World Cup 2026 Predictor")
    parser.add_argument("--list-teams", action="store_true",
                        help="Print all available team names")
    parser.add_argument("--match", nargs=2, metavar=("HOME", "AWAY"),
                        help="Predict a specific match: --match Brazil Argentina")
    parser.add_argument("--refresh-data", action="store_true",
                        help="Re-download match data even if cached")
    args = parser.parse_args()

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print("\n=== Step 1: Loading data ===")
    df = data.load()

    if args.list_teams:
        teams = data.get_all_teams(df)
        print(f"\n{len(teams)} teams found:\n")
        for i, t in enumerate(teams, 1):
            print(f"  {i:3}. {t}")
        return

    # ── Step 2: ELO ratings ───────────────────────────────────────────────────
    print("\n=== Step 2: Computing ELO ratings ===")
    df_elo, elo_ratings, elo_last_played = elo_module.compute_elo(df)

    # ── Step 3: Feature engineering ───────────────────────────────────────────
    print("\n=== Step 3: Building features ===")
    feat_df = feat_module.build_features(df_elo)

    # ── Step 4: Train models ──────────────────────────────────────────────────
    print("\n=== Step 4: Training models ===")
    outcome_model, goals_model = model_module.train_all(feat_df)

    print("\nTop feature importances (XGBoost):")
    fi = outcome_model.feature_importance()
    for feat, imp in fi.head(6).items():
        print(f"  {feat:<30} {imp:.4f}")

    # ── Step 5: Team stats lookup ─────────────────────────────────────────────
    team_stats = build_team_stats_lookup(feat_df)

    # ── Single match prediction mode ──────────────────────────────────────────
    if args.match:
        home, away = args.match
        predict_match(home, away, outcome_model, goals_model, elo_ratings, team_stats)
        return

    # ── Step 6: Simulate tournament ───────────────────────────────────────────
    print("\n=== Step 5: Simulating 2026 World Cup ===")
    results = sim_module.run_simulations(goals_model, outcome_model,
                                          elo_ratings, team_stats)

    # ── Step 7: Save and display ──────────────────────────────────────────────
    print("\n=== Step 6: Results ===")
    save_results(results)


if __name__ == "__main__":
    np.random.seed(42)
    main()
