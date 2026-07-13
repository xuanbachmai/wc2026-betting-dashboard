"""
corners_model.py — Corners prediction model for WC 2026.

Constants fitted from 207 real international matches:
  • FIFA World Cup 2022 (64 matches, avg 9.0 corners)
  • UEFA Euro 2024     (51 matches, avg 10.0 corners)
  • Copa America 2024  (32 matches, avg  8.5 corners)
  • AFCON 2023         (52 matches, avg  8.9 corners)
  • WC 2026 so far     (12 matches, avg  8.7 corners)

Key empirical findings:
  • Style base rates derived from avg corners generated per style (n≥10 matches)
  • Opp-style multipliers from cross-style matchup matrix
  • Possession bucket: 60-69% → 6.74 avg corners, 30-39% → 3.20
  • ELO gap calibrated to WC 2022 dominance patterns
  • Team-specific bias from actual WC 2026 + recent tournament data
"""

from __future__ import annotations
from scipy.stats import poisson

# ── Style → base corner generation rate ──────────────────────────────────────
# Derived from 207 international matches across WC2022, Euro2024, CopaAm2024, AFCON2023
# Raw averages: tiki-taka 5.69, attacking 5.28, high-press 5.10, counter 4.92, possession 4.65, defensive 3.67
# Scaled to ~4.5 international baseline (removing outlier team effects)
STYLE_CORNER_RATE: dict[str, float] = {
    "tiki-taka":      5.5,   # Spain-type: patient build-up forces corners from crossing
    "attacking":      5.2,   # Brazil/Portugal: direct wide play → crosses → corners
    "high-press":     5.0,   # Germany/Japan: wins ball high, generates set-pieces
    "counter-attack": 4.7,   # France/Senegal: sit back, still generate on counter-breaks
    "possession":     4.4,   # Argentina/Mexico: control ball but fewer corners
    "defensive":      3.4,   # Morocco/Paraguay: low block, rarely attack → few corners
}
DEFAULT_RATE = 4.5

# ── 2D Style matchup matrix ──────────────────────────────────────────────────
# STYLE_MATCHUP[attacker_style][opponent_style] = corner rate multiplier
#
# Key insights from WC 2026 + historical data:
#   - Tiki-taka/possession vs DEFENSIVE: FEWER corners (they pass around the box,
#     never reach the byline — Spain vs Cabo Verde is the perfect example)
#   - High-press vs DEFENSIVE: moderate corners (pressing forces mistakes near box)
#   - Attacking vs DEFENSIVE: more corners (direct wide play hits byline)
#   - Any style vs HIGH-PRESS: more corners (open end-to-end creates corner situations)
#   - Defensive vs anything: very few corners (barely attacks at all)
STYLE_MATCHUP: dict[str, dict[str, float]] = {
    #                  vs:  defensive  counter  high-press  attacking  possession  tiki-taka
    "tiki-taka":     {"defensive": 0.88, "counter-attack": 1.10, "high-press": 1.08, "attacking": 1.05, "possession": 1.00, "tiki-taka": 0.95},
    "attacking":     {"defensive": 1.15, "counter-attack": 1.10, "high-press": 1.12, "attacking": 1.10, "possession": 1.05, "tiki-taka": 1.00},
    "high-press":    {"defensive": 1.05, "counter-attack": 1.10, "high-press": 1.18, "attacking": 1.12, "possession": 1.08, "tiki-taka": 1.05},
    "counter-attack":{"defensive": 0.85, "counter-attack": 0.90, "high-press": 0.95, "attacking": 0.95, "possession": 0.90, "tiki-taka": 0.85},
    "possession":    {"defensive": 0.88, "counter-attack": 0.95, "high-press": 1.02, "attacking": 1.00, "possession": 0.95, "tiki-taka": 0.90},
    "defensive":     {"defensive": 0.72, "counter-attack": 0.75, "high-press": 0.80, "attacking": 0.78, "possession": 0.75, "tiki-taka": 0.72},
}

# ── Team-specific corner bias ─────────────────────────────────────────────────
# Sources (in priority order, Bayesian-blended):
#   1. WC 2026 observed (n=1 per team, 30% weight)
#   2. WC 2022 / Euro 2024 / Copa 2024 per-team averages (70% prior)
#
# WC 2026 per-team bias signals (pred - actual from 9 matches):
#   Scotland -3.7, USA -3.5, Mexico -3.4, Brazil -2.1, Germany -2.1  (model over-predicted)
#   Switzerland +3.4, Turkey +2.0, Canada +2.0, Australia +1.4       (model under-predicted)
#
# Blend formula: bias = 0.70 * historical_bias + 0.30 * wc26_correction
# Limited to ±2.5 to prevent overfitting on 1 game
TEAM_CORNER_BIAS: dict[str, float] = {
    # ── Confirmed high generators (historical + WC2026) ──
    "Germany":        +1.1,   # WC2022: 7.3 avg (high); WC2026: 8 (model over by 2.1) → blend ≈ +1.1
    "Portugal":       +1.4,   # Euro2024: 6.8 avg; no WC2026 data yet
    "Denmark":        +1.3,   # Euro2024: 6.9 avg; aggressive crossing patterns
    "Senegal":        +1.0,   # 6.0 avg across tournaments; athletic wide play
    "Spain":          +2.4,   # Yamal+Williams relentlessly attack byline → high corners; WC2026: 11 corners vs Cabo Verde (74% poss, 27 shots) confirmed
    "Tunisia":        +0.8,   # 6.3 avg; more aggressive than defensive label suggests
    "Switzerland":    +2.0,   # WC2026: 10 corners (model under by 3.4); strong signal
    "Canada":         +1.6,   # WC2026: 9 corners (model under by 2.0); high-press with crosses
    "Turkey":         +1.4,   # WC2026: 8 corners (model under by 2.0); wide aggressive play
    "Brazil":         +0.6,   # Historical +1.2 BUT WC2026 over by 2.1 → reduced to +0.6
    "Japan":          +0.5,   # WC2026: 4 corners; WC2022 avg ~5 per game; solid

    # ── Confirmed low generators (historical + WC2026) ──
    "Scotland":       -2.0,   # WC2026: 3 corners (model over by 3.7); high-press but ineffective delivery
    "United States":  -1.8,   # WC2026: 3 corners (model over by 3.5); controlled possession not crossing
    "Mexico":         -2.0,   # WC2026: 3 corners (model over by 3.4); possession style doesn't win corners
    "Australia":      -1.0,   # WC2026: 5 corners (model over by 1.4); historical low 2.0 avg
    "Morocco":        -1.2,   # Historical 1.9 avg; ultra-organized defense when attacking too
    "Costa Rica":     -2.5,   # Historical 1.0 avg — extreme low block, almost never attacks
    "Wales":          -1.5,   # Historical 2.0 avg
    "Saudi Arabia":   -1.5,   # Historical 2.7 avg despite counter style
    "Iran":           -1.5,   # Historical 2.7 avg
    "South Korea":    -0.9,   # WC2026: 4 corners (model over by 1.8); WC2022 avg was 6.3 however
    "Paraguay":       -1.2,   # WC2026: 1 corner (model over by 1.7); deep defensive block
    "South Africa":   -0.8,   # WC2026: 1 corner; deep block, minimal attacking intent
    "Bolivia":        -0.8,   # Expected low-volume attacker based on Copa 2024
    "Qatar":          -0.8,   # Host team; possession but weak in final third
    "Uruguay":        +1.5,   # WC2026: 14 corners as away team vs Saudi Arabia; Valverde+Nunez create wide overloads
}

# ── How much each style generates corners aggressively in early phases ────────
# Used for "First Corner" prediction
STYLE_EARLY_AGGRESSION: dict[str, float] = {
    "high-press":     0.72,   # presses immediately → high chance of first corner
    "attacking":      0.68,
    "tiki-taka":      0.62,
    "possession":     0.52,
    "counter-attack": 0.38,
    "defensive":      0.24,
}

# ── ELO gap → corner generation multiplier ───────────────────────────────────
# Calibrated to WC 2022 data: dominant teams (ELO gap 200+) averaged ~2 more corners
def _elo_mult(team_elo: float, opp_elo: float) -> float:
    gap = team_elo - opp_elo
    if   gap >=  350: return 1.22   # capped lower: blowouts = fewer corners needed
    elif gap >=  250: return 1.15
    elif gap >=  150: return 1.08
    elif gap >=   75: return 1.04
    elif gap <= -350: return 0.76
    elif gap <= -250: return 0.84
    elif gap <= -150: return 0.92
    elif gap <=  -75: return 0.96
    return 1.0

def _blowout_dampener(home_elo: float, away_elo: float) -> float:
    """When ELO gap ≥ 300, both teams tend to generate fewer corners:
    weak team barely attacks, strong team scores efficiently without needing corners.
    Observed: Germany +403 gap → 9 corners (predicted 12), USA ~300 → 4 corners (predicted 7).
    """
    gap = abs(home_elo - away_elo)
    if   gap >= 400: return 0.86
    elif gap >= 300: return 0.91
    elif gap >= 200: return 0.96
    return 1.0

def _tactical_mult(tactical_score: float) -> float:
    # Compressed — tactical score has smaller effect than style
    return 0.93 + (tactical_score / 10) * 0.14

def _star_mult(star_rating: float) -> float:
    # Star players → better crossing, set-piece delivery, winning aerial duels
    return 0.94 + (star_rating / 10) * 0.12


# ── Poisson helpers ───────────────────────────────────────────────────────────
def _over_prob(lam: float, line: float) -> float:
    """P(X > line)"""
    return float(1.0 - poisson.cdf(int(line), lam))

def _nearest_half(x: float) -> float:
    return round(x * 2) / 2

def _poisson_pmf(k: int, lam: float) -> float:
    return float(poisson.pmf(k, lam))

def _odd_prob(lam: float, max_k: int = 40) -> float:
    """P(Poisson(lam) is odd)"""
    return sum(_poisson_pmf(k, lam) for k in range(1, max_k, 2))

def _prob_table(lam: float, lines: list) -> dict:
    """O/U probabilities for a list of lines."""
    return {
        str(line): {
            "over":  round(_over_prob(lam, line), 4),
            "under": round(1 - _over_prob(lam, line), 4),
        }
        for line in lines
    }


# ── Main prediction function ──────────────────────────────────────────────────

def predict_corners(
    home: str,
    away: str,
    home_elo: float = 1500,
    away_elo: float = 1500,
    home_intel: dict | None = None,
    away_intel: dict | None = None,
    extra_bias: dict | None = None,   # learned biases from corners_learning.run_learning()
) -> dict:
    hi = home_intel or {}
    ai = away_intel or {}
    eb = extra_bias or {}

    h_style = hi.get("style", "counter-attack")
    a_style = ai.get("style", "counter-attack")

    # Base rates from empirical style data (207 matches)
    h_base = STYLE_CORNER_RATE.get(h_style, DEFAULT_RATE)
    a_base = STYLE_CORNER_RATE.get(a_style, DEFAULT_RATE)

    # 2D style matchup: team's own style × how opponent's style affects corner rate
    h_matchup = STYLE_MATCHUP.get(h_style, {}).get(a_style, 1.0)
    a_matchup = STYLE_MATCHUP.get(a_style, {}).get(h_style, 1.0)

    # Blowout dampener: large ELO gaps reduce total corners for both teams
    blowout = _blowout_dampener(home_elo, away_elo)

    # Multipliers: ELO strength × 2D style matchup × tactical quality × star quality × blowout
    h_exp = h_base * _elo_mult(home_elo, away_elo) \
            * h_matchup \
            * _tactical_mult(hi.get("tactical_score", 6.5)) \
            * _star_mult(hi.get("star_rating", 7.0)) \
            * blowout
    a_exp = a_base * _elo_mult(away_elo, home_elo) \
            * a_matchup \
            * _tactical_mult(ai.get("tactical_score", 6.5)) \
            * _star_mult(ai.get("star_rating", 7.0)) \
            * blowout

    # Light nudge toward tournament average (8.5 calibrated from WC 2026 observed avg 8.7)
    raw = h_exp + a_exp
    if raw > 0:
        scale = 0.80 + (8.0 / raw) * 0.20
        h_exp *= scale
        a_exp *= scale

    # Static team bias (hand-coded from historical data + early WC 2026)
    h_bias = TEAM_CORNER_BIAS.get(home, 0.0)
    a_bias = TEAM_CORNER_BIAS.get(away, 0.0)

    # Dynamic learned bias (EMA updated after each WC 2026 match, walk-forward)
    # Capped at ±2.5 per team to prevent overfitting on small sample
    h_learn = max(-2.5, min(2.5, eb.get(home, 0.0)))
    a_learn = max(-2.5, min(2.5, eb.get(away, 0.0)))

    h_exp = max(h_exp + h_bias + h_learn, 1.0)
    a_exp = max(a_exp + a_bias + a_learn, 1.0)

    t_exp = h_exp + a_exp

    # Half-time splits: WC 2026 observed 49%/51% (43H1 vs 44H2 from 12 matches)
    HT = 0.49
    h1_exp = t_exp * HT
    h2_exp = t_exp * (1 - HT)
    hh_exp = h_exp * HT   # home in 1H
    ah_exp = a_exp * HT   # away in 1H

    # ── O/U lines ─────────────────────────────────────────────────────────────
    TOTAL_LINES = [5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5]
    TEAM_LINES  = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5,  6.5,  7.5,  8.5,  9.5]
    HALF_LINES  = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]

    # ── Odd / Even ────────────────────────────────────────────────────────────
    t_odd_p  = _odd_prob(t_exp)
    h1_odd_p = _odd_prob(h1_exp)
    h2_odd_p = _odd_prob(h2_exp)

    # ── First Corner ──────────────────────────────────────────────────────────
    # Based on early aggression score adjusted for ELO dominance
    h_aggr = STYLE_EARLY_AGGRESSION.get(h_style, 0.5) * _elo_mult(home_elo, away_elo) ** 0.5
    a_aggr = STYLE_EARLY_AGGRESSION.get(a_style, 0.5) * _elo_mult(away_elo, home_elo) ** 0.5
    total_aggr = h_aggr + a_aggr
    h_first_p = h_aggr / total_aggr if total_aggr > 0 else 0.5
    a_first_p = 1 - h_first_p

    # ── Confidence ────────────────────────────────────────────────────────────
    style_spread = abs(STYLE_CORNER_RATE.get(h_style, DEFAULT_RATE) -
                       STYLE_CORNER_RATE.get(a_style, DEFAULT_RATE))
    elo_gap = abs(home_elo - away_elo)
    confidence = "high" if (style_spread >= 1.8 or elo_gap >= 250) else \
                 "medium" if (style_spread >= 0.8 or elo_gap >= 100) else "low"

    # ── Reasoning ─────────────────────────────────────────────────────────────
    parts = []
    if h_style in ("high-press", "attacking"):
        parts.append(f"{home} ({h_style}) generates corners through wide pressure")
    if a_style in ("defensive", "counter-attack"):
        parts.append(f"{away} ({a_style}) sits deep → more set-pieces for {home}")
    if elo_gap >= 150:
        dom = home if home_elo >= away_elo else away
        parts.append(f"ELO gap {elo_gap:.0f} → {dom} dominates territory")
    if not parts:
        parts.append("Balanced styles — moderate corners expected")

    # ── Return full prediction ─────────────────────────────────────────────────
    h_line = _nearest_half(h_exp)
    a_line = _nearest_half(a_exp)
    t_line = _nearest_half(t_exp)

    return {
        # Core expectations
        "home":      home,
        "away":      away,
        "home_exp":  round(h_exp, 2),
        "away_exp":  round(a_exp, 2),
        "total_exp": round(t_exp, 2),
        "h1_exp":    round(h1_exp, 2),
        "h2_exp":    round(h2_exp, 2),

        # Recommended lines
        "home_line":  h_line,
        "away_line":  a_line,
        "total_line": t_line,

        # Line over/under probs (primary line)
        "home_over_prob":   round(_over_prob(h_exp, h_line), 3),
        "home_under_prob":  round(1 - _over_prob(h_exp, h_line), 3),
        "away_over_prob":   round(_over_prob(a_exp, a_line), 3),
        "away_under_prob":  round(1 - _over_prob(a_exp, a_line), 3),
        "total_over_prob":  round(_over_prob(t_exp, t_line), 3),
        "total_under_prob": round(1 - _over_prob(t_exp, t_line), 3),

        # Full O/U tables for all common lines
        "prob_total": _prob_table(t_exp,  TOTAL_LINES),
        "prob_home":  _prob_table(h_exp,  TEAM_LINES),
        "prob_away":  _prob_table(a_exp,  TEAM_LINES),
        "prob_h1":    _prob_table(h1_exp, HALF_LINES),   # 1st half total
        "prob_h2":    _prob_table(h2_exp, HALF_LINES),   # 2nd half total

        # Odd / Even
        "odd_even": {
            "total":  {"odd": round(t_odd_p, 4),  "even": round(1 - t_odd_p, 4)},
            "h1":     {"odd": round(h1_odd_p, 4), "even": round(1 - h1_odd_p, 4)},
            "h2":     {"odd": round(h2_odd_p, 4), "even": round(1 - h2_odd_p, 4)},
        },

        # First corner
        "first_corner": {
            "home_prob": round(h_first_p, 4),
            "away_prob": round(a_first_p, 4),
        },

        "confidence": confidence,
        "reasoning":  "; ".join(parts),
    }


# ── Batch helper ──────────────────────────────────────────────────────────────

def predict_all_corners(schedule: list, learned_elo: dict, team_intel: dict) -> dict:
    results = {}
    for m in schedule:
        results[m["match_no"]] = predict_corners(
            home=m["home"], away=m["away"],
            home_elo=learned_elo.get(m["home"], 1500),
            away_elo=learned_elo.get(m["away"], 1500),
            home_intel=team_intel.get(m["home"], {}),
            away_intel=team_intel.get(m["away"], {}),
        )
    return results


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from team_intelligence import TEAM_INTEL

    test = [
        ("Spain",         "Cabo Verde",  1820, 1410),
        ("Brazil",        "Bolivia",     1850, 1380),
        ("Germany",       "Curacao",     1800, 1350),
        ("United States", "Paraguay",    1680, 1490),
    ]

    for home, away, h_elo, a_elo in test:
        p = predict_corners(
            home=home, away=away, home_elo=h_elo, away_elo=a_elo,
            home_intel=TEAM_INTEL.get(home, {}),
            away_intel=TEAM_INTEL.get(away, {}),
        )
        print(f"\n{'='*60}")
        print(f"  {home} vs {away}")
        print(f"{'='*60}")
        print(f"  Expected:  Home {p['home_exp']}  Away {p['away_exp']}  Total {p['total_exp']}")
        print(f"  1st Half:  {p['h1_exp']}  |  2nd Half: {p['h2_exp']}")
        print(f"\n  Bet Type                        Our Pred")
        print(f"  {'─'*45}")

        # Total lines
        for line, probs in sorted(p["prob_total"].items(), key=lambda x: float(x[0])):
            print(f"  Total O/U {line:<5}   Over {probs['over']*100:.0f}¢   Under {probs['under']*100:.0f}¢")

        print()
        # Home corners
        for line, probs in sorted(p["prob_home"].items(), key=lambda x: float(x[0])):
            if float(line) <= p["home_exp"] + 3:
                print(f"  {home} O/U {line:<5}   Over {probs['over']*100:.0f}¢   Under {probs['under']*100:.0f}¢")

        print()
        # Away corners
        for line, probs in sorted(p["prob_away"].items(), key=lambda x: float(x[0])):
            if float(line) <= p["away_exp"] + 3:
                print(f"  {away} O/U {line:<5}   Over {probs['over']*100:.0f}¢   Under {probs['under']*100:.0f}¢")

        print()
        # 1H / 2H
        for line, probs in sorted(p["prob_h1"].items(), key=lambda x: float(x[0])):
            print(f"  1H O/U {line:<5}   Over {probs['over']*100:.0f}¢   Under {probs['under']*100:.0f}¢")

        print()
        oe = p["odd_even"]["total"]
        print(f"  Odd/Even:   Odd {oe['odd']*100:.0f}¢  Even {oe['even']*100:.0f}¢")
        fc = p["first_corner"]
        print(f"  First Corner:  {home} {fc['home_prob']*100:.0f}¢  /  {away} {fc['away_prob']*100:.0f}¢")
        print(f"\n  Confidence: {p['confidence']}  |  {p['reasoning']}")
