"""
factor_engine.py — Computes contextual probability adjustments beyond the base ML model.

Seven factor categories:
  1. PLAYER PERFORMANCE   — injuries/suspensions reduce effective squad rating
  2. COACH STRATEGY       — pressing intensity vs opponent style, set-piece edge
  3. COUNTER-STRATEGY     — tactical matchup advantages (style vs style matrix)
  4. POLITICAL/MORALE     — travel distance, squad morale, political instability
  5. HOST NATION BONUS    — co-host crowd amplifier (USA/Canada/Mexico)
  6. CONFEDERATION DISCOUNT — ELO built on weak opponents gets haircut
  7. UNDERDOG LOW-BLOCK   — big underdogs park the bus → draw probability boost

Output: probability adjustments (delta applied to xG, then re-normalised through Poisson)
These are added ON TOP of the ELO + ML model output.
"""

from __future__ import annotations
import math
from player_data import get_player_data, COUNTER_MATRIX

# ── Tuning constants ───────────────────────────────────────────────────────────
# Max xG adjustment per factor category (keeps adjustments bounded)
MAX_INJURY_ADJ      = 0.25   # per missing key player → up to 3 players = 0.75
MAX_COACH_ADJ       = 0.15   # pressing/set-piece edge
MAX_COUNTER_ADJ     = 0.15   # tactical matchup ceiling
MAX_POLITICAL_ADJ   = 0.20   # morale/travel/instability

INJURY_PER_PLAYER   = 0.10   # xG reduction per injured key player
MORALE_SCALE        = 0.03   # per morale point above/below 7.0 (neutral)
TRAVEL_THRESHOLD_KM = 10000  # above this → fatigue factor kicks in
TRAVEL_FATIGUE      = 0.04   # xG penalty for extreme travel (>10,000 km)
POLITICAL_RISK_SCALE= 0.03   # per point of political_risk (0–10)

# ── 2026 WC rule changes ───────────────────────────────────────────────────────
# 6 substitutions allowed (up from 5 in 2022).
# Deeper squads can rotate more aggressively and maintain intensity late.
# This amplifies squad_depth advantage — wider squads gain more from this rule.
SUBS_2026           = 6      # (was 5) — affects squad depth factor weight
SQUAD_DEPTH_SCALE   = 0.018  # xG bonus per squad_depth point gap (was implicit 0.01)
# Yellow cards reset after group stage — no suspension carry-over into R32.
# This means yellow card accumulation is only relevant within the group stage.
YELLOW_RESET_AFTER_GROUP = True

# ── New factor constants ───────────────────────────────────────────────────────
# 5. Host nation bonus — co-hosts playing on home soil get ~0.30 extra xG
#    (Research: host nations average +0.35 goals/game over tournament expectation)
HOST_NATIONS        = {"United States", "Canada", "Mexico"}
HOST_XG_BONUS       = 0.30

# 6. Confederation ELO discount — ELO from weak confederations overstates quality
#    CONCACAF/OFC/CAF teams face weaker qualifying opponents than UEFA/CONMEBOL
WEAK_CONF_TEAMS = {
    # CONCACAF non-hosts (hosts get crowd boost instead, not discount)
    "Haiti", "Jamaica", "Panama", "El Salvador", "Honduras", "Cuba",
    "Costa Rica", "Trinidad and Tobago", "Curacao", "Martinique",
    # OFC
    "New Zealand",
    # OFC / CONCACAF minnows only — not broad CAF
    "Cape Verde", "Cabo Verde",
}
CONF_DISCOUNT       = 0.15   # ELO points to subtract from weak conf teams' effective ELO
ELO_DISCOUNT_ADJ    = 0.10   # xG boost for opponent facing a weak-conf team

# 7. Underdog low-block — when ELO gap ≥ 150, heavy underdog parks the bus
#    → reduce underdog's attacking xG slightly, reduce favourite's xG slightly (fewer spaces),
#      net effect: boost draw probability
LOWBLOCK_ELO_THRESHOLD  = 150   # ELO gap to trigger low-block assumption
LOWBLOCK_ATTACK_PENALTY = 0.04  # underdog loses attack (fewer chances trying)
LOWBLOCK_SPACE_PENALTY  = 0.03  # favourite loses some xG (opponent sitting deep)
# Also directly inflate draw probability weight (added in apply_factor_adjustments)
LOWBLOCK_DRAW_BOOST     = 0.06  # raw probability mass shifted toward draw


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def compute_factors(
    home: str,
    away: str,
    pred: dict,          # base model prediction dict
    elo_home: float,
    elo_away: float,
    factor_weights: dict | None = None,   # learned weights from OnlineLearner
) -> dict:
    """
    Compute all contextual factor adjustments.

    Returns:
        {
          'home_xg_adj': float,  # net xG adjustment for home team (can be negative)
          'away_xg_adj': float,
          'factor_cards': [
              {type, side, icon, label, text, magnitude}
          ],
          'summary': {
              'home_injury': bool, 'away_injury': bool,
              'counter_winner': 'home'|'away'|None,
              'political_risk_home': int, 'political_risk_away': int,
              'home_morale': float, 'away_morale': float,
          }
        }
    """
    # Helper: return learned weight for a factor type (default 1.0)
    def _fw(ftype: str) -> float:
        if not factor_weights:
            return 1.0
        return float(factor_weights.get(ftype, 1.0))

    hd = get_player_data(home)
    ad = get_player_data(away)

    home_xg_adj = 0.0
    away_xg_adj = 0.0
    draw_boost  = 0.0   # direct draw probability inflation
    cards: list[dict] = []

    # ── 1. PLAYER PERFORMANCE: Injuries & Suspensions ─────────────────────────
    h_missing = hd["injured"] + hd["suspended"]
    a_missing = ad["injured"] + ad["suspended"]

    if h_missing:
        adj = -_clamp(len(h_missing) * INJURY_PER_PLAYER, 0, MAX_INJURY_ADJ) * _fw("player")
        home_xg_adj += adj
        cards.append({
            "type":      "player",
            "side":      "away",   # benefits away team
            "icon":      "🏥",
            "label":     "Injury / Suspension",
            "text":      f"{home} missing: {', '.join(h_missing)} — reduced attacking threat",
            "magnitude": abs(adj),
        })

    if a_missing:
        adj = +_clamp(len(a_missing) * INJURY_PER_PLAYER, 0, MAX_INJURY_ADJ) * _fw("player")
        away_xg_adj -= adj
        cards.append({
            "type":      "player",
            "side":      "home",   # benefits home team
            "icon":      "🏥",
            "label":     "Injury / Suspension",
            "text":      f"{away} missing: {', '.join(a_missing)} — reduced attacking threat",
            "magnitude": abs(adj),
        })

    # ── 2. COACH STRATEGY: Pressing vs Set-Piece Edge ─────────────────────────
    h_press = hd["pressing_intensity"]
    a_press = ad["pressing_intensity"]
    press_diff = h_press - a_press

    if abs(press_diff) >= 1.5:
        # Better pressing team benefits when facing a ball-playing side
        h_style = hd["style_code"]
        a_style = ad["style_code"]
        if press_diff > 0 and a_style in ("TT", "PO", "AT"):
            adj = _clamp(press_diff * 0.01, 0, MAX_COACH_ADJ) * _fw("coach")
            home_xg_adj += adj
            cards.append({
                "type":      "coach",
                "side":      "home",
                "icon":      "🧠",
                "label":     "Coach Pressing Advantage",
                "text":      (f"{home} ({hd['coach']}, {hd['formation']}) presses at "
                              f"{h_press:.1f}/10 — disrupts {away}'s {a_style} style"),
                "magnitude": adj,
            })
        elif press_diff < 0 and h_style in ("TT", "PO", "AT"):
            adj = _clamp(abs(press_diff) * 0.01, 0, MAX_COACH_ADJ) * _fw("coach")
            away_xg_adj += adj
            cards.append({
                "type":      "coach",
                "side":      "away",
                "icon":      "🧠",
                "label":     "Coach Pressing Advantage",
                "text":      (f"{away} ({ad['coach']}, {ad['formation']}) presses at "
                              f"{a_press:.1f}/10 — disrupts {home}'s {h_style} style"),
                "magnitude": adj,
            })

    # Set-piece edge
    h_sp = hd["set_piece_threat"]
    a_sp = ad["set_piece_threat"]
    sp_diff = h_sp - a_sp
    if abs(sp_diff) >= 1.5:
        side   = "home" if sp_diff > 0 else "away"
        team   = home if sp_diff > 0 else away
        d_     = get_player_data(team)
        adj    = _clamp(abs(sp_diff) * 0.008, 0, 0.10) * _fw("coach")
        if side == "home":
            home_xg_adj += adj
        else:
            away_xg_adj += adj
        cards.append({
            "type":      "coach",
            "side":      side,
            "icon":      "⚽",
            "label":     "Set-Piece Threat",
            "text":      (f"{team} has superior set-piece delivery "
                          f"({h_sp if side=='home' else a_sp:.1f}/10 vs "
                          f"{a_sp if side=='home' else h_sp:.1f}/10)"),
            "magnitude": adj,
        })

    # ── 3. COUNTER-STRATEGY: Tactical Matchup ─────────────────────────────────
    h_code = hd["style_code"]
    a_code = ad["style_code"]

    # Check if home style counters away style
    h_attacks_a = COUNTER_MATRIX.get((h_code, a_code), 0.0)
    # Check if away style counters home style
    a_attacks_h = COUNTER_MATRIX.get((a_code, h_code), 0.0)

    counter_winner = None
    net_counter = h_attacks_a - a_attacks_h

    if abs(net_counter) >= 0.03:
        side = "home" if net_counter > 0 else "away"
        counter_winner = side
        team    = home if side == "home" else away
        opp     = away if side == "home" else home
        t_style = h_code if side == "home" else a_code
        o_style = a_code if side == "home" else h_code
        adj = _clamp(abs(net_counter), 0, MAX_COUNTER_ADJ) * _fw("counter")

        # Style labels
        style_names = {
            "HP": "High-Press",
            "TT": "Tiki-Taka",
            "PO": "Possession",
            "CA": "Counter-Attack",
            "AT": "Direct Attacking",
            "DF": "Deep Block",
        }
        counter_text = {
            ("HP","TT"): f"{team}'s high-press forces {opp} into errors — dangerous for possession team",
            ("HP","PO"): f"{team}'s pressing disrupts {opp}'s build-up play",
            ("CA","TT"): f"{team}'s counter-attack is ideal weapon vs {opp}'s high line",
            ("CA","PO"): f"{team} exploits space behind {opp}'s possession-heavy structure",
            ("AT","DF"): f"{team}'s direct attack can wear down {opp}'s deep block",
            ("HP","CA"): f"{team}'s high-press neutralises {opp}'s counter-threat",
        }.get((t_style, o_style),
              f"{team}'s {style_names.get(t_style,t_style)} system has tactical edge over "
              f"{opp}'s {style_names.get(o_style,o_style)}")

        if side == "home":
            home_xg_adj += adj
        else:
            away_xg_adj += adj

        cards.append({
            "type":      "counter",
            "side":      side,
            "icon":      "⚔️",
            "label":     "Tactical Counter-Strategy",
            "text":      counter_text,
            "magnitude": adj,
        })

    # ── 4. POLITICAL / MORALE ─────────────────────────────────────────────────
    # a) Squad morale
    h_morale = hd["squad_morale"]
    a_morale = ad["squad_morale"]
    morale_diff = h_morale - a_morale

    if abs(morale_diff) >= 1.0:
        adj  = _clamp(abs(morale_diff) * MORALE_SCALE, 0, 0.10) * _fw("morale")
        side = "home" if morale_diff > 0 else "away"
        team = home if side == "home" else away
        m    = h_morale if side == "home" else a_morale
        if side == "home":
            home_xg_adj += adj
        else:
            away_xg_adj += adj
        cards.append({
            "type":      "political",
            "side":      side,
            "icon":      "🔥",
            "label":     "Squad Morale",
            "text":      f"{team} squad morale {m:.1f}/10 — {hd['political_context'] if side=='home' else ad['political_context']}",
            "magnitude": adj,
        })

    # b) Political risk (instability, unrest, travel chaos)
    h_risk = hd["political_risk"]
    a_risk = ad["political_risk"]

    if h_risk >= 2:
        adj = _clamp(h_risk * POLITICAL_RISK_SCALE, 0, MAX_POLITICAL_ADJ) * _fw("political")
        home_xg_adj -= adj
        cards.append({
            "type":      "political",
            "side":      "away",  # benefits away
            "icon":      "⚠️",
            "label":     "Political Risk",
            "text":      f"{home} faces political instability (risk score {h_risk}/10) — impacts focus and preparation",
            "magnitude": adj,
        })
    if a_risk >= 2:
        adj = _clamp(a_risk * POLITICAL_RISK_SCALE, 0, MAX_POLITICAL_ADJ) * _fw("political")
        away_xg_adj -= adj
        cards.append({
            "type":      "political",
            "side":      "home",  # benefits home
            "icon":      "⚠️",
            "label":     "Political Risk",
            "text":      f"{away} faces political instability (risk score {a_risk}/10) — impacts focus and preparation",
            "magnitude": adj,
        })

    # c) Extreme travel fatigue
    h_travel = hd["travel_distance_km"]
    a_travel = ad["travel_distance_km"]

    if h_travel > TRAVEL_THRESHOLD_KM and a_travel < h_travel - 3000:
        adj = TRAVEL_FATIGUE
        home_xg_adj -= adj
        cards.append({
            "type":      "political",
            "side":      "away",
            "icon":      "✈️",
            "label":     "Travel Fatigue",
            "text":      f"{home} travelled {h_travel:,} km vs {a_travel:,} km for {away} — significant fatigue factor",
            "magnitude": adj,
        })
    elif a_travel > TRAVEL_THRESHOLD_KM and h_travel < a_travel - 3000:
        adj = TRAVEL_FATIGUE
        away_xg_adj -= adj
        cards.append({
            "type":      "political",
            "side":      "home",
            "icon":      "✈️",
            "label":     "Travel Fatigue",
            "text":      f"{away} travelled {a_travel:,} km vs {h_travel:,} km for {home} — significant fatigue factor",
            "magnitude": adj,
        })

    # ── 4b. SQUAD DEPTH — amplified by 2026 six-substitution rule ────────────
    # With 6 subs allowed, deeper squads can rotate more aggressively, maintain
    # intensity in the second half, and outlast narrower squads late in the game.
    # squad_depth is rated 0–10 in player_data; we apply a bonus for a gap ≥ 1.5.
    h_depth = hd.get("squad_depth", 7.0)
    a_depth = ad.get("squad_depth", 7.0)
    depth_diff = h_depth - a_depth

    if abs(depth_diff) >= 1.5:
        adj  = _clamp(abs(depth_diff) * SQUAD_DEPTH_SCALE, 0, 0.10) * _fw("player")
        side = "home" if depth_diff > 0 else "away"
        team = home if side == "home" else away
        d_   = h_depth if side == "home" else a_depth
        if side == "home":
            home_xg_adj += adj
        else:
            away_xg_adj += adj
        cards.append({
            "type":      "player",
            "side":      side,
            "icon":      "🔄",
            "label":     "Squad Depth (6-sub rule)",
            "text":      (f"{team} has significantly deeper squad ({d_:.1f}/10) — "
                          f"2026's 6-substitution rule amplifies rotation advantage late in game"),
            "magnitude": adj,
        })

    # ── 5. HOST NATION BONUS ─────────────────────────────────────────────────
    # Co-hosts (USA/Canada/Mexico) playing in their own country get massive crowd
    # lift. Research: USA/Mexico ~+0.30 goals/game, Canada +0.18 (lower football
    # culture intensity means slightly smaller home crowd impact than USA/Mexico).
    _HOST_BONUS = {"United States": 0.30, "Mexico": 0.30, "Canada": 0.18}
    if home in HOST_NATIONS:
        adj = _HOST_BONUS.get(home, HOST_XG_BONUS) * _fw("host")
        home_xg_adj += adj
        cards.append({
            "type":      "host",
            "side":      "home",
            "icon":      "🏟️",
            "label":     "Host Nation Crowd Boost",
            "text":      (f"{home} playing on home soil as WC co-host — "
                          f"estimated +{adj:.2f} xG from crowd energy & familiarity"),
            "magnitude": adj,
        })
    if away in HOST_NATIONS:
        adj = _HOST_BONUS.get(away, HOST_XG_BONUS) * _fw("host")
        away_xg_adj += adj
        cards.append({
            "type":      "host",
            "side":      "away",
            "icon":      "🏟️",
            "label":     "Host Nation Crowd Boost",
            "text":      (f"{away} playing as WC co-host — "
                          f"estimated +{adj:.2f} xG from home crowd advantage"),
            "magnitude": adj,
        })

    # ── 6. CONFEDERATION ELO DISCOUNT ────────────────────────────────────────
    # ELO built from CONCACAF/OFC qualifiers overstates true WC-level strength.
    # Give opponent a bonus when facing these teams.
    if home in WEAK_CONF_TEAMS:
        adj = ELO_DISCOUNT_ADJ * _fw("confederation")
        away_xg_adj += adj
        cards.append({
            "type":      "confederation",
            "side":      "away",
            "icon":      "📊",
            "label":     "Confederation ELO Discount",
            "text":      (f"{home}'s ELO rating inflated by weak qualifying opponents — "
                          f"{away} gets +{adj:.2f} xG adjustment"),
            "magnitude": adj,
        })
    if away in WEAK_CONF_TEAMS:
        adj = ELO_DISCOUNT_ADJ * _fw("confederation")
        home_xg_adj += adj
        cards.append({
            "type":      "confederation",
            "side":      "home",
            "icon":      "📊",
            "label":     "Confederation ELO Discount",
            "text":      (f"{away}'s ELO rating inflated by weak qualifying opponents — "
                          f"{home} gets +{adj:.2f} xG adjustment"),
            "magnitude": adj,
        })

    # ── 7. UNDERDOG LOW-BLOCK ADJUSTMENT ─────────────────────────────────────
    # When ELO gap ≥ 150, heavy underdog will park the bus:
    #   → underdog loses attacking xG (not trying to attack)
    #   → favourite loses xG too (opponent sitting deep = fewer spaces)
    #   → net: boosts draw probability significantly
    elo_gap = elo_home - elo_away
    if abs(elo_gap) >= LOWBLOCK_ELO_THRESHOLD:
        if elo_gap > 0:
            # Home is big favourite, away parks the bus
            away_xg_adj -= LOWBLOCK_ATTACK_PENALTY   # away attacks less
            home_xg_adj -= LOWBLOCK_SPACE_PENALTY     # home finds less space
            underdog, favourite = away, home
            underdog_elo, fav_elo = elo_away, elo_home
        else:
            # Away is big favourite, home parks the bus
            home_xg_adj -= LOWBLOCK_ATTACK_PENALTY
            away_xg_adj -= LOWBLOCK_SPACE_PENALTY
            underdog, favourite = home, away
            underdog_elo, fav_elo = elo_home, elo_away
        draw_boost += LOWBLOCK_DRAW_BOOST * _fw("lowblock")
        cards.append({
            "type":      "lowblock",
            "side":      "draw",
            "icon":      "🚌",
            "label":     "Underdog Low-Block",
            "text":      (f"{underdog} (ELO {underdog_elo:.0f}) expected to sit deep vs "
                          f"{favourite} (ELO {fav_elo:.0f}, gap {abs(elo_gap):.0f}) — "
                          f"draw probability boosted by {LOWBLOCK_DRAW_BOOST*100:.0f}%"),
            "magnitude": LOWBLOCK_ATTACK_PENALTY,
        })

    # ── Final clamp ───────────────────────────────────────────────────────────
    total_max = 0.5  # cap total adjustment to prevent dominating the model
    home_xg_adj = _clamp(home_xg_adj, -total_max, total_max)
    away_xg_adj = _clamp(away_xg_adj, -total_max, total_max)

    return {
        "home_xg_adj":  round(home_xg_adj, 3),
        "away_xg_adj":  round(away_xg_adj, 3),
        "draw_boost":   round(draw_boost, 3),
        "elo_gap":      elo_gap,          # passed through for mismatch amplifier
        "factor_cards": cards,
        "summary": {
            "home_injury":       bool(h_missing),
            "away_injury":       bool(a_missing),
            "counter_winner":    counter_winner,
            "political_risk_home": h_risk,
            "political_risk_away": a_risk,
            "home_morale":       h_morale,
            "away_morale":       a_morale,
            "home_coach":        hd["coach"],
            "away_coach":        ad["coach"],
            "home_formation":    hd["formation"],
            "away_formation":    ad["formation"],
            "home_style_code":   h_code,
            "away_style_code":   a_code,
            "home_injured":      h_missing,
            "away_injured":      a_missing,
        },
    }


# ── Dixon-Coles ρ correction ──────────────────────────────────────────────────
# Empirically fitted value from Dixon & Coles (1997).
# Negative ρ means low-scoring draws (0-0, 1-1) are slightly more correlated
# than independent Poisson assumes — so we boost them.
# Literature range: -0.10 to -0.15.  We use -0.13 (original paper value).
DC_RHO = -0.10   # moderate value — -0.13 over-predicted draws, -0.07 under-predicted


def _dc_tau(hg: int, ag: int, lam_h: float, lam_a: float, rho: float) -> float:
    """
    Dixon-Coles correction factor τ for scoreline (hg, ag).

    Only the four low-scoring cells need adjustment:
      τ(0,0) = 1 - λ_h × λ_a × ρ    → boosted when ρ < 0
      τ(1,0) = 1 + λ_a × ρ           → slightly reduced when ρ < 0
      τ(0,1) = 1 + λ_h × ρ           → slightly reduced when ρ < 0
      τ(1,1) = 1 - ρ                  → boosted when ρ < 0
      τ(x,y) = 1 for all other cells  → untouched

    With ρ = -0.13:
      • 0-0 and 1-1 probabilities increase  (more draws)
      • 1-0 and 0-1 probabilities decrease slightly
      • 2-0, 2-1, 3-0, etc. — completely unchanged

    After applying τ to every cell, we renormalise so everything sums to 1.
    """
    if hg == 0 and ag == 0:
        return 1.0 - lam_h * lam_a * rho
    if hg == 1 and ag == 0:
        return 1.0 + lam_a * rho
    if hg == 0 and ag == 1:
        return 1.0 + lam_h * rho
    if hg == 1 and ag == 1:
        return 1.0 - rho
    return 1.0   # all other scorelines untouched


def apply_factor_adjustments(pred: dict, factors: dict,
                             dc_rho: float = DC_RHO) -> dict:
    """
    Apply xG adjustments to a prediction dict and re-derive probabilities
    using the Dixon-Coles bivariate Poisson correction.

    Standard Poisson treats home and away goals as independent.
    Dixon-Coles adds a small correlation correction (ρ ≈ -0.13) that
    boosts the probability of 0-0 and 1-1, slightly reducing 1-0 and 0-1.
    This fixes the well-known draw under-prediction bug in plain Poisson.
    """
    from scipy.stats import poisson as _poisson

    h_adj = factors["home_xg_adj"]
    a_adj = factors["away_xg_adj"]

    raw_xg_h = pred.get("xg_home", 1.2)
    raw_xg_a = pred.get("xg_away", 1.0)

    # Adjusted xG (floor at 0.3 to avoid degenerate distributions)
    adj_xg_h = max(0.3, raw_xg_h + h_adj)
    adj_xg_a = max(0.3, raw_xg_a + a_adj)

    # ── Fix 2: Mismatch xG Amplifier ─────────────────────────────────────────
    # When ELO gap > 200, the model systematically underestimates the favourite's
    # goals.  Evidence: USA 4-1 (pred 1-1), Australia 2-0 (pred 1-1),
    # Germany 7-1 (pred 2-0).  Scale up the stronger team's xG proportionally.
    MISMATCH_ELO_THRESHOLD = 200
    elo_gap = factors.get("elo_gap", 0.0)
    if abs(elo_gap) > MISMATCH_ELO_THRESHOLD:
        gap_above = abs(elo_gap) - MISMATCH_ELO_THRESHOLD
        # +0.8% per ELO point above threshold, capped at +40%
        scale = 1.0 + min(gap_above * 0.008 / 10, 0.40)
        if elo_gap > 0:
            adj_xg_h = max(0.3, adj_xg_h * scale)
        else:
            adj_xg_a = max(0.3, adj_xg_a * scale)

    # ── Dixon-Coles scoreline grid ────────────────────────────────────────────
    # Build the full probability table for all scorelines 0-0 through 9-9.
    # Each cell = Poisson(hg|λ_h) × Poisson(ag|λ_a) × τ(hg, ag, λ_h, λ_a, ρ)
    # τ only differs from 1.0 for the four low-scoring cells (0-0,1-0,0-1,1-1).
    MAX_G = 10   # goals per team ceiling for enumeration
    score_probs: dict[tuple[int,int], float] = {}
    raw_total = 0.0

    for hg in range(MAX_G):
        p_hg = _poisson.pmf(hg, adj_xg_h)
        for ag in range(MAX_G):
            p_ag  = _poisson.pmf(ag, adj_xg_a)
            tau   = _dc_tau(hg, ag, adj_xg_h, adj_xg_a, dc_rho)
            p     = p_hg * p_ag * tau
            score_probs[(hg, ag)] = p
            raw_total += p

    # Renormalise — τ corrections mean the grid no longer sums to exactly 1
    ph, pd_, pa = 0.0, 0.0, 0.0
    normalised: dict[tuple[int,int], float] = {}
    for (hg, ag), p in score_probs.items():
        pn = p / raw_total
        normalised[(hg, ag)] = pn
        if   hg > ag:  ph  += pn
        elif hg == ag: pd_ += pn
        else:          pa  += pn

    # Apply direct draw boost (low-block inflation) — shift mass from H/A to draw
    draw_boost = factors.get("draw_boost", 0.0)
    if draw_boost > 0:
        take_from_h = draw_boost * ph / (ph + pa)
        take_from_a = draw_boost * pa / (ph + pa)
        ph  -= take_from_h
        pa  -= take_from_a
        pd_ += draw_boost
        total2 = ph + pd_ + pa
        ph /= total2; pd_ /= total2; pa /= total2

    # ── Small even-match draw nudge ───────────────────────────────────────────
    # Only trigger when teams are very evenly matched (|ph-pa| < 5%) AND
    # draw is already competitive (pd > 25%). Very small nudge (2%) to avoid
    # the 58%-draw-prediction problem we had with the old 4.5% / 10% version.
    if abs(ph - pa) < 0.05 and pd_ > 0.25:
        extra_draw = 0.02
        take_h = extra_draw * ph / (ph + pa) if (ph + pa) > 0 else extra_draw / 2
        take_a = extra_draw * pa / (ph + pa) if (ph + pa) > 0 else extra_draw / 2
        ph  -= take_h
        pa  -= take_a
        pd_ += extra_draw
        total3 = ph + pd_ + pa
        ph /= total3; pd_ /= total3; pa /= total3

    # Most likely score (mode) — pick from Dixon-Coles normalised table
    best_score = max(normalised, key=normalised.get)
    best_p     = normalised[best_score]
    best_score_str = f"{best_score[0]}-{best_score[1]}"

    # Fix 3: Expected score (mean-rounded) — better for high xG matches
    # Poisson mode is floor(λ) for λ not integer, but expected goals E[X]=λ.
    # round(λ) gives a better central estimate than floor(λ) for xG > 1.0.
    exp_h = round(adj_xg_h)
    exp_a = round(adj_xg_a)
    expected_score_str = f"{exp_h}-{exp_a}"

    # Direction-consistent score: most probable scoreline that matches predicted winner
    # ph/pa/pd_ already computed above (as fractions)
    predicted_dir = "home" if ph > pa and ph > pd_ else ("away" if pa > ph and pa > pd_ else "draw")
    def _direction_consistent_score(normalised_table, direction):
        filtered = {
            sc: p for sc, p in normalised_table.items()
            if (direction == "home" and sc[0] > sc[1])
            or (direction == "away" and sc[1] > sc[0])
            or (direction == "draw" and sc[0] == sc[1])
        }
        if not filtered:
            return best_score_str  # fallback
        best = max(filtered, key=filtered.get)
        return f"{best[0]}-{best[1]}"
    direction_score_str = _direction_consistent_score(normalised, predicted_dir)

    updated = dict(pred)
    updated["xg_home"]         = round(adj_xg_h, 2)
    updated["xg_away"]         = round(adj_xg_a, 2)
    updated["p_home_win"]      = round(ph  * 100, 1)
    updated["p_draw"]          = round(pd_ * 100, 1)
    updated["p_away_win"]      = round(pa  * 100, 1)
    updated["likely_score"]        = best_score_str       # overall mode (often 1-1)
    updated["expected_score"]      = expected_score_str   # mean-rounded
    updated["direction_score"]     = direction_score_str  # most probable score consistent with predicted winner
    updated["likely_score_%"]      = round(best_p * 100, 1)
    updated["favourite"]      = (
        "home" if ph > pa else ("away" if pa > ph else "draw")
    )
    updated["favourite_p"]    = round(max(ph, pa) * 100, 1)
    updated["factor_adj_home"] = h_adj
    updated["factor_adj_away"] = a_adj
    updated["factor_cards"]    = factors["factor_cards"]
    updated["factor_summary"]  = factors["summary"]
    return updated


if __name__ == "__main__":
    # ── Dixon-Coles before/after demo ────────────────────────────────────────
    from scipy.stats import poisson as _p
    import json

    def plain_poisson(lh, la):
        """Standard independent Poisson probabilities."""
        ph = pd = pa = 0.0
        for hg in range(10):
            for ag in range(10):
                p = _p.pmf(hg, lh) * _p.pmf(ag, la)
                if hg > ag:   ph += p
                elif hg == ag: pd += p
                else:          pa += p
        return ph, pd, pa

    print("=" * 58)
    print("  DIXON-COLES vs PLAIN POISSON — side-by-side comparison")
    print("=" * 58)

    test_cases = [
        ("Ivory Coast vs Ecuador", 1.35, 1.45),   # near-equal, many draws
        ("Brazil vs Morocco",      1.80, 0.95),   # clear favourite
        ("Germany vs Curacao",     2.50, 0.60),   # big mismatch → low-block
        ("Haiti vs Scotland",      0.95, 1.20),   # slight underdog
    ]

    for label, lh, la in test_cases:
        ph_plain, pd_plain, pa_plain = plain_poisson(lh, la)

        # Build a fake pred + empty factors for DC path
        fake_pred = {"xg_home": lh, "xg_away": la,
                     "p_home_win": ph_plain*100, "p_draw": pd_plain*100,
                     "p_away_win": pa_plain*100,
                     "likely_score": "1-0", "likely_score_%": 0,
                     "favourite": "home", "favourite_p": 0}
        fake_factors = {"home_xg_adj": 0.0, "away_xg_adj": 0.0,
                        "draw_boost": 0.0, "factor_cards": [],
                        "summary": {}}
        dc = apply_factor_adjustments(fake_pred, fake_factors)

        print(f"\n  {label}  (xG {lh:.2f} – {la:.2f})")
        print(f"  {'':18} {'Home Win':>10} {'Draw':>10} {'Away Win':>10}  {'Best Score':>12}")
        print(f"  {'Plain Poisson':18} {ph_plain*100:>9.1f}% {pd_plain*100:>9.1f}% {pa_plain*100:>9.1f}%")
        print(f"  {'Dixon-Coles':18} {dc['p_home_win']:>9.1f}% {dc['p_draw']:>9.1f}% {dc['p_away_win']:>9.1f}%  {dc['likely_score']:>12}")
        draw_delta = dc['p_draw'] - pd_plain*100
        print(f"  {'Δ draw':18} {draw_delta:>+9.1f}%  ← DC boosted draws")
    print()
