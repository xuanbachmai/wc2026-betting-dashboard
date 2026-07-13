"""
corners_learning.py — Online learning for the corners model.

Walk-forward approach:
  • For each match played (in date order), predict corners using the model
    with whatever biases have been learned SO FAR (no look-ahead).
  • After observing the actual result, update the per-team bias estimate.
  • Save learned biases → data/corners_bias_learned.json
  • Return walk-forward accuracy trace (MAE per match, rolling MAE) so the
    dashboard can show how the model improves over time.

Bias update rule (exponential moving average):
  new_bias = (1 - alpha) * old_bias  +  alpha * (actual - base_pred)
  alpha = 0.45  (fast-learning for small WC sample)

The learned biases are additive corrections applied on top of the static
TEAM_CORNER_BIAS in corners_model.py.  engine.py passes them
in as extra kwargs so the base model constants stay clean.
"""

from __future__ import annotations
import json
from pathlib import Path

from schedule import SCHEDULE

BIAS_PATH   = Path("data/corners_bias_learned.json")
ACTUAL_PATH = Path("data/actual_corners.json")

ALPHA = 0.45   # learning rate — higher = trust recent matches more


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return s.strip().lower()

def _match_key(home: str, away: str) -> str:
    return f"{_norm(home)} vs {_norm(away)}"


# ── Walk-forward learning loop ────────────────────────────────────────────────

def run_learning(
    elo: dict[str, float],
    team_intel: dict,
    force: bool = False,
) -> dict:
    """
    Run walk-forward bias learning over all played matches.

    Returns a dict with:
      learned_bias    : {team: float}   — additive bias to apply per team
      walk_forward    : list of per-match dicts (for dashboard accuracy chart)
      mae             : float           — overall MAE on played matches
      within2         : int
      within3         : int
      n               : int             — number of matches evaluated
    """
    from corners_model import predict_corners

    actual_data: dict = {}
    if ACTUAL_PATH.exists():
        try:
            actual_data = json.loads(ACTUAL_PATH.read_text())
        except Exception:
            pass

    # Played matches in date order
    played = sorted(
        [m for m in SCHEDULE if m.get("home_score") is not None],
        key=lambda m: (m["date"], m["match_no"])
    )

    learned_bias: dict[str, float] = {}   # team → current running bias
    walk_forward: list[dict] = []
    running_errors: list[float] = []

    for m in played:
        home, away = m["home"], m["away"]
        key = _match_key(home, away)

        rec = actual_data.get(key)
        if rec is None or rec.get("total") is None:
            continue   # no corner data yet for this match

        actual_total = rec["total"]
        act_h = rec.get("home_corners")
        act_a = rec.get("away_corners")

        # ── Predict using model + current learned biases (no look-ahead) ──
        pred = predict_corners(
            home=home, away=away,
            home_elo=elo.get(home, 1500),
            away_elo=elo.get(away, 1500),
            home_intel=team_intel.get(home, {}),
            away_intel=team_intel.get(away, {}),
            extra_bias=dict(learned_bias),   # only biases learned BEFORE this match
        )

        pred_total = pred["total_exp"]
        pred_h     = pred["home_exp"]
        pred_a     = pred["away_exp"]
        err        = pred_total - actual_total
        abs_err    = abs(err)
        running_errors.append(abs_err)
        rolling_mae = sum(running_errors) / len(running_errors)

        walk_forward.append({
            "match_no":    m["match_no"],
            "date":        m["date"],
            "home":        home,
            "away":        away,
            "pred_total":  round(pred_total, 2),
            "pred_h":      round(pred_h, 2),
            "pred_a":      round(pred_a, 2),
            "actual_total": actual_total,
            "actual_h":    act_h,
            "actual_a":    act_a,
            "error":       round(err, 2),
            "abs_error":   round(abs_err, 2),
            "rolling_mae": round(rolling_mae, 2),
            "within2":     abs_err <= 2,
            "within3":     abs_err <= 3,
            # Per-team residuals (for bias update)
            "res_h": round((act_h or 0) - pred_h, 2) if act_h is not None else None,
            "res_a": round((act_a or 0) - pred_a, 2) if act_a is not None else None,
            # Learned biases at time of prediction (snapshot)
            "bias_home_at_pred": round(learned_bias.get(home, 0.0), 3),
            "bias_away_at_pred": round(learned_bias.get(away, 0.0), 3),
        })

        # ── Update biases AFTER observing result (EMA) ──
        if act_h is not None:
            old = learned_bias.get(home, 0.0)
            res = act_h - pred_h
            learned_bias[home] = (1 - ALPHA) * old + ALPHA * res
            walk_forward[-1]["bias_home_after"] = round(learned_bias[home], 3)

        if act_a is not None:
            old = learned_bias.get(away, 0.0)
            res = act_a - pred_a
            learned_bias[away] = (1 - ALPHA) * old + ALPHA * res
            walk_forward[-1]["bias_away_after"] = round(learned_bias[away], 3)

    # ── Final stats ───────────────────────────────────────────────────────────
    n       = len(running_errors)
    mae     = round(sum(running_errors) / n, 3) if n else None
    within2 = sum(1 for wf in walk_forward if wf["within2"])
    within3 = sum(1 for wf in walk_forward if wf["within3"])

    result = {
        "learned_bias": {t: round(v, 3) for t, v in learned_bias.items()},
        "walk_forward": walk_forward,
        "mae":      mae,
        "within2":  within2,
        "within3":  within3,
        "n":        n,
        "alpha":    ALPHA,
    }

    BIAS_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


# ── Load saved biases (for use in predictions) ────────────────────────────────

def load_learned_bias() -> dict[str, float]:
    """Return {team: bias} from the last saved learning run, or {}."""
    if not BIAS_PATH.exists():
        return {}
    try:
        return json.loads(BIAS_PATH.read_text()).get("learned_bias", {})
    except Exception:
        return {}


# ── Pretty print for CLI ──────────────────────────────────────────────────────

def print_report(result: dict) -> None:
    wf  = result["walk_forward"]
    lb  = result["learned_bias"]
    n   = result["n"]
    mae = result["mae"]

    print(f"\n{'='*65}")
    print(f"  CORNERS LEARNING  —  {n} matches  |  MAE {mae}  |  "
          f"±2: {result['within2']}/{n}  ±3: {result['within3']}/{n}")
    print(f"{'='*65}")
    print(f"\n{'Match':<40} {'Pred':>5} {'Act':>4} {'Err':>5}  {'Rolling MAE':>11}")
    print("─" * 65)
    for wm in wf:
        flag = "★" if wm["within2"] else ("·" if wm["within3"] else " ")
        print(f"{flag} {wm['home']+' vs '+wm['away']:<38} "
              f"{wm['pred_total']:>5.1f} {wm['actual_total']:>4} "
              f"{wm['error']:>+5.1f}  {wm['rolling_mae']:>11.2f}")

    print(f"\n{'─'*65}")
    print(f"  Learned biases (applied to upcoming matches):\n")
    for team, bias in sorted(lb.items(), key=lambda x: -abs(x[1])):
        bar = ("+" if bias >= 0 else "") + f"{bias:+.2f}"
        direction = "▲ generates more" if bias > 0 else "▼ generates fewer"
        print(f"  {team:<28} {bar:>7}   {direction} corners than model base")


if __name__ == "__main__":
    import data as data_module
    import elo as elo_module
    from team_intelligence import TEAM_INTEL

    df = data_module.load()
    _, elo, _ = elo_module.compute_elo(df)

    result = run_learning(elo=elo, team_intel=TEAM_INTEL)
    print_report(result)
