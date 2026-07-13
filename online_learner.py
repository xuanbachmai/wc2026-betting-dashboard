"""
online_learner.py — Self-learning backend for WC 2026 predictions.

After every match result the system:
  1. Updates ELO ratings (K=60, goal-diff scaled)
  2. Calibrates factor weights (which factor types have been accurate?)
  3. Detects systematic biases (draw under-prediction, xG over/under-estimate)
  4. Fine-tunes XGBoost by adding WC matches at 10x sample weight
  5. Saves learned state → engine.py picks it up next run

State is persisted in  models/online_state.json
WC match data cache in data/wc_2026_results.csv

Usage (automatic inside engine.py):
    from online_learner import OnlineLearner
    learner = OnlineLearner()
    learner.process_all_played(SCHEDULE, pre_match_preds)
    adjustments = learner.get_adjustments()
    learner.save()
"""

from __future__ import annotations
import json, os, math, copy
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

STATE_FILE   = Path("models/online_state.json")
WC_CSV       = Path("data/wc_2026_results.csv")

# ── Hyper-parameters ──────────────────────────────────────────────────────────
ELO_K_BASE        = 60.0   # WC match ELO update weight
ELO_K_KO          = 80.0   # Higher for KO rounds
WC_SAMPLE_WEIGHT  = 10.0   # How much more a WC match counts vs historical in XGBoost
MIN_FIRES_CALIBRATE = 3    # Need this many fires before adjusting factor weight
BIAS_WINDOW       = 8      # Rolling window for xG / draw bias detection
DRAW_THRESHOLD    = 0.12   # Actual-pred draw rate diff needed to adjust inflation
WEIGHT_MIN        = 0.4    # Floor for factor weight
WEIGHT_MAX        = 2.5    # Ceiling for factor weight

# ── Outlier / live-match guards ───────────────────────────────────────────────
# If actual goals deviate > this from predicted xG, the match is a statistical
# outlier (e.g. Germany 7-1) — skip it for bias learning to avoid corrupting
# the xG calibration with a once-in-a-tournament freak result.
OUTLIER_GOAL_THRESHOLD = 3.0   # |actual - xg_pred| > 3.0 → skip bias learning

# A match is considered "live / not final" if fewer than MATCH_DURATION_MINS
# minutes have elapsed since kickoff.  Live scores must NOT affect the model.
MATCH_DURATION_MINS = 110       # 90 min + stoppage buffer


# ── Helpers ───────────────────────────────────────────────────────────────────

def _elo_expected(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))

def _goal_diff_mult(gd: int) -> float:
    """Larger winning margin → bigger ELO swing."""
    if gd <= 1: return 1.0
    if gd == 2: return 1.5
    if gd == 3: return 1.75
    return 2.0

def _outcome(hs: int, as_: int):
    if hs > as_: return 1.0, 0.0, "home"
    if as_ > hs: return 0.0, 1.0, "away"
    return 0.5, 0.5, "draw"


# ── Main class ────────────────────────────────────────────────────────────────

class OnlineLearner:
    def __init__(self):
        self.state = self._load_state()

    # ── State I/O ─────────────────────────────────────────────────────────────

    def _load_state(self) -> dict:
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                return json.load(f)
        return self._blank_state()

    def _blank_state(self) -> dict:
        factor_types = [
            "player", "coach", "counter", "political",
            "host", "confederation", "lowblock", "morale",
        ]
        return {
            "version": 3,
            "matches_processed": 0,
            # ── ELO deltas learned from WC 2026 results ──────────────────────
            "elo": {},          # {team: updated_elo}  (WC-only updates)
            # ── Factor calibration ────────────────────────────────────────────
            "factor_calibration": {
                ft: {"fires": 0, "correct": 0, "xg_adj_sum": 0.0, "weight": 1.0}
                for ft in factor_types
            },
            # ── Systematic bias correction ────────────────────────────────────
            "draw_inflation":    0.0,   # extra probability mass → draw
            "xg_bias_home":      0.0,   # learned offset: actual_goals - predicted_xg (home)
            "xg_bias_away":      0.0,
            "favourite_overconf":0.0,   # model overconfident about favourites?
            # ── History ───────────────────────────────────────────────────────
            "match_history": [],
            # ── Accuracy tracker ──────────────────────────────────────────────
            "accuracy": {
                "total": 0,
                "direction_correct": 0,
                "score_correct": 0,
                "draw_actual": 0,
                "draw_predicted": 0,
                "fav_win_actual": 0,
                "fav_win_predicted": 0,
                "total_pred_goals": 0.0,
                "total_actual_goals": 0,
            },
            "last_updated": None,
        }

    def save(self):
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(self.state, f, indent=2, default=str)
        print(f"[learner] State saved → {STATE_FILE}")

    # ── Main entry point ──────────────────────────────────────────────────────

    def process_all_played(
        self,
        schedule: list[dict],
        pre_match_preds: dict[int, dict],   # match_no → pred dict (from BEFORE the game)
        base_elo: dict[str, float],
    ) -> None:
        """
        Process all already-played matches.  Called once per dashboard build.
        Only processes matches not yet in history (idempotent).
        """
        already_done = {r["match_no"] for r in self.state["match_history"]}
        newly_played = [
            m for m in schedule
            if m["home_score"] is not None and m["match_no"] not in already_done
        ]

        if not newly_played:
            print("[learner] No new results to learn from.")
            return

        # Seed ELO from base if first run
        if not self.state["elo"]:
            self.state["elo"] = dict(base_elo)

        for m in sorted(newly_played, key=lambda x: x["match_no"]):
            # ── Guard 1: Live match — score not yet final ─────────────────────
            # Primary: ESPN match_state flag set by score_fetcher
            if m.get("match_state") == "in":
                print(f"[learner] M{m['match_no']} {m['home']} vs {m['away']} "
                      f"— LIVE (in progress), skipping until FT.")
                continue

            # Fallback: use kickoff time if match_state not available
            kickoff_str = m.get("kickoff_utc")
            if kickoff_str and not m.get("match_state"):
                try:
                    from datetime import timezone
                    kickoff_dt = datetime.fromisoformat(kickoff_str.replace("Z", "+00:00"))
                    now_utc = datetime.now(timezone.utc)
                    elapsed_mins = (now_utc - kickoff_dt).total_seconds() / 60
                    if elapsed_mins < MATCH_DURATION_MINS:
                        print(f"[learner] M{m['match_no']} {m['home']} vs {m['away']} "
                              f"— LIVE ({elapsed_mins:.0f} min elapsed), skipping until final.")
                        continue
                except Exception:
                    pass

            pred = pre_match_preds.get(m["match_no"], {})
            factors = pred.get("_factors", {})
            self._process_one(m, pred, factors)

        # After all new results → recompute biases
        self._recompute_biases()
        self.state["last_updated"] = datetime.now().isoformat()
        print(f"[learner] Processed {len(newly_played)} new result(s). "
              f"Total: {self.state['matches_processed']} matches.")

    def _process_one(self, m: dict, pred: dict, factors: dict):
        home, away     = m["home"], m["away"]
        hs, as_        = m["home_score"], m["away_score"]
        match_no       = m["match_no"]
        is_ko          = m.get("group", "") not in [chr(c) for c in range(65, 77)]

        # Determine actual vs predicted winner
        act_h, act_a, actual_dir = _outcome(hs, as_)

        # Use aggregate probabilities to determine predicted direction.
        # Rule: predict DRAW when the match is close enough that the draw
        # probability exceeds the gap between the two sides.
        # i.e.  P(draw) > |P(home) - P(away)|
        # This reflects: "if I can't confidently pick a winner, call draw."
        ph_p  = pred.get("p_home_win", 34)
        pd_p  = pred.get("p_draw",     33)
        pa_p  = pred.get("p_away_win", 33)
        # Predict draw only when draw prob clearly exceeds the H/A gap by a margin.
        # P(draw) must beat |P(home)-P(away)| by at least 5% to call draw.
        # This keeps draw predictions at realistic WC rate (~25-30%).
        if pd_p > abs(ph_p - pa_p) + 5:
            pred_dir = "draw"
        elif ph_p >= pa_p:
            pred_dir = "home"
        else:
            pred_dir = "away"

        # Keep likely_score for score accuracy tracking
        pred_score = pred.get("likely_score", "1-1")
        try:
            pg, ag = map(int, pred_score.split("-"))
        except Exception:
            pg, ag = 1, 1
        direction_correct = (pred_dir == actual_dir)
        score_correct     = (pg == hs and ag == as_)

        # 1. ELO update
        K = ELO_K_KO if is_ko else ELO_K_BASE
        self._update_elo(home, away, hs, as_, K)

        # 2. Factor calibration
        self._calibrate_factors(factors.get("factor_cards", []), direction_correct)

        # 3. Record in accuracy
        acc = self.state["accuracy"]
        acc["total"]             += 1
        acc["direction_correct"] += int(direction_correct)
        acc["score_correct"]     += int(score_correct)
        acc["draw_actual"]       += int(actual_dir == "draw")
        acc["draw_predicted"]    += int(pred_dir == "draw")
        acc["total_pred_goals"]  += pred.get("xg_home", 0) + pred.get("xg_away", 0)
        acc["total_actual_goals"]+= hs + as_

        fav_side = "home" if pred.get("p_home_win", 0) > pred.get("p_away_win", 0) else "away"
        acc["fav_win_predicted"] += 1
        acc["fav_win_actual"]    += int(
            (fav_side == "home" and actual_dir == "home") or
            (fav_side == "away" and actual_dir == "away")
        )

        # 4. Store match record
        self.state["match_history"].append({
            "match_no":          match_no,
            "home":              home,
            "away":              away,
            "home_score":        hs,
            "away_score":        as_,
            "actual_dir":        actual_dir,
            "pred_score":        pred_score,
            "pred_dir":          pred_dir,
            "direction_correct": direction_correct,
            "score_correct":     score_correct,
            "p_home_win":        pred.get("p_home_win", 0),
            "p_draw":            pred.get("p_draw", 0),
            "p_away_win":        pred.get("p_away_win", 0),
            "xg_home_pred":      pred.get("xg_home", 0),
            "xg_away_pred":      pred.get("xg_away", 0),
            "factor_adj_home":   pred.get("factor_adj_home", 0),
            "factor_adj_away":   pred.get("factor_adj_away", 0),
            "factors_fired":     [c.get("type") for c in factors.get("factor_cards", [])],
            "timestamp":         datetime.now().isoformat(),
        })
        self.state["matches_processed"] += 1

    # ── ELO update ────────────────────────────────────────────────────────────

    def _update_elo(self, home: str, away: str, hs: int, as_: int, K: float):
        elo   = self.state["elo"]
        h_elo = elo.get(home, 1500.0)
        a_elo = elo.get(away, 1500.0)

        exp_h  = _elo_expected(h_elo, a_elo)
        exp_a  = 1.0 - exp_h
        act_h, act_a, _ = _outcome(hs, as_)
        mult   = _goal_diff_mult(abs(hs - as_))

        elo[home] = h_elo + K * mult * (act_h - exp_h)
        elo[away] = a_elo + K * mult * (act_a - exp_a)

    # ── Factor calibration ────────────────────────────────────────────────────

    def _calibrate_factors(self, factor_cards: list[dict], direction_correct: bool):
        cal = self.state["factor_calibration"]
        seen_types = set()

        for card in factor_cards:
            ftype = card.get("type", "unknown")
            if ftype not in cal:
                cal[ftype] = {"fires": 0, "correct": 0, "xg_adj_sum": 0.0, "weight": 1.0}

            cal[ftype]["fires"]       += 1
            cal[ftype]["correct"]     += int(direction_correct)
            cal[ftype]["xg_adj_sum"]  += abs(card.get("magnitude", 0))
            seen_types.add(ftype)

        # Recompute weight for fired factors
        for ftype in seen_types:
            data = cal[ftype]
            if data["fires"] >= MIN_FIRES_CALIBRATE:
                accuracy = data["correct"] / data["fires"]
                # Scale: 50% accuracy → weight 1.0, 80% → 1.6, 30% → 0.6
                # But cap so one great match can't explode the weight
                raw_weight = accuracy / 0.50
                # Exponential smoothing: don't jump too fast
                old_w = data["weight"]
                data["weight"] = round(
                    max(WEIGHT_MIN, min(WEIGHT_MAX, 0.7 * old_w + 0.3 * raw_weight)), 3
                )

    # ── Bias detection ────────────────────────────────────────────────────────

    def _recompute_biases(self):
        history = self.state["match_history"]
        if len(history) < 3:
            return

        window = history[-BIAS_WINDOW:]

        # ── Draw inflation ────────────────────────────────────────────────────
        actual_draw_rate = sum(1 for r in window if r["actual_dir"] == "draw") / len(window)
        pred_draw_rate   = sum(r["p_draw"] for r in window if "p_draw" in r) / (len(window) * 100)

        draw_gap = actual_draw_rate - pred_draw_rate
        if draw_gap > DRAW_THRESHOLD:
            # Draws happening more than predicted → inflate proportional to gap
            # No hard cap — let the data speak (WC 2026 is genuinely draw-heavy)
            new_infl = min(0.40, draw_gap * 0.6)
        elif draw_gap < -0.05:
            # Draws happening less → deflate slowly
            new_infl = max(0.0, self.state["draw_inflation"] - 0.01)
        else:
            new_infl = self.state["draw_inflation"]  # no change
        self.state["draw_inflation"] = round(new_infl, 4)

        # ── xG bias ───────────────────────────────────────────────────────────
        # WC 2026 is played at neutral venues — there is no real home/away
        # structural advantage (co-host bonus is handled separately in factors).
        # We compute ONE combined bias from ALL goals and apply it equally to
        # both teams, avoiding a false "home team scores more" signal.
        #
        # Also exclude outlier matches (e.g. Germany 7-1) whose extreme
        # deviation would corrupt the calibration.
        def _is_outlier(r):
            h_dev = abs(r["home_score"] - r.get("xg_home_pred", 0))
            a_dev = abs(r["away_score"] - r.get("xg_away_pred", 0))
            return h_dev > OUTLIER_GOAL_THRESHOLD or a_dev > OUTLIER_GOAL_THRESHOLD

        normal_window = [r for r in window if not _is_outlier(r)]
        outlier_count = len(window) - len(normal_window)
        if outlier_count:
            outlier_matches = [f"M{r['match_no']} {r['home']} {r['home_score']}-{r['away_score']} {r['away']}"
                               for r in window if _is_outlier(r)]
            print(f"[learner] Excluding {outlier_count} outlier(s) from bias learning: {outlier_matches}")

        # Single combined bias: pool all goals from both sides
        all_errors = []
        for r in normal_window:
            if r.get("xg_home_pred", 0) > 0:
                all_errors.append(r["home_score"] - r["xg_home_pred"])
            if r.get("xg_away_pred", 0) > 0:
                all_errors.append(r["away_score"] - r["xg_away_pred"])

        if all_errors:
            # Use the full observed mean error as the bias correction
            combined_bias = round(float(np.mean(all_errors)), 4)
            self.state["xg_bias_home"] = combined_bias
            self.state["xg_bias_away"] = combined_bias

        self._log_biases(len(all_errors))

    def _log_biases(self, n_observations: int = 0):
        print(f"[learner] Bias update → "
              f"xg_bias={self.state['xg_bias_home']:+.3f}  "
              f"(from {n_observations} goal observations)")

    # ── XGBoost fine-tuning ───────────────────────────────────────────────────

    def retrain_with_wc_data(self, feat_df: pd.DataFrame, outcome_model, goals_model):
        """
        Fine-tune XGBoost by adding WC 2026 matches to training set with
        WC_SAMPLE_WEIGHT (10x) weight.  Returns (new_outcome_model, new_goals_model).
        """
        import model as model_module

        history = self.state["match_history"]
        if len(history) < 4:
            print("[learner] Not enough WC data yet (need ≥4 matches) — skipping retrain.")
            return outcome_model, goals_model

        # Build WC rows in the same feature format as feat_df
        wc_rows = []
        for r in history:
            wc_rows.append({
                "home_team":  r["home"],
                "away_team":  r["away"],
                "home_score": r["home_score"],
                "away_score": r["away_score"],
                "date":       r.get("timestamp", "2026-06-01")[:10],
                "tournament": "FIFA World Cup",
                "neutral":    True,
                # Features we know
                "home_elo":   self.state["elo"].get(r["home"], 1500),
                "away_elo":   self.state["elo"].get(r["away"], 1500),
            })

        if not wc_rows:
            return outcome_model, goals_model

        wc_df = pd.DataFrame(wc_rows)

        try:
            new_outcome, new_goals = model_module.retrain_with_extra(
                feat_df, wc_df, wc_weight=WC_SAMPLE_WEIGHT
            )
            print(f"[learner] XGBoost retrained with {len(wc_rows)} WC matches at {WC_SAMPLE_WEIGHT}x weight.")
            return new_outcome, new_goals
        except Exception as e:
            print(f"[learner] XGBoost retrain failed ({e}) — using original model.")
            return outcome_model, goals_model

    # ── Public API ────────────────────────────────────────────────────────────

    def get_adjustments(self) -> dict:
        """
        Return all learned adjustments to apply when predicting upcoming matches.
        These override / supplement the base model values.
        """
        return {
            # ELO overrides (apply on top of historical ELO)
            "elo_overrides":      dict(self.state["elo"]),

            # Factor weight multipliers {factor_type: weight}
            "factor_weights":     {
                ft: d["weight"]
                for ft, d in self.state["factor_calibration"].items()
            },

            # Probability mass to shift from H/A to Draw
            "draw_inflation":     self.state["draw_inflation"],

            # xG offset to add to base model predictions
            "xg_bias_home":       self.state["xg_bias_home"],
            "xg_bias_away":       self.state["xg_bias_away"],

            # Probability damping for the favourite
            "favourite_overconf": self.state["favourite_overconf"],
        }

    def get_summary(self) -> dict:
        """Human-readable summary of everything learned so far."""
        acc   = self.state["accuracy"]
        total = max(acc["total"], 1)
        cal   = self.state["factor_calibration"]

        factor_report = {}
        for ft, d in cal.items():
            if d["fires"] > 0:
                factor_report[ft] = {
                    "fires":          d["fires"],
                    "correct":        d["correct"],
                    "accuracy_pct":   round(100 * d["correct"] / d["fires"], 1),
                    "learned_weight": d["weight"],
                    "verdict": (
                        "🔥 Boosted" if d["weight"] > 1.2 else
                        "❌ Downweighted" if d["weight"] < 0.8 else
                        "✓ Neutral"
                    ),
                }

        return {
            "matches_processed":   self.state["matches_processed"],
            "direction_accuracy":  f"{acc['direction_correct']}/{total} ({100*acc['direction_correct']/total:.1f}%)",
            "score_accuracy":      f"{acc['score_correct']}/{total} ({100*acc['score_correct']/total:.1f}%)",
            "draws_actual_vs_pred":f"{acc['draw_actual']}/{total} actual vs {acc['draw_predicted']}/{total} predicted",
            "goals_actual_vs_pred":f"{acc['total_actual_goals']} actual vs {acc['total_pred_goals']:.1f} predicted",
            "learned_biases": {
                "draw_inflation":      self.state["draw_inflation"],
                "xg_bias_home":        self.state["xg_bias_home"],
                "xg_bias_away":        self.state["xg_bias_away"],
                "favourite_overconf":  self.state["favourite_overconf"],
            },
            "factor_weights":      factor_report,
            "last_updated":        self.state["last_updated"],
        }

    def print_summary(self):
        s = self.get_summary()
        print("\n" + "═"*60)
        print("  ONLINE LEARNER — What the model has learned so far")
        print("═"*60)
        print(f"  Matches processed : {s['matches_processed']}")
        print(f"  Direction accuracy: {s['direction_accuracy']}")
        print(f"  Exact score       : {s['score_accuracy']}")
        print(f"  Draws actual/pred : {s['draws_actual_vs_pred']}")
        print(f"  Goals actual/pred : {s['goals_actual_vs_pred']}")
        print(f"\n  Learned biases:")
        for k, v in s["learned_biases"].items():
            print(f"    {k:<22}: {v:+.4f}")
        if s["factor_weights"]:
            print(f"\n  Factor weights (learned):")
            for ft, d in s["factor_weights"].items():
                bar = "█" * int(d["learned_weight"] * 5)
                print(f"    {ft:<16} {d['accuracy_pct']:5.1f}% acc "
                      f"({d['fires']} fires)  w={d['learned_weight']:.2f}  {bar}  {d['verdict']}")
        print("═"*60 + "\n")
