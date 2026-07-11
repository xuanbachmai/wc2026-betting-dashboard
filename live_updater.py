"""
live_updater.py — Pulls real WC 2026 match results from the live GitHub CSV
and patches SCHEDULE in-memory so every module always uses up-to-date scores.

Source: https://github.com/martj42/international_results (updated within hours
        of each match finishing — same source used for model training).

Usage:
    from live_updater import patch_schedule_with_live_scores, fetch_fresh_csv

    # Call once at startup; mutates schedule.SCHEDULE in-place
    patch_schedule_with_live_scores()

Auto-refresh:
    Results are cached to data/live_scores.json (TTL = 30 min).
    The server.py rebuild loop calls this before every dashboard generation.
"""

from __future__ import annotations

import io
import json
import ssl
import time
import urllib.request
from pathlib import Path

import certifi
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
CSV_URL    = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
CACHE_FILE = Path("data/live_scores.json")
CACHE_TTL  = 30 * 60        # 30 minutes

# Name mapping: CSV spelling → our schedule.py spelling
CSV_TO_SCHEDULE: dict[str, str] = {
    "Cape Verde":   "Cabo Verde",
    "DR Congo":     "Congo DR",
    "Curaçao":      "Curacao",   # schedule uses ASCII version
    # everything else matches exactly between CSV and schedule.py
}

# Reverse map: schedule → CSV (for lookup)
SCHEDULE_TO_CSV: dict[str, str] = {v: k for k, v in CSV_TO_SCHEDULE.items()}


def _ssl_ctx() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=certifi.where())


def fetch_fresh_csv(force: bool = False) -> pd.DataFrame:
    """
    Download (or load from cache) the results CSV.
    Returns a DataFrame filtered to FIFA World Cup 2026 matches only.
    """
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()

    if not force and CACHE_FILE.exists():
        raw = json.loads(CACHE_FILE.read_text())
        if now - raw.get("fetched_at", 0) < CACHE_TTL:
            df = pd.DataFrame(raw["rows"])
            _log(f"[live] Using cached live scores ({len(df)} WC matches, "
                 f"{sum(df['home_score'].notna())} with results)")
            return df

    _log("[live] Fetching fresh results from GitHub...")
    try:
        req = urllib.request.Request(CSV_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=_ssl_ctx(), timeout=15) as r:
            content = r.read().decode("utf-8")
    except Exception as e:
        _log(f"[live] ⚠️  Fetch failed: {e}. Using cached data if available.")
        if CACHE_FILE.exists():
            df = pd.DataFrame(json.loads(CACHE_FILE.read_text())["rows"])
            return df
        return pd.DataFrame()

    full_df = pd.read_csv(io.StringIO(content))
    # Filter to WC 2026 only
    wc = full_df[
        (full_df["date"] >= "2026-06-01") &
        (full_df["tournament"] == "FIFA World Cup")
    ].copy()

    # Normalise scores (CSV uses floats for goals, "NA" for upcoming)
    def to_int_or_none(v):
        try:
            f = float(v)
            return int(f)
        except (ValueError, TypeError):
            return None

    wc["home_score"] = wc["home_score"].apply(to_int_or_none)
    wc["away_score"] = wc["away_score"].apply(to_int_or_none)

    # Convert to plain dicts — replace any surviving NaN with None for clean JSON
    import math
    def clean(v):
        if v is None: return None
        try:
            return None if math.isnan(float(v)) else v
        except (TypeError, ValueError):
            return v

    rows = [
        {k: clean(v) for k, v in r.items()}
        for r in wc[["date", "home_team", "away_team", "home_score", "away_score"]].to_dict("records")
    ]
    CACHE_FILE.write_text(json.dumps({"fetched_at": now, "rows": rows}, ensure_ascii=False))
    played = sum(1 for r in rows if r["home_score"] is not None)
    _log(f"[live] ✅ {len(rows)} WC matches fetched, {played} with results.")
    return pd.DataFrame(rows)


def patch_schedule_with_live_scores(force: bool = False) -> int:
    """
    Download latest scores and update schedule.SCHEDULE in-place.
    Tries ESPN first (real-time), falls back to GitHub CSV (lags ~2h).
    Returns the number of matches updated.
    """
    # Primary: ESPN — updates within ~1 min of a goal, no API key needed
    try:
        from score_fetcher import patch_schedule_with_espn
        espn_updated = patch_schedule_with_espn(force=force)
        if espn_updated > 0:
            _log(f"[live] ESPN patched {espn_updated} match(es) — skipping slow CSV.")
            return espn_updated
    except Exception as e:
        _log(f"[live] ESPN fetch failed ({e}), falling back to GitHub CSV...")

    import schedule as sched_mod     # local schedule.py

    live_df = fetch_fresh_csv(force=force)
    if live_df.empty:
        return 0

    # Build lookup: (csv_home, csv_away) → scores
    score_map: dict[tuple[str, str], tuple[int, int]] = {}
    for _, row in live_df.iterrows():
        if row["home_score"] is None or (hasattr(row["home_score"], '__class__') and row["home_score"] != row["home_score"]):
            continue
        import math
        try:
            if math.isnan(float(row["home_score"])):
                continue
        except (TypeError, ValueError):
            pass
        h = str(row["home_team"])
        a = str(row["away_team"])
        score_map[(h, a)] = (int(row["home_score"]), int(row["away_score"]))

    updated = 0
    for match in sched_mod.SCHEDULE:
        # Convert schedule names → CSV names for lookup
        h_csv = SCHEDULE_TO_CSV.get(match["home"], match["home"])
        a_csv = SCHEDULE_TO_CSV.get(match["away"], match["away"])

        key = (h_csv, a_csv)
        if key in score_map:
            hs, as_ = score_map[key]
            if match["home_score"] != hs or match["away_score"] != as_:
                match["home_score"] = hs
                match["away_score"] = as_
                updated += 1
            elif match["home_score"] is None:
                # Result exists but schedule still has None
                match["home_score"] = hs
                match["away_score"] = as_
                updated += 1

    if updated:
        _log(f"[live] 🔄 {updated} match score(s) updated from live data.")
    else:
        _log("[live] ✓ No new scores since last check.")

    return updated


def also_refresh_training_data(force: bool = False):
    """
    Re-download the full historical CSV so the ML model trains on
    the latest data (including WC 2026 matches already played).
    TTL = same as live scores cache.
    """
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    local = Path("data/results.csv")
    meta  = Path("data/results_meta.json")
    now   = time.time()

    if not force and meta.exists():
        m = json.loads(meta.read_text())
        if now - m.get("fetched_at", 0) < CACHE_TTL:
            _log("[live] Training CSV still fresh — skipping re-download.")
            return

    _log("[live] Re-downloading full training CSV...")
    try:
        req = urllib.request.Request(CSV_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=_ssl_ctx(), timeout=20) as r:
            content = r.read()
        local.write_bytes(content)
        meta.write_text(json.dumps({"fetched_at": now}))
        _log(f"[live] ✅ Training CSV refreshed ({len(content)//1024} KB).")
    except Exception as e:
        _log(f"[live] ⚠️  Could not refresh training CSV: {e}")


def _log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    also_refresh_training_data(force=force)
    patch_schedule_with_live_scores(force=force)

    # Show current state
    import schedule as sched_mod
    played = [(m["home"], m["away"], m["home_score"], m["away_score"])
              for m in sched_mod.SCHEDULE if m["home_score"] is not None]
    print(f"\n📋 {len(played)} matches with results:")
    for h, a, hs, as_ in played:
        print(f"   {h} {hs}–{as_} {a}")
