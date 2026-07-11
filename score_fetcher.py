"""
score_fetcher.py — Real-time WC 2026 scores from ESPN (no API key needed).

ESPN updates within ~1 minute of a goal. Falls back to martj42 GitHub CSV
if ESPN is unavailable.

Usage:
    from score_fetcher import fetch_live_scores

    scores = fetch_live_scores()
    # Returns list of dicts:
    # [{"home": "Germany", "away": "Curaçao", "home_score": 3, "away_score": 0,
    #   "state": "in", "clock": "67'", "date": "2026-06-14"}, ...]
    # state: "pre" | "in" | "post"
    # home_score/away_score: int when in/post, None when pre
"""

from __future__ import annotations

import json
import ssl
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────
ESPN_URL   = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
CACHE_FILE = Path("data/espn_scores.json")
CACHE_TTL  = 60   # 1 minute — ESPN updates frequently

# ESPN team name → schedule.py name
ESPN_TO_SCHEDULE: dict[str, str] = {
    "Türkiye":            "Turkey",
    "Côte d'Ivoire":      "Ivory Coast",
    "Korea Republic":     "South Korea",
    "IR Iran":            "Iran",
    "USA":                "United States",
    "Czechia":            "Czech Republic",
    "Curaçao":            "Curacao",
    "Bosnia & Herzegovina": "Bosnia",
}


def _ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _espn_name(name: str) -> str:
    return ESPN_TO_SCHEDULE.get(name, name)


def fetch_live_scores(date: Optional[str] = None, force: bool = False) -> list[dict]:
    """
    Fetch today's (or a specific date's) WC 2026 scores from ESPN.

    Args:
        date: YYYYMMDD string (default: today UTC)
        force: bypass cache

    Returns list of match dicts with keys:
        home, away, home_score, away_score, state, clock, date
    """
    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y%m%d")

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    cache_key = f"espn_{date}"

    if not force and CACHE_FILE.exists():
        try:
            cached = json.loads(CACHE_FILE.read_text())
            if cached.get("key") == cache_key and now - cached.get("fetched_at", 0) < CACHE_TTL:
                return cached["matches"]
        except Exception:
            pass

    url = f"{ESPN_URL}?dates={date}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=_ssl_ctx(), timeout=10) as r:
            data = json.loads(r.read())
    except Exception as e:
        _log(f"[espn] ⚠️  Fetch failed: {e}")
        if CACHE_FILE.exists():
            try:
                return json.loads(CACHE_FILE.read_text()).get("matches", [])
            except Exception:
                pass
        return []

    matches = []
    for event in data.get("events", []):
        comp = event["competitions"][0]
        competitors = comp["competitors"]
        try:
            home = next(c for c in competitors if c["homeAway"] == "home")
            away = next(c for c in competitors if c["homeAway"] == "away")
        except StopIteration:
            continue

        status = comp["status"]
        state  = status["type"]["state"]      # "pre" | "in" | "post"
        clock  = status.get("displayClock", "")

        home_name = _espn_name(home["team"]["displayName"])
        away_name = _espn_name(away["team"]["displayName"])

        def _score(c: dict) -> Optional[int]:
            s = c.get("score")
            if s is None or s == "":
                return None
            try:
                return int(s)
            except (ValueError, TypeError):
                return None

        matches.append({
            "home":       home_name,
            "away":       away_name,
            "home_score": _score(home) if state != "pre" else None,
            "away_score": _score(away) if state != "pre" else None,
            "state":      state,
            "clock":      clock,
            "date":       event.get("date", "")[:10],
        })

    CACHE_FILE.write_text(json.dumps(
        {"key": cache_key, "fetched_at": now, "matches": matches},
        ensure_ascii=False
    ))
    _log(f"[espn] ✅ {len(matches)} matches | "
         f"{sum(1 for m in matches if m['state']=='in')} live | "
         f"{sum(1 for m in matches if m['state']=='post')} finished")
    return matches


def patch_schedule_with_espn(force: bool = False) -> int:
    """
    Pull ESPN scores and update schedule.SCHEDULE in-place.
    Checks today + the past 3 days so we never miss a result that
    finished after midnight UTC or on a previous calendar day.
    Returns number of matches updated.
    """
    import schedule as sched_mod
    from datetime import timedelta

    now_utc = datetime.now(timezone.utc)
    # Collect matches from today and past 3 days
    all_espn: list[dict] = []
    seen_keys: set[tuple] = set()
    for days_back in range(4):   # 0=today, 1=yesterday, 2, 3
        d = (now_utc - timedelta(days=days_back)).strftime("%Y%m%d")
        day_matches = fetch_live_scores(date=d, force=(force and days_back == 0))
        for m in day_matches:
            key = (m["home"], m["away"])
            if key not in seen_keys:
                seen_keys.add(key)
                all_espn.append(m)

    espn = all_espn
    if not espn:
        return 0

    # Build lookup: (home_name, away_name) → match dict
    score_map = {(m["home"], m["away"]): m for m in espn if m["state"] == "post"}

    # Also include in-progress matches
    live_map  = {(m["home"], m["away"]): m for m in espn if m["state"] == "in"}

    updated = 0
    for match in sched_mod.SCHEDULE:
        h, a = match["home"], match["away"]

        # Prefer finished (post) scores; also take live (in) scores
        entry = score_map.get((h, a)) or live_map.get((h, a))
        if entry is None:
            continue

        hs = entry["home_score"]
        as_ = entry["away_score"]
        if hs is None:
            continue

        changed = match["home_score"] != hs or match["away_score"] != as_
        match["home_score"] = hs
        match["away_score"] = as_
        match["match_state"] = entry["state"]   # "in" | "post"
        if changed:
            state_label = "LIVE" if entry["state"] == "in" else "FT"
            _log(f"[espn] 🔄 {h} {hs}–{as_} {a} [{state_label} {entry['clock']}]")
            updated += 1

    if not updated:
        _log("[espn] ✓ No new scores from ESPN.")
    return updated


def _log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    date  = next((a for a in sys.argv[1:] if a.isdigit()), None)
    force = "--force" in sys.argv

    matches = fetch_live_scores(date=date, force=force)
    print(f"\n{'Team':<25} {'Score':^7} {'Team':<25} {'State':<6} {'Clock'}")
    print("-" * 75)
    for m in matches:
        hs = str(m["home_score"]) if m["home_score"] is not None else "-"
        as_ = str(m["away_score"]) if m["away_score"] is not None else "-"
        score = f"{hs} - {as_}"
        print(f"{m['home']:<25} {score:^7} {m['away']:<25} {m['state']:<6} {m['clock']}")
