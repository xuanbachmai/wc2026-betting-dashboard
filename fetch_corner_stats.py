"""
fetch_corner_stats.py — Pull actual corner stats for all played WC 2026 matches from ESPN.

ESPN boxscore API provides per-team 'wonCorners' for every finished match.
No API key needed.

Output: data/actual_corners.json
  {
    "mexico vs south africa": {
      "home": "Mexico", "away": "South Africa",
      "home_corners": 3, "away_corners": 1, "total": 4,
      "possession_home": 60.5, "possession_away": 39.5,
      "shots_home": 16, "shots_away": 3,
      "source": "espn", "event_id": "760415"
    }, ...
  }

Usage:
    python3 fetch_corner_stats.py
    python3 fetch_corner_stats.py --force
"""

from __future__ import annotations
import json, os, ssl, time, urllib.request
from pathlib import Path

from schedule import SCHEDULE

OUTPUT_PATH = Path("data/actual_corners.json")
ESPN_SCORE  = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
ESPN_SUMM   = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/summary"

_SSL = ssl.create_default_context()
_SSL.check_hostname = False
_SSL.verify_mode    = ssl.CERT_NONE

def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=_SSL, timeout=12) as r:
        return json.loads(r.read())

ESPN_NAME_MAP = {
    "czechia":             "czech republic",
    "bosnia-herzegovina":  "bosnia and herzegovina",
    "türkiye":             "turkey",
    "curaçao":             "curacao",
    "cape verde":          "cabo verde",   # ESPN uses "Cape Verde", schedule uses "Cabo Verde"
    # "ivory coast" stays as "ivory coast" to match schedule
    "korea republic":      "south korea",
}

def _norm(s: str) -> str:
    s = s.strip().lower()
    return ESPN_NAME_MAP.get(s, s)

# ── Step 1: Get ESPN event IDs for all WC dates ───────────────────────────────

def fetch_event_ids() -> dict[str, str]:
    """Return {match_key → espn_event_id} for COMPLETED (state=post) WC matches."""
    played = [m for m in SCHEDULE if m.get("home_score") is not None]
    dates  = sorted({m["date"] for m in played})

    event_map: dict[str, str] = {}   # "home vs away" (lower) → event_id

    for date in dates:
        d_str = date.replace("-", "")
        try:
            data   = _get(f"{ESPN_SCORE}?dates={d_str}")
            events = data.get("events", [])
            completed_count = 0
            for e in events:
                # Only include fully completed matches (state="post"), skip live/pre
                state = e.get("status", {}).get("type", {}).get("state", "")
                if state != "post":
                    continue
                name = e.get("name", "")   # "South Africa at Mexico"
                eid  = e.get("id", "")
                # Parse "Away at Home" format
                if " at " in name:
                    parts = name.split(" at ", 1)
                    away_n, home_n = parts[0].strip(), parts[1].strip()
                    key = f"{_norm(home_n)} vs {_norm(away_n)}"
                    event_map[key] = eid
                    completed_count += 1
            print(f"  {date}: {len(events)} events, {completed_count} completed")
        except Exception as ex:
            print(f"  {date}: error — {ex}")
        time.sleep(0.15)

    return event_map

# ── Step 2: Fetch boxscore stats for each event ───────────────────────────────

STAT_FIELDS = [
    "wonCorners", "possessionPct", "totalShots", "shotsOnTarget",
    "yellowCards", "redCards", "foulsCommitted", "offsides",
    "accuratePasses", "totalPasses", "accurateCrosses", "totalCrosses",
]

def fetch_match_stats(event_id: str, home: str, away: str) -> dict | None:
    try:
        data  = _get(f"{ESPN_SUMM}?event={event_id}")
        teams = data.get("boxscore", {}).get("teams", [])
        if len(teams) < 2:
            return None

        def parse_team(t: dict) -> dict:
            stats = {s["name"]: s.get("displayValue") for s in t.get("statistics", [])}
            def num(k):
                v = stats.get(k)
                try: return float(v) if v is not None else None
                except: return None
            return {
                "team_name": t.get("team", {}).get("name", ""),
                "corners":   int(num("wonCorners") or 0),
                "possession": num("possessionPct"),
                "shots":      int(num("totalShots") or 0),
                "shots_on":   int(num("shotsOnTarget") or 0),
                "crosses":    int(num("totalCrosses") or 0),
                "passes":     int(num("totalPasses") or 0),
                "fouls":      int(num("foulsCommitted") or 0),
                "yellows":    int(num("yellowCards") or 0),
                "reds":       int(num("redCards") or 0),
            }

        # ESPN returns teams in away/home order typically — match to schedule names
        t0 = parse_team(teams[0])
        t1 = parse_team(teams[1])

        # Identify which is home by matching name (fuzzy)
        def matches(espn_name: str, sched_name: str) -> bool:
            en, sn = espn_name.lower(), sched_name.lower()
            return en in sn or sn in en or en.split()[0] in sn or sn.split()[0] in en

        if matches(t0["team_name"], home):
            h, a = t0, t1
        elif matches(t1["team_name"], home):
            h, a = t1, t0
        else:
            # fallback: ESPN usually puts away first
            h, a = t1, t0

        return {
            "home":          home,
            "away":          away,
            "home_corners":  h["corners"],
            "away_corners":  a["corners"],
            "total":         h["corners"] + a["corners"],
            "home_poss":     h["possession"],
            "away_poss":     a["possession"],
            "home_shots":    h["shots"],
            "away_shots":    a["shots"],
            "home_shots_on": h["shots_on"],
            "away_shots_on": a["shots_on"],
            "home_crosses":  h["crosses"],
            "away_crosses":  a["crosses"],
            "home_passes":   h["passes"],
            "away_passes":   a["passes"],
            "home_fouls":    h["fouls"],
            "away_fouls":    a["fouls"],
            "home_yellows":  h["yellows"],
            "away_yellows":  a["yellows"],
            "source":        "espn",
            "event_id":      event_id,
        }
    except Exception as ex:
        print(f"    stats error for event {event_id}: {ex}")
        return None

# ── Main ──────────────────────────────────────────────────────────────────────

def fetch_all(force: bool = False) -> dict:
    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    # Load existing data
    existing: dict = {}
    if OUTPUT_PATH.exists() and not force:
        try:
            existing = json.loads(OUTPUT_PATH.read_text())
        except Exception:
            pass

    print(f"[corners] Fetching corner stats from ESPN...")
    print(f"[corners] {len([m for m in SCHEDULE if m.get('home_score') is not None])} played matches to check\n")

    # Live-rebuild mode (GD_LIVE=1): serve the cache, skip all network calls
    if os.environ.get("GD_LIVE") == "1":
        _valid = {f"{_norm(m['home'])} vs {_norm(m['away'])}"
                  for m in SCHEDULE if m.get("home_score") is not None}
        results = {k: v for k, v in existing.items() if k in _valid}
        print(f"[corners] Live mode — using {len(results)} cached entries, no network")
        return results

    event_ids = fetch_event_ids()
    print(f"\n[corners] Found {len(event_ids)} ESPN event IDs\n")

    # Only keep cached entries for played schedule matches (filter out live/future match data)
    _valid_keys = {
        f"{_norm(m['home'])} vs {_norm(m['away'])}"
        for m in SCHEDULE if m.get("home_score") is not None
    }
    results: dict = {k: v for k, v in existing.items() if k in _valid_keys}

    for m in SCHEDULE:
        if m.get("home_score") is None:
            continue
        home, away = m["home"], m["away"]
        key = f"{_norm(home)} vs {_norm(away)}"

        # Skip if already have ESPN data and not forcing
        if not force and key in results and results[key].get("source") == "espn":
            print(f"  ✓ {home} vs {away}: cached (H={results[key]['home_corners']} A={results[key]['away_corners']} T={results[key]['total']})")
            continue

        eid = event_ids.get(key)
        if not eid:
            # Try reversed key (ESPN sometimes flips)
            eid = event_ids.get(f"{_norm(away)} vs {_norm(home)}")
        if not eid:
            print(f"  ✗ {home} vs {away}: no ESPN event ID found")
            continue

        stats = fetch_match_stats(eid, home, away)
        if stats:
            results[key] = stats
            print(f"  ✓ {home} vs {away}: H={stats['home_corners']} A={stats['away_corners']} Total={stats['total']}  (poss {stats['home_poss']}%/{stats['away_poss']}%)")
        else:
            print(f"  ✗ {home} vs {away}: stats unavailable")

        time.sleep(0.2)

    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[corners] Saved {len(results)} matches → {OUTPUT_PATH}")

    # Print summary
    with_data = [v for v in results.values() if v.get("total") is not None]
    if with_data:
        avg = sum(v["total"] for v in with_data) / len(with_data)
        print(f"[corners] Average total corners so far: {avg:.1f}")
        print(f"\n{'Match':<35} {'H':>3} {'A':>3} {'Tot':>4}  {'Poss':>10}")
        print("─" * 60)
        for k, v in results.items():
            if v.get("total") is None: continue
            poss = f"{v.get('home_poss','?')}%/{v.get('away_poss','?')}%"
            hc = v['home_corners'] if v['home_corners'] is not None else '?'
            ac = v['away_corners'] if v['away_corners'] is not None else '?'
            tot = v['total'] if v['total'] is not None else '?'
            print(f"{v['home']+' vs '+v['away']:<35} {str(hc):>3} {str(ac):>3} {str(tot):>4}  {poss:>10}")

    return results


if __name__ == "__main__":
    import sys
    fetch_all(force="--force" in sys.argv)
