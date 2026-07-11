"""
fetch_historical_corners.py — Pull corner stats from major international tournaments
via ESPN's summary API.

Tournaments covered:
  • FIFA World Cup 2022 (64 matches)
  • UEFA Euro 2024 (51 matches)
  • Copa America 2024 (32 matches)
  • AFC Asian Cup 2023 (51 matches)
  • AFCON 2023 (52 matches)
  • WC Qualifiers 2022/2026 (sample)

Output: data/historical_corners.json
  [
    {
      "tournament": "FIFA World Cup 2022",
      "date": "2022-11-20",
      "home": "Qatar", "away": "Ecuador",
      "home_corners": 1, "away_corners": 3, "total_corners": 4,
      "home_shots": 5, "away_shots": 6,
      "home_poss": 47.1, "away_poss": 52.9,
      "home_crosses": 8, "away_crosses": 12,
      "home_passes": 412, "away_passes": 521,
      "home_fouls": 10, "away_fouls": 8,
      "home_yellows": 1, "away_yellows": 2,
      "event_id": "633790"
    }, ...
  ]
"""

from __future__ import annotations
import json, ssl, time, urllib.request
from datetime import date, timedelta
from pathlib import Path

OUTPUT_PATH = Path("data/historical_corners.json")

_SSL = ssl.create_default_context()
_SSL.check_hostname = False
_SSL.verify_mode    = ssl.CERT_NONE

def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=_SSL, timeout=15) as r:
        return json.loads(r.read())

def _date_range(start: str, end: str) -> list[str]:
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    out = []
    while s <= e:
        out.append(s.strftime("%Y%m%d"))
        s += timedelta(days=1)
    return out

# ── Tournament definitions ────────────────────────────────────────────────────
TOURNAMENTS = [
    {
        "slug":  "fifa.world",
        "name":  "FIFA World Cup 2022",
        "dates": _date_range("2022-11-20", "2022-12-18"),
    },
    {
        "slug":  "uefa.euro",
        "name":  "UEFA Euro 2024",
        "dates": _date_range("2024-06-14", "2024-07-14"),
    },
    {
        "slug":  "conmebol.america",
        "name":  "Copa America 2024",
        "dates": _date_range("2024-06-20", "2024-07-14"),
    },
    {
        "slug":  "afc.asian",
        "name":  "AFC Asian Cup 2023",
        "dates": _date_range("2024-01-12", "2024-02-10"),
    },
    {
        "slug":  "caf.nations",
        "name":  "AFCON 2023",
        "dates": _date_range("2024-01-13", "2024-02-11"),
    },
    {
        "slug":  "concacaf.nations.league",
        "name":  "CONCACAF Nations League 2023-24",
        "dates": _date_range("2024-03-21", "2024-03-25"),
    },
    {
        "slug":  "uefa.nations",
        "name":  "UEFA Nations League 2024-25 Finals",
        "dates": _date_range("2025-06-04", "2025-06-08"),
    },
]

# ── Fetch event IDs for a tournament on a date ────────────────────────────────
ESPN_SCORE = "https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard"
ESPN_SUMM  = "https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/summary"

def fetch_events_for_date(slug: str, d: str) -> list[dict]:
    """Return list of {event_id, home, away, date} for a given date."""
    try:
        url  = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard?dates={d}"
        data = _get(url)
        out  = []
        for e in data.get("events", []):
            name = e.get("name", "")
            eid  = e.get("id",   "")
            # Only include completed matches
            status = e.get("status", {}).get("type", {}).get("completed", False)
            if not status:
                continue
            if " at " in name:
                parts = name.split(" at ", 1)
                away_n, home_n = parts[0].strip(), parts[1].strip()
                date_str = e.get("date", d)[:10]
                out.append({"event_id": eid, "home": home_n, "away": away_n, "date": date_str})
        return out
    except Exception:
        return []


def fetch_match_stats(slug: str, event_id: str, home: str, away: str, tournament: str) -> dict | None:
    try:
        url   = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/summary?event={event_id}"
        data  = _get(url)
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
                "shots":     int(num("totalShots") or 0),
                "shots_on":  int(num("shotsOnTarget") or 0),
                "crosses":   int(num("totalCrosses") or 0),
                "passes":    int(num("totalPasses") or 0),
                "acc_passes":int(num("accuratePasses") or 0),
                "fouls":     int(num("foulsCommitted") or 0),
                "yellows":   int(num("yellowCards") or 0),
                "reds":      int(num("redCards") or 0),
                "poss":      num("possessionPct"),
                "offsides":  int(num("offsides") or 0),
            }

        t0, t1 = parse_team(teams[0]), parse_team(teams[1])

        # Match ESPN teams to home/away names
        def matches(espn: str, sched: str) -> bool:
            e, s = espn.lower(), sched.lower()
            return e in s or s in e or e.split()[0] in s or s.split()[0] in e

        if matches(t0["team_name"], home):
            h, a = t0, t1
        elif matches(t1["team_name"], home):
            h, a = t1, t0
        else:
            h, a = t1, t0  # fallback: ESPN usually puts away first

        # Skip if no corner data at all
        if h["corners"] == 0 and a["corners"] == 0 and h["shots"] == 0:
            return None

        return {
            "tournament":     tournament,
            "event_id":       event_id,
            "home":           home,
            "away":           away,
            "home_corners":   h["corners"],
            "away_corners":   a["corners"],
            "total_corners":  h["corners"] + a["corners"],
            "home_shots":     h["shots"],
            "away_shots":     a["shots"],
            "home_shots_on":  h["shots_on"],
            "away_shots_on":  a["shots_on"],
            "home_poss":      h["poss"],
            "away_poss":      a["poss"],
            "home_crosses":   h["crosses"],
            "away_crosses":   a["crosses"],
            "home_passes":    h["passes"],
            "away_passes":    a["passes"],
            "home_acc_passes":h["acc_passes"],
            "away_acc_passes":a["acc_passes"],
            "home_fouls":     h["fouls"],
            "away_fouls":     a["fouls"],
            "home_yellows":   h["yellows"],
            "away_yellows":   a["yellows"],
            "home_reds":      h["reds"],
            "away_reds":      a["reds"],
            "home_offsides":  h["offsides"],
            "away_offsides":  a["offsides"],
        }
    except Exception as ex:
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def fetch_all(force: bool = False) -> list[dict]:
    existing: dict[str, dict] = {}
    if OUTPUT_PATH.exists() and not force:
        try:
            for rec in json.loads(OUTPUT_PATH.read_text()):
                existing[rec["event_id"]] = rec
        except Exception:
            pass

    all_records: list[dict] = list(existing.values())
    seen_ids = set(existing.keys())

    for t in TOURNAMENTS:
        slug = t["slug"]
        name = t["name"]
        print(f"\n[hist] {name} ({len(t['dates'])} days to scan)...")

        new_events = []
        for d in t["dates"]:
            events = fetch_events_for_date(slug, d)
            for ev in events:
                if ev["event_id"] not in seen_ids:
                    ev["slug"]       = slug
                    ev["tournament"] = name
                    new_events.append(ev)
            time.sleep(0.1)

        print(f"  {len(new_events)} new completed events found")

        for ev in new_events:
            rec = fetch_match_stats(ev["slug"], ev["event_id"], ev["home"], ev["away"], ev["tournament"])
            if rec:
                rec["date"] = ev["date"]
                all_records.append(rec)
                seen_ids.add(ev["event_id"])
                print(f"  ✓ {ev['home']} vs {ev['away']}: {rec['home_corners']}+{rec['away_corners']}={rec['total_corners']} corners")
            time.sleep(0.15)

    OUTPUT_PATH.write_text(json.dumps(all_records, indent=2, ensure_ascii=False))
    print(f"\n[hist] Saved {len(all_records)} matches → {OUTPUT_PATH}")

    if all_records:
        totals = [r["total_corners"] for r in all_records]
        avg    = sum(totals) / len(totals)
        print(f"[hist] Average total corners: {avg:.1f}")

        # Per-tournament breakdown
        from collections import defaultdict
        by_t: dict[str, list] = defaultdict(list)
        for r in all_records:
            by_t[r["tournament"]].append(r["total_corners"])
        print("\n[hist] Per-tournament averages:")
        for tn, vals in sorted(by_t.items()):
            print(f"  {tn:<45} n={len(vals):>3}  avg={sum(vals)/len(vals):.1f}  "
                  f"min={min(vals)}  max={max(vals)}")

    return all_records


if __name__ == "__main__":
    import sys
    fetch_all(force="--force" in sys.argv)
