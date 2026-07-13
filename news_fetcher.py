"""
news_fetcher.py — Real news & sentiment for WC 2026 matches.

Sources (no API key required):
  • Google News RSS   — headline + source + timestamp
  • BBC Sport RSS     — additional football coverage

Sentiment (keyword-based):
  • Scans each headline for positive / negative / neutral signals
    relative to each team name → shows media bias vs model prediction

Caching:
  • Results cached to data/news_cache.json (TTL = 3 hours)
  • Re-fetch if cache is stale or match not cached yet
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()  # system certs fallback

CACHE_FILE = Path("data/news_cache.json")
CACHE_TTL  = 3 * 3600          # 3 hours in seconds
REQUEST_DELAY = 0.6             # seconds between requests (polite)

# ── Keyword sentiment dictionaries ────────────────────────────────────────────
POSITIVE_KW = [
    "win", "won", "victory", "dominant", "strong", "confident",
    "ready", "form", "sharp", "star", "brilliant", "unstoppable",
    "advance", "qualify", "favourit", "favorite", "top", "best",
    "recover", "fit", "available", "returned", "unbeaten",
    "momentum", "danger", "threat", "impressive", "quality",
]
NEGATIVE_KW = [
    "injur", "suspend", "doubt", "doubt", "crisis", "struggle",
    "concern", "loss", "lost", "defeat", "poor", "worst",
    "absent", "miss", "ruled out", "drop", "trouble", "ban",
    "fired", "sacked", "chaos", "conflict", "protest", "unrest",
    "disqualif", "fine", "warning",
]


def _sentiment_score(text: str, team: str) -> str:
    """Return 'positive', 'negative', or 'neutral' for a team in a headline."""
    lower = text.lower()
    # Check if team is mentioned in this headline
    team_variants = [team.lower(), team.split()[-1].lower()]
    mentioned = any(v in lower for v in team_variants)
    if not mentioned:
        return "neutral"

    pos = sum(1 for kw in POSITIVE_KW if kw in lower)
    neg = sum(1 for kw in NEGATIVE_KW if kw in lower)
    if pos > neg:
        return "positive"
    elif neg > pos:
        return "negative"
    return "neutral"


def _fetch_google_news(query: str, max_items: int = 5) -> list[dict]:
    """Fetch headlines from Google News RSS for a search query."""
    encoded = urllib.parse.quote(query)
    url = (
        f"https://news.google.com/rss/search"
        f"?q={encoded}&hl=en-US&gl=US&ceid=US:en"
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        req  = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10, context=_SSL_CTX) as resp:
            content = resp.read()
        root  = ET.fromstring(content)
        items = []
        for item in root.findall(".//item")[:max_items]:
            raw_title = item.findtext("title", "") or ""
            # Google News embeds source after " - "
            parts  = raw_title.rsplit(" - ", 1)
            title  = parts[0].strip()
            source = parts[1].strip() if len(parts) > 1 else item.findtext("source", "") or ""
            pub    = item.findtext("pubDate", "") or ""
            link   = item.findtext("link", "") or ""
            # Parse date → ISO string
            try:
                dt = datetime.strptime(pub[:25], "%a, %d %b %Y %H:%M:%S")
                pub_iso = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                pub_iso = pub[:16]

            items.append({
                "title":    title,
                "source":   source,
                "pub_date": pub_iso,
                "link":     link,
            })
        return items
    except (urllib.error.URLError, ET.ParseError, Exception):
        return []


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_cache(cache: dict):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_match_news(home: str, away: str, date: str,
                     match_no: int, max_items: int = 5) -> list[dict]:
    """
    Fetch news for one match.  Returns a list of article dicts:
        title, source, pub_date, link, home_sentiment, away_sentiment
    Results are cached per match_no for CACHE_TTL seconds.
    """
    cache     = _load_cache()
    cache_key = f"match_{match_no}"
    now_ts    = time.time()

    # Live-rebuild mode (GD_LIVE=1): never hit the network — cache at any age
    if os.environ.get("GD_LIVE") == "1":
        return cache.get(cache_key, {}).get("articles", [])

    if cache_key in cache:
        entry = cache[cache_key]
        if now_ts - entry.get("fetched_at", 0) < CACHE_TTL:
            return entry["articles"]

    # Build a focused search query
    query = f'"{home}" "{away}" World Cup 2026'
    articles = _fetch_google_news(query, max_items=max_items)

    # Fall back to a looser query if nothing found
    if not articles:
        query = f"{home} {away} FIFA 2026"
        articles = _fetch_google_news(query, max_items=max_items)

    # Attach sentiment for each team
    for a in articles:
        text = a["title"] + " " + a["source"]
        a["home_sentiment"] = _sentiment_score(text, home)
        a["away_sentiment"] = _sentiment_score(text, away)

    # Persist to cache
    cache[cache_key] = {"fetched_at": now_ts, "articles": articles}
    _save_cache(cache)

    return articles


def fetch_all_upcoming_news(
    schedule: list[dict],
    max_matches: int = 20,
    delay: float = REQUEST_DELAY,
) -> dict[int, list[dict]]:
    """
    Fetch news for upcoming (unplayed) matches.

    schedule   — list of match dicts from schedule.py
    max_matches — only fetch for the next N unplayed matches
    Returns    — {match_no: [article, ...]}
    """
    upcoming = [m for m in schedule if m["home_score"] is None][:max_matches]
    result   = {}

    total = len(upcoming)
    for i, m in enumerate(upcoming):
        match_no = m["match_no"]
        home, away, date = m["home"], m["away"], m["date"]
        print(f"  [news] {i+1}/{total} Match {match_no}: {home} vs {away} ... ", end="", flush=True)

        articles = fetch_match_news(home, away, date, match_no)
        result[match_no] = articles
        print(f"{len(articles)} articles")

        # Be polite — don't hammer the RSS endpoint
        if i < total - 1:
            time.sleep(delay)

    return result


def aggregate_sentiment(articles: list[dict], home: str, away: str) -> dict:
    """
    Aggregate article sentiments into a summary for both teams.
    Returns:
        {
          home_positive: int, home_negative: int, home_neutral: int,
          away_positive: int, away_negative: int, away_neutral: int,
          home_bias: 'bullish'|'bearish'|'neutral',
          away_bias: 'bullish'|'bearish'|'neutral',
        }
    """
    def counts(key):
        pos = sum(1 for a in articles if a.get(key) == "positive")
        neg = sum(1 for a in articles if a.get(key) == "negative")
        neu = sum(1 for a in articles if a.get(key) == "neutral")
        return pos, neg, neu

    hp, hn, hne = counts("home_sentiment")
    ap, an, ane = counts("away_sentiment")

    def bias(p, n):
        if p > n:     return "bullish"
        elif n > p:   return "bearish"
        return "neutral"

    return {
        "home_positive": hp, "home_negative": hn, "home_neutral": hne,
        "away_positive": ap, "away_negative": an, "away_neutral": ane,
        "home_bias": bias(hp, hn),
        "away_bias": bias(ap, an),
    }


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    home = sys.argv[1] if len(sys.argv) > 2 else "Brazil"
    away = sys.argv[2] if len(sys.argv) > 2 else "Morocco"
    arts = fetch_match_news(home, away, "2026-06-13", match_no=7)
    for a in arts:
        print(f"  [{a['pub_date']}] {a['title']}  ({a['source']})")
        print(f"    {home} sentiment: {a['home_sentiment']}  |  {away} sentiment: {a['away_sentiment']}")
