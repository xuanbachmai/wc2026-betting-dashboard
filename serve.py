"""
serve.py — All-in-one dashboard server + live score watcher.

  http://localhost:8765/          → main dashboard
  http://localhost:8765/events    → SSE stream (live predictions + auto-reload)
  http://localhost:8765/api/live  → JSON corner stats
  http://localhost:8765/api/rebuild → manual rebuild trigger

Polls ESPN every 3 seconds. Auto-rebuilds on goals/FT (in a background thread).
Backfills missed results at startup and hourly. Browser reloads itself via SSE.
"""

from __future__ import annotations
import json, os, re, subprocess, sys, threading, time, urllib.request
from datetime import datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
HTTPServer = ThreadingHTTPServer  # alias so rest of code unchanged
from pathlib import Path
from scipy.stats import poisson

from update_results import (norm, espn_get, patch_schedule,
                            backfill_results, ESPN_SCORE, _SSL)

PORT = int(os.environ.get("PORT", 8765))       # cloud hosts inject PORT
HOST = os.environ.get("HOST", "0.0.0.0" if "PORT" in os.environ else "localhost")
ESPN_SUMM  = "https://site.api.espn.com/apis/site/v2/sports/soccer/FIFA.WORLD/summary"
OUTPUT_DIR    = Path("output")
POLL_INTERVAL = 3

# ── SSE clients ────────────────────────────────────────────────────────────────
_sse_clients: list = []
_sse_lock = threading.Lock()

def broadcast(data: dict):
    msg = ("data: " + json.dumps(data) + "\n\n").encode()
    with _sse_lock:
        dead = []
        for wfile in _sse_clients:
            try:
                wfile.write(msg)
                wfile.flush()
            except Exception:
                dead.append(wfile)
        for d in dead:
            _sse_clients.remove(d)

# ── ESPN helpers ───────────────────────────────────────────────────────────────
def parse_minute(clock: str, period: int) -> int:
    if not clock:
        return 45 if period == 1 else 90
    et = re.search(r"(\d+)'\+(\d+)'", clock)
    if et:
        return int(et.group(1)) + int(et.group(2))
    m = re.search(r"(\d+)", clock)
    if m:
        mins = int(m.group(1))
        if period == 2 and mins < 45:
            mins += 45
        return mins
    return 45

# ── Live corner cache ──────────────────────────────────────────────────────────
_corner_cache: dict = {"data": [], "ts": 0}

def fetch_corners() -> list[dict]:
    if time.time() - _corner_cache["ts"] < 30:
        return _corner_cache["data"]
    results = []
    try:
        data = espn_get(ESPN_SCORE)
        for ev in data.get("events", []):
            state = ev.get("status", {}).get("type", {}).get("state", "")
            if state not in ("in", "post"):
                continue
            eid  = ev.get("id", "")
            comp = (ev.get("competitions") or [{}])[0]
            competitors = comp.get("competitors", [])
            scores = {}; names = {}
            for c in competitors:
                hoa = c.get("homeAway", "")
                scores[hoa] = c.get("score", "?")
                names[hoa]  = c.get("team", {}).get("displayName", "?")
            ch = ca = None
            try:
                summ = espn_get(f"{ESPN_SUMM}?event={eid}")
                for t in summ.get("boxscore", {}).get("teams", []):
                    hoa   = t.get("homeAway", "")
                    stats = {s["name"]: s.get("displayValue") for s in t.get("statistics", [])}
                    val   = stats.get("wonCorners")
                    v = int(float(val)) if val is not None else None
                    if hoa == "home": ch = v
                    else:             ca = v
            except Exception:
                pass
            results.append({
                "home": names.get("home","?"), "away": names.get("away","?"),
                "home_score": scores.get("home","?"), "away_score": scores.get("away","?"),
                "minute": ev.get("status",{}).get("displayClock","?"),
                "completed": ev.get("status",{}).get("type",{}).get("completed", False),
                "corners_home": ch, "corners_away": ca,
                "total_corners": (ch + ca) if ch is not None and ca is not None else None,
            })
            time.sleep(0.1)
    except Exception as e:
        log(f"Corner fetch error: {e}")
    _corner_cache["data"] = results
    _corner_cache["ts"]   = time.time()
    return results

# ── Live prediction ────────────────────────────────────────────────────────────
_prematch: dict[str, dict] = {}

def get_prematch(home: str, away: str) -> dict:
    key = f"{home}|{away}"
    if key not in _prematch:
        try:
            from engine import predict_enhanced, _load_cached_pipeline
            pl = _load_cached_pipeline()
            _prematch[key] = predict_enhanced(home, away,
                pl["outcome_model"], pl["goals_model"],
                pl["elo_ratings"],   pl["team_stats"])
        except Exception:
            _prematch[key] = {"xg_home": 1.3, "xg_away": 1.0,
                              "p_home_win": 40, "p_draw": 30, "p_away_win": 30}
    return _prematch[key]

def live_probs(xg_h, xg_a, sh, sa, minute):
    frac  = max(0, 90 - min(minute, 90)) / 90
    rh, ra = xg_h * frac, xg_a * frac
    ph = pd = pa = 0.0
    for gh in range(9):
        for ga in range(9):
            p = poisson.pmf(gh, max(rh, 1e-6)) * poisson.pmf(ga, max(ra, 1e-6))
            fh, fa = sh + gh, sa + ga
            if fh > fa:   ph += p
            elif fh == fa: pd += p
            else:          pa += p
    t = ph + pd + pa
    return round(ph/t*100,1), round(pd/t*100,1), round(pa/t*100,1)

def winner_label(home, away, ph, pd, pa):
    if ph >= pd and ph >= pa: return f"{home} ({ph}%)"
    if pa >= ph and pa >= pd: return f"{away} ({pa}%)"
    return f"Draw ({pd}%)"

# ── Debounced background rebuild ───────────────────────────────────────────────
_rebuild_requested = threading.Event()
_rebuild_full      = threading.Event()   # set → next rebuild refreshes news/corners

def request_rebuild(full: bool = False):
    if full:
        _rebuild_full.set()
    _rebuild_requested.set()

def rebuild(live: bool = True):
    """live=True skips news/corners network refresh (~1 min instead of ~2.5)."""
    if os.environ.get("NO_REBUILD") == "1":
        log("Rebuild skipped (NO_REBUILD=1 — updates arrive via repo redeploys)")
        return False
    env = dict(os.environ, GD_LIVE="1") if live else dict(os.environ)
    try:
        subprocess.run([sys.executable, "engine.py"],
                       timeout=900, check=True, capture_output=True, env=env)
        return True
    except Exception as e:
        log(f"Rebuild error: {e}")
        return False

def rebuild_worker():
    """Runs rebuilds off the polling thread so score checks never stall."""
    while True:
        _rebuild_requested.wait()
        time.sleep(5)                # debounce: batch rapid goal bursts
        _rebuild_requested.clear()
        full = _rebuild_full.is_set()
        _rebuild_full.clear()
        log(f"🔄 Rebuilding dashboard ({'full' if full else 'live'} mode)...")
        if rebuild(live=not full):
            log("✅ Rebuilt → dashboard updated")
            broadcast({"type": "reload"})

def full_refresh_loop():
    """Every 6 h, refresh news + corners with a full rebuild."""
    while True:
        time.sleep(6 * 3600)
        request_rebuild(full=True)

# ── Backfill: recover results missed while the server was down ─────────────────
def backfill_loop():
    """Backfill missing past results at startup, then hourly."""
    while True:
        try:
            if backfill_results():
                request_rebuild()
        except Exception as e:
            log(f"Backfill error: {e}")
        time.sleep(3600)

# ── Score watcher thread ───────────────────────────────────────────────────────
_finalised:  set[str]         = set()
_last_score: dict[str, tuple] = {}
_last_pred:  dict[str, tuple] = {}

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def watcher_loop():
    log("🟢 Score watcher active — polling ESPN every 3s")
    while True:
        try:
            req = urllib.request.Request(ESPN_SCORE, headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req, context=_SSL, timeout=10) as r:
                data = json.loads(r.read())

            changed = False
            live_updates = []

            for ev in data.get("events", []):
                comp   = ev["competitions"][0]
                status = comp["status"]
                state  = status["type"]["state"]
                clock  = status.get("displayClock", "")
                period = status.get("period", 1)
                teams  = comp["competitors"]
                t_home = next(t for t in teams if t["homeAway"]=="home")
                t_away = next(t for t in teams if t["homeAway"]=="away")
                home   = norm(t_home["team"]["displayName"])
                away   = norm(t_away["team"]["displayName"])
                hs     = int(t_home.get("score", 0)) if state != "pre" else None
                as_    = int(t_away.get("score", 0)) if state != "pre" else None
                # Penalty shootout scores (present only when match went to pens)
                hp = t_home.get("shootoutScore")
                ap = t_away.get("shootoutScore")
                hp = int(hp) if hp is not None else None
                ap = int(ap) if ap is not None else None
                key    = f"{home}|{away}"

                if state == "pre":
                    continue

                if state == "post":
                    if key not in _finalised:
                        if patch_schedule(home, away, hs, as_, hp, ap):
                            pens = f" (pens {hp}–{ap})" if hp is not None else ""
                            log(f"✅ FT: {home} {hs}–{as_} {away}{pens}")
                            changed = True
                        _finalised.add(key)
                    continue

                # In progress
                minute = parse_minute(clock, period)
                prev_score = _last_score.get(key)
                score_changed = prev_score != (hs, as_)

                if score_changed:
                    if prev_score is not None:
                        old_h, old_a = prev_score
                        scorer = home if hs > old_h else (away if as_ > old_a else "?")
                        log(f"⚽ GOAL! {scorer}  →  {home} {hs}–{as_} {away}  [{clock}]")
                    _last_score[key] = (hs, as_)
                    patch_schedule(home, away, hs, as_)
                    changed = True

                # Recalculate live probs
                pred = get_prematch(home, away)
                ph, pd, pa = live_probs(pred.get("xg_home",1.3), pred.get("xg_away",1.0), hs, as_, minute)

                prev_pred = _last_pred.get(key)
                if prev_pred:
                    old_ph, old_pd, old_pa = prev_pred
                    if abs(ph-old_ph) >= 3 or abs(pa-old_pa) >= 3:
                        log(f"📈 PRED [{clock}] {home} vs {away}: "
                            f"{home} {old_ph}%→{ph}% | Draw {old_pd}%→{pd}% | {away} {old_pa}%→{pa}%")
                        log(f"   → {winner_label(home, away, ph, pd, pa)}")
                else:
                    log(f"🔴 LIVE [{clock}]: {home} {hs}–{as_} {away}  "
                        f"| {home} {ph}% Draw {pd}% {away} {pa}%")
                _last_pred[key] = (ph, pd, pa)

                live_updates.append({
                    "home": home, "away": away,
                    "score_h": hs, "score_a": as_,
                    "clock": clock, "minute": minute,
                    "ph": ph, "pd": pd, "pa": pa,
                    "xg_h": round(pred.get("xg_home",1.3),2),
                    "xg_a": round(pred.get("xg_away",1.0),2),
                    "winner": winner_label(home, away, ph, pd, pa),
                })

            if live_updates:
                broadcast({"type": "live", "matches": live_updates})

            if changed:
                request_rebuild()

        except Exception as e:
            log(f"Watcher error: {e}")

        time.sleep(POLL_INTERVAL)

# ── HTTP Handler ───────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        if int(args[1]) >= 400:
            super().log_message(fmt, *args)

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]

        # SSE stream (also handle /stream alias for old clients)
        if path == "/events" or path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with _sse_lock:
                _sse_clients.append(self.wfile)
            try:
                while True:
                    time.sleep(25)
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
            except Exception:
                with _sse_lock:
                    if self.wfile in _sse_clients:
                        _sse_clients.remove(self.wfile)
            return

        if path == "/" or path == "/index.html":
            p = OUTPUT_DIR / "dashboard.html"
            if not p.exists():
                self._send(404, "text/plain", b"Not built yet. Run: python3 engine.py")
                return
            html = p.read_text(encoding="utf-8")
            # Inject SSE client at serve time (rebuilds regenerate the file without it)
            if "ES.onmessage" not in html:
                html = html.replace("</body>", SSE_JS + "\n</body>")
            self._send(200, "text/html; charset=utf-8", html.encode("utf-8"))

        elif path == "/api/live":
            body = json.dumps({"matches": fetch_corners(), "ts": int(time.time())}).encode()
            self._send(200, "application/json", body)

        elif path == "/api/rebuild":
            log("Manual rebuild triggered")
            ok = rebuild()
            self._send(200, "application/json", json.dumps({"ok": ok}).encode())
            if ok:
                broadcast({"type": "reload"})

        else:
            fpath = OUTPUT_DIR / path.lstrip("/")
            if fpath.exists() and fpath.is_file():
                ext = fpath.suffix
                ctype = ("text/html; charset=utf-8" if ext == ".html" else
                         "text/css" if ext == ".css" else
                         "application/javascript" if ext == ".js" else
                         "application/octet-stream")
                self._send(200, ctype, fpath.read_bytes())
            else:
                self._send(404, "text/plain", b"Not found")

# ── SSE inject ─────────────────────────────────────────────────────────────────
SSE_JS = """
<script>
(function(){
  const ES = new EventSource('/events');
  let panel = null;

  function color(p){ return p>=65?'#22c55e':p>=45?'#f59e0b':'#ef4444'; }

  function renderMatch(m){
    return `<div style="margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid #1e293b">
      <div style="display:flex;justify-content:space-between">
        <span style="font-size:12px;color:#f59e0b;font-weight:700">🔴 ${m.clock}</span>
        <span style="font-size:10px;color:#475569">xG ${m.xg_h}–${m.xg_a}</span>
      </div>
      <div style="font-size:15px;color:#e2e8f0;font-weight:800;margin:4px 0">
        ${m.home} <span style="color:#fff">${m.score_h}–${m.score_a}</span> ${m.away}
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;margin-top:6px">
        <div style="text-align:center;background:#1e293b;border-radius:6px;padding:5px 2px">
          <div style="font-size:17px;font-weight:800;color:${color(m.ph)}">${m.ph}%</div>
          <div style="font-size:9px;color:#64748b">${m.home.split(' ')[0]}</div>
        </div>
        <div style="text-align:center;background:#1e293b;border-radius:6px;padding:5px 2px">
          <div style="font-size:17px;font-weight:800;color:#94a3b8">${m.pd}%</div>
          <div style="font-size:9px;color:#64748b">Draw</div>
        </div>
        <div style="text-align:center;background:#1e293b;border-radius:6px;padding:5px 2px">
          <div style="font-size:17px;font-weight:800;color:${color(m.pa)}">${m.pa}%</div>
          <div style="font-size:9px;color:#64748b">${m.away.split(' ')[0]}</div>
        </div>
      </div>
      <div style="font-size:11px;color:#22c55e;margin-top:5px">→ <b>${m.winner}</b></div>
    </div>`;
  }

  ES.onmessage = function(e){
    const msg = JSON.parse(e.data);

    if(msg.type==='reload'){
      setTimeout(()=>location.reload(), 1500);
      return;
    }

    if(msg.type==='live' && msg.matches && msg.matches.length){
      if(!panel || !document.contains(panel)){
        panel = document.createElement('div');
        panel.style.cssText='position:fixed;top:16px;right:16px;z-index:9999;width:300px;'+
          'background:#0f172a;border:2px solid #f59e0b;border-radius:14px;'+
          'padding:14px 16px;box-shadow:0 8px 32px rgba(0,0,0,.75);font-family:system-ui,sans-serif;';
        document.body.appendChild(panel);
      }
      const ts = new Date().toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',second:'2-digit'});
      panel.innerHTML =
        `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
          <span style="color:#f59e0b;font-weight:800;font-size:13px">⚽ LIVE PREDICTIONS</span>
          <div style="display:flex;gap:8px;align-items:center">
            <span style="font-size:10px;color:#475569">${ts}</span>
            <button onclick="this.closest('div[style]').remove()"
              style="background:none;border:none;color:#64748b;cursor:pointer;font-size:18px;line-height:1">✕</button>
          </div>
        </div>
        ${msg.matches.map(renderMatch).join('')}
        <div style="font-size:10px;color:#334155;text-align:right;margin-top:4px">Updates every 10s · auto-reloads on goal</div>`;
    }
  };

  ES.onerror = function(){ setTimeout(()=>location.reload(),8000); };
})();
</script>
"""

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    # First boot on a fresh host: build synchronously if missing, otherwise
    # refresh in the background (the committed copy may be stale)
    if not (OUTPUT_DIR / "dashboard.html").exists():
        log("No dashboard found — building initial dashboard...")
        rebuild()
    else:
        request_rebuild()

    # Start watcher + rebuild + backfill + periodic full-refresh threads
    threading.Thread(target=watcher_loop, daemon=True).start()
    threading.Thread(target=rebuild_worker, daemon=True).start()
    threading.Thread(target=backfill_loop, daemon=True).start()
    threading.Thread(target=full_refresh_loop, daemon=True).start()

    HTTPServer.allow_reuse_address = True
    server = HTTPServer((HOST, PORT), Handler)
    log(f"🌐 Server → http://{HOST}:{PORT}")
    log(f"   SSE auto-reload + live predictions active")
    log("   Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("Stopped.")
