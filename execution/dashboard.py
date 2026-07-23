"""
execution/dashboard.py — Mobile-friendly live trading dashboard.

Run alongside live_trader.py:
    python -m execution.dashboard

Then tunnel with ngrok:
    ngrok http 5050
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify

STATUS_FILE = Path("experiments/live_status.json")
REFRESH_SEC = 55   # page auto-refresh interval

app = Flask(__name__)

_HTML = """\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <meta http-equiv="refresh" content="{refresh}"/>
  <title>MWM Live</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{background:#0d0d0d;color:#e8e8e8;font-family:'Segoe UI',sans-serif;padding:16px;max-width:480px;margin:auto}}
    h1{{font-size:1.1rem;color:#aaa;margin-bottom:12px;letter-spacing:.05em}}
    .card{{background:#1a1a1a;border-radius:12px;padding:14px 16px;margin-bottom:12px}}
    .label{{font-size:.7rem;color:#666;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px}}
    .value{{font-size:1.5rem;font-weight:700}}
    .badge{{display:inline-block;padding:4px 14px;border-radius:20px;font-size:1rem;font-weight:700}}
    .LONG{{background:#0e4a2a;color:#2dca72}}
    .SHORT{{background:#4a0e0e;color:#e85252}}
    .FLAT{{background:#2a2a2a;color:#999}}
    .row{{display:flex;gap:10px}}
    .row .card{{flex:1}}
    .good{{color:#2dca72}}.warn{{color:#e8a532}}.bad{{color:#e85252}}
    table{{width:100%;border-collapse:collapse;font-size:.8rem}}
    th{{color:#666;font-weight:500;text-align:left;padding:6px 4px;border-bottom:1px solid #2a2a2a}}
    td{{padding:6px 4px;border-bottom:1px solid #1e1e1e}}
    .action-hold{{color:#999}}
    .action-long{{color:#2dca72}}
    .action-short{{color:#e85252}}
    .action-flat{{color:#e8a532}}
    .stale{{color:#e8a532;font-size:.75rem;margin-top:6px}}
    footer{{text-align:center;color:#444;font-size:.7rem;margin-top:16px}}
  </style>
</head>
<body>
  <h1>&#9660; MWM LIVE &mdash; {symbol} {mode}</h1>

  {stale_banner}

  <div class="card">
    <div class="label">Position</div>
    <span class="badge {pos_class}">{position}</span>
  </div>

  <div class="row">
    <div class="card">
      <div class="label">Dir Prob</div>
      <div class="value {dir_color}">{dir_prob}</div>
    </div>
    <div class="card">
      <div class="label">S&#x209C; Surprise</div>
      <div class="value {st_color}">{s_t}</div>
    </div>
  </div>

  <div class="row">
    <div class="card">
      <div class="label">Signal</div>
      <div class="value">{signal}</div>
    </div>
    <div class="card">
      <div class="label">Last Action</div>
      <div class="value {action_color}">{action}</div>
    </div>
  </div>

  <div class="card">
    <div class="label">Last Bar</div>
    <div style="font-size:.85rem;color:#aaa">{last_ts}</div>
    <div class="label" style="margin-top:8px">Next bar in</div>
    <div id="cdown" style="font-size:1.1rem;font-weight:600;color:#aaa">&mdash;</div>
  </div>

  <div class="card">
    <div class="label">Recent Bars</div>
    {history_table}
  </div>

  <footer>Surprise buf: {buf_len} bars &middot; refreshes every {refresh}s</footer>

  <script>
    (function(){{
      var lastTs = "{last_ts_iso}";
      if(!lastTs) return;
      function tick(){{
        var now = new Date();
        var next = new Date(now);
        next.setUTCMinutes(0); next.setUTCSeconds(0); next.setUTCMilliseconds(0);
        next.setUTCHours(next.getUTCHours()+1);
        var diff = Math.round((next-now)/1000);
        var m = Math.floor(diff/60), s = diff%60;
        document.getElementById('cdown').textContent =
          m+'m '+String(s).padStart(2,'0')+'s';
      }}
      tick(); setInterval(tick,1000);
    }})();
  </script>
</body>
</html>
"""


def _color_dir(prob: float) -> str:
    if prob > 0.55:
        return "good"
    if prob < 0.45:
        return "bad"
    return "warn"


def _color_st(s_t: float, history: list) -> str:
    if not history:
        return ""
    vals = [b.get("S_t", 0) for b in history]
    p60 = sorted(vals)[int(len(vals) * 0.6)]
    p90 = sorted(vals)[int(len(vals) * 0.9)]
    if s_t < p60:
        return "good"
    if s_t < p90:
        return "warn"
    return "bad"


def _action_color(action: str) -> str:
    a = action.lower()
    if "long" in a:
        return "action-long"
    if "short" in a:
        return "action-short"
    if "flat" in a:
        return "action-flat"
    return "action-hold"


def _build_page(data: dict) -> str:
    history = data.get("history", [])
    last    = history[0] if history else {}

    dir_prob = last.get("dir_prob", float("nan"))
    s_t      = last.get("S_t",      float("nan"))
    signal   = last.get("effective_signal", float("nan"))
    action   = last.get("action_taken", data.get("message", "—"))
    last_ts  = last.get("timestamp", "—")
    last_ts_iso = last_ts if last_ts != "—" else ""

    pos      = data.get("position", "FLAT")
    symbol   = data.get("symbol", "XAUUSD")
    mode     = "(DRY RUN)" if data.get("dry_run") else "(LIVE)"
    buf_len  = data.get("surprise_buf_len", 0)

    # Stale check — warn if last update > 90 min ago
    stale_banner = ""
    try:
        updated = datetime.fromisoformat(data["updated"])
        age_min = (datetime.now(tz=timezone.utc) - updated).total_seconds() / 60
        if age_min > 90:
            stale_banner = f'<div class="stale">&#9888; No update for {age_min:.0f} min — trader may be down</div>'
    except Exception:
        pass

    # History table (last 10)
    rows = ""
    for b in history[:10]:
        ts   = b.get("timestamp", "")[-8:-3] if b.get("timestamp") else "—"  # HH:MM
        dp   = f'{b.get("dir_prob", 0):.3f}'
        st   = f'{b.get("S_t", 0):.5f}'
        act  = b.get("action_taken", "—")
        ac   = _action_color(act)
        rows += f"<tr><td>{ts}</td><td>{dp}</td><td>{st}</td><td class='{ac}'>{act}</td></tr>"

    history_table = (
        "<table><thead><tr><th>Time</th><th>Dir</th><th>S_t</th><th>Action</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        if rows else "<div style='color:#555;font-size:.85rem'>No bars yet.</div>"
    )

    return _HTML.format(
        refresh      = REFRESH_SEC,
        symbol       = symbol,
        mode         = mode,
        stale_banner = stale_banner,
        pos_class    = pos,
        position     = pos,
        dir_prob     = f"{dir_prob:.3f}" if dir_prob == dir_prob else "—",
        dir_color    = _color_dir(dir_prob) if dir_prob == dir_prob else "",
        s_t          = f"{s_t:.5f}" if s_t == s_t else "—",
        st_color     = _color_st(s_t, history) if s_t == s_t else "",
        signal       = f"{signal:.3f}" if signal == signal else "—",
        action       = action,
        action_color = _action_color(str(action)),
        last_ts      = last_ts,
        last_ts_iso  = last_ts_iso,
        buf_len      = buf_len,
        history_table= history_table,
    )


@app.route("/")
def index():
    try:
        data = json.loads(STATUS_FILE.read_text())
    except FileNotFoundError:
        data = {"status": "waiting", "message": "Trader not yet started.", "history": []}
    except Exception as exc:
        data = {"status": "error",   "message": str(exc), "history": []}
    return _build_page(data)


@app.route("/api/status")
def api_status():
    try:
        return jsonify(json.loads(STATUS_FILE.read_text()))
    except FileNotFoundError:
        return jsonify({"error": "status file not found"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
