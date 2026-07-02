"""Flask dashboard for the AFL multis tracker (Stage 2D + Phase 3 CLV).

Launch: python -m afl_bot.cli dashboard
Opens:  http://127.0.0.1:8765

Panels:
  1. Round View — per-game cards with Model + Sportsbet ladders; filter/sort controls.
  2. Place a Bet — "I placed this" form: stake + taken_odds -> appends to ledger.
  3. Tracker / P&L — open bets, settled bets, season summary + Chart.js cumulative profit.
  4. CLV — rolling CLV stats, t-stat, min-detectable-edge, per-market breakdown.
  "Settle now" runs settle_bets for all completed pending rounds and refreshes.

Data:
  reads  reports/{year}_r{round}_multis.json  (written by round-report)
  reads/writes  reports/bets_ledger.json
"""

from __future__ import annotations

import glob
import json
import os
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template_string, request, url_for

from afl_bot.config import BANKROLL, KELLY_PER_ROUND_CAP, ROOT_DIR, UNIT_SIZE
from afl_bot.dashboard.ledger import (
    add_bet,
    add_manual_bet,
    cumulative_profit,
    load_ledger,
    manual_settle_bet,
    pnl_summary,
    save_ledger,
)
from afl_bot.dashboard.clv import clv_breakdown_by_market, clv_stats
from afl_bot.dashboard.settle import settle_bets

REPORTS_DIR = ROOT_DIR / "reports"
LEDGER_PATH = REPORTS_DIR / "bets_ledger.json"

app = Flask(__name__)
app.secret_key = "afl_dashboard_secret"


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_multis_files() -> dict[str, dict]:
    """Return {'{year}_r{round}': {"records": [...], "generated_at": "..."}} for all multis JSON files."""
    result: dict[str, dict] = {}
    for p in sorted(glob.glob(str(REPORTS_DIR / "*_multis.json"))):
        name = Path(p).stem.replace("_multis", "")   # e.g. "2026_r16"
        try:
            raw = json.loads(Path(p).read_text(encoding="utf-8"))
            if isinstance(raw, list):
                # Legacy format: plain list of records, no generated_at
                result[name] = {"records": raw, "generated_at": None}
            else:
                result[name] = {"records": raw.get("records", []), "generated_at": raw.get("generated_at")}
        except Exception:
            pass
    return result


def _latest_round_key(all_files: dict) -> str | None:
    keys = sorted(all_files.keys())
    return keys[-1] if keys else None


def _round_options(all_files: dict) -> list[str]:
    return sorted(all_files.keys(), reverse=True)


def _group_by_game(records: list[dict]) -> list[dict]:
    """Group multis records into game cards for the UI."""
    games: dict[str, dict] = {}
    for r in records:
        gid = r["game"]
        if gid not in games:
            games[gid] = {"game": gid, "model": [], "sportsbet": [], "pull_em": None,
                          "greasiness": r.get("greasiness", 0.0)}
        ladder = r.get("ladder", "model")
        if ladder == "pull_em":
            games[gid]["pull_em"] = r
        else:
            games[gid][ladder].append(r)
    for g in games.values():
        g["model"].sort(key=lambda r: r["band"])
        g["sportsbet"].sort(key=lambda r: r["band"])
    return list(games.values())


# ── routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    all_files = _load_multis_files()
    selected = request.args.get("round_key") or _latest_round_key(all_files)
    file_entry = all_files.get(selected, {"records": [], "generated_at": None}) if selected else {"records": [], "generated_at": None}
    records = file_entry["records"]
    generated_at = file_entry["generated_at"]
    # Stale check: compare generated_at timestamp against the file mtime on disk.
    stale_warning = None
    if selected and generated_at:
        multis_path = REPORTS_DIR / f"{selected}_multis.json"
        if multis_path.exists():
            import os
            file_mtime = os.path.getmtime(str(multis_path))
            from datetime import datetime, timezone as _tz
            try:
                gen_ts = datetime.fromisoformat(generated_at).timestamp()
                if file_mtime > gen_ts + 5:  # >5s newer → stale in memory
                    stale_warning = f"Data may be stale — file updated after last load. Reload to refresh."
            except Exception:
                pass
    games = _group_by_game(records)
    rounds = _round_options(all_files)
    bets = load_ledger(LEDGER_PATH)
    summary = pnl_summary(bets)
    chart_data = cumulative_profit(bets)
    clv_avail = [b for b in bets if b.get("clv_available")]
    clv_all = clv_stats([b["clv_pct"] for b in clv_avail])
    clv_by_mkt = clv_breakdown_by_market(bets)
    n_clv_pending = sum(1 for b in bets
                        if b["status"] == "pending" and not b.get("close_captured_at"))
    n_clv_unavail = sum(1 for b in bets if b.get("close_captured_at")
                        and not b.get("clv_available"))
    round_total_units = sum(r.get("units", 0.0) for r in records if r.get("units", 0.0) > 0)
    round_cap_units = KELLY_PER_ROUND_CAP * BANKROLL / UNIT_SIZE
    round_total_dollars = round_total_units * UNIT_SIZE
    return render_template_string(
        _TEMPLATE,
        games=games, rounds=rounds, selected=selected,
        bets=bets, summary=summary, chart_data=json.dumps(chart_data),
        all_multis={r["id"]: r for r in records},
        clv_all=clv_all, clv_by_mkt=clv_by_mkt,
        n_clv_pending=n_clv_pending, n_clv_unavail=n_clv_unavail,
        round_total_units=round_total_units,
        round_total_dollars=round_total_dollars,
        round_cap_units=round_cap_units,
        generated_at=generated_at,
        stale_warning=stale_warning,
    )


@app.route("/place", methods=["POST"])
def place_bet():
    multi_id = request.form.get("multi_id")
    stake = request.form.get("stake", type=float)
    taken_odds = request.form.get("taken_odds", type=float)
    round_key = request.form.get("round_key", "")

    all_files = _load_multis_files()
    records = all_files.get(round_key, {}).get("records", [])
    record = next((r for r in records if r["id"] == multi_id), None)
    if record is None or not stake or not taken_odds:
        return redirect(url_for("index", round_key=round_key))

    LEDGER_PATH.parent.mkdir(exist_ok=True)
    add_bet(LEDGER_PATH, record, stake, taken_odds)
    return redirect(url_for("index", round_key=round_key) + "#tracker")


@app.route("/settle", methods=["POST"])
def settle():
    round_key = request.form.get("round_key", "")
    LEDGER_PATH.parent.mkdir(exist_ok=True)
    settle_bets(LEDGER_PATH)
    return redirect(url_for("index", round_key=round_key) + "#tracker")


@app.route("/add-manual-bet", methods=["POST"])
def add_manual_bet_route():
    round_key = request.form.get("round_key", "")
    try:
        year = int(request.form.get("manual_year", 0))
        round_no = int(request.form.get("manual_round", 0))
        game = request.form.get("manual_game", "").strip()
        stake = float(request.form.get("manual_stake", 0))
        taken_odds = float(request.form.get("manual_odds", 0))
        label = request.form.get("manual_label", "").strip()
        book = request.form.get("manual_book", "other").strip() or "other"
    except (ValueError, TypeError):
        return redirect(url_for("index", round_key=round_key) + "#tracker")

    if not year or not round_no or not game or not stake or not taken_odds:
        return redirect(url_for("index", round_key=round_key) + "#tracker")

    # Build legs from the dynamic repeating form fields.
    # Each leg is submitted as leg_type_N, leg_player_N, leg_stat_N, leg_line_N, leg_text_N.
    legs = []
    for i in range(1, 8):   # support up to 7 legs
        leg_type = request.form.get(f"leg_type_{i}", "")
        if not leg_type:
            break
        if leg_type == "prop":
            player = request.form.get(f"leg_player_{i}", "").strip()
            stat = request.form.get(f"leg_stat_{i}", "disposals")
            try:
                line = int(request.form.get(f"leg_line_{i}", 0))
            except (ValueError, TypeError):
                line = 0
            if player and line:
                legs.append({
                    "player": player,
                    "market": f"player_{stat}",
                    "line": line,
                    "name": f"{player} {line}+ {stat}",
                    "book_odds": None,
                })
        elif leg_type == "h2h":
            team = request.form.get(f"leg_player_{i}", "").strip()
            if team:
                legs.append({
                    "player": team,
                    "market": "h2h",
                    "line": None,
                    "name": f"{team} to win",
                    "book_odds": None,
                })
        elif leg_type == "total":
            try:
                line = float(request.form.get(f"leg_line_{i}", 0))
            except (ValueError, TypeError):
                line = 0
            if line:
                legs.append({
                    "player": "total",
                    "market": "total_points",
                    "line": line,
                    "name": f"Total points {line}+",
                    "book_odds": None,
                })
        elif leg_type == "other":
            text = request.form.get(f"leg_text_{i}", "").strip()
            if text:
                legs.append({
                    "player": "",
                    "market": "other",
                    "line": None,
                    "name": text,
                    "book_odds": None,
                })
        # attach per-leg odds if supplied
        if legs:
            try:
                leg_odds_val = float(request.form.get(f"leg_odds_{i}", "") or "")
                if leg_odds_val >= 1.01:
                    legs[-1]["book_odds"] = leg_odds_val
            except (ValueError, TypeError):
                pass

    if not legs:
        return redirect(url_for("index", round_key=round_key) + "#tracker")

    LEDGER_PATH.parent.mkdir(exist_ok=True)
    add_manual_bet(LEDGER_PATH, year=year, round_no=round_no, game=game,
                   stake=stake, taken_odds=taken_odds, legs=legs,
                   label=label or None, book=book)
    return redirect(url_for("index", round_key=round_key) + "#tracker")


@app.route("/manual-settle", methods=["POST"])
def manual_settle():
    bet_id = request.form.get("bet_id", "")
    outcome = request.form.get("outcome", "")
    round_key = request.form.get("round_key", "")
    if bet_id and outcome in ("won", "lost", "void"):
        LEDGER_PATH.parent.mkdir(exist_ok=True)
        manual_settle_bet(LEDGER_PATH, bet_id, outcome=outcome)
    return redirect(url_for("index", round_key=round_key) + "#tracker")


@app.route("/api/multis")
def api_multis():
    all_files = _load_multis_files()
    return jsonify(all_files)


@app.route("/api/ledger")
def api_ledger():
    return jsonify(load_ledger(LEDGER_PATH))


def run_dashboard(port: int = 8765, open_browser: bool = True) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    print(f"AFL Multis Dashboard -> http://127.0.0.1:{port}")
    if open_browser:
        webbrowser.open(f"http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)


# ── HTML template ─────────────────────────────────────────────────────────────

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AFL Multis Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  :root{--bg:#0d1117;--card:#161b22;--border:#30363d;--accent:#58a6ff;--green:#3fb950;
    --red:#f85149;--yellow:#d29922;--text:#e6edf3;--muted:#8b949e;--radius:6px;}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;
    font-size:14px;line-height:1.5}
  h1{font-size:1.4rem;color:var(--accent);padding:16px 20px;border-bottom:1px solid var(--border)}
  h2{font-size:1.1rem;margin-bottom:12px;color:var(--accent)}
  h3{font-size:.95rem;margin:10px 0 6px;color:var(--muted)}
  .tabs{display:flex;border-bottom:1px solid var(--border);padding:0 20px}
  .tab{padding:10px 18px;cursor:pointer;border-bottom:2px solid transparent;color:var(--muted)}
  .tab.active{color:var(--accent);border-color:var(--accent)}
  .panel{display:none;padding:20px}.panel.active{display:block}
  .card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
    padding:14px;margin-bottom:14px}
  .game-title{font-size:1rem;font-weight:600;margin-bottom:10px}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th{text-align:left;padding:6px 8px;border-bottom:1px solid var(--border);
    color:var(--muted);font-weight:600;white-space:nowrap}
  td{padding:5px 8px;border-bottom:1px solid var(--border);vertical-align:top}
  tr:last-child td{border:none}
  .value{color:var(--green);font-weight:600}
  .neg{color:var(--red)}
  .badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;font-weight:600}
  .badge-won{background:#1a3a2a;color:var(--green)}
  .badge-lost{background:#3a1a1a;color:var(--red)}
  .badge-pending{background:#2a2a1a;color:var(--yellow)}
  .badge-void{background:#1a1a2a;color:var(--muted)}
  .btn{display:inline-block;padding:6px 14px;border:none;border-radius:var(--radius);
    cursor:pointer;font-size:13px;font-weight:600}
  .btn-primary{background:var(--accent);color:#0d1117}
  .btn-sm{padding:3px 9px;font-size:12px}
  .btn-settle{background:#1a3a2a;color:var(--green);border:1px solid var(--green)}
  select,input[type=text],input[type=number]{background:#0d1117;color:var(--text);
    border:1px solid var(--border);border-radius:var(--radius);padding:5px 9px;font-size:13px}
  .controls{display:flex;gap:10px;align-items:center;margin-bottom:16px;flex-wrap:wrap}
  .summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;
    margin-bottom:18px}
  .stat-box{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
    padding:12px;text-align:center}
  .stat-box .val{font-size:1.3rem;font-weight:700;color:var(--accent)}
  .stat-box .lbl{font-size:11px;color:var(--muted);margin-top:3px}
  .chart-wrap{max-width:720px;margin:10px 0 20px}
  .place-form{display:none;margin-top:8px;padding:10px;background:#0d1117;
    border:1px solid var(--border);border-radius:var(--radius)}
  .place-form.open{display:block}
  .form-row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:6px}
  .form-row label{color:var(--muted);font-size:12px}
  .section-label{font-size:11px;font-weight:600;text-transform:uppercase;
    letter-spacing:.05em;color:var(--muted);margin:10px 0 4px}
  .filter-row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
  #no-data{color:var(--muted);padding:30px 0;text-align:center}
  .badge-manual{background:#1a2a3a;color:#79c0ff}
  .manual-form{display:none;padding:14px;background:#0d1117;border:1px solid var(--border);
    border-radius:var(--radius);margin-bottom:14px}
  .manual-form.open{display:block}
  .leg-row{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-bottom:6px;
    padding:8px;background:var(--card);border:1px solid var(--border);border-radius:4px}
  .leg-row select,.leg-row input{font-size:12px;padding:4px 7px}
  .settle-controls{display:flex;gap:6px;margin-top:8px;flex-wrap:wrap}
</style>
</head>
<body>
<h1>AFL Multis Dashboard</h1>

<div class="tabs">
  <div class="tab active" onclick="showTab('rounds')">Round View</div>
  <div class="tab" onclick="showTab('tracker')" id="tab-tracker">Tracker / P&L</div>
  <div class="tab" onclick="showTab('clv')" id="tab-clv">CLV</div>
</div>

<!-- ── ROUND VIEW ─────────────────────────────── -->
<div class="panel active" id="panel-rounds">
  <div class="controls">
    <form method="get" action="/" style="display:contents">
      <label style="color:var(--muted)">Round:</label>
      <select name="round_key" onchange="this.form.submit()">
        {% for rk in rounds %}
        <option value="{{ rk }}" {% if rk==selected %}selected{% endif %}>
          {{ rk.replace('_r', ' Round ').replace('_', ' ') }}
        </option>
        {% endfor %}
        {% if not rounds %}<option value="">— no reports yet —</option>{% endif %}
      </select>
    </form>
    <label style="color:var(--muted)">Filter:</label>
    <select id="ladder-filter" onchange="applyFilter()">
      <option value="all">All ladders</option>
      <option value="model">Model only</option>
      <option value="sportsbet">Sportsbet only</option>
    </select>
    <label><input type="checkbox" id="value-only" onchange="applyFilter()"> Value picks only</label>
  </div>

  {% if stale_warning %}
  <div style="margin:8px 0 10px 0;padding:8px 12px;background:rgba(210,153,34,0.15);border:1px solid var(--yellow);border-radius:4px;font-size:12px;color:var(--yellow);display:flex;align-items:center;gap:10px">
    ⚠ {{ stale_warning }}
    <a href="/?round_key={{ selected }}" style="color:var(--yellow);text-decoration:underline">Reload data</a>
  </div>
  {% endif %}
  {% if generated_at %}
  <div style="font-size:11px;color:var(--muted);margin-bottom:6px">Generated: {{ generated_at[:19].replace('T',' ') }} UTC · <a href="/?round_key={{ selected }}" style="color:var(--muted);text-decoration:underline">↻ Reload</a></div>
  {% endif %}

  {% if not games %}
  <div id="no-data">No multis data for this round. Run <code>round-report --sportsbet</code> first.</div>
  {% else %}
  <div style="margin:8px 0 12px 0;font-size:12px;color:var(--muted)">
    Round total: <strong style="color:{% if round_total_units > round_cap_units %}var(--red){% else %}var(--green){% endif %}">{{ '%.2f'|format(round_total_units) }}u / ${{ '%.2f'|format(round_total_dollars) }}</strong>
    of {{ '%.0f'|format(round_cap_units) }}u cap
  </div>
  {% endif %}

  {% for g in games %}
  <div class="card" data-game="{{ g.game }}">
    <div class="game-title">
      {{ g.game }}
      {% set gv = g.greasiness if g.greasiness is not none else 0.0 %}
      {% if gv >= 0.5 %}
        <span style="margin-left:10px;color:var(--red);font-size:12px;font-weight:400">WET g={{ '%.2f'|format(gv) }}</span>
      {% elif gv > 0.1 %}
        <span style="margin-left:10px;color:var(--yellow);font-size:12px;font-weight:400">greasy g={{ '%.2f'|format(gv) }}</span>
      {% else %}
        <span style="margin-left:10px;color:var(--muted);font-size:12px;font-weight:400">dry g={{ '%.2f'|format(gv) }}</span>
      {% endif %}
    </div>

    {% if g.model %}
    <div class="section-label">Model ladder (fair odds)</div>
    <p style="font-size:11px;color:var(--muted);margin:4px 0 8px 0">
      Edge = raw model vs book price. <strong>Total EV</strong> includes the stake-back
      refund — that's the number to bet on. Stake = suggested % of bankroll (capped Kelly).
    </p>
    <table>
      <tr><th>Legs</th><th>Band</th><th>Joint%</th><th>Fair</th>
        <th>Book combo</th><th>Edge</th><th>Total EV</th><th>Stake</th><th>Units</th><th>$</th><th>Pick</th><th></th></tr>
      {% for r in g.model %}
      <tr class="rung-row" data-ladder="model" data-value="{{ 'true' if r.value_pick else 'false' }}">
        <td>{% for leg in r.legs %}{{ leg.name }}{% if leg.hit_prob %} <span style="color:var(--muted);font-size:11px">({{ '%.0f'|format(leg.hit_prob * 100) }}%)</span>{% else %} <span style="color:var(--muted);font-size:11px">(—)</span>{% endif %}{% if not loop.last %} + {% endif %}{% endfor %}</td>
        <td>${{ '%.2f'|format(r.band) }}</td>
        <td>{{ '%.0f'|format(r.model_joint * 100) }}%</td>
        <td>${{ '%.2f'|format(r.model_fair) }}</td>
        <td>{% if r.book_combo %}${{ '%.2f'|format(r.book_combo) }}{% else %}—{% endif %}</td>
        <td>
          {% if r.edge is not none %}
          <span class="{{ 'value' if r.edge > 0 else 'neg' }}">{{ '%+.1f'|format(r.edge*100) }}%</span>
          {% else %}—{% endif %}
        </td>
        <td>
          {% set tev = r.get('total_ev') %}
          {% if tev is not none %}
          {% set p1l = r.get('p_one_loss') %}{% set pev = r.get('promo_ev') %}
          <span class="{{ 'value' if tev > 0 else 'neg' }}"
            {% if p1l is not none and pev is not none %}title="P(one loss)={{ '%.0f'|format(p1l*100) }}% · Promo EV={{ '%+.1f'|format(pev*100) }}%"{% endif %}>{{ '%+.1f'|format(tev*100) }}%</span>
          {% else %}—{% endif %}
        </td>
        <td>
          {% set stk = r.get('suggested_stake') %}
          {% if stk %}
          <span class="value">{{ '%.1f'|format(stk*100) }}%</span>
          {% else %}—{% endif %}
        </td>
        <td>{% set utag = r.get('units_tag') %}{% if utag and utag != 'MODEL-ONLY' and utag != 'NO BET' %}<span class="value">{{ utag }}</span>{% else %}<span style="color:var(--muted)">{{ utag or '—' }}</span>{% endif %}</td>
        <td>{% set uval = r.get('units', 0) %}{% if uval > 0 %}<span class="value">${{ '%.2f'|format(uval * 15) }}</span>{% else %}—{% endif %}</td>
        <td>{% if r.value_pick %}<span class="value">★ VALUE</span>{% else %}—{% endif %}</td>
        <td>
          <button class="btn btn-sm btn-primary" onclick="openPlace('{{ r.id }}','{{ r.game }}',{{ r.book_combo or r.model_fair }},{{ r.get('units', 0) * 15 }})">Place</button>
        </td>
      </tr>
      {% endfor %}
    </table>
    {% endif %}

    {% if g.sportsbet %}
    <div class="section-label" style="margin-top:12px">Sportsbet ladder (real prices)</div>
    <p style="font-size:11px;color:var(--muted);margin:4px 0 8px 0">
      Edge = raw model vs book price. <strong>Total EV</strong> includes the stake-back
      refund — that's the number to bet on. Stake = suggested % of bankroll (capped Kelly).
    </p>
    <table>
      <tr><th>Legs</th><th>Band</th><th>Book combo</th><th>Model joint%</th><th>Model fair</th>
        <th>Edge</th><th>Total EV</th><th>Stake</th><th>Units</th><th>$</th><th>Pick</th><th></th></tr>
      {% for r in g.sportsbet %}
      <tr class="rung-row" data-ladder="sportsbet" data-value="{{ 'true' if r.value_pick else 'false' }}">
        <td>{% for leg in r.legs %}{{ leg.name }}{% if leg.hit_prob %} <span style="color:var(--muted);font-size:11px">({{ '%.0f'|format(leg.hit_prob * 100) }}%)</span>{% else %} <span style="color:var(--muted);font-size:11px">(—)</span>{% endif %}{% if not loop.last %} + {% endif %}{% endfor %}</td>
        <td>${{ '%.2f'|format(r.band) }}</td>
        <td>{% if r.book_combo %}${{ '%.2f'|format(r.book_combo) }}{% else %}—{% endif %}</td>
        <td>{{ '%.0f'|format(r.model_joint * 100) }}%</td>
        <td>${{ '%.2f'|format(r.model_fair) }}</td>
        <td>
          {% if r.edge is not none %}
          <span class="{{ 'value' if r.edge > 0 else 'neg' }}">{{ '%+.1f'|format(r.edge*100) }}%</span>
          {% else %}—{% endif %}
        </td>
        <td>
          {% set tev = r.get('total_ev') %}
          {% if tev is not none %}
          {% set p1l = r.get('p_one_loss') %}{% set pev = r.get('promo_ev') %}
          <span class="{{ 'value' if tev > 0 else 'neg' }}"
            {% if p1l is not none and pev is not none %}title="P(one loss)={{ '%.0f'|format(p1l*100) }}% · Promo EV={{ '%+.1f'|format(pev*100) }}%"{% endif %}>{{ '%+.1f'|format(tev*100) }}%</span>
          {% else %}—{% endif %}
        </td>
        <td>
          {% set stk = r.get('suggested_stake') %}
          {% if stk %}
          <span class="value">{{ '%.1f'|format(stk*100) }}%</span>
          {% else %}—{% endif %}
        </td>
        <td>{% set utag = r.get('units_tag') %}{% if utag and utag != 'MODEL-ONLY' and utag != 'NO BET' %}<span class="value">{{ utag }}</span>{% else %}<span style="color:var(--muted)">{{ utag or '—' }}</span>{% endif %}</td>
        <td>{% set uval = r.get('units', 0) %}{% if uval > 0 %}<span class="value">${{ '%.2f'|format(uval * 15) }}</span>{% else %}—{% endif %}</td>
        <td>{% if r.value_pick %}<span class="value">★ VALUE</span>{% else %}—{% endif %}</td>
        <td>
          <button class="btn btn-sm btn-primary" onclick="openPlace('{{ r.id }}','{{ r.game }}',{{ r.book_combo or r.model_fair }},{{ r.get('units', 0) * 15 }})">Place</button>
        </td>
      </tr>
      {% endfor %}
    </table>
    {% endif %}

    {% if g.pull_em %}
    {% set pe = g.pull_em %}
    <div class="section-label" style="margin-top:12px">PointsBet Pull 'Em</div>
    <p style="font-size:11px;color:var(--muted);margin:4px 0 8px 0">
      Option EV uses <strong>PULL_DETECTION_PROB=0.70 (assumed prior — not fitted)</strong>.
      Book combo: ${{ '%.2f'|format(pe.book_combo) }} ·
      Option EV: <strong class="{{ 'value' if pe.option_ev > 0 else 'neg' }}">{{ '%+.2f'|format(pe.option_ev) }}%</strong>
    </p>
    <table>
      <tr><th>Leg</th><th>Role</th><th>Prob</th><th>Leg odds</th></tr>
      {% set anchor_set = pe.anchor_names %}
      {% for i in range(pe.leg_names|length) %}
      <tr>
        <td>{{ pe.leg_names[i] }}</td>
        <td>{% if pe.leg_names[i] in anchor_set %}Anchor{% else %}Booster{% endif %}</td>
        <td>{{ '%.0f'|format((pe.anchor_probs + [pe.booster_prob])[i] * 100) }}%</td>
        <td>{{ '%.2f'|format(pe.book_odds_per_leg[i]) }}</td>
      </tr>
      {% endfor %}
    </table>
    <p style="font-size:11px;color:var(--muted);margin:6px 0 4px 0"><em>{{ pe.pull_decision_rule }}</em></p>
    {% endif %}

    <!-- inline place-bet form -->
    <div class="place-form" id="place-{{ loop.index }}">
      <form method="post" action="/place" id="pform-{{ loop.index }}">
        <input type="hidden" name="multi_id" id="pmid-{{ loop.index }}">
        <input type="hidden" name="round_key" value="{{ selected }}">
        <b id="pleg-{{ loop.index }}" style="font-size:12px;color:var(--muted)"></b>
        <div class="form-row">
          <label>Stake ($)</label>
          <input type="number" name="stake" id="pstake-{{ loop.index }}" min="1" step="0.50" value="25" style="width:90px">
          <label>Odds taken</label>
          <input type="number" name="taken_odds" id="podds-{{ loop.index }}" min="1.01" step="0.01" style="width:90px">
          <button type="submit" class="btn btn-primary btn-sm">Confirm</button>
          <button type="button" class="btn btn-sm" style="background:var(--border)" onclick="closePlace({{ loop.index }})">Cancel</button>
        </div>
      </form>
    </div>
  </div>
  {% endfor %}
</div>

<!-- ── TRACKER ────────────────────────────────── -->
<div class="panel" id="panel-tracker">
  <div class="controls">
    <form method="post" action="/settle">
      <input type="hidden" name="round_key" value="{{ selected }}">
      <button class="btn btn-settle">⟳ Settle now</button>
    </form>
    <button class="btn btn-primary btn-sm" onclick="toggleManualForm()">+ Add my own bet</button>
    <span style="color:var(--muted);font-size:12px;margin-left:10px">Filter:</span>
    <select id="source-filter" onchange="applySourceFilter()">
      <option value="all">All bets</option>
      <option value="bot">Bot only</option>
      <option value="manual">Manual only</option>
    </select>
  </div>

  <!-- ── Add my own bet form ── -->
  <div class="manual-form" id="manual-form">
    <b style="font-size:13px;color:var(--accent)">Add my own bet</b>
    <form method="post" action="/add-manual-bet" id="manual-bet-form">
      <input type="hidden" name="round_key" value="{{ selected }}">
      <div class="form-row" style="margin-top:10px">
        <label>Year</label>
        <input type="number" name="manual_year" value="{{ selected[:4] if selected else '2026' }}" style="width:72px" required>
        <label>Round</label>
        <input type="number" name="manual_round" value="{{ selected.split('_r')[1] if selected and '_r' in selected else '' }}" style="width:60px" required>
        <label>Game</label>
        <input type="text" name="manual_game" placeholder="Home vs Away" style="width:200px" required>
        <label>Stake ($)</label>
        <input type="number" name="manual_stake" min="0.01" step="0.50" value="25" style="width:80px" required>
        <label>Odds</label>
        <input type="number" name="manual_odds" id="manual_odds" min="1.01" step="0.01" style="width:80px" required>
        <label>Bookie</label>
        <select name="manual_book" style="width:120px">
          <option value="sportsbet">Sportsbet</option>
          <option value="pointsbet">PointsBet</option>
          <option value="tab">TAB</option>
          <option value="other" selected>Other</option>
        </select>
        <label>Label</label>
        <input type="text" name="manual_label" placeholder="optional" style="width:120px">
      </div>
      <div style="margin-top:10px;margin-bottom:6px;color:var(--muted);font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.05em">Legs</div>
      <div id="leg-rows"></div>
      <div class="form-row" style="margin-top:8px">
        <button type="button" class="btn btn-sm" style="background:var(--border)" onclick="addLeg()">+ Add leg</button>
        <button type="submit" class="btn btn-primary btn-sm">Save bet</button>
        <button type="button" class="btn btn-sm" style="background:var(--border)" onclick="toggleManualForm()">Cancel</button>
      </div>
    </form>
  </div>

  <h2>Season Summary</h2>
  <div class="summary-grid">
    <div class="stat-box"><div class="val">${{ '%.2f'|format(summary.total_staked) }}</div><div class="lbl">Total staked</div></div>
    <div class="stat-box"><div class="val">${{ '%.2f'|format(summary.total_returned) }}</div><div class="lbl">Total returned</div></div>
    <div class="stat-box">
      <div class="val {{ 'value' if summary.net_profit >= 0 else 'neg' }}">${{ '%+.2f'|format(summary.net_profit) }}</div>
      <div class="lbl">Net profit</div>
    </div>
    <div class="stat-box">
      <div class="val {{ 'value' if summary.roi_pct >= 0 else 'neg' }}">{{ '%+.1f'|format(summary.roi_pct) }}%</div>
      <div class="lbl">ROI</div>
    </div>
    <div class="stat-box"><div class="val">{{ '%.0f'|format(summary.strike_rate*100) }}%</div><div class="lbl">Strike rate ({{ summary.n_won }}/{{ summary.n_settled }})</div></div>
  </div>

  <div class="chart-wrap"><canvas id="profitChart"></canvas></div>

  {% set open_bets = bets | selectattr('status','eq','pending') | list %}
  {% if open_bets %}
  <h2>Open bets ({{ open_bets|length }})</h2>
  {% for b in open_bets %}
  {% set is_manual = b.get('source') == 'manual' %}
  {% set has_other = b.legs | selectattr('market','eq','other') | list | length > 0 %}
  {% set ungradeable = b.get('ungradeable_legs', []) %}
  {% set needs_manual_settle = is_manual or has_other or ungradeable | length > 0 %}
  <div class="card" data-source="{{ b.get('source','bot') }}">
    <div style="display:flex;justify-content:space-between;align-items:start">
      <div>
        <b>{{ b.game }}</b>
        {% if is_manual %} — <span style="color:#79c0ff">manual</span>{% else %} — {{ b.ladder }} ladder{% endif %}
        {% if b.get('label') %}<span style="color:var(--muted);font-size:11px;margin-left:6px">[{{ b.label }}]</span>{% endif %}
        <br>
        <span style="color:var(--muted);font-size:12px">{{ b.legs|map(attribute='name')|join(' + ') }}</span>
      </div>
      <div style="display:flex;gap:6px;align-items:center">
        {% if is_manual %}<span class="badge badge-manual">manual</span>{% endif %}
        <span class="badge badge-pending">pending</span>
      </div>
    </div>
    <div style="margin-top:6px;color:var(--muted);font-size:12px">
      Stake ${{ '%.2f'|format(b.stake) }} @ {{ '%.2f'|format(b.taken_odds) }}
      · placed {{ b.placed_at[:16] }}
    </div>
    {% if ungradeable %}
    <div style="margin-top:5px;font-size:11px;color:var(--yellow)">
      Ungradeable: {{ ungradeable | join(', ') }}
    </div>
    {% endif %}
    {% if b.leg_results %}
    <div style="margin-top:5px;font-size:11px;color:var(--muted)">
      {% for lr in b.leg_results %}
      <span style="margin-right:8px">
        {% if lr.hit is none %}⬜{% elif lr.hit %}✅{% else %}❌{% endif %}
        {{ lr.name }}
      </span>
      {% endfor %}
    </div>
    {% endif %}
    {% if needs_manual_settle %}
    <div class="settle-controls">
      <span style="font-size:11px;color:var(--muted)">Manual settle:</span>
      <form method="post" action="/manual-settle" style="display:contents">
        <input type="hidden" name="bet_id" value="{{ b.bet_id }}">
        <input type="hidden" name="round_key" value="{{ selected }}">
        <button name="outcome" value="won" class="btn btn-sm" style="background:#1a3a2a;color:var(--green);border:1px solid var(--green)">Won</button>
        <button name="outcome" value="lost" class="btn btn-sm" style="background:#3a1a1a;color:var(--red);border:1px solid var(--red)">Lost</button>
        <button name="outcome" value="void" class="btn btn-sm" style="background:var(--border);color:var(--muted)">Void</button>
      </form>
    </div>
    {% endif %}
  </div>
  {% endfor %}
  {% endif %}

  {% set settled_bets = bets | rejectattr('status','eq','pending') | list %}
  {% if settled_bets %}
  <h2>Settled bets</h2>
  {% for b in settled_bets | sort(attribute='settled_at', reverse=True) %}
  {% set is_manual = b.get('source') == 'manual' %}
  <div class="card" data-source="{{ b.get('source','bot') }}">
    <div style="display:flex;justify-content:space-between;align-items:start">
      <div>
        <b>{{ b.game }}</b>
        {% if is_manual %} — <span style="color:#79c0ff">manual</span>{% else %} — {{ b.ladder }} ladder{% endif %}
        {% if b.get('label') %}<span style="color:var(--muted);font-size:11px;margin-left:6px">[{{ b.label }}]</span>{% endif %}
        <br>
        <span style="color:var(--muted);font-size:12px">{{ b.legs|map(attribute='name')|join(' + ') }}</span>
      </div>
      <div style="display:flex;gap:6px;align-items:center">
        {% if is_manual %}<span class="badge badge-manual">manual</span>{% endif %}
        <span class="badge badge-{{ b.status }}">{{ b.status }}</span>
      </div>
    </div>
    <div style="margin-top:6px;font-size:12px;color:var(--muted)">
      Stake ${{ '%.2f'|format(b.stake) }} @ {{ '%.2f'|format(b.taken_odds) }}
      · payout
      <span class="{{ 'value' if b.status=='won' else ('neg' if b.status=='lost' else '') }}">
        ${{ '%.2f'|format(b.payout or 0) }}
      </span>
      · profit
      <span class="{{ 'value' if (b.payout or 0)-b.stake >= 0 else 'neg' }}">
        ${{ '%+.2f'|format((b.payout or 0) - b.stake) }}
      </span>
    </div>
    {% if b.leg_results %}
    <div style="margin-top:5px;font-size:11px;color:var(--muted)">
      {% for lr in b.leg_results %}
      <span style="margin-right:8px">
        {% if lr.hit is none %}⬜{% elif lr.hit %}✅{% else %}❌{% endif %}
        {{ lr.name }}
      </span>
      {% endfor %}
    </div>
    {% endif %}
  </div>
  {% endfor %}
  {% endif %}
  {% if not open_bets and not settled_bets %}
  <div style="color:var(--muted);padding:30px 0;text-align:center">No bets recorded yet. Use the Place button in Round View or Add my own bet.</div>
  {% endif %}
</div>

<!-- ── CLV ──────────────────────────────────────── -->
<div class="panel" id="panel-clv">
  <h2>Closing Line Value</h2>
  <p style="color:var(--muted);font-size:12px;margin-bottom:16px">
    CLV = (1/close_ref_odds) - (1/open_odds). Positive = you beat the closing line.<br>
    Sharp reference: de-vigged consensus across Sportsbet + TAB (props &amp; H2H).
    Betfair exchange is the future upgrade for sharper H2H reference.<br>
    Run <code>capture-close</code> near bounce to record closing prices.
  </p>

  {% if n_clv_pending > 0 %}
  <div class="card" style="border-color:var(--yellow)">
    <span style="color:var(--yellow)">{{ n_clv_pending }} pending bet(s) not yet captured — run <code>capture-close</code> before bounce.</span>
  </div>
  {% endif %}

  {% if n_clv_unavail > 0 %}
  <div class="card" style="border-color:var(--muted)">
    <span style="color:var(--muted)">{{ n_clv_unavail }} bet(s) captured but CLV unavailable (no sharp reference).</span>
  </div>
  {% endif %}

  {% if clv_all.n > 0 %}
  <div class="summary-grid">
    <div class="stat-box">
      <div class="val {{ 'value' if (clv_all.mean_clv or 0) > 0 else 'neg' }}">
        {% if clv_all.mean_clv is not none %}{{ '%+.2f'|format(clv_all.mean_clv*100) }}pp{% else %}—{% endif %}
      </div>
      <div class="lbl">Mean CLV (n={{ clv_all.n }})</div>
    </div>
    <div class="stat-box">
      <div class="val">
        {% if clv_all.pct_positive is not none %}{{ '%.0f'|format(clv_all.pct_positive*100) }}%{% else %}—{% endif %}
      </div>
      <div class="lbl">% positive CLV</div>
    </div>
    <div class="stat-box">
      <div class="val {{ 'value' if clv_all.significant else '' }}">
        {% if clv_all.t_stat is not none %}{{ '%.2f'|format(clv_all.t_stat) }}{% else %}—{% endif %}
        {% if clv_all.significant %}<span style="font-size:11px"> *</span>{% endif %}
      </div>
      <div class="lbl">t-stat (one-sided 5%)</div>
    </div>
    <div class="stat-box">
      <div class="val" style="color:var(--yellow)">
        {% if clv_all.min_detectable_edge is not none %}{{ '%+.2f'|format(clv_all.min_detectable_edge*100) }}pp{% else %}—{% endif %}
      </div>
      <div class="lbl">Min detectable edge (80% power)</div>
    </div>
  </div>
  {% if not clv_all.significant %}
  <p style="color:var(--muted);font-size:12px;margin-bottom:12px">
    Too soon to tell — need ~{{ ((2.487 * (clv_all.sd_clv or 0.05) / (clv_all.mean_clv or 0.001)) ** 2) | int }} bets at current mean CLV for significance.
  </p>
  {% endif %}

  {% if clv_by_mkt %}
  <h3>By market type</h3>
  <table>
    <tr><th>Market</th><th>n</th><th>Mean CLV</th><th>t-stat</th><th>Sig?</th><th>MDE</th></tr>
    {% for mkt, s in clv_by_mkt.items() %}
    <tr>
      <td>{{ mkt }}</td>
      <td>{{ s.n }}</td>
      <td>{% if s.mean_clv is not none %}<span class="{{ 'value' if s.mean_clv > 0 else 'neg' }}">{{ '%+.2f'|format(s.mean_clv*100) }}pp</span>{% else %}—{% endif %}</td>
      <td>{% if s.t_stat is not none %}{{ '%.2f'|format(s.t_stat) }}{% else %}—{% endif %}</td>
      <td>{% if s.significant %}<span class="value">yes *</span>{% else %}<span style="color:var(--muted)">no</span>{% endif %}</td>
      <td>{% if s.min_detectable_edge is not none %}{{ '%+.2f'|format(s.min_detectable_edge*100) }}pp{% else %}—{% endif %}</td>
    </tr>
    {% endfor %}
  </table>
  {% endif %}

  {% else %}
  <div style="color:var(--muted);padding:30px 0;text-align:center">
    No CLV data yet — run <code>capture-close</code> near bounce.<br>
    Prop &amp; H2H CLV: active via Sportsbet + TAB consensus.<br>
    H2H future upgrade: Betfair exchange reference.
  </div>
  {% endif %}
</div>

<script>
function showTab(name){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  const idx=['rounds','tracker','clv'].indexOf(name);
  document.querySelectorAll('.tab')[idx].classList.add('active');
  document.getElementById('panel-'+name).classList.add('active');
}
// Check URL hash on load
if(location.hash==='#tracker') showTab('tracker');
if(location.hash==='#clv') showTab('clv');

function openPlace(multiId, game, defaultOdds, defaultStake){
  document.querySelectorAll('.place-form').forEach(f=>{f.classList.remove('open')});
  const btns = document.querySelectorAll('[onclick*="'+multiId+'"]');
  if(!btns.length) return;
  const card = btns[0].closest('.card');
  const pform = card.querySelector('.place-form');
  if(!pform) return;
  const idx = pform.id.split('-')[1];
  document.getElementById('pmid-'+idx).value = multiId;
  document.getElementById('pleg-'+idx).textContent = game + ' / ' + multiId.split('-').slice(4).join(' ');
  const oddsInput = document.getElementById('podds-'+idx);
  oddsInput.value = parseFloat(defaultOdds).toFixed(2);
  if(defaultStake && parseFloat(defaultStake) > 0){
    const stakeInput = document.getElementById('pstake-'+idx);
    if(stakeInput) stakeInput.value = parseFloat(defaultStake).toFixed(2);
  }
  pform.classList.add('open');
  pform.scrollIntoView({behavior:'smooth',block:'nearest'});
}
function closePlace(idx){
  document.getElementById('place-'+idx).classList.remove('open');
}

function applyFilter(){
  const ladder=document.getElementById('ladder-filter').value;
  const valueOnly=document.getElementById('value-only').checked;
  document.querySelectorAll('.rung-row').forEach(row=>{
    const ld=row.dataset.ladder;
    const vp=row.dataset.value==='true';
    const showLadder=(ladder==='all'||ld===ladder);
    const showValue=(!valueOnly||vp);
    row.style.display=(showLadder&&showValue)?'':'none';
  });
}

// ── Manual bet form ──────────────────────────────────────────────────────────
function toggleManualForm(){
  const f=document.getElementById('manual-form');
  f.classList.toggle('open');
  if(f.classList.contains('open')&&document.getElementById('leg-rows').children.length===0){
    addLeg(); // start with one leg row
  }
}

let _legCount=0;
function addLeg(){
  _legCount++;
  const n=_legCount;
  const wrap=document.createElement('div');
  wrap.className='leg-row';
  wrap.id='leg-row-'+n;
  wrap.innerHTML=`
    <input type="hidden" name="leg_type_${n}" id="lt_${n}" value="prop">
    <select onchange="changeLegType(${n},this.value)" style="width:120px">
      <option value="prop">Player prop</option>
      <option value="h2h">Team to win</option>
      <option value="total">Total points</option>
      <option value="other">Other</option>
    </select>
    <span id="lf_${n}">
      <input type="text" name="leg_player_${n}" placeholder="Player name" style="width:140px">
      <select name="leg_stat_${n}" style="width:100px">
        <option value="disposals">Disposals</option>
        <option value="goals">Goals</option>
        <option value="marks">Marks</option>
        <option value="tackles">Tackles</option>
      </select>
      <input type="number" name="leg_line_${n}" placeholder="Line" min="1" style="width:60px">
    </span>
    <span id="lf_other_${n}" style="display:none">
      <input type="text" name="leg_text_${n}" placeholder="Free text description" style="width:220px">
    </span>
    <input type="number" name="leg_odds_${n}" placeholder="Leg odds" min="1.01" step="0.01" style="width:80px" oninput="recalcComboOdds()" title="Per-leg price (optional, auto-multiplies into combined odds)">
    <button type="button" class="btn btn-sm" style="background:var(--border);color:var(--red)" onclick="removeLeg(${n})">✕</button>
  `;
  document.getElementById('leg-rows').appendChild(wrap);
}

function recalcComboOdds(){
  const inputs=[...document.querySelectorAll('#leg-rows input[name^="leg_odds_"]')];
  let product=1.0;
  let anyFilled=false;
  inputs.forEach(el=>{
    const v=parseFloat(el.value);
    if(v>=1.01){product*=v;anyFilled=true;}
  });
  const oddsField=document.getElementById('manual_odds');
  if(oddsField&&anyFilled) oddsField.value=product.toFixed(2);
}

function changeLegType(n,val){
  document.getElementById('lt_'+n).value=val;
  const stdFields=document.getElementById('lf_'+n);
  const otherFields=document.getElementById('lf_other_'+n);
  if(val==='prop'){
    stdFields.style.display='';
    stdFields.innerHTML=`
      <input type="text" name="leg_player_${n}" placeholder="Player name" style="width:140px">
      <select name="leg_stat_${n}" style="width:100px">
        <option value="disposals">Disposals</option>
        <option value="goals">Goals</option>
        <option value="marks">Marks</option>
        <option value="tackles">Tackles</option>
      </select>
      <input type="number" name="leg_line_${n}" placeholder="Line" min="1" style="width:60px">
    `;
    otherFields.style.display='none';
  } else if(val==='h2h'){
    stdFields.style.display='';
    stdFields.innerHTML=`<input type="text" name="leg_player_${n}" placeholder="Team name" style="width:180px">`;
    otherFields.style.display='none';
  } else if(val==='total'){
    stdFields.style.display='';
    stdFields.innerHTML=`<input type="number" name="leg_line_${n}" placeholder="Total line" min="1" step="0.5" style="width:100px">`;
    otherFields.style.display='none';
  } else {
    stdFields.style.display='none';
    otherFields.style.display='';
  }
}

function removeLeg(n){
  const row=document.getElementById('leg-row-'+n);
  if(row) row.remove();
}

// ── Source filter (manual / bot) ─────────────────────────────────────────────
function applySourceFilter(){
  const src=document.getElementById('source-filter').value;
  document.querySelectorAll('.card[data-source]').forEach(card=>{
    const cs=card.dataset.source||'bot';
    card.style.display=(src==='all'||cs===src)?'':'none';
  });
}

// Chart.js cumulative profit line
const chartData = {{ chart_data|safe }};
if(chartData.length){
  const ctx = document.getElementById('profitChart').getContext('2d');
  new Chart(ctx,{
    type:'line',
    data:{
      labels: chartData.map(d=>d.settled_at.slice(0,10)),
      datasets:[{
        label:'Cumulative profit ($)',
        data: chartData.map(d=>d.cumulative_profit),
        borderColor:'#58a6ff',
        backgroundColor:'rgba(88,166,255,.1)',
        fill:true,
        tension:.3,
        pointRadius:3,
      }]
    },
    options:{
      responsive:true,
      plugins:{legend:{display:false}},
      scales:{
        y:{grid:{color:'#21262d'},ticks:{color:'#8b949e',callback:v=>'$'+v}},
        x:{grid:{color:'#21262d'},ticks:{color:'#8b949e'}}
      }
    }
  });
}
</script>
</body>
</html>
"""
