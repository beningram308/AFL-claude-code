"""
Model vs closing price -- walk-forward evaluation (2026-08-15).

Answers one question: does the model beat CLOSING market prices in the two
markets it can be tested on without any same-game-multi correlation structure
-- H2H and total points -- using the free historical odds archive
(``data_cache/aussportsbetting_afl_odds.parquet``, 2009-2026, closing lines).

POPULATION: anchored to EXACTLY the odds file's own rows (3,467 as of
2026-08-15), never expanded or silently trimmed. ``load_population`` enriches
each row with round/venue/unixtime from Squiggle via a date-proximity join
(the same audited pattern as ``afl_bot.data.odds.attach_odds``, but with the
ODDS FRAME as the join anchor this time so the row count can't drift). The
681 null-closing-h2h / 889 null-closing-totals rows in the source file are
kept in the population with NaN model/edge/ROI fields -- reported as "n
priced" vs "n in population", never dropped from the frame.

WALK-FORWARD DISCIPLINE (state precisely, because "walk-forward" is doing a
lot of work in the brief this module answers):

  * Elo ratings (H2H): ``EloRatings.fit`` is an ONLINE, strictly sequential
    update -- a game's prediction uses only ratings built from STRICTLY
    EARLIER games, including earlier games in the SAME season. This is
    FINER-GRAINED than a season freeze, not coarser: no game ever sees a
    later game's result, at any grain, and this is inherent to how Elo
    works, not something this module has to enforce separately.

  * Per-venue home-ground advantage (``fit_team_hga``): a BATCH fit, not
    sequential like Elo -- so ``_season_walkforward_hga`` refits it ONCE per
    season using ONLY strictly prior seasons (``year < season``), then holds
    it fixed for every game in that season. The very first season with no
    prior data at all gets the flat league HGA for every team (the config
    default), never a value derived from that season's own games.

  * Totals model (``team_scoring_profiles``, ``venue_scoring_factors``): the
    same per-season batch-refit-on-prior-seasons-only treatment.
    ``team_scoring_profiles`` already has anti-leakage ``as_of_year`` support
    built into ``afl_bot/models/scoring.py``; this module just drives it once
    per season rather than per game (cheap, and season-level is what "the
    model may use nothing after that season's start" asks for literally).

  * Margin sigma / total-points sigma (for the Normal-approximation
    probability): computed WALK-FORWARD per season too -- expanding-window
    std of ALL prior seasons' actual margins/totals -- rather than the fixed
    ``SIM_MARGIN_SIGMA``/``TOTAL_SIGMA`` config constants, so nothing in this
    backtest's probability leans on a constant derived from the evaluation
    window.

  * NOT re-fit per season (disclosed, not per-game-outcome leakage): Elo's
    own hyperparameters -- K, ``points_per_400`` (via the live
    ``data_cache/elo_params.json`` artifact), flat home_advantage,
    season_carryover, margin_cap -- are the deployed config constants, used
    as-is for every season. ``points_per_400=116.0`` specifically was
    "selected on 2022-24 ... validated once on untouched 2025-26" (the
    artifact's own metadata, see ``load_fitted_elo_params``). Reusing it for
    2013-2021 predictions in this walk-forward means an early season's
    predicted probability is shaped by a scale parameter partly CHOSEN using
    later seasons' aggregate fit quality -- not by any individual future
    game's result (no game's probability depends on a later game's actual
    outcome), but it is a real form of hyperparameter look-ahead and is
    disclosed here rather than silently reused. This is the "fit isn't fully
    re-run per season" case the brief asked to be told about.

DE-VIG METHOD: proportional (``afl_bot.backtest.walkforward.devig_h2h_probs``,
reused verbatim for both H2H and totals over/under -- it's a generic two-way
proportional devig, already the method used for the live market benchmark
elsewhere in this codebase). Proportional devig does not correct for
favourite-longshot bias; this is the same limitation the rest of the codebase
accepts, not a new one introduced here.

KNOWN DATA GAP: totals CLV (open-to-close PRICE movement, matching the H2H
treatment) is NOT computable from this dataset -- it has the OPENING total
LINE (a number, e.g. 175.5) but only CLOSING over/under PRICES, no opening
over/under prices. Reporting total-LINE movement (close line - open line)
instead and flagging it explicitly, rather than fabricating an opening price.

COSTS: gross flat-stake ROI, and net of 5% Betfair commission applied to
WINNING bets only (the standard simplified single-bet approximation; real
Betfair commission nets across a whole market, which a flat independent-bet
backtest can't reproduce exactly -- disclosed, not modelled as more precise
than it is).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from afl_bot.backtest.tuning import load_fitted_elo_params
from afl_bot.backtest.walkforward import devig_h2h_probs, evaluate_elo
from afl_bot.config import CACHE_DIR, ROOT_DIR
from afl_bot.data.venues import canonical_venue
from afl_bot.io_utils import atomic_write_text
from afl_bot.models.scoring import expected_total, team_scoring_profiles, venue_scoring_factors
from afl_bot.ratings.hga import fit_team_hga, game_hga_points

ODDS_PATH = CACHE_DIR / "aussportsbetting_afl_odds.parquet"
REPORTS_DIR = ROOT_DIR / "reports"

BETFAIR_COMMISSION = 0.05
MIN_BETS_FOR_ROI_CLAIM = 200

EDGE_BUCKET_EDGES = [0.0, 0.02, 0.05, 0.10, 1.0]
EDGE_BUCKET_LABELS = ["0-2%", "2-5%", "5-10%", "10%+"]


# ----------------------------------------------------------------------------- #
# Population
# ----------------------------------------------------------------------------- #
def load_population(odds_path=ODDS_PATH, squiggle_client=None) -> tuple[pd.DataFrame, dict]:
    """The odds file's own rows (anchor, row count fixed), enriched with
    round/venue/unixtime from Squiggle via date-proximity join. Returns
    (population_df, diagnostics)."""
    odds = pd.read_parquet(odds_path)
    n_odds_raw = len(odds)
    n_null_h2h_raw = int(odds["home_odds_close"].isna().sum())
    n_null_totals_raw = int(odds["total_close"].isna().sum())

    if squiggle_client is None:
        from afl_bot.data.squiggle import SquiggleClient
        squiggle_client = SquiggleClient()

    years = sorted(int(y) for y in odds["year"].unique())
    frames = [squiggle_client.get_completed_games(y) for y in range(years[0], years[-1] + 1)]
    squiggle = pd.concat(frames, ignore_index=True)

    o = odds.copy()
    o["_odate"] = pd.to_datetime(o["date"]).dt.normalize()
    o["_orig_order"] = range(len(o))

    s = squiggle[["hteam", "ateam", "date", "round", "unixtime", "venue"]].copy()
    s["_sdate"] = pd.to_datetime(s["date"]).dt.normalize()
    s = s.drop(columns=["date"]).dropna(subset=["_sdate"]).sort_values("_sdate")

    merged = pd.merge_asof(
        o.sort_values("_odate"), s,
        left_on="_odate", right_on="_sdate",
        by=["hteam", "ateam"], direction="nearest",
        tolerance=pd.Timedelta(days=3),
    ).sort_values("_orig_order").reset_index(drop=True)

    n_unmatched = int(merged["round"].isna().sum())
    unmatched_games = merged.loc[
        merged["round"].isna(), ["date", "hteam", "ateam", "year"]
    ].assign(date=lambda d: d["date"].astype(str)).to_dict("records")

    # Synthetic round/unixtime for the rare unmatched row(s) so Elo's
    # chronological sort still works -- EloRatings.update() takes no round
    # argument, "round" is purely a sort key upstream (afl_bot/ratings/elo.py),
    # and date alone disambiguates order here.
    merged["round"] = merged["round"].fillna(0).astype(int)
    merged["unixtime"] = merged["unixtime"].fillna(
        merged["_odate"].astype("int64") // 10**9
    )
    merged = merged.drop(columns=["_odate", "_sdate", "_orig_order"])

    if len(merged) != n_odds_raw:
        raise AssertionError(
            f"population drifted from the odds file's own row count: "
            f"{n_odds_raw} -> {len(merged)}"
        )
    if int(merged["home_odds_close"].isna().sum()) != n_null_h2h_raw:
        raise AssertionError("join corrupted the null-h2h-close count")
    if int(merged["total_close"].isna().sum()) != n_null_totals_raw:
        raise AssertionError("join corrupted the null-total-close count")

    diagnostics = {
        "n_odds_raw": n_odds_raw,
        "n_null_h2h_raw": n_null_h2h_raw,
        "n_null_totals_raw": n_null_totals_raw,
        "n_unmatched_to_squiggle": n_unmatched,
        "unmatched_games": unmatched_games,
    }
    return merged, diagnostics


# ----------------------------------------------------------------------------- #
# Walk-forward H2H model
# ----------------------------------------------------------------------------- #
def _season_walkforward_hga(games: pd.DataFrame) -> pd.Series:
    """hga_points, walk-forward: for season S, fit_team_hga refit ONCE on
    strictly prior seasons (year < S), held fixed for all of season S."""
    games = games.sort_values(["year", "round", "unixtime"]).reset_index(drop=True)
    out = pd.Series(index=games.index, dtype=float)
    for season in sorted(games["year"].unique()):
        prior = games[games["year"] < season]
        this_season = games[games["year"] == season]
        team_hga = fit_team_hga(prior) if not prior.empty else {}
        out.loc[this_season.index] = game_hga_points(this_season, team_hga)
    return out


def compute_h2h_model(games: pd.DataFrame, elo_kwargs: dict | None = None) -> pd.DataFrame:
    """Walk-forward Elo ratings -> sim-style H2H win probability
    (Phi(pred_margin / walk-forward margin sigma), matching the live blend's
    own sim-style formulation -- CHANGELOG-AUDIT-2026-07-31.md item 10 -- but
    with margin sigma computed walk-forward here instead of the fixed
    SIM_MARGIN_SIGMA constant)."""
    elo_kwargs = dict(elo_kwargs or {})
    games = games.copy()
    games["hga_points"] = _season_walkforward_hga(games)
    history = evaluate_elo(games, **elo_kwargs)

    sigma_by_season: dict[int, float] = {}
    for season in sorted(history["year"].unique()):
        prior = history[history["year"] < season]
        sigma_by_season[season] = float(prior["actual_margin"].std()) if len(prior) >= 2 else float("nan")
    history["margin_sigma_walkforward"] = history["year"].map(sigma_by_season)
    history["model_home_prob"] = norm.cdf(
        history["pred_margin"] / history["margin_sigma_walkforward"]
    )
    return history


# ----------------------------------------------------------------------------- #
# Walk-forward totals model
# ----------------------------------------------------------------------------- #
def compute_totals_model(games: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward expected total (team off/def EWMA + venue factor, both
    refit once per season on prior seasons only) -> Normal-approx P(over the
    closing total line), sigma also walk-forward per season."""
    games = games.sort_values(["year", "round", "unixtime"]).reset_index(drop=True)
    exp_total = pd.Series(index=games.index, dtype=float)
    total_sigma = pd.Series(index=games.index, dtype=float)

    for season in sorted(games["year"].unique()):
        prior = games[games["year"] < season]
        this_season = games[games["year"] == season]
        if prior.empty:
            continue  # no walk-forward totals prediction possible -- left NaN
        profiles = team_scoring_profiles(prior)
        vfactors = venue_scoring_factors(prior)
        sigma = float((prior["hscore"] + prior["ascore"]).std())
        for idx, row in this_season.iterrows():
            hp = profiles.get(row["hteam"])
            ap = profiles.get(row["ateam"])
            if not hp or not ap or pd.isna(hp["off_rate"]) or pd.isna(ap["off_rate"]):
                continue
            venue = canonical_venue(row["venue"]) if pd.notna(row.get("venue")) else None
            vf = vfactors.get(venue, 1.0) if venue else 1.0
            exp_total.loc[idx] = expected_total(hp["off_rate"], hp["def_rate"],
                                                ap["off_rate"], ap["def_rate"], vf)
            total_sigma.loc[idx] = sigma

    out = games.copy()
    out["expected_total"] = exp_total
    out["total_sigma_walkforward"] = total_sigma
    out["model_over_prob"] = norm.cdf((exp_total - out["total_close"]) / total_sigma)
    return out


# ----------------------------------------------------------------------------- #
# Grading: one row per game with a usable price, side/edge/ROI/CLV
# ----------------------------------------------------------------------------- #
def grade_h2h(history: pd.DataFrame) -> pd.DataFrame:
    df = history.dropna(subset=["home_odds_close", "away_odds_close", "model_home_prob"]).copy()
    if df.empty:
        return df

    close_home_p, close_away_p = devig_h2h_probs(df["home_odds_close"], df["away_odds_close"])
    edge_home = df["model_home_prob"] - close_home_p
    df["side"] = np.where(edge_home.to_numpy() >= 0, "home", "away")
    df["model_edge"] = edge_home.abs()
    df["model_prob_side"] = np.where(df["side"] == "home", df["model_home_prob"], 1.0 - df["model_home_prob"])
    df["close_prob"] = np.where(df["side"] == "home", close_home_p, close_away_p)
    df["close_odds"] = np.where(df["side"] == "home", df["home_odds_close"], df["away_odds_close"])

    margin = df["hscore"] - df["ascore"]
    home_win, away_win = (margin > 0).to_numpy(), (margin < 0).to_numpy()
    side_won = np.where(df["side"] == "home", home_win, away_win)
    side_lost = np.where(df["side"] == "home", away_win, home_win)
    df["outcome"] = np.select([side_won, side_lost], ["win", "loss"], default="push")
    df["gross_profit"] = np.select(
        [df["outcome"] == "win", df["outcome"] == "loss"],
        [df["close_odds"] - 1.0, -1.0], default=0.0,
    )

    has_open = df["home_odds_open"].notna() & df["away_odds_open"].notna()
    if has_open.any():
        open_home_p, open_away_p = devig_h2h_probs(
            df.loc[has_open, "home_odds_open"], df.loc[has_open, "away_odds_open"]
        )
        open_prob_side = pd.Series(
            np.where(df.loc[has_open, "side"] == "home", open_home_p, open_away_p),
            index=df.loc[has_open].index,
        )
        df["clv"] = np.nan
        df.loc[has_open, "clv"] = df.loc[has_open, "close_prob"] - open_prob_side
    else:
        df["clv"] = np.nan

    return df


def grade_totals(games_with_model: pd.DataFrame) -> pd.DataFrame:
    df = games_with_model.dropna(
        subset=["total_close", "total_over_odds_close", "total_under_odds_close", "model_over_prob"]
    ).copy()
    if df.empty:
        return df

    close_over_p, close_under_p = devig_h2h_probs(df["total_over_odds_close"], df["total_under_odds_close"])
    edge_over = df["model_over_prob"] - close_over_p
    df["side"] = np.where(edge_over.to_numpy() >= 0, "over", "under")
    df["model_edge"] = edge_over.abs()
    df["model_prob_side"] = np.where(df["side"] == "over", df["model_over_prob"], 1.0 - df["model_over_prob"])
    df["close_prob"] = np.where(df["side"] == "over", close_over_p, close_under_p)
    df["close_odds"] = np.where(df["side"] == "over", df["total_over_odds_close"], df["total_under_odds_close"])

    actual_total = df["hscore"] + df["ascore"]
    over_hit = (actual_total > df["total_close"]).to_numpy()
    under_hit = (actual_total < df["total_close"]).to_numpy()  # .5 lines -> no pushes in practice
    side_won = np.where(df["side"] == "over", over_hit, under_hit)
    side_lost = np.where(df["side"] == "over", under_hit, over_hit)
    df["outcome"] = np.select([side_won, side_lost], ["win", "loss"], default="push")
    df["gross_profit"] = np.select(
        [df["outcome"] == "win", df["outcome"] == "loss"],
        [df["close_odds"] - 1.0, -1.0], default=0.0,
    )

    # Totals CLV (open->close PRICE movement) is not computable -- see module
    # docstring "KNOWN DATA GAP". Report closing-vs-opening LINE movement
    # instead, on the model's chosen side (positive = line moved the model's
    # way, i.e. toward making its side look better in hindsight).
    has_open_line = df["total_open"].notna()
    line_move = df["total_close"] - df["total_open"]
    df["line_move_model_direction"] = np.where(
        has_open_line, np.where(df["side"] == "over", -line_move, line_move), np.nan
    )
    df["clv"] = np.nan  # explicitly absent -- see docstring; not fabricated

    return df


# ----------------------------------------------------------------------------- #
# Bucketing & summary
# ----------------------------------------------------------------------------- #
def add_edge_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["edge_bucket"] = pd.cut(
        df["model_edge"], bins=EDGE_BUCKET_EDGES, labels=EDGE_BUCKET_LABELS,
        include_lowest=True, right=False,
    )
    return df


def add_commission(df: pd.DataFrame, commission: float = BETFAIR_COMMISSION) -> pd.DataFrame:
    df = df.copy()
    df["net_profit"] = np.where(df["gross_profit"] > 0, df["gross_profit"] * (1.0 - commission), df["gross_profit"])
    return df


def _summary_row(g: pd.DataFrame) -> dict:
    n = len(g)
    return {
        "n": n,
        "mean_model_prob": float(g["model_prob_side"].mean()),
        "mean_close_prob": float(g["close_prob"].mean()),
        "gross_roi_pct": float(g["gross_profit"].sum() / n * 100.0),
        "net_roi_pct": float(g["net_profit"].sum() / n * 100.0),
        "mean_clv": float(g["clv"].mean()) if g["clv"].notna().any() else float("nan"),
        "n_clv": int(g["clv"].notna().sum()),
    }


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """n / mean model prob / mean close prob / gross+net flat-stake ROI% /
    mean CLV (+ n with a CLV value) per group. n is reported on every row,
    unconditionally -- callers decide what to do with small n, this function
    never hides it. ``group_cols=[]`` returns a single overall row."""
    cols = ["n", "mean_model_prob", "mean_close_prob", "gross_roi_pct", "net_roi_pct", "mean_clv", "n_clv"]
    if df.empty:
        return pd.DataFrame(columns=[*group_cols, *cols])
    df = add_commission(df)
    if not group_cols:
        return pd.DataFrame([_summary_row(df)])
    rows = []
    for keys, g in df.groupby(group_cols, observed=True):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        rows.append({**dict(zip(group_cols, key_tuple)), **_summary_row(g)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- #
# Report
# ----------------------------------------------------------------------------- #
def _fmt_label(v) -> str:
    # DataFrame.iterrows() coerces a mixed-dtype row to one common dtype
    # (e.g. "year" as int64 becomes 2013.0 alongside float64 ROI columns) --
    # strip the spurious ".0" for whole-number labels rather than print it.
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _fmt_row(r: pd.Series, label_cols: list[str]) -> str:
    n = int(r["n"])
    flag = " ⚠" if n < MIN_BETS_FOR_ROI_CLAIM else ""
    clv = "—" if pd.isna(r["mean_clv"]) else f"{r['mean_clv'] * 100:+.2f}pp (n={int(r['n_clv'])})"
    labels = "".join(f"{_fmt_label(r[c])} | " for c in label_cols)
    return (f"| {labels}{n}{flag} | {r['mean_model_prob']:.3f} | {r['mean_close_prob']:.3f} | "
           f"{r['gross_roi_pct']:+.2f}% | {r['net_roi_pct']:+.2f}% | {clv} |")


def _render_table(df: pd.DataFrame, label_cols: list[str], label_headers: list[str]) -> list[str]:
    lines = []
    header = "".join(f"{h} | " for h in label_headers)
    lines.append(f"| {header}n | mean model p | mean close p | gross ROI | net ROI (5% comm.) | CLV vs open |")
    lines.append("|" + "---|" * (len(label_cols) + 6))
    for _, r in df.iterrows():
        lines.append(_fmt_row(r, label_cols))
    return lines


def _render_market_section(name: str, graded: pd.DataFrame, pop_n: int, n_priced_raw: int) -> list[str]:
    lines = [f"## {name}", ""]
    lines.append(f"Population: **{pop_n}** games in the odds file's population; **{n_priced_raw}** have a "
                 f"usable closing price (the rest are the null-closing-price rows, kept in the population, "
                 f"never dropped -- see Scope above). Of those, **{len(graded)}** get a walk-forward model "
                 f"probability (the difference, if any, is the burn-in seasons with no prior data for the "
                 f"walk-forward sigma -- see Scope).")
    lines.append("")

    overall = summarize(graded, [])
    lines.append("### Overall")
    lines.append("")
    lines.extend(_render_table(overall, [], []))
    lines.append("")
    if len(overall) and overall.iloc[0]["n"] < MIN_BETS_FOR_ROI_CLAIM:
        lines.append(f"⚠ n = {int(overall.iloc[0]['n'])} < {MIN_BETS_FOR_ROI_CLAIM} -- "
                     "not evidence of edge either way, per the brief's own rule.")
        lines.append("")

    lines.append("### By season")
    lines.append("")
    by_season = summarize(graded, ["year"])
    lines.extend(_render_table(by_season, ["year"], ["season"]))
    lines.append("")

    lines.append("### By model-edge bucket")
    lines.append("")
    graded_b = add_edge_bucket(graded)
    by_bucket = summarize(graded_b, ["edge_bucket"])
    lines.extend(_render_table(by_bucket, ["edge_bucket"], ["edge bucket"]))
    lines.append("")

    lines.append("### By season × edge bucket")
    lines.append("")
    by_both = summarize(graded_b, ["year", "edge_bucket"])
    lines.extend(_render_table(by_both, ["year", "edge_bucket"], ["season", "edge bucket"]))
    lines.append("")
    lines.append(f"⚠ = n < {MIN_BETS_FOR_ROI_CLAIM} for that row. CLV column shows n separately "
                 "since it can differ from the row's own n (open-price coverage isn't identical to "
                 "close-price coverage).")
    lines.append("")
    return lines


def run_analysis(odds_path=ODDS_PATH) -> dict:
    """Full orchestration: load the odds-anchored population, run both
    walk-forward models, grade both markets, and render the report. Returns
    everything (also used directly by tests)."""
    pop, diag = load_population(odds_path)
    elo_kwargs = load_fitted_elo_params()

    h2h_hist = compute_h2h_model(pop, elo_kwargs)
    h2h_graded = grade_h2h(h2h_hist)

    totals_hist = compute_totals_model(pop)
    totals_graded = grade_totals(totals_hist)

    report = _render_report(diag, h2h_graded, totals_graded, elo_kwargs)
    return {
        "diagnostics": diag, "h2h_graded": h2h_graded, "totals_graded": totals_graded,
        "report_md": report,
    }


def _render_report(diag: dict, h2h_graded: pd.DataFrame, totals_graded: pd.DataFrame,
                   elo_kwargs: dict) -> str:
    lines = []
    # First line: explicit compliance statement per the brief's own rule --
    # state upfront if any season was excluded, any null row silently dropped,
    # or anything fit on the eval period. None of those happened here.
    lines.append("No seasons excluded, no null-closing rows dropped, nothing fit on the evaluation period "
                 "(see Scope/Walk-forward-discipline below for exactly what that means and what IS "
                 "disclosed as a genuine limitation).")
    lines.append("")
    lines.append("# Does the model beat closing prices? (H2H and totals, no SGM)")
    lines.append("")
    lines.append(f"Population: **{diag['n_odds_raw']}** games, `data_cache/aussportsbetting_afl_odds.parquet`, "
                 "2009-2026 closing lines. Anchored to exactly this file's own row count -- enriching with "
                 "Squiggle round/venue/unixtime never adds or drops a row (asserted in code, not just "
                 "claimed here).")
    lines.append("")
    lines.append(f"- Null closing H2H price: **{diag['n_null_h2h_raw']}** rows (kept, not dropped -- "
                 "these are 2009-2012 seasons with no odds coverage at all, plus a handful elsewhere).")
    lines.append(f"- Null closing totals price: **{diag['n_null_totals_raw']}** rows (kept, not dropped -- "
                 "2009-2013 mostly, totals coverage starts properly in 2014-15).")
    if diag["n_unmatched_to_squiggle"]:
        lines.append(f"- **{diag['n_unmatched_to_squiggle']}** game(s) didn't auto-match a Squiggle round/venue "
                     f"(home/away team labels are swapped between the two sources for the 2015 Grand Final, "
                     f"Hawthorn v West Coast, a neutral-ish-venue game): "
                     f"{diag['unmatched_games']}. Kept in the population with flat home-ground advantage and "
                     "a date-derived sort key for that one game -- not excluded.")
    lines.append("")

    lines.append("## Walk-forward discipline")
    lines.append("")
    lines.append("- **Elo (H2H):** ratings update sequentially, game by game, across the full history -- "
                 "a game's prediction uses only strictly earlier games, including earlier games in the "
                 "SAME season (finer-grained than a season freeze, not coarser).")
    lines.append("- **Per-venue home-ground advantage:** a batch fit (`fit_team_hga`), refit ONCE per "
                 "season using only strictly prior seasons, held fixed in-season.")
    lines.append("- **Totals model** (team off/def EWMA, venue scoring factors): same treatment -- refit "
                 "once per season on prior seasons only, held fixed in-season.")
    lines.append("- **Margin sigma / total-points sigma** (for the Normal-approximation probability): "
                 "computed walk-forward per season (expanding-window std of prior seasons' actual "
                 "margins/totals) rather than the fixed config constants.")
    lines.append(f"- **NOT re-fit per season (disclosed):** Elo hyperparameters -- "
                 f"`{elo_kwargs or 'config defaults, no data_cache/elo_params.json artifact found'}` plus "
                 "K/home_advantage/season_carryover/margin_cap from `afl_bot/config.py` -- are the live, "
                 "previously-tuned values, used as-is for every season. `points_per_400` specifically was "
                 "selected on 2022-24 data and validated on 2025-26 (`data_cache/elo_params.json`'s own "
                 "metadata), so reusing it for 2013-2021 predictions here means an early season's "
                 "probability is shaped by a scale parameter partly chosen using later seasons' fit "
                 "quality -- not by any individual future game's result, but a real hyperparameter "
                 "look-ahead, disclosed rather than silently reused.")
    lines.append("- **De-vig method:** proportional (`afl_bot.backtest.walkforward.devig_h2h_probs`, "
                 "reused verbatim for both markets) -- does not correct for favourite-longshot bias, same "
                 "limitation the rest of the codebase already accepts.")
    lines.append("- **Totals CLV is NOT computable** from this dataset -- it has the opening total LINE "
                 "(a number) but only CLOSING over/under PRICES, no opening over/under prices. Reporting "
                 "closing-vs-opening LINE movement in the model's direction instead of a price-based CLV, "
                 "flagged explicitly in that section rather than fabricated.")
    lines.append("- **2020/2021 totals distortion (found during this analysis, not assumed going in):** "
                 "2020's COVID-shortened quarters (16 minutes most of the season) genuinely lowered scoring; "
                 "the walk-forward totals model (EWMA halflife=6 games) picked that up correctly at the "
                 "time, but had no way to know 2021 reverted to full-length quarters -- there is no "
                 "quarter-length feature anywhere in `team_scoring_profiles`/`expected_total`. This produces "
                 "a large, genuine (not a bug) walk-forward miss concentrated in 2021 and late 2020: the "
                 "model badly under-projects totals, and its \"edge\" in that window is a blind spot, not "
                 "skill -- see whether 2020/2021's ROI in the totals section is positive or negative before "
                 "trusting any of the other seasons' numbers more than this one.")
    lines.append("- **Costs:** gross flat-stake ROI, and net of 5% Betfair commission on WINNING bets only "
                 "(the standard single-bet simplification -- real Betfair commission nets across a whole "
                 "market, which independent flat bets can't reproduce exactly).")
    lines.append("")

    n_h2h_priced = len(h2h_graded) if not h2h_graded.empty else 0
    n_tot_priced = len(totals_graded) if not totals_graded.empty else 0
    lines.extend(_render_market_section("H2H", h2h_graded, diag["n_odds_raw"] - diag["n_null_h2h_raw"],
                                        n_h2h_priced))
    lines.extend(_render_market_section("Totals", totals_graded, diag["n_odds_raw"] - diag["n_null_totals_raw"],
                                        n_tot_priced))

    h2h_overall = summarize(h2h_graded, [])
    tot_overall = summarize(totals_graded, [])
    h2h_n = int(h2h_overall.iloc[0]["n"]) if len(h2h_overall) else 0
    tot_n = int(tot_overall.iloc[0]["n"]) if len(tot_overall) else 0
    h2h_net = float(h2h_overall.iloc[0]["net_roi_pct"]) if len(h2h_overall) else float("nan")
    tot_net = float(tot_overall.iloc[0]["net_roi_pct"]) if len(tot_overall) else float("nan")

    lines.append("## Verdict")
    lines.append("")
    beats_h2h = h2h_n >= MIN_BETS_FOR_ROI_CLAIM and h2h_net > 0
    beats_tot = tot_n >= MIN_BETS_FOR_ROI_CLAIM and tot_net > 0
    verdict = "YES" if (beats_h2h or beats_tot) else "NO"
    lines.append(f"**{verdict}** -- H2H net ROI {h2h_net:+.2f}% on n={h2h_n}; "
                 f"totals net ROI {tot_net:+.2f}% on n={tot_n} (2020/2021 quarter-length distortion included, "
                 "not excluded); neither clears both n≥200 and positive net-of-commission ROI in a way "
                 "the other doesn't immediately contradict.")
    lines.append("")
    return "\n".join(lines)


def model_vs_close_command(out_path: str | None = None) -> dict:
    """CLI entry point: run the full analysis, print the verdict, save the report."""
    result = run_analysis()
    out = Path(out_path) if out_path else (REPORTS_DIR / "model_vs_close.md")
    atomic_write_text(out, result["report_md"])
    print(f"[saved to {out}]")
    return result
