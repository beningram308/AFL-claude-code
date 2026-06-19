"""Project-wide configuration: paths, tunables, and constants from the master plan."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT_DIR / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

SQUIGGLE_BASE_URL = "https://api.squiggle.com.au/"
SQUIGGLE_USER_AGENT = "afl-multi-builder (https://github.com/; contact via repo issues)"

# Historical odds workbook updates weekly in-season — re-download a cache older
# than this many days rather than caching forever (round-2 §7.5).
ODDS_MAX_AGE_DAYS = 3.0

# ----------------------------------------------------------------------------- #
# Elo (margin-based) — plan §3.1
# ----------------------------------------------------------------------------- #
ELO_INITIAL = 1500.0
ELO_K = 35.0                 # responsiveness; tune via backtest (afl_bot.backtest.tuning)
ELO_HOME_ADVANTAGE = 10.0    # generic home-ground points value
ELO_SEASON_CARRYOVER = 0.65  # fraction of rating-vs-1500 gap retained at season start
ELO_SCALE = 400.0            # standard Elo logistic scale
ELO_POINTS_PER_400 = 92.0    # Elo points <-> expected margin mapping (tunable)
ELO_MARGIN_CAP = 80.0        # cap for margin->Elo "score" squashing function (tunable)
# Update rule: "margin" = clipped-linear margin->result squash (default);
# "mov" = FiveThirtyEight margin-of-victory multiplier on a binary result with
# an autocorrelation correction (afl_bot.ratings.elo.mov_multiplier).
ELO_UPDATE_MODE = "margin"
ELO_MOV_CORRECTION = 2.2     # 538-style MOV autocorrelation-correction constant

# ----------------------------------------------------------------------------- #
# Match scoring — plan §2.1/§2.2 (scoring-shots model)
# ----------------------------------------------------------------------------- #
# Legacy reference values (empirical AFL margin/total spread, 2015+). The
# scoring-shots model in afl_bot.sim.engine no longer samples margin/total
# directly from these -- sigma now emerges from SHOT_DISPERSION (NB variance
# grows with mu_shots, giving heteroscedastic sigma "for free", §2.2) and
# SHOT_ACCURACY_SIGMA. Kept here as calibration targets / for reference.
MARGIN_SIGMA = 37.0   # empirical game-to-game spread of AFL margins (points)
TOTAL_SIGMA = 25.0    # spread of total points

# Scoring shots (goals + behinds) per team per game ~ NB(mu_shots, SHOT_DISPERSION).
# Calibrated against 2015+ Squiggle results: mean ~22.3, std ~5.8 shots/team/game.
SHOT_DISPERSION = 42.5
# Per-iteration goal-conversion ("accuracy") noise around a team's EWMA accuracy,
# clipped to SHOT_ACCURACY_BOUNDS. Combined with SHOT_DISPERSION this calibrates
# team points variance to the empirical ~633 (std ~25), giving margin/total
# sigma ~36 -- between the old fixed 37/25, but now correlated draw/integer
# scorelines instead of a cosmetic post-hoc split.
SHOT_ACCURACY_SIGMA = 0.06
SHOT_ACCURACY_BOUNDS = (0.30, 0.75)
DEFAULT_SHOT_ACCURACY = 0.525  # league-average goal conversion, used when a team has no history
POINTS_PER_GOAL = 6.8

# ----------------------------------------------------------------------------- #
# Player props — plan §3.3
# ----------------------------------------------------------------------------- #
PROP_FORM_GAMES = 20          # rolling window for EWMA rate (legacy; see PLAYER_FORM_WINDOW)
PROP_EWMA_HALFLIFE = 6        # games; current-season weighted up
PROP_MIN_DISPERSION = 4.0     # floor on NB dispersion to avoid degenerate fits

# --- Player form window (last-N games, not all-history) ---
PLAYER_FORM_WINDOW = 40          # project each player off their most recent 40 games
TOG_RETURN_DEFAULT = 0.75        # expected TOG for a player flagged returning-from-injury / managed

# ----------------------------------------------------------------------------- #
# Boundary throw-ins / out-of-bounds market — plan §1.6c
# ----------------------------------------------------------------------------- #
# Real per-game boundary-throw-in counts need the Champion-Data-fed AFL stats
# feed (not free/stable here), so until that flows the model prices off a league
# prior. ~36 boundary throw-ins per game (both teams) with game-to-game std ~9
# => NB dispersion r ~= 36^2/(81-36) ~= 29.
LEAGUE_OOB_PER_GAME = 36.0
OOB_DISPERSION = 29.0
# Negative coupling to the match total (congestion lowers totals and raises
# OOB): per-iteration OOB mean scales with (mean_total/total_iter)**this.
# 0.5 gives a sensible OOB/total correlation (~-0.36) and CoV (~0.27) while
# keeping the marginal mean on the prior.
OOB_TOTAL_COUPLING = 0.5
# Wet weather lifts out-of-bounds (slippery ball, scrappy play).
OOB_RAIN_MULTIPLIER = 1.20

# ----------------------------------------------------------------------------- #
# Weather — plan §1.8, §3.4
# ----------------------------------------------------------------------------- #
WET_THRESHOLD_MM = 5.0   # daily rainfall (mm) at/above which a game counts as "wet".
                         # Daily totals are a coarse proxy for in-game conditions
                         # (morning rain != an evening bounce), so the threshold is
                         # set to mean "clearly rained", not drizzle.
# Wet effect applied INSIDE simulate_match so H2H/totals/margin move coherently
# with the props in the same multi (round-2 §4.1). The total multiplier is
# data-backed (2022-25 wet/dry total ratio ~0.92-0.94); the accuracy cut lowers
# goal conversion (more behinds) — daily rain understates it (~0.4pp), so this
# uses the genuinely-wet ~2pp. NB: lowering accuracy alone preserves expected
# total (it only shifts goals<->behinds), so both knobs are needed and don't
# double-count — together they reproduce the ~0.92 wet goal ratio.
WET_TOTAL_MULTIPLIER = 0.93
WET_ACCURACY_PENALTY = 0.02

# ----------------------------------------------------------------------------- #
# Hierarchical player priors & role adjustments — plan §3.1, §3.2
# ----------------------------------------------------------------------------- #
# Empirical-Bayes shrinkage as pseudo-games: a player with this many games is
# weighted 50/50 with their role prior, a debutant sits near the prior, a
# long-history player near their own EWMA (plan §3.1).
PROP_PRIOR_STRENGTH = 8.0
PROP_DISPERSION_PRIOR_STRENGTH = 5.0   # pseudo-games for pooling NB r toward the role prior
PROP_CALIBRATION_LOOKBACK = 3          # seasons of walk-forward prop backtest to fit calibrators on
# Minimum walk-forward samples for a (stat, line) cell to get its own
# IsotonicCalibrator (model-upgrade audit Phase 3.2); below this, fall back to
# the pooled per-stat curve -- the tail lines (35+ disposals, 8+ marks; the
# $5 multi legs) are the ones a single per-stat curve under-serves, but they
# also have the fewest historical hits, so they need their own floor.
PROP_CALIBRATION_MIN_SAMPLES = 200
# Era-matching: opponent-matchup league baselines and role priors use only the
# last N seasons (stat levels drift with rule changes), keeping the full 2012+
# history only where sample size matters, e.g. dispersion (round-2 §5.1/§5.2).
PROP_RECENT_SEASONS = 3

# Coarse, data-driven role classification (per-game averages) — there is no
# position label common to Fryzigg history and DFS, so roles are inferred from
# the box score. Order: ruck -> forward -> midfielder -> general.
ROLE_RUCK_HITOUTS_MIN = 8.0
ROLE_FORWARD_GOALS_MIN = 1.2
ROLE_MID_DISPOSALS_MIN = 21.0

# TOG (time-on-ground %) minutes multiplier: expected counts scale with
# projected/historical TOG, clipped to these bounds (plan §3.2).
TOG_MULT_BOUNDS = (0.70, 1.15)

# CBA (centre bounce attendance) role-change detector: each extra CBA/game vs a
# player's own baseline lifts disposals ~1% (a wing -> centre jump of ~15 CBA
# ~= +15% disposals ~= +3-5 on a 25 base, plan §3.2), clipped to the bounds.
# CBA is only in the DFS current-season data, so this fires for current players.
CBA_ROLE_SENSITIVITY = 0.01
CBA_MULT_BOUNDS = (0.85, 1.25)

# ----------------------------------------------------------------------------- #
# Environment latent factors & within-team allocation — plan §2.5, §3.3
# ----------------------------------------------------------------------------- #
# Shared per-iteration "pace" multiplier (mean 1) that scales BOTH teams' volume
# stat totals (disposals/marks/tackles), so they move together — a fast, open
# game lifts everyone's disposals. Lognormal sigma; ~0.07 => ~7% game-to-game
# pace swing shared across the match.
PACE_SIGMA = 0.07
# NB dispersion for a team's per-iteration volume-stat total around its
# pace-scaled expected total (higher => tighter around the mean).
TEAM_STAT_DISPERSION = 150.0
# Dirichlet concentration for allocating a team's volume-stat total among its
# players. Calibrated (~200) so a top mid's disposal CoV is ~0.26 (realistic);
# this is what couples teammates' draws to the shared team total (mildly
# positive net correlation via pace, with the Dirichlet sum constraint adding
# the share-reallocation give-and-take) and keeps SGM joint prices coherent.
# Per-player marginal dispersion matching is a §3.1 hierarchical-priors job
# (build-order step 7); this is a single global approximation.
SHARE_CONCENTRATION = 200.0
# Gaussian-copula correlation between the two teams' scoring-shot draws in
# simulate_match. Negative because AFL scores are negatively correlated
# (territory is ~zero-sum). Calibrated at a symmetric matchup so margin sigma
# ~39.3 / total sigma ~31.4 / corr ~-0.22 match the empirical 2015+ split
# (margin 39.4 / total 31.3 / corr -0.224); NB marginals are preserved by the
# copula, so per-team means/variance are unchanged from the independent case.
SCORE_SHOT_CORRELATION = -0.32

# ----------------------------------------------------------------------------- #
# Edge / leg classification — plan §4.2
# ----------------------------------------------------------------------------- #
ANCHOR_MIN_PROB = 0.85
VALUE_MIN_EDGE = 0.08
VALUE_PROB_RANGE = (0.40, 0.72)

# ----------------------------------------------------------------------------- #
# Promo-aware EV — plan §4.4
# ----------------------------------------------------------------------------- #
BONUS_BET_FACTOR = 0.75   # value of a bonus bet vs cash
DEFAULT_STAKE = 50.0

# ----------------------------------------------------------------------------- #
# Kelly staking & bankroll sims — plan §4.4
# ----------------------------------------------------------------------------- #
KELLY_FRACTION = 0.25       # fractional Kelly (0.25x) — full Kelly is too volatile
KELLY_PER_BET_CAP = 0.05    # max fraction of bankroll on any one bet
KELLY_PER_ROUND_CAP = 0.15  # max total fraction of bankroll staked across a round
DEFAULT_BANKROLL = 1000.0
# Player props are noisier than H2H and compound multiplicatively in multis, so
# stake them at half the Kelly fraction even after calibration (round-2 §2.5).
PROP_KELLY_MULTIPLIER = 0.5
# Before staking a multi, pull each leg's model prob this far toward its
# market-implied prob — per-leg overestimates compound multiplicatively across
# legs (round-2 §8.2).
MULTI_MARKET_SHRINK = 0.25
# Target Monte Carlo standard error for an anchor's probability; n_sims is
# auto-bumped so the tightest anchor clears it (round-2 §8.3).
MC_SE_TARGET = 0.002

# ----------------------------------------------------------------------------- #
# Leg probability gate — bettable-window filter (REAL-MULTIS §2B)
# ----------------------------------------------------------------------------- #
# Lines outside this window are either too short to post (near-cert, prob > MAX)
# or too long to appear on standard book menus (prob < MIN). Replaces the loose
# 0.05/0.97 gate that admitted "15+ disposals" for ball-magnets (~99% prob).
LEG_PROB_MIN = 0.30
LEG_PROB_MAX = 0.78

# Target combined odds for the three rungs of the same-game multi ladder
# (REAL-MULTIS ADDENDUM 1). Selection uses closest fair-odds to each target.
MULTI_TARGET_ODDS = (1.75, 3.50, 5.00)

# Player-prop lines priced live (round-report/run-round) -- single source of
# truth (model-upgrade audit Phase 3.1). `afl_bot.backtest.props` and
# `afl_bot.backtest.multis` import this too, so the prop backtest/calibrators
# are always fit on exactly the lines actually priced; nothing here can drift
# out of sync with a separate backtest-only line set the way the old
# `backtest/props.py DEFAULT_PROP_LINES` did.
PROP_LINES = {
    "disposals": [15, 20, 25, 30, 35],
    "goals": [1, 2, 3],
    "marks": [4, 5, 6, 7, 8],
    "tackles": [3, 4, 5, 6, 7],
}

# ----------------------------------------------------------------------------- #
# Simulation
# ----------------------------------------------------------------------------- #
SIM_ITERATIONS = 50_000
RNG_SEED = 42
