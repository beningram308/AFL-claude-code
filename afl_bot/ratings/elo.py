"""
Margin-based Elo ratings (plan §3.1, §2.3).

AFL adaptations on top of vanilla chess Elo:
  * Margin, not win/loss — two interchangeable update rules:
      - "margin" (default): the match margin is squashed into a 0-1 "score" via
        a clipped-linear function over a +/-ELO_MARGIN_CAP point range, so big
        wins move ratings more than narrow ones.
      - "mov" (plan §2.3): a binary win/loss result scaled by a
        FiveThirtyEight-style margin-of-victory multiplier,
        ln(|margin|+1) * autocorrelation-correction, so the squash no longer
        discards information past the cap and runaway favourites don't inflate.
  * Home-ground / travel adjustment — a flat home-ground bonus plus each team's
    travel penalty is folded into the expected-margin calc.
  * Season carry-over — at the start of each season every rating is regressed
    toward 1500 by ELO_SEASON_CARRYOVER.
  * k-factor / points_per_400 / margin_cap / home_advantage / season_carryover
    are all instance attributes so they can be tuned against out-of-sample log
    loss + margin MAE (see afl_bot.backtest.tuning / walkforward.py).

``EloRatings.fit`` processes games in chronological order and returns per-game
*pre-match* ratings (the only ones safe to use as features — anti-leakage,
plan §2) alongside the final rating table. It is written as a tight numpy-array
loop (not ``DataFrame.iterrows``) so thousands of tuning fits stay cheap — the
ratings update is inherently sequential (game t+1 depends on game t's result)
so it cannot be fully vectorised across games, but the per-row pandas overhead
is removed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from afl_bot.config import (
    CACHE_DIR,
    ELO_HOME_ADVANTAGE,
    ELO_INITIAL,
    ELO_K,
    ELO_MARGIN_CAP,
    ELO_MOV_CORRECTION,
    ELO_POINTS_PER_400,
    ELO_SCALE,
    ELO_SEASON_CARRYOVER,
    ELO_UPDATE_MODE,
)

RATINGS_PATH = CACHE_DIR / "elo_ratings.json"


def margin_to_result(margin: float, cap: float = ELO_MARGIN_CAP) -> float:
    """Squash a home-team margin (points) into a 0-1 'result' for the Elo update.

    +cap or beyond -> 1.0 (max win credit), -cap or beyond -> 0.0, 0 -> 0.5.
    Linear in between, matching the plan's "normalise margin over a +/-80 range".
    """
    return float(np.clip((margin / (2.0 * cap)) + 0.5, 0.0, 1.0))


def expected_result(rating_diff: float, scale: float = ELO_SCALE) -> float:
    """Standard Elo logistic expectation from a rating difference."""
    return 1.0 / (1.0 + 10.0 ** (-rating_diff / scale))


def mov_multiplier(margin: float, winner_rating_diff: float,
                   correction: float = ELO_MOV_CORRECTION) -> float:
    """FiveThirtyEight-style margin-of-victory multiplier (plan §2.3).

    ``margin`` is the match margin (points; sign ignored). ``winner_rating_diff``
    is the pre-match Elo difference from the *winner's* perspective (winner's
    rating, including home advantage, minus loser's) — positive when the
    favourite won.

    ``ln(|margin|+1)`` scales the update by how decisive the win was, while the
    ``correction / (winner_rating_diff*0.001 + correction)`` term shrinks the
    multiplier when a strong favourite wins big and inflates it for upsets,
    which damps the rating autocorrelation/feedback that an uncorrected MOV term
    would introduce. A draw (margin 0) gives 0 -> no update.
    """
    return float(np.log(abs(margin) + 1.0) * (correction / (winner_rating_diff * 0.001 + correction)))


@dataclass
class EloRatings:
    ratings: dict[str, float] = field(default_factory=dict)
    k: float = ELO_K
    home_advantage: float = ELO_HOME_ADVANTAGE
    season_carryover: float = ELO_SEASON_CARRYOVER
    scale: float = ELO_SCALE
    margin_cap: float = ELO_MARGIN_CAP
    points_per_400: float = ELO_POINTS_PER_400
    initial: float = ELO_INITIAL
    update_mode: str = ELO_UPDATE_MODE          # "margin" | "mov"
    mov_correction: float = ELO_MOV_CORRECTION
    _last_season: int | None = field(default=None, repr=False)

    def get(self, team: str) -> float:
        return self.ratings.get(team, self.initial)

    def _maybe_carry_over_season(self, season: int) -> None:
        if self._last_season is not None and season != self._last_season:
            for team, rating in self.ratings.items():
                self.ratings[team] = self.initial + self.season_carryover * (rating - self.initial)
        self._last_season = season

    def _hga_in_elo(self, hga: float | None = None) -> float:
        """Home-ground advantage (points -> Elo rating units). Per-game ``hga``
        overrides the flat ``home_advantage`` (round-2 §6.1)."""
        pts = self.home_advantage if hga is None else hga
        return pts / self.points_per_400 * self.scale

    def expected_margin(self, home: str, away: str, travel_penalty: float = 0.0,
                        hga: float | None = None) -> float:
        """Predicted home margin (points) from the current ratings + home/travel
        adj. ``hga`` (points) overrides the flat home advantage for venue/travel/
        rest-aware predictions (§6.1)."""
        rating_gap_pts = (self.get(home) - self.get(away)) / self.scale * self.points_per_400
        home_adv = self.home_advantage if hga is None else hga
        return rating_gap_pts + home_adv + travel_penalty

    def update(self, home: str, away: str, margin: float, season: int,
               hga: float | None = None) -> tuple[float, float]:
        """Process one completed match. Returns (pre_match_home_rating, pre_match_away_rating)."""
        self._maybe_carry_over_season(season)

        pre_home = self.get(home)
        pre_away = self.get(away)

        rating_diff = (pre_home + self._hga_in_elo(hga)) - pre_away
        expected = expected_result(rating_diff, self.scale)

        if self.update_mode == "mov":
            actual = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
            winner_rating_diff = rating_diff if margin > 0 else -rating_diff
            mult = mov_multiplier(margin, winner_rating_diff, self.mov_correction)
            delta = self.k * mult * (actual - expected)
        else:
            actual = margin_to_result(margin, self.margin_cap)
            delta = self.k * (actual - expected)

        self.ratings[home] = pre_home + delta
        self.ratings[away] = pre_away - delta
        return pre_home, pre_away

    def fit(self, games: pd.DataFrame) -> pd.DataFrame:
        """Process completed games in chronological order, returning a copy of
        ``games`` with added ``home_elo_pre`` / ``away_elo_pre`` columns containing
        the *pre-match* ratings (safe features for modelling).

        Implemented as a numpy-array loop rather than ``iterrows`` — the update
        is sequential (each game uses the ratings the previous games produced)
        so it can't be vectorised across games, but pulling the columns out to
        arrays first avoids per-row pandas Series construction, the dominant
        cost when tuning runs thousands of fits.
        """
        games = games.sort_values(["year", "round", "unixtime"]).reset_index(drop=True)
        n = len(games)
        hteams = games["hteam"].to_numpy()
        ateams = games["ateam"].to_numpy()
        margins = games["hscore"].to_numpy(dtype=float) - games["ascore"].to_numpy(dtype=float)
        years = games["year"].to_numpy(dtype=int)
        # Per-game home advantage (venue/travel/rest) when supplied (§6.1).
        hgas = games["hga_points"].to_numpy(dtype=float) if "hga_points" in games.columns else None

        home_pre = np.empty(n)
        away_pre = np.empty(n)
        update = self.update  # bind once to skip attribute lookup per game
        for i in range(n):
            hga = float(hgas[i]) if hgas is not None else None
            home_pre[i], away_pre[i] = update(hteams[i], ateams[i], float(margins[i]), int(years[i]), hga)

        out = games.copy()
        out["home_elo_pre"] = home_pre
        out["away_elo_pre"] = away_pre
        return out

    def save(self, path=RATINGS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "ratings": self.ratings,
            "k": self.k,
            "home_advantage": self.home_advantage,
            "season_carryover": self.season_carryover,
            "scale": self.scale,
            "margin_cap": self.margin_cap,
            "points_per_400": self.points_per_400,
            "initial": self.initial,
            "update_mode": self.update_mode,
            "mov_correction": self.mov_correction,
            "last_season": self._last_season,
        }, indent=2))

    @classmethod
    def load(cls, path=RATINGS_PATH) -> "EloRatings":
        data = json.loads(path.read_text())
        obj = cls(
            ratings=data["ratings"],
            k=data["k"],
            home_advantage=data["home_advantage"],
            season_carryover=data["season_carryover"],
            scale=data["scale"],
            margin_cap=data["margin_cap"],
            points_per_400=data.get("points_per_400", ELO_POINTS_PER_400),
            initial=data["initial"],
            update_mode=data.get("update_mode", ELO_UPDATE_MODE),
            mov_correction=data.get("mov_correction", ELO_MOV_CORRECTION),
        )
        obj._last_season = data.get("last_season")
        return obj


def build_ratings_from_history(games: pd.DataFrame, **elo_kwargs) -> tuple[EloRatings, pd.DataFrame]:
    """Convenience: fit a fresh EloRatings over a full history of completed games."""
    elo = EloRatings(**elo_kwargs)
    history = elo.fit(games)
    return elo, history
