"""Elo ratings, with margin of victory and a proper three-way option.

Elo is the right first model for almost any sport: it needs nothing but the
results, it updates online, and it is hard to overfit. It will not beat a
liquid closing line on its own -- nothing simple will -- but it gives you an
independent number to blend against the market, and its rating differences are
a clean feature for anything more elaborate you build later.

Two refinements matter enough to be on by default:

*Margin of victory*, damped. A 40-point win says more than a 1-point win, but
much less than 40 times as much, and blowouts by heavy favourites say least of
all. The multiplier used here is FiveThirtyEight's: it grows with the log of the
margin and shrinks as the winner's pre-game rating edge grows, which stops
favourites from running away with the ratings.

*Between-season regression*. Rosters turn over. Pulling every rating a fraction
of the way back to the mean each season is crude but reliably better than not
doing it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

__all__ = ["EloConfig", "Elo", "GameResult"]


@dataclass
class EloConfig:
    k: float = 20.0
    home_advantage: float = 55.0  # in rating points
    scale: float = 400.0  # rating points per 10:1 odds
    initial: float = 1500.0
    use_mov: bool = True
    season_regression: float = 0.25  # fraction pulled back to the mean
    draw_nu: float = 0.0  # Davidson tie parameter; > 0 enables draws


@dataclass(frozen=True)
class GameResult:
    home: str
    away: str
    home_score: float
    away_score: float
    neutral: bool = False
    season: str | None = None
    weight: float = 1.0


class Elo:
    """Online Elo with optional margin-of-victory scaling and draws.

    >>> elo = Elo(EloConfig(k=20))
    >>> elo.update(GameResult("A", "B", 3, 1))
    >>> round(elo.win_probability("A", "B"), 3) > 0.5
    True
    """

    def __init__(
        self,
        config: EloConfig | None = None,
        ratings: Mapping[str, float] | None = None,
    ) -> None:
        self.config = config or EloConfig()
        self.ratings: dict[str, float] = dict(ratings or {})
        self._current_season: str | None = None
        self.history: list[dict] = []

    # -- ratings access -------------------------------------------------

    def rating(self, team: str) -> float:
        return self.ratings.setdefault(team, self.config.initial)

    def rating_diff(self, home: str, away: str, neutral: bool = False) -> float:
        edge = 0.0 if neutral else self.config.home_advantage
        return self.rating(home) - self.rating(away) + edge

    # -- prediction -----------------------------------------------------

    def win_probability(self, home: str, away: str, neutral: bool = False) -> float:
        """Two-way probability that the home team wins (draws excluded)."""
        diff = self.rating_diff(home, away, neutral)
        return 1.0 / (1.0 + 10.0 ** (-diff / self.config.scale))

    def probabilities(
        self, home: str, away: str, neutral: bool = False
    ) -> dict[str, float]:
        """Full outcome distribution.

        With ``draw_nu`` at zero this is ``{home, away}``. With ``draw_nu``
        above zero it is ``{home, draw, away}`` under Davidson's tie model,
        which keeps the home/away ratio exactly as Elo implies while carving
        out a draw probability that is largest when the teams are close --
        the behaviour you actually observe. Around 1.0 is a reasonable
        starting point for football/soccer.
        """
        nu = self.config.draw_nu
        diff = self.rating_diff(home, away, neutral)
        h = 10.0 ** (diff / (2.0 * self.config.scale))
        a = 10.0 ** (-diff / (2.0 * self.config.scale))
        if nu <= 0:
            total = h + a
            return {"home": h / total, "away": a / total}
        d = nu * math.sqrt(h * a)
        total = h + a + d
        return {"home": h / total, "draw": d / total, "away": a / total}

    def spread(self, home: str, away: str, points_per_elo: float = 0.04,
               neutral: bool = False) -> float:
        """Expected margin in points, positive for the home team.

        ``points_per_elo`` is sport-specific: roughly 0.04 for the NFL (25
        rating points per point of spread), 0.028 for the NBA, 0.05 for
        college football. Calibrate it against closing spreads on your own
        history rather than trusting these.
        """
        return self.rating_diff(home, away, neutral) * points_per_elo

    # -- updating -------------------------------------------------------

    def _mov_multiplier(self, margin: float, winner_diff: float) -> float:
        if not self.config.use_mov:
            return 1.0
        return math.log(abs(margin) + 1.0) * (2.2 / (winner_diff * 0.001 + 2.2))

    def update(self, game: GameResult) -> dict:
        """Apply one result and return the rating changes."""
        if game.season is not None and game.season != self._current_season:
            if self._current_season is not None:
                self.regress_to_mean(self.config.season_regression)
            self._current_season = game.season

        expected = self.win_probability(game.home, game.away, game.neutral)
        if game.home_score > game.away_score:
            actual = 1.0
        elif game.home_score < game.away_score:
            actual = 0.0
        else:
            actual = 0.5

        margin = game.home_score - game.away_score
        diff = self.rating_diff(game.home, game.away, game.neutral)
        # The winner's rating edge, which is what damps blowout inflation.
        winner_diff = diff if actual >= 0.5 else -diff
        mult = self._mov_multiplier(margin, winner_diff) if margin != 0 else 1.0

        delta = self.config.k * mult * (actual - expected) * game.weight
        self.ratings[game.home] = self.rating(game.home) + delta
        self.ratings[game.away] = self.rating(game.away) - delta

        record = {
            "home": game.home,
            "away": game.away,
            "expected": expected,
            "actual": actual,
            "delta": delta,
        }
        self.history.append(record)
        return record

    def fit(self, games: Iterable[GameResult]) -> "Elo":
        """Run through a chronological sequence of results.

        Order matters -- this is an online algorithm, so feed it oldest first.
        """
        for game in games:
            self.update(game)
        return self

    def regress_to_mean(self, fraction: float) -> None:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction must be in [0, 1]")
        if not self.ratings:
            return
        mean = sum(self.ratings.values()) / len(self.ratings)
        for team in self.ratings:
            self.ratings[team] += fraction * (mean - self.ratings[team])

    # -- evaluation -----------------------------------------------------

    def backtest(self, games: Sequence[GameResult]) -> dict:
        """Walk-forward evaluation: predict each game *before* learning from it.

        This is the only way to measure Elo honestly. Fitting on a season and
        scoring the same season will flatter it enormously.
        """
        preds: list[float] = []
        results: list[int] = []
        for game in games:
            if game.home_score == game.away_score:
                self.update(game)
                continue
            preds.append(self.win_probability(game.home, game.away, game.neutral))
            results.append(1 if game.home_score > game.away_score else 0)
            self.update(game)
        if not preds:
            return {"n": 0}
        brier = sum((p - o) ** 2 for p, o in zip(preds, results)) / len(preds)
        logloss = -sum(
            math.log(max(p if o else 1 - p, 1e-12)) for p, o in zip(preds, results)
        ) / len(preds)
        return {
            "n": len(preds),
            "brier": brier,
            "log_loss": logloss,
            "accuracy": sum(
                1 for p, o in zip(preds, results) if (p > 0.5) == bool(o)
            ) / len(preds),
            "predictions": preds,
            "outcomes": results,
        }

    def table(self, top: int | None = None) -> list[tuple[str, float]]:
        ranked = sorted(self.ratings.items(), key=lambda kv: -kv[1])
        return ranked[:top] if top else ranked


def tune_k(
    games: Sequence[GameResult],
    candidates: Sequence[float] = (8, 12, 16, 20, 24, 28, 32, 40),
    base: EloConfig | None = None,
) -> tuple[float, dict]:
    """Pick K by walk-forward log loss. Cheap, and worth doing per sport.

    Returns the best K and the full scoreboard. Beware of tuning on the same
    games you then claim an edge on -- hold out a season.
    """
    scores: dict[float, float] = {}
    for k in candidates:
        cfg = EloConfig(**{**(base.__dict__ if base else {}), "k": float(k)})
        result = Elo(cfg).backtest(games)
        if result.get("n"):
            scores[float(k)] = result["log_loss"]
    if not scores:
        raise ValueError("no decisive games to tune on")
    best = min(scores, key=lambda k: scores[k])
    return best, scores
