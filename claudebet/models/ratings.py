"""Ridge-regularised power ratings from margins of victory.

Where Elo learns online and forgets slowly, this fits every team at once by
least squares: find the rating vector that best explains all observed margins,
with a ridge penalty pulling unproven teams toward average. For sports with
meaningful point margins -- the NFL, NBA, college basketball and football -- it
is a stronger baseline than Elo, and it produces a spread directly, which is
the number those markets actually trade.

The output that matters is not the rating, it is the *residual standard
deviation*. Predicting a 6.5-point favourite is easy; knowing that NFL margins
scatter around the spread with a standard deviation near 13.2 points is what
converts that into a win probability, and getting it wrong by a point moves
your probabilities by more than most edges are worth. The model estimates sigma
from its own residuals and falls back to :data:`SPORT_SIGMA` when the sample is
too thin to trust.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

__all__ = ["Game", "PowerRatings", "SPORT_SIGMA", "solve_ridge"]

# Residual standard deviation of final margin around a fair line, in the sport's
# own scoring units. These are the well-established values; re-estimate on your
# own data when you have a few hundred games.
SPORT_SIGMA: dict[str, float] = {
    "nfl": 13.2,
    "ncaaf": 16.0,
    "nba": 11.5,
    "ncaab": 10.5,
    "wnba": 11.0,
    "nhl": 1.95,
    "mlb": 3.15,
    "soccer": 1.55,
}


@dataclass(frozen=True)
class Game:
    home: str
    away: str
    home_score: float
    away_score: float
    neutral: bool = False
    weight: float = 1.0

    @property
    def margin(self) -> float:
        return self.home_score - self.away_score


def solve_ridge(
    matrix: list[list[float]], target: list[float], penalty: list[float]
) -> list[float]:
    """Solve ``(X'WX + diag(penalty)) b = X'Wy`` by Gaussian elimination with
    partial pivoting. Small enough problems that anything fancier is wasted."""
    n = len(matrix)
    aug = [row[:] + [target[i]] for i, row in enumerate(matrix)]
    for i in range(n):
        aug[i][i] += penalty[i]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            continue
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_val = aug[col][col]
        for row in range(col + 1, n):
            factor = aug[row][col] / pivot_val
            if factor == 0.0:
                continue
            for k in range(col, n + 1):
                aug[row][k] -= factor * aug[col][k]

    out = [0.0] * n
    for row in range(n - 1, -1, -1):
        if abs(aug[row][row]) < 1e-12:
            out[row] = 0.0
            continue
        acc = aug[row][n] - sum(aug[row][k] * out[k] for k in range(row + 1, n))
        out[row] = acc / aug[row][row]
    return out


class PowerRatings:
    """Least-squares team ratings on the margin scale.

    >>> games = [Game("A", "B", 24, 17), Game("B", "C", 21, 20), Game("C", "A", 10, 31)]
    >>> pr = PowerRatings(sport="nfl").fit(games)
    >>> pr.expected_margin("A", "B") > 0
    True
    """

    def __init__(self, sport: str = "nfl", ridge: float = 4.0) -> None:
        self.sport = sport
        self.ridge = ridge
        self.ratings: dict[str, float] = {}
        self.home_advantage: float = 0.0
        self.sigma: float = SPORT_SIGMA.get(sport, 12.0)
        self.n_games: int = 0
        self.r_squared: float = 0.0

    def fit(self, games: Sequence[Game], cap_margin: float | None = None) -> "PowerRatings":
        """Fit ratings and home advantage.

        ``cap_margin`` truncates blowouts before fitting. A 45-point win is
        weak evidence of being 45 points better -- teams stop trying. Capping
        at roughly two sigma is standard practice and measurably improves
        out-of-sample fit.
        """
        games = list(games)
        if len(games) < 2:
            raise ValueError("need at least two games to fit")
        teams = sorted({g.home for g in games} | {g.away for g in games})
        index = {t: i for i, t in enumerate(teams)}
        n = len(teams) + 1  # + home advantage
        hfa = len(teams)

        xtx = [[0.0] * n for _ in range(n)]
        xty = [0.0] * n

        for g in games:
            margin = g.margin
            if cap_margin is not None:
                margin = max(-cap_margin, min(cap_margin, margin))
            w = g.weight
            row: dict[int, float] = {index[g.home]: 1.0, index[g.away]: -1.0}
            if not g.neutral:
                row[hfa] = row.get(hfa, 0.0) + 1.0
            for i, vi in row.items():
                xty[i] += w * vi * margin
                for j, vj in row.items():
                    xtx[i][j] += w * vi * vj

        # Penalise team ratings, leave the home-advantage term unpenalised;
        # shrinking it would bias the estimate toward zero for no good reason.
        penalty = [self.ridge] * len(teams) + [1e-9]
        beta = solve_ridge(xtx, xty, penalty)

        raw = {t: beta[index[t]] for t in teams}
        mean = sum(raw.values()) / len(raw)
        self.ratings = {t: v - mean for t, v in raw.items()}
        self.home_advantage = beta[hfa]
        self.n_games = len(games)

        residuals = []
        for g in games:
            pred = self.expected_margin(g.home, g.away, g.neutral)
            residuals.append(g.margin - pred)
        dof = max(1, len(games) - len(teams) - 1)
        rss = sum(r * r for r in residuals)
        # Only trust the empirical sigma once there is real data behind it;
        # otherwise the in-sample fit will understate it badly.
        empirical = math.sqrt(rss / dof)
        if len(games) >= 4 * len(teams):
            self.sigma = empirical
        else:
            prior = SPORT_SIGMA.get(self.sport, 12.0)
            trust = len(games) / (4.0 * len(teams))
            self.sigma = trust * empirical + (1 - trust) * prior

        mean_margin = sum(g.margin for g in games) / len(games)
        tss = sum((g.margin - mean_margin) ** 2 for g in games)
        self.r_squared = 1.0 - rss / tss if tss > 0 else 0.0
        return self

    # -- prediction -----------------------------------------------------

    def rating(self, team: str) -> float:
        return self.ratings.get(team, 0.0)

    def expected_margin(self, home: str, away: str, neutral: bool = False) -> float:
        edge = 0.0 if neutral else self.home_advantage
        return self.rating(home) - self.rating(away) + edge

    def win_probability(self, home: str, away: str, neutral: bool = False) -> float:
        from ..correlation import norm_cdf

        return norm_cdf(self.expected_margin(home, away, neutral) / self.sigma)

    def cover_probability(
        self, home: str, away: str, spread: float, neutral: bool = False
    ) -> dict[str, float]:
        """Probability the home team covers ``spread``.

        ``spread`` is quoted from the home team's perspective, so -6.5 means
        the home side is laying 6.5. Whole numbers return a push probability
        estimated from the discrete distribution of margins around the line.
        """
        from ..correlation import norm_cdf

        margin = self.expected_margin(home, away, neutral)
        z = (margin + spread) / self.sigma
        push = 0.0
        if float(spread).is_integer():
            # Approximate the point mass at the exact number with the normal
            # density over a one-point interval. Crude for football, where key
            # numbers dominate, but better than pretending pushes cannot happen.
            density = math.exp(-0.5 * ((margin + spread) / self.sigma) ** 2) / (
                self.sigma * math.sqrt(2 * math.pi)
            )
            push = min(0.15, density)
        cover = norm_cdf(z) * (1 - push)
        return {"home": cover, "away": 1.0 - cover - push, "push": push}

    def table(self, top: int | None = None) -> list[tuple[str, float]]:
        ranked = sorted(self.ratings.items(), key=lambda kv: -kv[1])
        return ranked[:top] if top else ranked
