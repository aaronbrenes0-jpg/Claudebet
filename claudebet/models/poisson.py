"""Dixon-Coles: a scoreline model for low-scoring sports.

Fit an attack and a defence strength for every team plus a home-field term,
model each side's goals as Poisson, and you get a full distribution over
scorelines. From one such matrix you can price *every* market on the game at
once -- match result, totals at any line, both teams to score, Asian handicaps,
correct score -- and, crucially, price them *consistently*. That internal
consistency is where the edge usually is: books price these markets on separate
models and separate schedules, so they disagree with each other, and a joint
model tells you which one is stale.

Two departures from plain independent Poisson, both from Dixon and Coles (1997):

* **A low-score correction.** Independent Poisson understates 0-0 and 1-1 and
  overstates 1-0 and 0-1. One parameter, ``rho``, fixes the four cells where
  the error is concentrated.
* **Time decay.** Recent matches count more, with weight ``exp(-xi · days)``.
  A ``xi`` of 0.0045 halves a match's weight after about five months, which is
  in the range Dixon and Coles found optimal and still holds up.

The model is fit by gradient ascent on the weighted log likelihood. Attack and
defence parameters are fit under the Poisson likelihood, then ``rho`` is fit by
golden-section search on the full corrected likelihood. Splitting it this way
costs almost nothing in fit quality -- ``rho`` touches only four cells -- and
removes all of the numerical fragility from the joint optimisation.

This is a soccer model in its bones, but it transfers to any sport where scores
are small counts: hockey, and with a larger ``max_goals`` it is serviceable for
baseball runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Sequence

__all__ = ["Match", "Scoreline", "DixonColes", "tau_correction", "scoreline_from_rates"]


def tau_correction(h: int, a: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles adjustment to the four low-scoring cells.

    Independent Poisson understates 0-0 and 1-1 and overstates 1-0 and 0-1.
    ``rho`` is the single parameter that repairs it; everything else is
    untouched.
    """
    if h == 0 and a == 0:
        return max(1e-9, 1.0 - lam * mu * rho)
    if h == 0 and a == 1:
        return max(1e-9, 1.0 + lam * rho)
    if h == 1 and a == 0:
        return max(1e-9, 1.0 + mu * rho)
    if h == 1 and a == 1:
        return max(1e-9, 1.0 - rho)
    return 1.0


def scoreline_from_rates(
    home: str,
    away: str,
    lam: float,
    mu: float,
    rho: float = 0.0,
    max_goals: int = 12,
) -> "Scoreline":
    """Build a joint scoreline distribution straight from two scoring rates.

    Useful when the rates come from somewhere other than a fitted Dixon-Coles
    -- recent form, expected goals, or a hand-entered view.
    """
    n = max_goals + 1
    matrix = [[0.0] * n for _ in range(n)]
    for h in range(n):
        ph = _poisson(h, lam)
        for a in range(n):
            matrix[h][a] = ph * _poisson(a, mu) * tau_correction(h, a, lam, mu, rho)
    total = sum(sum(row) for row in matrix)
    if total <= 0:
        raise ValueError("degenerate scoreline distribution")
    matrix = [[v / total for v in row] for row in matrix]
    return Scoreline(home=home, away=away, matrix=matrix, home_xg=lam, away_xg=mu)

_LOG_FACT = [0.0]
for _i in range(1, 60):
    _LOG_FACT.append(_LOG_FACT[-1] + math.log(_i))


def _log_poisson(k: int, lam: float) -> float:
    return k * math.log(max(lam, 1e-12)) - lam - _LOG_FACT[k]


def _poisson(k: int, lam: float) -> float:
    return math.exp(_log_poisson(k, lam))


@dataclass(frozen=True)
class Match:
    home: str
    away: str
    home_goals: int
    away_goals: int
    when: date | datetime | None = None
    neutral: bool = False


@dataclass
class Scoreline:
    """A full joint distribution over final scores."""

    home: str
    away: str
    matrix: list[list[float]]  # matrix[h][a]
    home_xg: float
    away_xg: float

    def prob(self, home_goals: int, away_goals: int) -> float:
        if home_goals >= len(self.matrix) or away_goals >= len(self.matrix[0]):
            return 0.0
        return self.matrix[home_goals][away_goals]

    def match_odds(self) -> dict[str, float]:
        """1X2 probabilities."""
        home = draw = away = 0.0
        for h, row in enumerate(self.matrix):
            for a, p in enumerate(row):
                if h > a:
                    home += p
                elif h == a:
                    draw += p
                else:
                    away += p
        return {"home": home, "draw": draw, "away": away}

    def totals(self, line: float) -> dict[str, float]:
        """Over/under a total-goals line. Half-lines have no push; whole lines
        return the push probability separately so you can price the
        two-way-with-push market properly instead of silently dropping it."""
        over = under = push = 0.0
        for h, row in enumerate(self.matrix):
            for a, p in enumerate(row):
                total = h + a
                if total > line:
                    over += p
                elif total < line:
                    under += p
                else:
                    push += p
        return {"over": over, "under": under, "push": push}

    def asian_handicap(self, line: float) -> dict[str, float]:
        """Home team giving/receiving ``line`` goals. Negative favours the
        home side, matching how these are quoted."""
        home = away = push = 0.0
        for h, row in enumerate(self.matrix):
            for a, p in enumerate(row):
                margin = (h - a) + line
                if margin > 0:
                    home += p
                elif margin < 0:
                    away += p
                else:
                    push += p
        return {"home": home, "away": away, "push": push}

    def both_teams_to_score(self) -> dict[str, float]:
        yes = sum(
            p
            for h, row in enumerate(self.matrix)
            for a, p in enumerate(row)
            if h > 0 and a > 0
        )
        return {"yes": yes, "no": 1.0 - yes}

    def clean_sheet(self) -> dict[str, float]:
        return {
            "home": sum(self.matrix[h][0] for h in range(len(self.matrix))),
            "away": sum(self.matrix[0]),
        }

    def top_scorelines(self, n: int = 5) -> list[tuple[str, float]]:
        cells = [
            (f"{h}-{a}", p)
            for h, row in enumerate(self.matrix)
            for a, p in enumerate(row)
        ]
        cells.sort(key=lambda kv: -kv[1])
        return cells[:n]


class DixonColes:
    """Fit team strengths from historical scorelines and price any market.

    >>> model = DixonColes()
    >>> _ = model.fit([Match("A", "B", 2, 0), Match("B", "A", 1, 1)])
    >>> sl = model.predict("A", "B")
    >>> abs(sum(sum(row) for row in sl.matrix) - 1.0) < 1e-6
    True
    """

    def __init__(
        self,
        max_goals: int = 12,
        xi: float = 0.0045,
        rho_bounds: tuple[float, float] = (-0.25, 0.25),
    ) -> None:
        self.max_goals = max_goals
        self.xi = xi
        self.rho_bounds = rho_bounds
        self.attack: dict[str, float] = {}
        self.defence: dict[str, float] = {}
        self.home_advantage: float = 0.25
        self.rho: float = 0.0
        self.baseline: float = 0.0  # log of league mean goals per team
        self.n_matches: int = 0
        self.log_likelihood: float = float("-nan")

    # -- fitting --------------------------------------------------------

    def _weights(self, matches: Sequence[Match]) -> list[float]:
        stamps = [m.when for m in matches if m.when is not None]
        if not stamps or self.xi <= 0:
            return [1.0] * len(matches)
        latest = max(
            s if isinstance(s, datetime) else datetime.combine(s, datetime.min.time())
            for s in stamps
        )
        weights = []
        for m in matches:
            if m.when is None:
                weights.append(1.0)
                continue
            when = (
                m.when
                if isinstance(m.when, datetime)
                else datetime.combine(m.when, datetime.min.time())
            )
            days = max(0.0, (latest - when).total_seconds() / 86400.0)
            weights.append(math.exp(-self.xi * days))
        return weights

    def fit(
        self,
        matches: Sequence[Match],
        iterations: int = 400,
        learning_rate: float = 0.06,
        ridge: float = 0.02,
    ) -> "DixonColes":
        """Estimate attack, defence, home advantage and rho.

        ``ridge`` shrinks team strengths toward the league average, which keeps
        a team with three games from being rated as an outlier. Leave it on
        unless every team has 30-plus matches.
        """
        matches = list(matches)
        if len(matches) < 2:
            raise ValueError("need at least two matches to fit")
        teams = sorted({m.home for m in matches} | {m.away for m in matches})
        weights = self._weights(matches)

        total_goals = sum(m.home_goals + m.away_goals for m in matches)
        mean_goals = max(0.05, total_goals / (2.0 * len(matches)))
        self.baseline = math.log(mean_goals)

        self.attack = {t: 0.0 for t in teams}
        self.defence = {t: 0.0 for t in teams}
        home_adv = 0.25

        # Adam keeps this well behaved without any tuning per dataset.
        m1: dict[str, float] = {}
        m2: dict[str, float] = {}
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        def step(key: str, grad: float, t: int) -> float:
            m1[key] = beta1 * m1.get(key, 0.0) + (1 - beta1) * grad
            m2[key] = beta2 * m2.get(key, 0.0) + (1 - beta2) * grad * grad
            mhat = m1[key] / (1 - beta1**t)
            vhat = m2[key] / (1 - beta2**t)
            return learning_rate * mhat / (math.sqrt(vhat) + eps)

        for it in range(1, iterations + 1):
            g_attack = {t: 0.0 for t in teams}
            g_defence = {t: 0.0 for t in teams}
            g_home = 0.0

            for match, w in zip(matches, weights):
                adv = 0.0 if match.neutral else home_adv
                lam = math.exp(
                    self.baseline + self.attack[match.home] - self.defence[match.away] + adv
                )
                mu = math.exp(
                    self.baseline + self.attack[match.away] - self.defence[match.home]
                )
                res_h = w * (match.home_goals - lam)
                res_a = w * (match.away_goals - mu)
                g_attack[match.home] += res_h
                g_defence[match.away] -= res_h
                g_attack[match.away] += res_a
                g_defence[match.home] -= res_a
                if not match.neutral:
                    g_home += res_h

            for t in teams:
                g_attack[t] -= ridge * self.attack[t]
                g_defence[t] -= ridge * self.defence[t]

            for t in teams:
                self.attack[t] += step(f"a:{t}", g_attack[t], it)
                self.defence[t] += step(f"d:{t}", g_defence[t], it)
            home_adv += step("home", g_home, it)

            # Identifiability: attack and defence are only defined up to a
            # shared shift, so re-centre each one on zero every iteration.
            a_mean = sum(self.attack.values()) / len(teams)
            d_mean = sum(self.defence.values()) / len(teams)
            for t in teams:
                self.attack[t] -= a_mean
                self.defence[t] -= d_mean
            self.baseline += a_mean - d_mean

        self.home_advantage = home_adv
        self.n_matches = len(matches)
        self.rho = self._fit_rho(matches, weights)
        self.log_likelihood = self._loglik(matches, weights, self.rho)
        return self

    def _rates(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        adv = 0.0 if neutral else self.home_advantage
        lam = math.exp(
            self.baseline
            + self.attack.get(home, 0.0)
            - self.defence.get(away, 0.0)
            + adv
        )
        mu = math.exp(
            self.baseline + self.attack.get(away, 0.0) - self.defence.get(home, 0.0)
        )
        return lam, mu

    _tau = staticmethod(tau_correction)

    def _loglik(
        self, matches: Sequence[Match], weights: Sequence[float], rho: float
    ) -> float:
        total = 0.0
        for match, w in zip(matches, weights):
            lam, mu = self._rates(match.home, match.away, match.neutral)
            ll = _log_poisson(min(match.home_goals, 59), lam) + _log_poisson(
                min(match.away_goals, 59), mu
            )
            ll += math.log(
                self._tau(match.home_goals, match.away_goals, lam, mu, rho)
            )
            total += w * ll
        return total

    def _fit_rho(self, matches: Sequence[Match], weights: Sequence[float]) -> float:
        """Golden-section search on the one-dimensional corrected likelihood."""
        lo, hi = self.rho_bounds
        phi = (math.sqrt(5.0) - 1.0) / 2.0
        a, b = lo, hi
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        fc = self._loglik(matches, weights, c)
        fd = self._loglik(matches, weights, d)
        for _ in range(60):
            if fc > fd:
                b, d, fd = d, c, fc
                c = b - phi * (b - a)
                fc = self._loglik(matches, weights, c)
            else:
                a, c, fc = c, d, fd
                d = a + phi * (b - a)
                fd = self._loglik(matches, weights, d)
            if abs(b - a) < 1e-6:
                break
        return 0.5 * (a + b)

    # -- prediction -----------------------------------------------------

    def predict(
        self,
        home: str,
        away: str,
        neutral: bool = False,
        home_xg: float | None = None,
        away_xg: float | None = None,
    ) -> Scoreline:
        """Joint scoreline distribution for one fixture.

        Pass ``home_xg``/``away_xg`` to override the fitted rates with your own
        expected-goals numbers -- useful when you have injury or lineup
        information the fitted strengths cannot know about.
        """
        lam, mu = self._rates(home, away, neutral)
        if home_xg is not None:
            lam = float(home_xg)
        if away_xg is not None:
            mu = float(away_xg)

        n = self.max_goals + 1
        matrix = [[0.0] * n for _ in range(n)]
        for h in range(n):
            ph = _poisson(h, lam)
            for a in range(n):
                matrix[h][a] = ph * _poisson(a, mu) * self._tau(h, a, lam, mu, self.rho)

        # Renormalise: the truncation at max_goals and the tau correction both
        # cost a little mass, and every derived market assumes it sums to one.
        total = sum(sum(row) for row in matrix)
        if total <= 0:
            raise ValueError("degenerate scoreline distribution")
        matrix = [[v / total for v in row] for row in matrix]
        return Scoreline(home=home, away=away, matrix=matrix, home_xg=lam, away_xg=mu)

    def strengths(self) -> list[dict]:
        """Team parameters, strongest attack first. ``attack`` and ``defence``
        are in log-goals: +0.3 attack means roughly 35% more goals scored than
        an average team, and a *higher* defence value means a better defence."""
        rows = [
            {
                "team": t,
                "attack": self.attack[t],
                "defence": self.defence[t],
                "net": self.attack[t] + self.defence[t],
            }
            for t in self.attack
        ]
        rows.sort(key=lambda r: -r["net"])
        return rows
