"""Streaks and trends over a team's recent games -- and whether they mean anything.

"They have gone over 4.5 shots on target in five straight games" is the kind of
line that sells betting tips. This module finds those streaks for you, and then
does the thing the tip sellers never do: it asks how often a streak that long
would happen by pure chance.

Two separate traps sit behind every trend, and both are handled here.

**A streak is not a rate.** If a team's true rate is 75%, then five in a row
happens 24% of the time. That is not rare, it is Tuesday. Every trend here is
reported with the probability of seeing at least that many hits given the
team's underlying rate, so you can tell a real signal from an ordinary run of
luck.

**You are looking at hundreds of trends at once.** Check forty conditions
across ten teams and you have run four hundred lotteries; some of them will
come up 5-from-5 no matter what. That is called multiple testing, and it is why
trend screens feel so productive and pay so badly. :func:`scan_trends` reports
how many conditions it examined and how many perfect streaks you should have
expected to find by chance alone, so a screen that turns up six 5/5 streaks
when chance predicts five is correctly unimpressive.

None of this makes trends useless. It makes them a *screen* -- a way to decide
what to look at -- rather than a reason to bet. Whether a bet is good is
decided by the price, and that lives in :mod:`claudebet.scan`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

from .form import MatchLog, MatchRecord

__all__ = [
    "Condition",
    "TrendResult",
    "hit_rate",
    "team_trends",
    "scan_trends",
    "binomial_at_least",
    "standard_conditions",
]


def binomial_at_least(hits: int, trials: int, prob: float) -> float:
    """P(at least ``hits`` successes in ``trials``) at rate ``prob``.

    This is the number that deflates most streaks. Five from five at a true
    75% rate has a probability of 0.24 -- nothing has been discovered.
    """
    if trials <= 0:
        return 1.0
    prob = min(max(prob, 1e-12), 1 - 1e-12)
    total = 0.0
    for k in range(int(hits), trials + 1):
        log_coef = (
            math.lgamma(trials + 1) - math.lgamma(k + 1) - math.lgamma(trials - k + 1)
        )
        total += math.exp(
            log_coef + k * math.log(prob) + (trials - k) * math.log(1 - prob)
        )
    return min(1.0, total)


@dataclass(frozen=True)
class Condition:
    """Something that either happened or did not, in a given match.

    ``market_key`` links the condition to a bet you can actually place, using
    the same grammar :mod:`claudebet.scan` uses to read your odds file --
    ``"corners_9.5"``, ``"shots_home_4.5"``, ``"btts"``. ``None`` means the
    trend is descriptive only and has no matching market.
    """

    name: str
    test: Callable[[MatchRecord, bool], bool | None]
    market_key: str | None = None
    selection: str | None = None


@dataclass
class TrendResult:
    """One condition, measured over a team's recent games."""

    team: str
    condition: str
    hits: int
    played: int
    window: int
    baseline: float  # the team's underlying rate, from the whole log
    sequence: str  # most recent last, e.g. "YYNYY"
    market_key: str | None = None
    selection: str | None = None

    @property
    def rate(self) -> float:
        return self.hits / self.played if self.played else 0.0

    @property
    def luck_probability(self) -> float:
        """Chance of a run this good (or better) if nothing special is going
        on. Large values mean the streak is unremarkable."""
        return binomial_at_least(self.hits, self.played, self.baseline)

    @property
    def is_notable(self) -> bool:
        """Would this streak be surprising at the team's normal rate?"""
        return self.luck_probability < 0.05

    def line(self) -> str:
        verdict = (
            "genuinely unusual"
            if self.is_notable
            else f"ordinary -- happens {self.luck_probability:.0%} of the time anyway"
        )
        return (
            f"{self.condition:<38} {self.hits}/{self.played}  "
            f"[{self.sequence}]  season rate {self.baseline:.0%}  -> {verdict}"
        )


def hit_rate(
    log: MatchLog, team: str, condition: Condition, window: int | None = None
) -> tuple[int, int, str]:
    """Count how often a condition held in a team's last ``window`` matches."""
    games = log.last(team, window) if window else log.appearances(team)
    hits = played = 0
    letters = []
    for record, is_home in games:
        outcome = condition.test(record, is_home)
        if outcome is None:
            continue
        played += 1
        if outcome:
            hits += 1
            letters.append("Y")
        else:
            letters.append("N")
    return hits, played, "".join(letters)


def _over_team(stat: str, line: float, side: str | None = None):
    """Condition factory: a team's own count of ``stat`` exceeds ``line``."""

    def test(record: MatchRecord, is_home: bool) -> bool | None:
        if side == "home" and not is_home:
            return None
        if side == "away" and is_home:
            return None
        value = record.value(stat, is_home)
        return None if value is None else value > line

    return test


def _over_match(stat: str, line: float):
    """Condition factory: both teams' counts of ``stat`` combined exceed
    ``line``."""

    def test(record: MatchRecord, is_home: bool) -> bool | None:
        total = record.total(stat)
        return None if total is None else total > line

    return test


def _btts(record: MatchRecord, is_home: bool) -> bool | None:
    h = record.home_stats.get("goals")
    a = record.away_stats.get("goals")
    if h is None or a is None:
        return None
    return h > 0 and a > 0


def _team_wins(record: MatchRecord, is_home: bool) -> bool | None:
    result = record.result
    if result == "?":
        return None
    return result == ("H" if is_home else "A")


def _team_unbeaten(record: MatchRecord, is_home: bool) -> bool | None:
    result = record.result
    if result == "?":
        return None
    return result == "D" or result == ("H" if is_home else "A")


def _clean_sheet(record: MatchRecord, is_home: bool) -> bool | None:
    conceded = record.value("goals", not is_home)
    return None if conceded is None else conceded == 0


def standard_conditions(
    stats: Sequence[str],
    team_lines: dict[str, Sequence[float]] | None = None,
    match_lines: dict[str, Sequence[float]] | None = None,
) -> list[Condition]:
    """Build the usual set of trends for whatever statistics you have.

    Lines default to sensible ones for goals, corners and cards, and for any
    other statistic you can pass your own. Team-level and match-level versions
    of each are generated, because books price both.
    """
    team_lines = dict(team_lines or {})
    match_lines = dict(match_lines or {})
    defaults_team = {
        "goals": (0.5, 1.5, 2.5),
        "corners": (3.5, 4.5, 5.5, 6.5),
        "cards": (0.5, 1.5, 2.5),
    }
    defaults_match = {
        "goals": (1.5, 2.5, 3.5),
        "corners": (8.5, 9.5, 10.5, 11.5),
        "cards": (2.5, 3.5, 4.5, 5.5),
    }

    conditions: list[Condition] = [
        Condition("wins the match", _team_wins, "1X2", None),
        Condition("avoids defeat", _team_unbeaten, "dc", None),
        Condition("keeps a clean sheet", _clean_sheet, None, None),
    ]
    if "goals" in stats:
        conditions.append(Condition("both teams score", _btts, "btts", "yes"))

    for stat in stats:
        for line in team_lines.get(stat, defaults_team.get(stat, ())):
            conditions.append(
                Condition(
                    f"team over {line:g} {stat}",
                    _over_team(stat, line),
                    f"{stat}_TEAMSIDE_{line:g}",
                    "over",
                )
            )
        for line in match_lines.get(stat, defaults_match.get(stat, ())):
            conditions.append(
                Condition(
                    f"match over {line:g} {stat}",
                    _over_match(stat, line),
                    f"{stat}_{line:g}",
                    "over",
                )
            )
    return conditions


def team_trends(
    log: MatchLog,
    team: str,
    window: int = 5,
    conditions: Sequence[Condition] | None = None,
    min_hits: int | None = None,
) -> list[TrendResult]:
    """Every trend for one team, strongest first.

    ``min_hits`` filters to streaks of at least that many hits. The default
    keeps anything that hit in all but one game of the window, which is what
    people mean by "a trend".
    """
    if conditions is None:
        conditions = standard_conditions(log.stats())
    if min_hits is None:
        min_hits = max(1, window - 1)

    results: list[TrendResult] = []
    for condition in conditions:
        hits, played, sequence = hit_rate(log, team, condition, window)
        if played == 0 or hits < min_hits:
            continue
        # Baseline from the team's whole history, which is what makes the
        # luck calculation meaningful -- comparing a streak to the same games
        # that produced it would always look impressive.
        base_hits, base_played, _ = hit_rate(log, team, condition, None)
        baseline = (base_hits + 1.0) / (base_played + 2.0) if base_played else 0.5
        results.append(
            TrendResult(
                team=team,
                condition=condition.name,
                hits=hits,
                played=played,
                window=window,
                baseline=baseline,
                sequence=sequence,
                market_key=condition.market_key,
                selection=condition.selection,
            )
        )
    results.sort(key=lambda r: (r.luck_probability, -r.rate))
    return results


def scan_trends(
    log: MatchLog,
    teams: Sequence[str] | None = None,
    window: int = 5,
    conditions: Sequence[Condition] | None = None,
) -> dict:
    """Run the trend screen across many teams and report honestly on it.

    The ``expected_by_chance`` figure is the point of this function. It is the
    number of perfect streaks a screen this size would throw up even if every
    team were completely ordinary. Compare it to ``perfect_streaks`` before
    getting excited about anything in the list.
    """
    if conditions is None:
        conditions = standard_conditions(log.stats())
    teams = list(teams) if teams else log.teams()

    all_results: list[TrendResult] = []
    checked = 0
    expected_perfect = 0.0
    for team in teams:
        for condition in conditions:
            hits, played, sequence = hit_rate(log, team, condition, window)
            if played == 0:
                continue
            checked += 1
            base_hits, base_played, _ = hit_rate(log, team, condition, None)
            baseline = (base_hits + 1.0) / (base_played + 2.0) if base_played else 0.5
            expected_perfect += baseline**played
            if hits < max(1, window - 1):
                continue
            all_results.append(
                TrendResult(
                    team=team,
                    condition=condition.name,
                    hits=hits,
                    played=played,
                    window=window,
                    baseline=baseline,
                    sequence=sequence,
                    market_key=condition.market_key,
                    selection=condition.selection,
                )
            )

    all_results.sort(key=lambda r: (r.luck_probability, -r.rate))
    perfect = [r for r in all_results if r.hits == r.played]
    notable = [r for r in all_results if r.is_notable]
    return {
        "trends": all_results,
        "notable": notable,
        "conditions_checked": checked,
        "perfect_streaks": len(perfect),
        "expected_by_chance": expected_perfect,
        "window": window,
    }
