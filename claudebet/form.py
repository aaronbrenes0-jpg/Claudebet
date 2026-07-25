"""Recent form: turning the last 5 or 10 games into probabilities.

You give this a match log -- dates, teams, and whatever columns you have: goals,
corners, cards, shots, anything -- and it produces a probability for every
market you can find on a betting slip. Match result, over/under at any goal
line, both teams to score, Asian handicaps, team totals, corner and card lines,
correct score.

The hard part is not the arithmetic, it is that **five games is almost no
data**. A team averaging 7.0 corners across five matches has a standard error
near 1.2 corners; the difference between "this team gets 7 corners" and "this
team gets 5.8 corners" is invisible at that sample size, and it is exactly the
difference that decides a 10.5 line. Reading raw last-5 averages as if they
were true rates is the single most reliable way to lose money with form data.

Three corrections do the work, and none of them are optional:

**Shrinkage toward the season, not toward the league.** A team's last-5 rate is
pulled back toward its own longer-run level, by an amount derived from the data
via empirical Bayes: how much do teams genuinely differ, versus how much would
they appear to differ from noise alone? If the spread across teams is no wider
than sampling noise predicts, form gets almost no weight. Nothing is
hand-tuned.

**Opponent adjustment.** Seven corners against the league's worst defence is
not seven corners against the best. Every count is scaled by who it came
against and where it was played, so a soft run of fixtures does not read as
form.

**Honest distributions.** Corners and cards are overdispersed -- their variance
exceeds their mean -- so pricing them as Poisson understates both tails and
makes extreme lines look like value. Dispersion is measured from your data and
a negative binomial is used when it is warranted. See :mod:`claudebet.counts`.

One thing this deliberately does *not* do: derive match-result probabilities
from recent win/loss records. Wins are a coarse, low-information summary of a
match, and five of them is a sample of five. Goals scored and conceded carry
far more signal per game, so the 1X2 price is built from the goal rates and the
win/draw/loss record is reported as description only. That is not a shortcut,
it is the more accurate route.

Finally: recent form is a *weak* predictor of match results once long-run
strength is accounted for -- this is one of the better-replicated findings in
the field. It carries more signal for corners and cards, which are driven by
stable team style, than for who wins. The blending machinery in
:mod:`claudebet.blend` exists precisely so this model moves the market price by
an amount its track record justifies, which at the start is very little.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterable, Mapping, Sequence

from .counts import CountModel, NegativeBinomial, Poisson
from .models.poisson import Scoreline, scoreline_from_rates, tau_correction

__all__ = [
    "MatchRecord",
    "MatchLog",
    "TeamForm",
    "StatModel",
    "FormModel",
    "MatchProjection",
    "DEFAULT_GOAL_LINES",
    "DEFAULT_CORNER_LINES",
    "DEFAULT_CARD_LINES",
]

DEFAULT_GOAL_LINES = (0.5, 1.5, 2.5, 3.5, 4.5)
DEFAULT_CORNER_LINES = (7.5, 8.5, 9.5, 10.5, 11.5, 12.5)
DEFAULT_CARD_LINES = (2.5, 3.5, 4.5, 5.5)
_OVERDISPERSION_THRESHOLD = 1.15
_MIN_MATCHES_TO_FIT = 4


def _as_datetime(value: date | datetime | None) -> datetime:
    if value is None:
        return datetime.min
    if isinstance(value, datetime):
        return value
    return datetime.combine(value, datetime.min.time())


@dataclass
class MatchRecord:
    """One completed match, with whatever statistics you have for it.

    ``home_stats`` and ``away_stats`` are free-form: ``{"goals": 2, "corners":
    7, "cards": 3}``. Only ``goals`` is treated specially, because the match
    result and every goal-derived market come from it. Any other key you supply
    is modelled and priced the same way.
    """

    home: str
    away: str
    home_stats: dict[str, float]
    away_stats: dict[str, float]
    when: date | datetime | None = None
    competition: str = ""
    neutral: bool = False

    @property
    def result(self) -> str:
        """``"H"``, ``"D"`` or ``"A"`` from the goals, if present."""
        h = self.home_stats.get("goals")
        a = self.away_stats.get("goals")
        if h is None or a is None:
            return "?"
        return "H" if h > a else ("A" if a > h else "D")

    def value(self, stat: str, home_side: bool) -> float | None:
        source = self.home_stats if home_side else self.away_stats
        v = source.get(stat)
        return None if v is None else float(v)

    def total(self, stat: str) -> float | None:
        h, a = self.home_stats.get(stat), self.away_stats.get(stat)
        if h is None or a is None:
            return None
        return float(h) + float(a)


@dataclass
class MatchLog:
    """A collection of match records, ordered oldest first."""

    records: list[MatchRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.records.sort(key=lambda r: _as_datetime(r.when))

    def __len__(self) -> int:
        return len(self.records)

    def add(self, record: MatchRecord) -> None:
        self.records.append(record)
        self.records.sort(key=lambda r: _as_datetime(r.when))

    def teams(self) -> list[str]:
        found: set[str] = set()
        for r in self.records:
            found.add(r.home)
            found.add(r.away)
        return sorted(found)

    def stats(self) -> list[str]:
        """Stat names present on both sides of at least one match."""
        found: set[str] = set()
        for r in self.records:
            found |= set(r.home_stats) & set(r.away_stats)
        return sorted(found)

    def appearances(self, team: str) -> list[tuple[MatchRecord, bool]]:
        """Every match involving ``team``, with a flag for whether they were
        at home. Oldest first."""
        return [
            (r, r.home == team)
            for r in self.records
            if r.home == team or r.away == team
        ]

    def last(self, team: str, n: int) -> list[tuple[MatchRecord, bool]]:
        """The team's most recent ``n`` matches, oldest first."""
        games = self.appearances(team)
        return games[-n:] if n > 0 else games

    def team_form(self, team: str, window: int = 5) -> "TeamForm":
        """Plain recent-form summary: results and raw per-game rates.

        Description only, and deliberately available without fitting anything
        -- you should be able to look at a team's last five games even when
        there is nowhere near enough data to model them.
        """
        recent = self.last(team, window)
        wins = draws = losses = 0
        scored = conceded = 0.0
        letters: list[str] = []
        totals: dict[str, list[float]] = {}

        for record, is_home in recent:
            for stat in record.home_stats:
                if stat not in record.away_stats:
                    continue
                mine = record.value(stat, is_home)
                theirs = record.value(stat, not is_home)
                if mine is None or theirs is None:
                    continue
                totals.setdefault(stat, [0.0, 0.0])
                totals[stat][0] += mine
                totals[stat][1] += theirs
            result = record.result
            if result == "?":
                continue
            if result == "D":
                draws += 1
                letters.append("D")
            elif (result == "H") == is_home:
                wins += 1
                letters.append("W")
            else:
                losses += 1
                letters.append("L")
            scored += record.value("goals", is_home) or 0.0
            conceded += record.value("goals", not is_home) or 0.0

        played = len(recent)
        divisor = max(1, played)
        return TeamForm(
            team=team,
            window=window,
            played=played,
            wins=wins,
            draws=draws,
            losses=losses,
            scored=scored / divisor,
            conceded=conceded / divisor,
            per_stat={
                name: {"for": vals[0] / divisor, "against": vals[1] / divisor}
                for name, vals in totals.items()
            },
            results="".join(letters),
        )


@dataclass
class TeamForm:
    """Descriptive summary of a team's recent run. Reporting, not modelling."""

    team: str
    window: int
    played: int
    wins: int
    draws: int
    losses: int
    scored: float
    conceded: float
    per_stat: dict[str, dict[str, float]] = field(default_factory=dict)
    results: str = ""  # e.g. "WWDLW", most recent last

    @property
    def points(self) -> int:
        return 3 * self.wins + self.draws

    @property
    def points_per_game(self) -> float:
        return self.points / self.played if self.played else 0.0

    def line(self) -> str:
        head = (
            f"{self.team:<22} last {self.played:>2}: "
            f"{self.wins}W-{self.draws}D-{self.losses}L "
            f"({self.points_per_game:.2f} ppg, {self.results})"
        )
        parts = [
            f"{name} {vals['for']:.2f}/{vals['against']:.2f}"
            for name, vals in sorted(self.per_stat.items())
        ]
        return head + ("  |  " + "  ".join(parts) if parts else "")


_NO_POOLING = 1e6
_DEFAULT_ALPHA = 0.01


def _chi2_critical(df: int, alpha: float) -> float:
    """Upper ``alpha`` critical value of a chi-square, Wilson-Hilferty.

    Accurate to well under a percent for any ``df`` above about two, which is
    far more than a gate on believing form needs, and it avoids a dependency.
    """
    from .correlation import norm_ppf

    z = norm_ppf(1.0 - alpha)
    term = 1.0 - 2.0 / (9.0 * df) + z * math.sqrt(2.0 / (9.0 * df))
    return df * term**3


def _empirical_bayes_k(
    observed: Sequence[float],
    expected: Sequence[float],
    dispersion: float = 1.0,
    alpha: float = _DEFAULT_ALPHA,
) -> float:
    """Shrinkage strength for a set of counts, in units of expected count.

    Returns ``k`` for the posterior mean ``(C + k) / (E + k)``. Large ``k``
    means "these teams are indistinguishable, trust the prior"; small ``k``
    means the differences are real and the data should speak.

    The method is a two-step: gate, then estimate.

    **Gate.** Under the null that every team sits exactly at its expected rate,
    the Pearson statistic ``sum((C-E)^2 / (phi·E))`` follows a chi-square on
    ``n-1`` degrees of freedom. If the observed value does not clear the upper
    ``alpha`` critical point, the apparent spread is consistent with pure noise
    and ``k`` is returned at the no-pooling sentinel -- every multiplier
    collapses to exactly 1. This is what keeps a five-game window from
    inventing form, and it fails at a stated rate rather than an unknown one.

    **Estimate.** Past the gate, the between-team variance follows from
    matching moments: ``Var(C) = phi·E + tau^2·E^2``, so

        tau^2 = (sum((C-E)^2 / E) - phi·(n-1)) / sum(E)

    and ``k = 1 / tau^2``, which is the Gamma prior with mean 1 and variance
    ``tau^2``.

    ``dispersion`` is ``phi``, the measured variance-to-mean ratio. Passing 1.0
    for corners, which run near 1.5, would under-count the noise by a third and
    hand the difference to form that is not there.
    """
    pairs = [(c, e) for c, e in zip(observed, expected) if e > 1e-9]
    n = len(pairs)
    if n < 3:
        return _NO_POOLING
    phi = max(1.0, dispersion)
    df = n - 1

    pearson = sum((c - e) ** 2 / e for c, e in pairs)
    if pearson / phi <= _chi2_critical(df, alpha):
        return _NO_POOLING

    tau2 = (pearson - phi * df) / sum(e for _, e in pairs)
    if tau2 <= 1e-9:
        return _NO_POOLING
    return max(0.5, min(_NO_POOLING, 1.0 / tau2))


@dataclass
class StatModel:
    """Fitted multiplicative model for one statistic.

    ``attack`` above 1 means the team produces more of this stat than average;
    ``concede`` above 1 means it allows more of it. Both are on a ratio scale
    around 1, and both already carry the season-long shrinkage.
    """

    stat: str
    baseline: float  # league mean per team per match
    home_factor: float
    away_factor: float
    attack: dict[str, float] = field(default_factory=dict)
    concede: dict[str, float] = field(default_factory=dict)
    form_attack: dict[str, float] = field(default_factory=dict)
    form_concede: dict[str, float] = field(default_factory=dict)
    dispersion_team: float = 1.0
    dispersion_total: float = 1.0
    n_matches: int = 0
    shrinkage_season: float = 0.0
    shrinkage_form: float = 0.0

    def expected(
        self, home: str, away: str, neutral: bool = False, use_form: bool = True
    ) -> tuple[float, float]:
        """Expected count for each side in a fixture."""
        hf = 1.0 if neutral else self.home_factor
        af = 1.0 if neutral else self.away_factor
        a_home = self.attack.get(home, 1.0)
        a_away = self.attack.get(away, 1.0)
        c_home = self.concede.get(home, 1.0)
        c_away = self.concede.get(away, 1.0)
        if use_form:
            a_home *= self.form_attack.get(home, 1.0)
            a_away *= self.form_attack.get(away, 1.0)
            c_home *= self.form_concede.get(home, 1.0)
            c_away *= self.form_concede.get(away, 1.0)
        return (
            self.baseline * a_home * c_away * hf,
            self.baseline * a_away * c_home * af,
        )

    def distribution(self, mean: float, total: bool = False) -> CountModel:
        """Poisson or negative binomial, whichever the measured dispersion
        justifies."""
        phi = self.dispersion_total if total else self.dispersion_team
        if phi <= _OVERDISPERSION_THRESHOLD:
            return Poisson(mean)
        return NegativeBinomial(mean, phi * mean)


class FormModel:
    """Fit recent-form-aware rates for every statistic in a match log.

    Two layers are fitted per statistic. The *season* layer uses the whole log
    to estimate each team's underlying attack and concede multipliers, plus the
    league baseline and home advantage. The *form* layer then looks only at each
    team's last ``window`` matches and asks whether they have over- or
    under-performed their own season level, shrinking that answer toward "no
    change" by an amount the data decides.

    Splitting it this way is deliberate. Opponent strength and home advantage
    are far better estimated from the full history, while form is by definition
    a recent-window question -- fitting both on five games would produce
    opponent adjustments as noisy as the thing they are meant to correct.

    >>> log = MatchLog([
    ...     MatchRecord("A", "B", {"goals": 2}, {"goals": 1}),
    ...     MatchRecord("B", "A", {"goals": 0}, {"goals": 3}),
    ... ])
    >>> model = FormModel(window=5).fit(log)
    >>> lam, mu = model.stat_models["goals"].expected("A", "B")
    >>> lam > mu
    True
    """

    def __init__(
        self,
        window: int = 5,
        stats: Sequence[str] | None = None,
        iterations: int = 12,
        max_goals: int = 12,
        alpha: float = _DEFAULT_ALPHA,
    ) -> None:
        self.window = window
        self.requested_stats = list(stats) if stats else None
        self.iterations = iterations
        self.max_goals = max_goals
        self.alpha = alpha
        self.stat_models: dict[str, StatModel] = {}
        self.goal_rho: float = 0.0
        self.log: MatchLog | None = None

    # -- fitting --------------------------------------------------------

    def fit(self, log: MatchLog) -> "FormModel":
        if not log.records:
            raise ValueError("match log is empty")
        self.log = log
        names = self.requested_stats or log.stats()
        if not names:
            raise ValueError(
                "no statistic appears on both sides of any match; check that "
                "your columns are paired, e.g. home_corners and away_corners"
            )
        self.stat_models = {}
        for name in names:
            model = self._fit_stat(log, name)
            if model is not None:
                self.stat_models[name] = model
        if not self.stat_models:
            raise ValueError(
                f"no statistic has the {_MIN_MATCHES_TO_FIT} complete matches "
                "needed to fit. Descriptive form is still available without "
                "fitting: use MatchLog.team_form(team, window)."
            )
        if "goals" in self.stat_models:
            self.goal_rho = self._fit_rho(log)
        return self

    def _fit_stat(self, log: MatchLog, stat: str) -> StatModel | None:
        rows = [
            (r, float(r.home_stats[stat]), float(r.away_stats[stat]))
            for r in log.records
            if stat in r.home_stats and stat in r.away_stats
        ]
        if len(rows) < _MIN_MATCHES_TO_FIT:
            return None

        n = len(rows)
        home_mean = sum(h for _, h, _ in rows) / n
        away_mean = sum(a for _, _, a in rows) / n
        baseline = 0.5 * (home_mean + away_mean)
        if baseline <= 0:
            return None
        # Neutral-venue matches carry no venue signal, so the factors are
        # estimated from the rest and simply not applied to them.
        home_factor = home_mean / baseline
        away_factor = away_mean / baseline

        teams = log.teams()

        def fit_multipliers(dispersion: float):
            attack = {t: 1.0 for t in teams}
            concede = {t: 1.0 for t in teams}
            k_attack = k_concede = _NO_POOLING
            for iteration in range(self.iterations):
                obs_for = {t: 0.0 for t in teams}
                exp_for = {t: 0.0 for t in teams}
                obs_against = {t: 0.0 for t in teams}
                exp_against = {t: 0.0 for t in teams}

                for record, hv, av in rows:
                    hf = 1.0 if record.neutral else home_factor
                    af = 1.0 if record.neutral else away_factor
                    h, a = record.home, record.away
                    obs_for[h] += hv
                    obs_against[a] += hv
                    exp_for[h] += baseline * concede[a] * hf
                    exp_against[a] += baseline * attack[h] * hf
                    obs_for[a] += av
                    obs_against[h] += av
                    exp_for[a] += baseline * concede[h] * af
                    exp_against[h] += baseline * attack[a] * af

                if iteration == 0:
                    k_attack = _empirical_bayes_k(
                        [obs_for[t] for t in teams],
                        [exp_for[t] for t in teams],
                        dispersion,
                        self.alpha,
                    )
                    k_concede = _empirical_bayes_k(
                        [obs_against[t] for t in teams],
                        [exp_against[t] for t in teams],
                        dispersion,
                        self.alpha,
                    )

                for t in teams:
                    if exp_for[t] > 0:
                        attack[t] = (obs_for[t] + k_attack) / (exp_for[t] + k_attack)
                    if exp_against[t] > 0:
                        concede[t] = (obs_against[t] + k_concede) / (
                            exp_against[t] + k_concede
                        )

                # Resolve the scale degeneracy: attack and concede are only
                # identified up to a common multiplicative shift.
                mean_attack = sum(attack.values()) / len(attack)
                if mean_attack > 0:
                    for t in teams:
                        attack[t] /= mean_attack
                        concede[t] *= mean_attack
            return attack, concede, k_attack

        def build(attack, concede, k_attack) -> StatModel:
            return StatModel(
                stat=stat,
                baseline=baseline,
                home_factor=home_factor,
                away_factor=away_factor,
                attack=attack,
                concede=concede,
                n_matches=n,
                shrinkage_season=k_attack,
            )

        def measure_dispersion(model: StatModel) -> tuple[float, float]:
            """Pearson dispersion of the fitted model, per team and per match
            total. This is what decides Poisson versus negative binomial, and
            it also corrects the shrinkage on the second pass."""
            team_chi = team_n = 0.0
            total_chi = total_n = 0.0
            for record, hv, av in rows:
                eh, ea = model.expected(
                    record.home, record.away, record.neutral, use_form=False
                )
                for observed, expected in ((hv, eh), (av, ea)):
                    if expected > 0:
                        team_chi += (observed - expected) ** 2 / expected
                        team_n += 1
                expected_total = eh + ea
                if expected_total > 0:
                    total_chi += (hv + av - expected_total) ** 2 / expected_total
                    total_n += 1
            params = 2 * len(teams)
            return (
                team_chi / max(1.0, team_n - params) if team_n > params else 1.0,
                total_chi / max(1.0, total_n - params) if total_n > params else 1.0,
            )

        # Two passes: the first assumes Poisson noise to get a dispersion
        # estimate, the second uses it. Overdispersed counts look like large
        # team differences to a Poisson-assuming shrinkage rule, so skipping
        # this systematically under-shrinks corners and cards.
        provisional = build(*fit_multipliers(1.0))
        phi_team, phi_total = measure_dispersion(provisional)
        model = build(*fit_multipliers(phi_team))
        model.dispersion_team, model.dispersion_total = measure_dispersion(model)

        self._fit_form_layer(log, model, rows)
        return model

    def _fit_form_layer(
        self,
        log: MatchLog,
        model: StatModel,
        rows: Sequence[tuple[MatchRecord, float, float]],
    ) -> None:
        """Compare each team's last ``window`` matches to its own season level."""
        by_record = {id(r): (hv, av) for r, hv, av in rows}
        teams = log.teams()

        obs_for: dict[str, float] = {}
        exp_for: dict[str, float] = {}
        obs_against: dict[str, float] = {}
        exp_against: dict[str, float] = {}

        for team in teams:
            recent = [
                (r, is_home)
                for r, is_home in log.last(team, self.window)
                if id(r) in by_record
            ]
            of = ef = oa = ea = 0.0
            for record, is_home in recent:
                hv, av = by_record[id(record)]
                eh, eaway = model.expected(
                    record.home, record.away, record.neutral, use_form=False
                )
                if is_home:
                    of += hv
                    ef += eh
                    oa += av
                    ea += eaway
                else:
                    of += av
                    ef += eaway
                    oa += hv
                    ea += eh
            obs_for[team], exp_for[team] = of, ef
            obs_against[team], exp_against[team] = oa, ea

        k_form_attack = _empirical_bayes_k(
            [obs_for[t] for t in teams],
            [exp_for[t] for t in teams],
            model.dispersion_team,
            self.alpha,
        )
        k_form_concede = _empirical_bayes_k(
            [obs_against[t] for t in teams],
            [exp_against[t] for t in teams],
            model.dispersion_team,
            self.alpha,
        )
        model.shrinkage_form = k_form_attack

        for team in teams:
            ef, ea = exp_for[team], exp_against[team]
            # When the estimator finds no detectable form, say so exactly.
            # Returning 0.99999 instead of 1.0 would leave a phantom signal in
            # every downstream price and make use_form=True quietly differ from
            # use_form=False even when the data supports no difference at all.
            model.form_attack[team] = (
                1.0
                if k_form_attack >= _NO_POOLING or ef <= 0
                else (obs_for[team] + k_form_attack) / (ef + k_form_attack)
            )
            model.form_concede[team] = (
                1.0
                if k_form_concede >= _NO_POOLING or ea <= 0
                else (obs_against[team] + k_form_concede) / (ea + k_form_concede)
            )

    def _fit_rho(self, log: MatchLog) -> float:
        """Low-score correction for goals, by golden-section search."""
        rows = [
            r
            for r in log.records
            if "goals" in r.home_stats and "goals" in r.away_stats
        ]
        if len(rows) < 20:
            return 0.0
        goals = self.stat_models["goals"]
        cached = []
        for r in rows:
            lam, mu = goals.expected(r.home, r.away, r.neutral, use_form=False)
            cached.append((int(r.home_stats["goals"]), int(r.away_stats["goals"]), lam, mu))

        def loglik(rho: float) -> float:
            total = 0.0
            for h, a, lam, mu in cached:
                total += math.log(max(1e-12, tau_correction(h, a, lam, mu, rho)))
            return total

        phi = (math.sqrt(5.0) - 1.0) / 2.0
        lo, hi = -0.2, 0.2
        c, d = hi - phi * (hi - lo), lo + phi * (hi - lo)
        fc, fd = loglik(c), loglik(d)
        for _ in range(50):
            if fc > fd:
                hi, d, fd = d, c, fc
                c = hi - phi * (hi - lo)
                fc = loglik(c)
            else:
                lo, c, fc = c, d, fd
                d = lo + phi * (hi - lo)
                fd = loglik(d)
            if hi - lo < 1e-5:
                break
        return 0.5 * (lo + hi)

    # -- description ----------------------------------------------------

    def team_form(self, team: str, window: int | None = None) -> TeamForm:
        """Descriptive form summary. See :meth:`MatchLog.team_form`."""
        if self.log is None:
            raise ValueError("fit a match log first")
        return self.log.team_form(team, self.window if window is None else window)

    # -- projection -----------------------------------------------------

    def project(
        self, home: str, away: str, neutral: bool = False, use_form: bool = True
    ) -> "MatchProjection":
        """Full probability set for one fixture."""
        if self.log is None:
            raise ValueError("fit a match log first")
        rates: dict[str, tuple[float, float]] = {}
        for name, model in self.stat_models.items():
            rates[name] = model.expected(home, away, neutral, use_form)
        return MatchProjection(
            home=home,
            away=away,
            rates=rates,
            models=self.stat_models,
            rho=self.goal_rho,
            max_goals=self.max_goals,
            used_form=use_form,
            window=self.window,
        )


@dataclass
class MatchProjection:
    """Every market for one fixture, derived from the fitted rates."""

    home: str
    away: str
    rates: dict[str, tuple[float, float]]
    models: Mapping[str, StatModel]
    rho: float = 0.0
    max_goals: int = 12
    used_form: bool = True
    window: int = 5

    def __post_init__(self) -> None:
        self._scoreline: Scoreline | None = None

    # -- goals ----------------------------------------------------------

    @property
    def has_goals(self) -> bool:
        return "goals" in self.rates

    def scoreline(self) -> Scoreline:
        if not self.has_goals:
            raise ValueError("no goals data: cannot build a scoreline")
        if self._scoreline is None:
            lam, mu = self.rates["goals"]
            self._scoreline = scoreline_from_rates(
                self.home, self.away, lam, mu, self.rho, self.max_goals
            )
        return self._scoreline

    def match_odds(self) -> dict[str, float]:
        return self.scoreline().match_odds()

    def double_chance(self) -> dict[str, float]:
        o = self.match_odds()
        return {
            "1X": o["home"] + o["draw"],
            "12": o["home"] + o["away"],
            "X2": o["draw"] + o["away"],
        }

    def btts(self) -> dict[str, float]:
        return self.scoreline().both_teams_to_score()

    def asian_handicap(self, line: float) -> dict[str, float]:
        return self.scoreline().asian_handicap(line)

    def correct_score(self, n: int = 6) -> list[tuple[str, float]]:
        return self.scoreline().top_scorelines(n)

    def clean_sheet(self) -> dict[str, float]:
        return self.scoreline().clean_sheet()

    def win_to_nil(self) -> dict[str, float]:
        matrix = self.scoreline().matrix
        return {
            "home": sum(matrix[h][0] for h in range(1, len(matrix))),
            "away": sum(matrix[0][a] for a in range(1, len(matrix[0]))),
        }

    # -- any counted statistic -------------------------------------------

    def total_distribution(self, stat: str) -> CountModel:
        """Distribution of the match total for a statistic."""
        if stat not in self.rates:
            raise KeyError(f"no model for {stat!r}; have {sorted(self.rates)}")
        home_rate, away_rate = self.rates[stat]
        return self.models[stat].distribution(home_rate + away_rate, total=True)

    def team_distribution(self, stat: str, side: str) -> CountModel:
        if stat not in self.rates:
            raise KeyError(f"no model for {stat!r}; have {sorted(self.rates)}")
        if side not in ("home", "away"):
            raise ValueError("side must be 'home' or 'away'")
        rate = self.rates[stat][0 if side == "home" else 1]
        return self.models[stat].distribution(rate, total=False)

    def totals(self, stat: str, lines: Iterable[float]) -> dict[float, dict[str, float]]:
        """Over/under prices across several lines for a match total.

        Goals use the scoreline (so the correlation between the two teams is
        respected); everything else uses the fitted count distribution.
        """
        out: dict[float, dict[str, float]] = {}
        if stat == "goals" and self.has_goals:
            sl = self.scoreline()
            for line in lines:
                out[line] = sl.totals(line)
            return out
        dist = self.total_distribution(stat)
        for line in lines:
            out[line] = dist.total_market(line)
        return out

    def team_totals(
        self, stat: str, lines: Iterable[float]
    ) -> dict[str, dict[float, dict[str, float]]]:
        out: dict[str, dict[float, dict[str, float]]] = {"home": {}, "away": {}}
        for side in ("home", "away"):
            dist = self.team_distribution(stat, side)
            for line in lines:
                out[side][line] = dist.total_market(line)
        return out

    def expected(self, stat: str) -> tuple[float, float]:
        return self.rates[stat]

    # -- packaging -------------------------------------------------------

    def two_way(self, market: dict[str, float], a: str, b: str) -> dict[str, float]:
        """Renormalise a market that can push into the two-way price you can
        actually bet, conditional on the bet not being voided."""
        total = market[a] + market[b]
        if total <= 0:
            return {a: 0.5, b: 0.5}
        return {a: market[a] / total, b: market[b] / total}

    def markets(
        self,
        goal_lines: Sequence[float] = DEFAULT_GOAL_LINES,
        corner_lines: Sequence[float] = DEFAULT_CORNER_LINES,
        card_lines: Sequence[float] = DEFAULT_CARD_LINES,
    ) -> dict[str, dict]:
        """Everything at once, keyed by market name.

        Feed any two-way entry straight into :func:`claudebet.pipeline.analyze`
        as ``model_probs``.
        """
        out: dict[str, dict] = {"expected_rates": dict(self.rates)}
        if self.has_goals:
            out["1X2"] = self.match_odds()
            out["double_chance"] = self.double_chance()
            out["btts"] = self.btts()
            out["clean_sheet"] = self.clean_sheet()
            out["win_to_nil"] = self.win_to_nil()
            out["correct_score"] = dict(self.correct_score(8))
            out["goals"] = self.totals("goals", goal_lines)
            out["team_goals"] = self.team_totals("goals", (0.5, 1.5, 2.5))
            out["asian_handicap"] = {
                line: self.asian_handicap(line)
                for line in (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)
            }
        for stat, lines in (("corners", corner_lines), ("cards", card_lines)):
            if stat in self.rates:
                out[stat] = self.totals(stat, lines)
                out[f"team_{stat}"] = self.team_totals(
                    stat, (3.5, 4.5, 5.5, 6.5) if stat == "corners" else (0.5, 1.5, 2.5)
                )
        for stat in self.rates:
            if stat in ("goals", "corners", "cards"):
                continue
            mean = sum(self.rates[stat])
            span = max(0.5, round(mean) - 2.5)
            out[stat] = self.totals(
                stat, [span + i for i in range(6)]
            )
        return out

    def report(self, top_lines: int = 3) -> str:
        lines = [
            f"== {self.home} v {self.away} ==",
            f"projection uses the last {self.window} matches"
            if self.used_form
            else "projection uses season-long strength only",
        ]
        for stat, (h, a) in sorted(self.rates.items()):
            phi = self.models[stat].dispersion_total
            note = "" if phi <= _OVERDISPERSION_THRESHOLD else f"  [overdispersed x{phi:.2f}]"
            lines.append(
                f"  expected {stat:<10} {h:5.2f} - {a:5.2f}   total {h + a:5.2f}{note}"
            )
        if self.has_goals:
            o = self.match_odds()
            lines.append(
                f"  1X2        {o['home']:.1%} / {o['draw']:.1%} / {o['away']:.1%}"
            )
            btts = self.btts()
            lines.append(f"  BTTS       {btts['yes']:.1%}")
            for line, market in self.totals("goals", (1.5, 2.5, 3.5)).items():
                lines.append(
                    f"  goals {line:<4} over {market['over']:.1%}  "
                    f"under {market['under']:.1%}"
                )
            lines.append(
                "  top scores " + ", ".join(
                    f"{s} {p:.1%}" for s, p in self.correct_score(4)
                )
            )
        for stat in ("corners", "cards"):
            if stat not in self.rates:
                continue
            dist = self.total_distribution(stat)
            shown = 0
            for line in (
                DEFAULT_CORNER_LINES if stat == "corners" else DEFAULT_CARD_LINES
            ):
                if abs(line - dist.mean) > 2.5 or shown >= top_lines:
                    continue
                market = dist.total_market(line)
                lines.append(
                    f"  {stat} {line:<5} over {market['over']:.1%}  "
                    f"under {market['under']:.1%}"
                )
                shown += 1
        return "\n".join(lines)
