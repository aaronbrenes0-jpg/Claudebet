import math
import random
from datetime import date, timedelta

import pytest

from claudebet.counts import (
    Discrete,
    NegativeBinomial,
    Poisson,
    convolve,
    dispersion_ratio,
    fit_counts,
)
from claudebet.data.sources import load_match_log, write_match_log_template
from claudebet.form import FormModel, MatchLog, MatchRecord


def poisson_sample(rng: random.Random, lam: float) -> int:
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1
        if k > 60:
            return k


TEAMS = ["Alianza", "Cristal", "Universitario", "Melgar", "Cienciano", "Boys",
         "Garcilaso", "Grau", "Huancayo", "Vallejo", "Binacional", "Municipal"]

TRUE_ATTACK = {t: 1.35 - 0.06 * i for i, t in enumerate(TEAMS)}
TRUE_CONCEDE = {t: 0.75 + 0.06 * i for i, t in enumerate(TEAMS)}
BASE_GOALS, BASE_CORNERS, HFA = 1.30, 5.0, 1.15


def synthetic_league(
    n: int = 700,
    seed: int = 5,
    corner_tempo: float = 0.28,
    form_swing: float = 0.0,
) -> MatchLog:
    """A league with known team strengths, overdispersed corners, and an
    optional genuine form swing confined to each team's recent games."""
    rng = random.Random(seed)
    hot = {
        t: (1.0 + form_swing if i % 2 == 0 else 1.0 - form_swing)
        for i, t in enumerate(TEAMS)
    }
    records = []
    start = date(2025, 1, 1)
    for i in range(n):
        h, a = rng.sample(TEAMS, 2)
        recent = i > n - len(TEAMS) * 3
        fh = hot[h] if recent else 1.0
        fa = hot[a] if recent else 1.0
        lam = BASE_GOALS * TRUE_ATTACK[h] * TRUE_CONCEDE[a] * HFA * fh
        mu = BASE_GOALS * TRUE_ATTACK[a] * TRUE_CONCEDE[h] / HFA * fa
        tempo = math.exp(rng.gauss(0, corner_tempo)) if corner_tempo else 1.0
        records.append(
            MatchRecord(
                home=h,
                away=a,
                home_stats={
                    "goals": poisson_sample(rng, lam),
                    "corners": poisson_sample(rng, BASE_CORNERS * HFA * tempo),
                    "cards": poisson_sample(rng, 2.0),
                },
                away_stats={
                    "goals": poisson_sample(rng, mu),
                    "corners": poisson_sample(rng, BASE_CORNERS / HFA * tempo),
                    "cards": poisson_sample(rng, 2.0),
                },
                when=start + timedelta(days=i // 4),
            )
        )
    return MatchLog(records)


def sustained_form_league(
    swing: float, seed: int = 4, n: int = 1100, tail_games: int = 40
) -> MatchLog:
    """A league where the form swing is sustained over each team's last
    ``tail_games`` matches, so windows of 5, 10 and 20 all sit inside it.

    Distinct from :func:`synthetic_league`, whose swing covers only about six
    games per team -- that one rewards a short window, this one lets window
    length be compared cleanly.
    """
    rng = random.Random(seed)
    hot = {t: (1 + swing if i % 2 == 0 else 1 - swing) for i, t in enumerate(TEAMS)}
    cutoff = n - (tail_games * len(TEAMS)) // 2
    records, start = [], date(2025, 1, 1)
    for i in range(n):
        h, a = rng.sample(TEAMS, 2)
        recent = i > cutoff
        fh, fa = (hot[h] if recent else 1.0), (hot[a] if recent else 1.0)
        records.append(
            MatchRecord(
                h, a,
                {"goals": poisson_sample(rng, 1.45 * fh)},
                {"goals": poisson_sample(rng, 1.15 * fa)},
                when=start + timedelta(days=i // 5),
            )
        )
    return MatchLog(records)


class TestCountModels:
    def test_poisson_pmf_is_a_distribution(self):
        p = Poisson(2.4)
        assert sum(p.pmf(k) for k in range(60)) == pytest.approx(1.0, abs=1e-9)
        assert p.pmf(0) == pytest.approx(math.exp(-2.4))
        assert p.mean == pytest.approx(2.4)
        assert p.variance == pytest.approx(2.4)

    def test_negative_binomial_matches_its_moments(self):
        nb = NegativeBinomial(9.5, 14.0)
        mean = sum(k * nb.pmf(k) for k in range(200))
        var = sum((k - mean) ** 2 * nb.pmf(k) for k in range(200))
        assert mean == pytest.approx(9.5, rel=1e-3)
        assert var == pytest.approx(14.0, rel=1e-2)

    def test_negative_binomial_collapses_to_poisson(self):
        # Variance barely above the mean should look almost exactly Poisson.
        nb = NegativeBinomial(4.0, 4.0000001)
        p = Poisson(4.0)
        for k in range(15):
            assert nb.pmf(k) == pytest.approx(p.pmf(k), abs=1e-6)

    def test_overdispersion_fattens_both_tails(self):
        p, nb = Poisson(10.0), NegativeBinomial(10.0, 16.0)
        assert nb.over(14.5) > p.over(14.5)
        assert nb.under(5.5) > p.under(5.5)

    def test_total_market_is_exhaustive(self):
        for model in (Poisson(9.7), NegativeBinomial(9.7, 13.0)):
            for line in (8.5, 9.0, 9.5, 10.0):
                m = model.total_market(line)
                assert sum(m.values()) == pytest.approx(1.0, abs=1e-9)

    def test_half_lines_cannot_push(self):
        assert Poisson(9.0).push(9.5) == 0.0
        assert Poisson(9.0).push(9.0) == pytest.approx(Poisson(9.0).pmf(9))

    def test_whole_line_over_excludes_the_push(self):
        p = Poisson(9.0)
        assert p.over(9.0) == pytest.approx(p.sf(9))
        assert p.over(9.0) + p.pmf(9) == pytest.approx(p.over(8.5))

    def test_fit_picks_poisson_for_equidispersed_data(self):
        rng = random.Random(1)
        samples = [poisson_sample(rng, 3.0) for _ in range(4000)]
        assert isinstance(fit_counts(samples), Poisson)

    def test_fit_picks_negative_binomial_for_overdispersed_data(self):
        rng = random.Random(2)
        samples = [
            poisson_sample(rng, 3.0 * math.exp(rng.gauss(0, 0.5)))
            for _ in range(4000)
        ]
        model = fit_counts(samples)
        assert isinstance(model, NegativeBinomial)
        assert dispersion_ratio(samples) > 1.15

    def test_convolution_adds_means(self):
        total = convolve(Poisson(3.0), Poisson(4.0))
        assert total.mean == pytest.approx(7.0, rel=1e-6)
        # Two independent Poissons sum to a Poisson.
        reference = Poisson(7.0)
        for k in range(20):
            assert total.pmf(k) == pytest.approx(reference.pmf(k), abs=1e-9)

    def test_discrete_normalises(self):
        d = Discrete([1.0, 2.0, 1.0])
        assert sum(d.probs) == pytest.approx(1.0)
        assert d.mean == pytest.approx(1.0)

    def test_at_least(self):
        p = Poisson(2.0)
        assert p.at_least(1) == pytest.approx(1 - p.pmf(0))


class TestMatchLog:
    def test_orders_by_date_and_finds_teams(self):
        log = MatchLog([
            MatchRecord("B", "A", {"goals": 0}, {"goals": 1}, when=date(2026, 2, 1)),
            MatchRecord("A", "B", {"goals": 2}, {"goals": 1}, when=date(2026, 1, 1)),
        ])
        assert [r.home for r in log.records] == ["A", "B"]
        assert log.teams() == ["A", "B"]
        assert log.stats() == ["goals"]

    def test_results_are_read_from_goals(self):
        assert MatchRecord("A", "B", {"goals": 2}, {"goals": 1}).result == "H"
        assert MatchRecord("A", "B", {"goals": 1}, {"goals": 1}).result == "D"
        assert MatchRecord("A", "B", {"goals": 0}, {"goals": 1}).result == "A"
        assert MatchRecord("A", "B", {"corners": 5}, {"corners": 3}).result == "?"

    def test_last_n_takes_the_most_recent(self):
        log = MatchLog([
            MatchRecord("A", "B", {"goals": i}, {"goals": 0}, when=date(2026, 1, i + 1))
            for i in range(8)
        ])
        recent = log.last("A", 3)
        assert len(recent) == 3
        assert [r.home_stats["goals"] for r, _ in recent] == [5, 6, 7]

    def test_appearances_flag_home_and_away(self):
        log = MatchLog([
            MatchRecord("A", "B", {"goals": 1}, {"goals": 0}, when=date(2026, 1, 1)),
            MatchRecord("B", "A", {"goals": 2}, {"goals": 2}, when=date(2026, 1, 8)),
        ])
        assert [is_home for _, is_home in log.appearances("A")] == [True, False]

    def test_stats_needs_both_sides(self):
        log = MatchLog([
            MatchRecord("A", "B", {"goals": 1, "corners": 4}, {"goals": 0})
        ])
        assert log.stats() == ["goals"]


class TestTeamForm:
    def test_counts_results_from_both_perspectives(self):
        log = MatchLog([
            MatchRecord("A", "B", {"goals": 2}, {"goals": 0}, when=date(2026, 1, 1)),
            MatchRecord("B", "A", {"goals": 3}, {"goals": 1}, when=date(2026, 1, 8)),
            MatchRecord("A", "B", {"goals": 1}, {"goals": 1}, when=date(2026, 1, 15)),
        ])
        form = log.team_form("A")
        assert (form.wins, form.draws, form.losses) == (1, 1, 1)
        assert form.results == "WLD"
        assert form.points == 4
        assert form.points_per_game == pytest.approx(4 / 3)
        assert form.scored == pytest.approx(4 / 3)
        assert form.conceded == pytest.approx(4 / 3)

    def test_window_limits_the_summary(self):
        log = synthetic_league(n=200)
        model = FormModel(window=5).fit(log)
        assert model.team_form("Alianza", window=5).played <= 5
        assert model.team_form("Alianza", window=10).played <= 10

    def test_line_renders(self):
        model = FormModel(window=5).fit(synthetic_league(n=200))
        text = model.team_form("Alianza").line()
        assert "Alianza" in text and "ppg" in text


@pytest.fixture(scope="module")
def model():
    return FormModel(window=5).fit(synthetic_league())


class TestFormModelFitting:

    def test_recovers_baseline_and_home_advantage(self, model):
        goals = model.stat_models["goals"]
        assert goals.baseline == pytest.approx(BASE_GOALS, rel=0.20)
        assert goals.home_factor > goals.away_factor
        assert goals.home_factor == pytest.approx(1.14, abs=0.12)

    def test_recovers_the_ordering_of_team_strength(self, model):
        goals = model.stat_models["goals"]
        ranked = sorted(TEAMS, key=lambda t: -goals.attack[t])
        # Best and worst should land in the right halves of the table.
        assert ranked.index(TEAMS[0]) < len(TEAMS) / 2
        assert ranked.index(TEAMS[-1]) >= len(TEAMS) / 2

    def test_detects_overdispersion_only_where_it_exists(self, model):
        assert model.stat_models["goals"].dispersion_total < 1.15
        assert model.stat_models["corners"].dispersion_total > 1.15

    def test_uses_negative_binomial_for_overdispersed_stats(self, model):
        corners = model.stat_models["corners"]
        assert isinstance(corners.distribution(9.5, total=True), NegativeBinomial)
        goals = model.stat_models["goals"]
        assert isinstance(goals.distribution(2.6, total=True), Poisson)

    def test_no_form_signal_is_fully_discounted(self):
        # Strengths are constant through the log, so any apparent hot streak is
        # noise. Every form multiplier must come back at exactly 1.
        model = FormModel(window=5).fit(synthetic_league(form_swing=0.0))
        for stat_model in model.stat_models.values():
            assert stat_model.shrinkage_form >= 1e5
            assert all(
                v == pytest.approx(1.0) for v in stat_model.form_attack.values()
            )

    def test_overdispersion_alone_does_not_create_phantom_form(self):
        # Corners here have heavy shared match-tempo noise but no team form.
        # A shrinkage rule that assumed Poisson would read that as form.
        model = FormModel(window=5).fit(
            synthetic_league(corner_tempo=0.35, form_swing=0.0)
        )
        corners = model.stat_models["corners"]
        assert corners.dispersion_team > 1.15
        assert all(v == pytest.approx(1.0) for v in corners.form_attack.values())

    def test_real_form_is_detected(self):
        # The swing in this fixture spans roughly the last six games per team,
        # so a five-game window is the one that lines up with it.
        model = FormModel(window=5).fit(synthetic_league(form_swing=0.40, seed=11))
        goals = model.stat_models["goals"]
        assert goals.shrinkage_form < 1e5
        spread = max(goals.form_attack.values()) - min(goals.form_attack.values())
        assert spread > 0.2

    @pytest.mark.parametrize("seed", [1, 3, 5, 7])
    def test_a_short_window_can_sit_under_the_noise_floor(self, seed):
        # A 20% swing sustained across each team's recent run. Measured over
        # this fixture, a five-game window never clears the gate while a
        # fifteen-game window does about half the time -- the shorter window
        # simply does not contain enough matches to separate the swing from
        # sampling noise. That is an information limit, not a tuning choice,
        # and the model reports it instead of producing a confident number.
        log = sustained_form_league(swing=0.20, seed=seed)
        short = FormModel(window=5).fit(log).stat_models["goals"]
        longer = FormModel(window=15).fit(log).stat_models["goals"]
        assert short.shrinkage_form >= 1e5
        assert all(v == pytest.approx(1.0) for v in short.form_attack.values())
        assert longer.shrinkage_form < 1e5

    def test_longer_windows_never_have_less_evidence(self):
        log = sustained_form_league(swing=0.30, seed=6)
        ks = [
            FormModel(window=w).fit(log).stat_models["goals"].shrinkage_form
            for w in (5, 10, 20)
        ]
        assert ks[0] >= ks[1] >= ks[2]

    def test_alpha_controls_how_readily_form_is_believed(self):
        log = sustained_form_league(swing=0.20, seed=1)
        strict = FormModel(window=15, alpha=1e-9).fit(log).stat_models["goals"]
        loose = FormModel(window=15, alpha=0.30).fit(log).stat_models["goals"]
        assert strict.shrinkage_form >= loose.shrinkage_form


class TestHeterogeneityGate:
    @pytest.mark.parametrize(
        "df,alpha,expected",
        [(5, 0.05, 11.070), (9, 0.05, 16.919), (19, 0.05, 30.144),
         (29, 0.05, 42.557), (19, 0.01, 36.191), (9, 0.01, 21.666)],
    )
    def test_chi_square_critical_values_match_published_tables(self, df, alpha, expected):
        from claudebet.form import _chi2_critical

        assert _chi2_critical(df, alpha) == pytest.approx(expected, abs=0.05)

    def test_identical_teams_produce_no_shrinkage_signal(self):
        from claudebet.form import _NO_POOLING, _empirical_bayes_k

        # Every team exactly at expectation: nothing to detect.
        assert _empirical_bayes_k([10.0] * 20, [10.0] * 20) == _NO_POOLING

    def test_wildly_different_teams_are_detected(self):
        from claudebet.form import _empirical_bayes_k

        observed = [2.0, 30.0] * 10
        k = _empirical_bayes_k(observed, [16.0] * 20)
        assert k < 100

    def test_overdispersion_raises_the_bar(self):
        from claudebet.form import _empirical_bayes_k

        observed = [8.0, 13.0] * 10
        expected = [10.0] * 20
        assumed_poisson = _empirical_bayes_k(observed, expected, dispersion=1.0)
        knowing_overdispersed = _empirical_bayes_k(observed, expected, dispersion=3.0)
        assert knowing_overdispersed >= assumed_poisson

    def test_too_few_groups_declines_to_guess(self):
        from claudebet.form import _NO_POOLING, _empirical_bayes_k

        assert _empirical_bayes_k([1.0, 40.0], [10.0, 10.0]) == _NO_POOLING

    def test_empty_log_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            FormModel().fit(MatchLog([]))

    def test_log_with_no_paired_stats_rejected(self):
        log = MatchLog([MatchRecord("A", "B", {"goals": 1}, {"corners": 2})])
        with pytest.raises(ValueError):
            FormModel().fit(log)

    def test_projection_requires_a_fit(self):
        with pytest.raises(ValueError, match="fit a match log"):
            FormModel().project("A", "B")


class TestProjection:

    def test_match_odds_are_a_distribution(self, model):
        odds = model.project("Alianza", "Municipal").match_odds()
        assert sum(odds.values()) == pytest.approx(1.0)
        assert all(0 < v < 1 for v in odds.values())

    def test_stronger_team_is_favoured_and_venue_matters(self, model):
        strong_home = model.project("Alianza", "Municipal").match_odds()
        strong_away = model.project("Municipal", "Alianza").match_odds()
        assert strong_home["home"] > strong_home["away"]
        assert strong_away["away"] > strong_away["home"]
        # Home advantage: the same fixture at a neutral venue favours the
        # stronger side slightly less.
        neutral = model.project("Alianza", "Municipal", neutral=True).match_odds()
        assert neutral["home"] < strong_home["home"]

    def test_double_chance_is_consistent_with_1x2(self, model):
        p = model.project("Alianza", "Cristal")
        o, dc = p.match_odds(), p.double_chance()
        assert dc["1X"] == pytest.approx(o["home"] + o["draw"])
        assert sum(dc.values()) == pytest.approx(2.0)

    def test_goal_totals_are_exhaustive_and_monotone(self, model):
        totals = model.project("Alianza", "Cristal").totals("goals", (1.5, 2.5, 3.5))
        for market in totals.values():
            assert sum(market.values()) == pytest.approx(1.0)
        assert totals[1.5]["over"] > totals[2.5]["over"] > totals[3.5]["over"]

    def test_corner_totals_use_the_overdispersed_model(self, model):
        p = model.project("Alianza", "Cristal")
        dist = p.total_distribution("corners")
        assert dist.variance > dist.mean
        for line, market in p.totals("corners", (8.5, 9.5, 10.5)).items():
            assert sum(market.values()) == pytest.approx(1.0)

    def test_btts_and_clean_sheets(self, model):
        p = model.project("Alianza", "Municipal")
        assert sum(p.btts().values()) == pytest.approx(1.0)
        cs = p.clean_sheet()
        # The stronger side keeps more clean sheets.
        assert cs["home"] > cs["away"]
        wtn = p.win_to_nil()
        assert wtn["home"] < cs["home"]  # winning to nil implies a clean sheet

    def test_asian_handicap_is_exhaustive(self, model):
        for line in (-1.5, -0.5, 0.5, 1.5):
            market = model.project("Alianza", "Cristal").asian_handicap(line)
            assert sum(market.values()) == pytest.approx(1.0)

    def test_team_totals(self, model):
        tt = model.project("Alianza", "Municipal").team_totals("goals", (0.5, 1.5))
        assert set(tt) == {"home", "away"}
        assert tt["home"][0.5]["over"] > tt["away"][0.5]["over"]

    def test_correct_score_is_ordered(self, model):
        scores = model.project("Alianza", "Municipal").correct_score(5)
        assert len(scores) == 5
        assert [p for _, p in scores] == sorted((p for _, p in scores), reverse=True)

    def test_markets_bundle_has_everything(self, model):
        markets = model.project("Alianza", "Cristal").markets()
        for key in ("1X2", "btts", "goals", "corners", "cards", "asian_handicap",
                    "double_chance", "team_goals", "correct_score",
                    "expected_rates"):
            assert key in markets, key

    def test_two_way_renormalises_away_the_push(self, model):
        p = model.project("Alianza", "Cristal")
        market = p.totals("goals", (3.0,))[3.0]
        assert market["push"] > 0
        two = p.two_way(market, "over", "under")
        assert sum(two.values()) == pytest.approx(1.0)

    def test_unknown_stat_raises(self, model):
        with pytest.raises(KeyError):
            model.project("Alianza", "Cristal").total_distribution("offsides")

    def test_disabling_form_changes_nothing_when_form_is_noise(self):
        model = FormModel(window=5).fit(synthetic_league(form_swing=0.0))
        with_form = model.project("Alianza", "Cristal", use_form=True).match_odds()
        without = model.project("Alianza", "Cristal", use_form=False).match_odds()
        assert with_form["home"] == pytest.approx(without["home"])

    def test_report_renders(self, model):
        text = model.project("Alianza", "Cristal").report()
        assert "1X2" in text and "expected goals" in text

    def test_arbitrary_stats_are_modelled_too(self):
        rng = random.Random(3)
        records = []
        for i in range(200):
            h, a = rng.sample(TEAMS[:6], 2)
            records.append(
                MatchRecord(h, a,
                            {"goals": poisson_sample(rng, 1.4),
                             "offsides": poisson_sample(rng, 2.2)},
                            {"goals": poisson_sample(rng, 1.1),
                             "offsides": poisson_sample(rng, 2.0)},
                            when=date(2025, 1, 1) + timedelta(days=i))
            )
        model = FormModel(window=5).fit(MatchLog(records))
        assert "offsides" in model.stat_models
        markets = model.project(TEAMS[0], TEAMS[1]).markets()
        assert "offsides" in markets


class TestMatchLogLoading:
    def test_template_round_trips(self, tmp_path):
        path = write_match_log_template(tmp_path / "m.csv")
        log = load_match_log(path)
        assert "goals" in log.stats() and "corners" in log.stats()
        assert "Alianza Lima" in log.teams()
        # The shipped template must be big enough to actually fit, or the
        # first thing a new user runs fails.
        model = FormModel(window=5).fit(log)
        assert set(model.stat_models) >= {"goals", "corners", "cards"}
        assert model.project("Alianza Lima", "Boys").match_odds()

    def test_paired_columns_become_stats(self, tmp_path):
        path = tmp_path / "m.csv"
        path.write_text(
            "date,home,away,home_goals,away_goals,home_corners,away_corners\n"
            "2026-01-01,A,B,2,1,7,4\n2026-01-08,B,A,0,0,5,5\n"
        )
        log = load_match_log(path)
        assert log.stats() == ["corners", "goals"]
        assert log.records[0].home_stats["corners"] == 7

    def test_unpaired_columns_are_ignored(self, tmp_path):
        path = tmp_path / "m.csv"
        path.write_text(
            "date,home,away,home_goals,away_goals,home_xg\n2026-01-01,A,B,2,1,1.7\n"
        )
        log = load_match_log(path)
        assert log.stats() == ["goals"]

    def test_missing_values_drop_only_that_pair(self, tmp_path):
        path = tmp_path / "m.csv"
        path.write_text(
            "date,home,away,home_goals,away_goals,home_corners,away_corners\n"
            "2026-01-01,A,B,2,1,7,4\n"
            "2026-01-08,B,A,1,1,,3\n"
        )
        log = load_match_log(path)
        assert "corners" not in log.records[1].home_stats
        assert log.records[1].home_stats["goals"] == 1

    def test_file_without_paired_columns_rejected(self, tmp_path):
        path = tmp_path / "m.csv"
        path.write_text("date,home,away,score\n2026-01-01,A,B,2\n")
        with pytest.raises(ValueError, match="paired"):
            load_match_log(path)

    def test_empty_file_rejected(self, tmp_path):
        path = tmp_path / "m.csv"
        path.write_text("date,home,away,home_goals,away_goals\n")
        with pytest.raises(ValueError, match="no rows"):
            load_match_log(path)


class TestFormCli:
    def _log_file(self, tmp_path):
        log = synthetic_league(n=300)
        path = tmp_path / "matches.csv"
        header = "date,home,away,home_goals,away_goals,home_corners,away_corners,home_cards,away_cards\n"
        rows = [
            f"{r.when.isoformat()},{r.home},{r.away},"
            f"{int(r.home_stats['goals'])},{int(r.away_stats['goals'])},"
            f"{int(r.home_stats['corners'])},{int(r.away_stats['corners'])},"
            f"{int(r.home_stats['cards'])},{int(r.away_stats['cards'])}"
            for r in log.records
        ]
        path.write_text(header + "\n".join(rows) + "\n")
        return path

    def test_form_command(self, tmp_path, capsys):
        from claudebet.cli import main

        path = self._log_file(tmp_path)
        assert main(["form", str(path), "--home", "Alianza",
                     "--away", "Municipal", "--last", "5"]) == 0
        out = capsys.readouterr().out
        assert "recent form (last 5)" in out
        assert "recent form (last 10)" in out
        assert "1X2" in out
        assert "form multiplier after shrinkage" in out

    def test_form_command_json(self, tmp_path, capsys):
        import json

        from claudebet.cli import main

        path = self._log_file(tmp_path)
        assert main(["form", str(path), "--home", "Alianza",
                     "--away", "Cristal", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out.split("\n\n")[-1])
        assert "1X2" in payload
        assert sum(payload["1X2"].values()) == pytest.approx(1.0)

    def test_unknown_team_is_reported(self, tmp_path):
        from claudebet.cli import main

        path = self._log_file(tmp_path)
        with pytest.raises(SystemExit, match="unknown team"):
            main(["form", str(path), "--home", "Nowhere", "--away", "Cristal"])

    def test_template_log_flag(self, tmp_path, capsys):
        from claudebet.cli import main

        assert main(["template", str(tmp_path / "m.csv"), "--log"]) == 0
        assert "claudebet form" in capsys.readouterr().out


class TestExampleDataGuard:
    def test_shipped_template_is_marked_as_example(self, tmp_path):
        path = write_match_log_template(tmp_path / "m.csv")
        log = load_match_log(path)
        assert log.is_example_data, (
            "the shipped sample must be detectable, or someone will bet on "
            "invented matches"
        )

    def test_real_looking_data_is_not_flagged(self, tmp_path):
        path = tmp_path / "m.csv"
        path.write_text(
            "date,home,away,home_goals,away_goals\n"
            "2026-01-01,A,B,2,1\n2026-01-08,B,A,0,0\n"
        )
        assert not load_match_log(path).is_example_data

    def test_clearing_the_marker_clears_the_flag(self, tmp_path):
        path = write_match_log_template(tmp_path / "m.csv")
        cleaned = path.read_text().replace("EXAMPLE-DATA-DO-NOT-BET", "")
        path.write_text(cleaned)
        assert not load_match_log(path).is_example_data

    def test_commands_shout_about_example_data(self, tmp_path, capsys):
        from claudebet.cli import main

        path = write_match_log_template(tmp_path / "m.csv")
        assert main(["ask", str(path), "--home", "Alianza Lima",
                     "--away", "Boys"]) == 0
        out = capsys.readouterr().out
        assert "EXAMPLE FILE" in out
        assert "made up" in out
