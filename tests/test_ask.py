import math
import random
from datetime import date, timedelta

import pytest

from claudebet.ask import (
    check_price,
    match_team,
    odds_card,
    parse_fixture,
    parse_market_phrase,
    required_price,
)
from claudebet.form import FormModel, MatchLog, MatchRecord
from claudebet.scan import ScanConfig

TEAMS = ["Inter Miami", "Chicago Fire", "Orlando City", "Atlanta United",
         "Real Salt Lake", "LA Galaxy"]


def poisson_sample(rng, lam):
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1
        if k > 40:
            return k


def league(n=500, seed=12):
    rng = random.Random(seed)
    atk = {t: 1.28 - 0.09 * i for i, t in enumerate(TEAMS)}
    con = {t: 0.80 + 0.09 * i for i, t in enumerate(TEAMS)}
    records, start = [], date(2025, 1, 1)
    for i in range(n):
        h, a = rng.sample(TEAMS, 2)
        tempo = math.exp(rng.gauss(0, 0.25))
        records.append(MatchRecord(
            h, a,
            {"goals": poisson_sample(rng, 1.35 * atk[h] * con[a] * 1.15),
             "corners": poisson_sample(rng, 5.4 * tempo),
             "shots_on_target": poisson_sample(rng, 5.0 * atk[h])},
            {"goals": poisson_sample(rng, 1.35 * atk[a] * con[h] / 1.15),
             "corners": poisson_sample(rng, 4.6 * tempo),
             "shots_on_target": poisson_sample(rng, 4.0 * atk[a])},
            when=start + timedelta(days=i // 4)))
    return MatchLog(records)


@pytest.fixture(scope="module")
def model():
    return FormModel(window=5).fit(league())


@pytest.fixture(scope="module")
def projection(model):
    return model.project("Inter Miami", "Chicago Fire")


STATS = ["corners", "goals", "shots_on_target"]


class TestTeamMatching:
    def test_exact_and_case_insensitive(self):
        assert match_team("Inter Miami", TEAMS) == "Inter Miami"
        assert match_team("inter miami", TEAMS) == "Inter Miami"

    def test_partial_names(self):
        assert match_team("miami", TEAMS) == "Inter Miami"
        assert match_team("chicago", TEAMS) == "Chicago Fire"
        assert match_team("galaxy", TEAMS) == "LA Galaxy"

    def test_typos(self):
        assert match_team("chigago fire", TEAMS) == "Chicago Fire"

    def test_blank_and_stub_queries_match_nothing(self):
        # Regression: an empty leftover substring-matches every team, which
        # silently turned "over 2.5" into one team's total.
        assert match_team("", TEAMS) is None
        assert match_team("   ", TEAMS) is None
        assert match_team(".", TEAMS) is None
        assert match_team("a", TEAMS) is None

    def test_unknown_team(self):
        assert match_team("Barcelona", TEAMS) is None


class TestFixtureParsing:
    @pytest.mark.parametrize("text", [
        "Inter Miami vs Chicago Fire",
        "inter miami v chicago fire",
        "miami versus chicago",
        "Inter Miami - Chicago Fire",
        "miami against chicago",
    ])
    def test_separators(self, text):
        assert parse_fixture(text, TEAMS) == ("Inter Miami", "Chicago Fire")

    def test_rejects_nonsense(self):
        assert parse_fixture("hello there", TEAMS) is None
        assert parse_fixture("miami vs miami", TEAMS) is None
        assert parse_fixture("just one team", TEAMS) is None


class TestMarketPhrases:
    def test_bare_over_under_is_the_match_total(self):
        # Regression: this used to be read as a team total.
        phrase = parse_market_phrase("over 2.5", STATS, "Inter Miami", "Chicago Fire")
        assert phrase.key == "goals_2.5"
        assert phrase.selection == "over"

    def test_named_statistic(self):
        phrase = parse_market_phrase("corners over 9.5", STATS, "Inter Miami", "Chicago Fire")
        assert phrase.key == "corners_9.5"
        assert parse_market_phrase("under 9.5 corners", STATS).key == "corners_9.5"
        assert parse_market_phrase("under 9.5 corners", STATS).selection == "under"

    def test_multi_word_statistic_beats_its_prefix(self):
        phrase = parse_market_phrase("shots on target over 4.5", STATS,
                                     "Inter Miami", "Chicago Fire")
        assert phrase.key == "shots_on_target_4.5"

    def test_team_totals_by_name_and_by_side(self):
        named = parse_market_phrase("shots on target miami over 4.5", STATS,
                                    "Inter Miami", "Chicago Fire")
        assert named.key == "shots_on_target_home_4.5"
        sided = parse_market_phrase("corners away over 4.5", STATS,
                                    "Inter Miami", "Chicago Fire")
        assert sided.key == "corners_away_4.5"

    def test_btts(self):
        assert parse_market_phrase("btts", STATS).selection == "yes"
        assert parse_market_phrase("btts no", STATS).selection == "no"
        assert parse_market_phrase("both teams to score", STATS).key == "btts"

    def test_match_result(self):
        assert parse_market_phrase("home", STATS).selection == "home"
        assert parse_market_phrase("draw", STATS).selection == "draw"
        by_name = parse_market_phrase("miami to win", STATS,
                                      "Inter Miami", "Chicago Fire")
        assert by_name.key == "1X2" and by_name.selection == "home"

    def test_handicap(self):
        phrase = parse_market_phrase("ah -0.5", STATS)
        assert phrase.key == "ah_-0.5" and phrase.selection == "home"

    def test_prices_in_several_notations(self):
        assert parse_market_phrase("over 2.5 @ 1.85", STATS).odds == pytest.approx(1.85)
        assert parse_market_phrase("over 2.5 @ +150", STATS).odds == pytest.approx(2.5)
        assert parse_market_phrase("over 2.5 = 1.85", STATS).odds == pytest.approx(1.85)

    def test_both_sides_of_a_price(self):
        phrase = parse_market_phrase("corners over 9.5 @ 2.60/1.55", STATS)
        assert phrase.odds == pytest.approx(2.60)
        assert phrase.other_odds == pytest.approx(1.55)

    def test_no_price_is_fine(self):
        assert parse_market_phrase("over 2.5", STATS).odds is None

    def test_gibberish(self):
        assert parse_market_phrase("banana", STATS) is None
        assert parse_market_phrase("", STATS) is None


class TestRequiredPrice:
    def test_sits_above_the_fair_price(self):
        config = ScanConfig()
        for prob in (0.35, 0.5, 0.65):
            threshold = required_price(prob, 0.40, config)
            assert threshold is not None
            assert threshold > 1.0 / prob

    def test_more_trust_means_a_lower_bar(self):
        config = ScanConfig()
        trusting = required_price(0.55, 0.60, config)
        sceptical = required_price(0.55, 0.20, config)
        assert trusting < sceptical

    def test_a_bigger_required_edge_raises_the_bar(self):
        lenient = required_price(0.55, 0.40, ScanConfig(min_edge=0.01))
        strict = required_price(0.55, 0.40, ScanConfig(min_edge=0.08))
        assert strict > lenient

    def test_the_threshold_actually_clears(self):
        # Whatever it returns must genuinely be a bet when offered.
        config = ScanConfig()
        prob, trust = 0.55, 0.40
        threshold = required_price(prob, trust, config)
        from claudebet.ask import _ASSUMED_MARGIN
        from claudebet.blend import expit, logit

        book = (1.0 / threshold) / (_ASSUMED_MARGIN / 2.0 + 0.5)
        blended = expit(trust * logit(prob) + (1 - trust) * logit(book))
        used = expit(logit(blended) - config.confidence_z * config.extra_stderr)
        assert used - 1.0 / threshold == pytest.approx(config.min_edge, abs=1e-3)

    def test_impossible_probabilities_return_none(self):
        assert required_price(0.0, 0.4, ScanConfig()) is None
        assert required_price(1.0, 0.4, ScanConfig()) is None


class TestOddsCard:
    def test_renders_the_expected_sections(self, projection):
        card = odds_card(projection, ScanConfig(), STATS)
        assert "Inter Miami v Chicago Fire" in card
        assert "MATCH RESULT" in card
        assert "GOALS" in card
        assert "CORNERS" in card
        assert "TAKE ABOVE" in card

    def test_take_above_is_never_below_fair(self, projection):
        card = odds_card(projection, ScanConfig(), STATS)
        for line in card.split("\n"):
            parts = line.split()
            if len(parts) >= 3 and parts[-1] not in ("never",) and "%" in line:
                try:
                    take, fair = float(parts[-1]), float(parts[-2])
                except ValueError:
                    continue
                assert take >= fair


class TestCheckPrice:
    def test_a_generous_price_is_a_bet(self, projection):
        phrase = parse_market_phrase("corners over 9.5 @ 6.00/1.10", STATS)
        verdict = check_price(projection, phrase, ScanConfig(), bankroll=200)
        assert not isinstance(verdict, str)
        assert verdict.decision in ("bet", "check")

    def test_a_short_price_is_refused(self, projection):
        # Short enough to be bad value, but close enough to our own number
        # that it is an EV judgement rather than a suspected data problem.
        phrase = parse_market_phrase("corners over 9.5 @ 1.50/2.55", STATS)
        verdict = check_price(projection, phrase, ScanConfig(), bankroll=200)
        assert verdict.decision == "no-edge", verdict.reason
        assert verdict.ev < 0
        assert verdict.stake == 0

    def test_missing_price_is_explained(self, projection):
        phrase = parse_market_phrase("over 2.5", STATS)
        assert isinstance(check_price(projection, phrase, ScanConfig()), str)

    def test_unknown_market_is_explained(self, projection):
        phrase = parse_market_phrase("offsides over 3.5", ["offsides"])
        result = check_price(projection, phrase, ScanConfig())
        assert isinstance(result, str)
        assert "no model" in result

    def test_two_sided_price_removes_the_margin_exactly(self, projection):
        one = parse_market_phrase("corners over 9.5 @ 2.10", STATS)
        two = parse_market_phrase("corners over 9.5 @ 2.10/1.75", STATS)
        a = check_price(projection, one, ScanConfig())
        b = check_price(projection, two, ScanConfig())
        # Same offered price, different inferred book opinion.
        assert a.odds == b.odds
        assert a.blended_prob != b.blended_prob

    def test_one_sided_flag_suggests_supplying_the_other_side(self, projection):
        phrase = parse_market_phrase("corners over 9.5 @ 12.0", STATS)
        verdict = check_price(projection, phrase, ScanConfig())
        if verdict.decision == "check":
            assert "other side" in verdict.reason

    def test_render_includes_the_verdict(self, projection):
        phrase = parse_market_phrase("corners over 9.5 @ 1.50/2.55", STATS)
        text = check_price(projection, phrase, ScanConfig(), 200).render(200)
        assert "we make it" in text
        assert "NO-EDGE" in text, text


class TestAskCli:
    def _log_file(self, tmp_path):
        log = league(n=400)
        path = tmp_path / "matches.csv"
        header = ("date,home,away,home_goals,away_goals,home_corners,away_corners,"
                  "home_shots_on_target,away_shots_on_target\n")
        rows = [
            f"{r.when.isoformat()},{r.home},{r.away},"
            f"{int(r.home_stats['goals'])},{int(r.away_stats['goals'])},"
            f"{int(r.home_stats['corners'])},{int(r.away_stats['corners'])},"
            f"{int(r.home_stats['shots_on_target'])},"
            f"{int(r.away_stats['shots_on_target'])}"
            for r in log.records
        ]
        path.write_text(header + "\n".join(rows) + "\n")
        return path

    def test_one_shot_card(self, tmp_path, capsys):
        from claudebet.cli import main

        path = self._log_file(tmp_path)
        assert main(["ask", str(path), "--home", "miami", "--away", "chicago"]) == 0
        out = capsys.readouterr().out
        assert "Inter Miami v Chicago Fire" in out
        assert "TAKE ABOVE" in out

    def test_one_shot_with_prices(self, tmp_path, capsys):
        from claudebet.cli import main

        path = self._log_file(tmp_path)
        assert main(["ask", str(path), "--home", "miami", "--away", "chicago",
                     "--price", "over 2.5 @ 1.85",
                     "--price", "corners over 9.5 @ 2.10/1.75",
                     "--bankroll", "200"]) == 0
        out = capsys.readouterr().out
        assert "we make it" in out
        assert "goals over 2.5" in out

    def test_unreadable_price_is_reported(self, tmp_path, capsys):
        from claudebet.cli import main

        path = self._log_file(tmp_path)
        assert main(["ask", str(path), "--home", "miami", "--away", "chicago",
                     "--price", "banana"]) == 0
        assert "could not read" in capsys.readouterr().out

    def test_unknown_team_exits_cleanly(self, tmp_path):
        from claudebet.cli import main

        path = self._log_file(tmp_path)
        with pytest.raises(SystemExit, match="could not find"):
            main(["ask", str(path), "--home", "Barcelona", "--away", "chicago"])

    def test_interactive_session(self, tmp_path, capsys, monkeypatch):
        from claudebet.cli import main

        path = self._log_file(tmp_path)
        script = iter([
            "teams",
            "help",
            "over 2.5 @ 1.85",          # before naming a game
            "miami vs chicago",
            "corners over 9.5 @ 2.10/1.75",
            "banana",
            "quit",
        ])
        monkeypatch.setattr("builtins.input", lambda _="": next(script))
        assert main(["ask", str(path), "--bankroll", "200"]) == 0
        out = capsys.readouterr().out
        assert "Inter Miami" in out
        assert "name a game first" in out
        assert "did not understand" in out
        assert "we make it" in out

    def test_interactive_handles_eof(self, tmp_path, monkeypatch):
        from claudebet.cli import main

        path = self._log_file(tmp_path)

        def raise_eof(_=""):
            raise EOFError

        monkeypatch.setattr("builtins.input", raise_eof)
        assert main(["ask", str(path)]) == 0
