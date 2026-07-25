"""Ask about one match in plain language and get odds back.

This is the conversational front door. You name a fixture, it prints a full
card of fair prices; you type a price you have been offered, it tells you
whether to take it::

    > Inter Miami vs Chicago Fire
    > over 2.5 @ 1.85
    > corners over 9.5 @ 2.10
    > shots inter over 4.5 @ 1.30

Team names are matched loosely, so "miami" finds "Inter Miami", and markets are
read from ordinary phrasing rather than a fixed syntax.

The column that matters on the card is **TAKE ABOVE**: the price at which a bet
becomes worth making. It is not the fair price. It sits meaningfully above it,
for two reasons that both cost you money if ignored -- the estimate carries
error, and the answer gets pulled toward whatever the book thinks once you see
their number. Quoting the fair price alone would invite betting at prices that
lose slowly, so the card quotes the threshold you can actually act on.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Sequence

from .blend import expit, logit
from .form import FormModel, MatchProjection
from .odds import decimal_to_american, parse_odds
from .scan import ScanConfig, _trust_for, resolve_market

__all__ = [
    "match_team",
    "parse_market_phrase",
    "required_price",
    "odds_card",
    "check_price",
    "Verdict",
]

# What a two-way market at a recreational book typically holds. Used to guess
# the book's own opinion from a single quoted price, when the other side of the
# market has not been supplied.
_ASSUMED_MARGIN = 1.06


_MIN_FUZZY = 3


def match_team(name: str, teams: Sequence[str]) -> str | None:
    """Find the team someone meant. Exact, then substring, then near-miss.

    Anything shorter than three characters only matches exactly. Without that
    guard an empty leftover -- which is what "over 2.5" reduces to once the
    market words are stripped -- substring-matches every team in the league and
    silently turns a match total into one team's total.
    """
    cleaned = (name or "").strip().lower()
    if not cleaned:
        return None
    lookup = {t.lower(): t for t in teams}
    if cleaned in lookup:
        return lookup[cleaned]
    if len(cleaned) < _MIN_FUZZY:
        return None
    contains = [t for t in teams if cleaned in t.lower()]
    if len(contains) == 1:
        return contains[0]
    # Any word of the query matching a whole word of the team name.
    words = {w for w in re.findall(r"\w+", cleaned) if len(w) >= _MIN_FUZZY}
    if words:
        scored = [t for t in teams if words & set(re.findall(r"\w+", t.lower()))]
        if len(scored) == 1:
            return scored[0]
    close = difflib.get_close_matches(cleaned, list(lookup), 1, 0.6)
    if close:
        return lookup[close[0]]
    return contains[0] if contains else None


_VS = re.compile(r"\s+(?:vs\.?|v\.?|versus|against|-|x)\s+", re.IGNORECASE)


def parse_fixture(text: str, teams: Sequence[str]) -> tuple[str, str] | None:
    """Read "Inter Miami vs Chicago Fire" into two known team names."""
    parts = _VS.split(text.strip(), maxsplit=1)
    if len(parts) != 2:
        return None
    home = match_team(parts[0], teams)
    away = match_team(parts[1], teams)
    if home and away and home != away:
        return home, away
    return None


_PRICE = re.compile(
    r"[@=]\s*([+-]?\d+(?:\.\d+)?%?)"        # the price you are taking
    r"(?:\s*/\s*([+-]?\d+(?:\.\d+)?%?))?"   # optionally the other side
    r"\s*$"
)
_NUMBER = re.compile(r"(\d+(?:\.\d+)?)")


@dataclass
class MarketPhrase:
    key: str
    selection: str
    odds: float | None = None
    label: str = ""
    other_odds: float | None = None
    """The opposite side of the same market, if you supplied it.

    Giving both sides ("@ 2.60/1.55") lets the book's margin be removed
    exactly instead of assumed, which materially sharpens the read.
    """


def parse_market_phrase(
    text: str,
    stats: Sequence[str],
    home: str = "",
    away: str = "",
) -> MarketPhrase | None:
    """Turn ordinary phrasing into a market key and selection.

    Understands, among others::

        over 2.5                 -> goals_2.5 over
        under 9.5 corners        -> corners_9.5 under
        corners over 10.5        -> corners_10.5 over
        shots miami over 4.5     -> shots_home_4.5 over
        btts / btts no           -> btts yes / no
        home / draw / inter win  -> 1X2 home / draw / home
        ah -0.5                  -> ah_-0.5 home

    A trailing ``@ 1.85`` or ``= +150`` is read as the price on offer.
    """
    raw = text.strip()
    if not raw:
        return None

    odds = other_odds = None
    price_match = _PRICE.search(raw)
    if price_match:
        try:
            odds = parse_odds(price_match.group(1))
            if price_match.group(2):
                other_odds = parse_odds(price_match.group(2))
        except ValueError:
            return None
        raw = raw[: price_match.start()].strip()

    lowered = raw.lower().replace("+", " +").strip()
    if not lowered:
        return None

    # Both teams to score.
    if "btts" in lowered or "both teams" in lowered:
        selection = "no" if re.search(r"\bno\b", lowered) else "yes"
        return MarketPhrase("btts", selection, odds, f"btts {selection}", other_odds)

    # Asian handicap.
    handicap = re.search(r"\bah\s*(-?\+?\d+(?:\.\d+)?)", lowered)
    if handicap:
        line = float(handicap.group(1).replace("+", ""))
        side = "away" if "away" in lowered else "home"
        return MarketPhrase(f"ah_{line:g}", side, odds, f"ah {line:g} {side}", other_odds)

    # Double chance.
    if "double" in lowered or re.fullmatch(r"(1x|12|x2)", lowered.replace(" ", "")):
        token = lowered.replace(" ", "").replace("doublechance", "") or "1x"
        selection = {"1x": "1X", "12": "12", "x2": "X2"}.get(token, "1X")
        return MarketPhrase("dc", selection, odds, f"double chance {selection}", other_odds)

    over_under = None
    if re.search(r"\bover\b|\bo\b|\+", lowered):
        over_under = "over"
    elif re.search(r"\bunder\b|\bu\b", lowered):
        over_under = "under"

    if over_under:
        numbers = _NUMBER.findall(lowered)
        if not numbers:
            return None
        line = float(numbers[-1])
        # Which statistic? Longest matching name wins, so "shots_on_target"
        # beats "shots".
        stat = "goals"
        for candidate in sorted(stats, key=len, reverse=True):
            if candidate.replace("_", " ") in lowered or candidate in lowered:
                stat = candidate
                break
        # A team mentioned by name or by side makes it a team total.
        side = ""
        if re.search(r"\bhome\b", lowered):
            side = "home"
        elif re.search(r"\baway\b", lowered):
            side = "away"
        elif home or away:
            named = match_team(
                re.sub(r"\b(over|under|o|u)\b|\d+(\.\d+)?", " ", lowered),
                [t for t in (home, away) if t],
            )
            if named == home:
                side = "home"
            elif named == away:
                side = "away"
        key = f"{stat}_{side}_{line:g}" if side else f"{stat}_{line:g}"
        label = f"{stat.replace('_', ' ')} {over_under} {line:g}"
        if side:
            label = f"{(home if side == 'home' else away)} {label}"
        return MarketPhrase(key, over_under, odds, label, other_odds)

    # Match result.
    if re.search(r"\bdraw\b|\btie\b|\bx\b", lowered):
        return MarketPhrase("1X2", "draw", odds, "draw", other_odds)
    if re.search(r"\bhome\b", lowered):
        return MarketPhrase("1X2", "home", odds, "home win", other_odds)
    if re.search(r"\baway\b", lowered):
        return MarketPhrase("1X2", "away", odds, "away win", other_odds)
    if home or away:
        named = match_team(re.sub(r"\b(to )?win\b|\bwins\b", " ", lowered),
                           [t for t in (home, away) if t])
        if named == home:
            return MarketPhrase("1X2", "home", odds, f"{home} win", other_odds)
        if named == away:
            return MarketPhrase("1X2", "away", odds, f"{away} win", other_odds)
    return None


def required_price(
    model_prob: float, trust: float, config: ScanConfig, margin: float = _ASSUMED_MARGIN
) -> float | None:
    """The lowest price at which this selection becomes worth backing.

    Solving this properly matters. The naive answer -- one divided by the model
    probability -- ignores both the safety margin on the estimate and the fact
    that seeing the book's price drags our number toward theirs. The longer the
    price they offer, the lower their own opinion of it, and the more that pulls
    our estimate down. So the threshold is found by searching for the price at
    which the blended, discounted probability still clears the required edge,
    which lands well above the naive fair price.

    Returns ``None`` when no price clears, which is a real answer rather than a
    failure: it means the model cannot justify this selection at any price, so
    the market is one to leave alone.

    Note that the test is *not* monotone in the price. Raising the offered odds
    lowers the break-even bar, but it also tells us the book rates the outcome
    less likely, which drags the blended estimate down with it. At extreme
    prices the second effect wins and the edge disappears again, so the region
    where a bet makes sense is a band rather than everything above a line. The
    search below scans upward for the bottom of that band instead of bisecting
    on an assumption of monotonicity that does not hold.
    """
    if not 0.0 < model_prob < 1.0:
        return None

    def clears(price: float) -> bool:
        book_prob = min(0.999999, (1.0 / price) / (margin / 2.0 + 0.5))
        blended = expit(trust * logit(model_prob) + (1 - trust) * logit(book_prob))
        used = expit(logit(blended) - config.confidence_z * config.extra_stderr)
        return (used - 1.0 / price) >= config.min_edge

    steps = 400
    low, high = 1.005, 200.0
    ratio = (high / low) ** (1.0 / steps)
    previous = low
    for i in range(1, steps + 1):
        price = low * ratio**i
        if clears(price):
            lo, hi = previous, price
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if clears(mid):
                    hi = mid
                else:
                    lo = mid
                if hi - lo < 1e-6:
                    break
            return hi
        previous = price
    return None


@dataclass
class Verdict:
    """The answer to "should I take this price?"."""

    label: str
    odds: float
    model_prob: float
    blended_prob: float
    needs: float
    edge: float
    ev: float
    take_above: float | None
    stake: float
    decision: str
    reason: str

    def render(self, bankroll: float = 0.0) -> str:
        lines = [
            f"  {self.label} @ {self.odds:.2f} "
            f"({decimal_to_american(self.odds):+.0f})",
            f"    we make it        {self.model_prob:.1%}",
            f"    price needs       {self.needs:.1%}",
            f"    edge              {self.edge:+.1%}   "
            f"EV {self.ev:+.1%} per unit staked",
        ]
        if self.take_above:
            lines.append(f"    worth taking at   {self.take_above:.2f} or better")
        lines.append(f"    -> {self.decision.upper()}: {self.reason}")
        if self.stake > 0:
            lines.append(f"    -> stake {self.stake:.2f}")
        return "\n".join(lines)


def check_price(
    projection: MatchProjection,
    phrase: MarketPhrase,
    config: ScanConfig,
    bankroll: float = 0.0,
) -> Verdict | str:
    """Judge one offered price. Returns a :class:`Verdict` or an error string."""
    from .edge import evaluate

    market = resolve_market(projection, phrase.key)
    if market is None:
        return (
            f"no model for {phrase.key!r} -- either the statistic is not in your "
            "results file, or the line is one I cannot build"
        )
    if phrase.selection not in market:
        return f"{phrase.selection!r} is not an option on {phrase.key!r}"
    if phrase.odds is None:
        return "no price given -- add '@ 1.85' to check whether it is worth taking"

    model_prob = market[phrase.selection]
    trust = _trust_for(phrase.key, config)
    if phrase.other_odds:
        from .devig import devig

        book_prob = devig([phrase.odds, phrase.other_odds]).probs[0]
    else:
        book_prob = min(0.999999, (1.0 / phrase.odds) / (_ASSUMED_MARGIN / 2.0 + 0.5))
    blended = expit(trust * logit(model_prob) + (1 - trust) * logit(book_prob))

    result = evaluate(
        blended,
        phrase.odds,
        outcome=phrase.selection,
        bankroll=bankroll,
        prob_stderr=config.extra_stderr,
        kelly_multiplier=config.kelly_multiplier,
        confidence_z=config.confidence_z,
        min_edge=config.min_edge,
        max_stake_fraction=config.max_stake_fraction,
    )

    decision, reason = result.verdict, result.reason
    gap = abs(model_prob - book_prob)
    tolerance = config.max_disagreement * (0.5 + 2.0 * trust)
    stake = result.stake
    if gap > tolerance:
        decision = "check"
        reason = (
            f"we and the book are {gap:.0%} apart, past the {tolerance:.0%} this "
            "market allows -- suspect your results file before the price"
        )
        if not phrase.other_odds:
            reason += (
                ". Add the other side of the market ('@ 2.60/1.55') for an exact "
                "read; without it the book's margin is only estimated"
            )
        stake = 0.0
    if decision == "bet":
        reason = "worth taking"
    elif decision == "no-edge":
        reason = "the price is too short for how likely we make it"

    return Verdict(
        label=phrase.label or f"{phrase.key} {phrase.selection}",
        odds=phrase.odds,
        model_prob=model_prob,
        blended_prob=blended,
        needs=1.0 / phrase.odds,
        edge=blended - 1.0 / phrase.odds,
        ev=blended * phrase.odds - 1.0,
        take_above=required_price(model_prob, trust, config),
        stake=stake,
        decision=decision,
        reason=reason,
    )


# Beyond this multiple of the fair price, a "worth taking" number is not a
# realistic target -- it is the model conceding that the book is better than it
# is on this market.
_UNREACHABLE_MULTIPLE = 2.0


def _row(label: str, prob: float, trust: float, config: ScanConfig) -> str:
    fair = 1.0 / prob if prob > 0 else float("inf")
    threshold = required_price(prob, trust, config)
    if threshold is None or threshold > fair * _UNREACHABLE_MULTIPLE:
        target = f"{'never':>11}"
    else:
        target = f"{threshold:>11.2f}"
    return f"    {label:<24}{prob:>8.1%}{fair:>9.2f}{target}"


def odds_card(
    projection: MatchProjection,
    config: ScanConfig | None = None,
    stats: Sequence[str] = (),
) -> str:
    """A full card of fair prices for one fixture."""
    cfg = config or ScanConfig()
    home, away = projection.home, projection.away
    out = [f"{home} v {away}", ""]
    out.append(f"    {'':<24}{'CHANCE':>8}{'FAIR':>9}{'TAKE ABOVE':>11}")

    def section(title: str) -> None:
        out.append("")
        out.append(f"  {title}")

    if projection.has_goals:
        section("MATCH RESULT")
        result = projection.match_odds()
        for label, key in ((home, "home"), ("Draw", "draw"), (away, "away")):
            out.append(_row(label, result[key], _trust_for("1X2", cfg), cfg))

        section("GOALS")
        for line in (1.5, 2.5, 3.5):
            market = projection.totals("goals", [line])[line]
            for side in ("over", "under"):
                out.append(
                    _row(f"{side} {line:g}", market[side],
                         _trust_for("goals", cfg), cfg)
                )

        section("BOTH TEAMS TO SCORE")
        btts = projection.btts()
        for side in ("yes", "no"):
            out.append(_row(side, btts[side], _trust_for("btts", cfg), cfg))

    for stat in stats:
        if stat in ("goals",) or stat not in projection.rates:
            continue
        total = sum(projection.rates[stat])
        centre = round(total * 2) / 2
        lines = [centre - 1.0, centre, centre + 1.0]
        section(f"{stat.replace('_', ' ').upper()}  (we expect about {total:.1f})")
        for line in lines:
            if line <= 0:
                continue
            half = line if line % 1 else line + 0.5
            market = projection.totals(stat, [half])[half]
            for side in ("over", "under"):
                out.append(
                    _row(f"{side} {half:g}", market[side], _trust_for(stat, cfg), cfg)
                )

    out.append("")
    out.append("  CHANCE is how often we think it happens. FAIR is the break-even")
    out.append("  price. TAKE ABOVE is what the book must beat before the bet is")
    out.append("  worth making -- always well above fair, because the estimate has")
    out.append("  error in it and you get paid for that or you do not bet.")
    out.append("")
    out.append("  'never' means no price would justify it: on that market the book")
    out.append("  is simply better informed than this model, and the honest move is")
    out.append("  to leave it alone. Expect to see it on match results and goals,")
    out.append("  and to see real numbers on corners, cards and shots.")
    return "\n".join(out)


def build_model(log, window: int = 5) -> FormModel:
    return FormModel(window=window).fit(log)
