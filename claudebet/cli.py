"""Command line interface.

    claudebet devig 2.05 1.87
    claudebet analyze market.json --bankroll 1000
    claudebet kelly --prob 0.55 --odds 2.00 --bankroll 1000
    claudebet parlay --leg "KC ML:0.62:1.65" --leg "over 47.5:0.52:1.91" --corr "KC ML,over 47.5:0.25"
    claudebet log report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .correlation import ParlayLeg, parlay
from .devig import METHODS, devig, devig_spread
from .edge import evaluate, risk_of_drawdown
from .market import Market
from .odds import decimal_to_american, decimal_to_prob, hold, overround, parse_odds
from .pipeline import AnalysisConfig, analyze, analyze_slate
from .store import BetLog


def _load_markets(path: str) -> list[Market]:
    from .data.sources import load_csv, load_json

    p = Path(path)
    if not p.exists():
        raise SystemExit(f"no such file: {path}")
    return load_csv(p) if p.suffix.lower() == ".csv" else load_json(p)


# -- commands -----------------------------------------------------------


def cmd_devig(args: argparse.Namespace) -> int:
    prices = [parse_odds(o) for o in args.odds]
    labels = args.labels or [f"outcome {i + 1}" for i in range(len(prices))]
    if len(labels) != len(prices):
        raise SystemExit("--labels must match the number of prices")

    print(f"overround {overround(prices):.4f}   hold {hold(prices):.2%}")
    methods = METHODS if args.all else (args.method,)
    print()
    header = f"{'outcome':<20}" + "".join(f"{m:>14}" for m in methods)
    print(header)
    print("-" * len(header))
    results = {m: devig(prices, m) for m in methods}
    for i, label in enumerate(labels):
        row = f"{label:<20}"
        for m in methods:
            row += f"{results[m].probs[i]:>13.2%} "
        print(row)
    print()
    for m in methods:
        params = results[m].params
        if params:
            detail = ", ".join(f"{k}={v:.4f}" for k, v in params.items())
            print(f"  {m}: {detail}")
    if args.all:
        spread = devig_spread(prices)
        print(
            f"\nmax disagreement between methods: {spread['max_spread']:.2%} "
            "-- any edge smaller than this is not measurable"
        )
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    markets = _load_markets(args.file)
    cfg = AnalysisConfig(
        devig_method=args.method,
        model_weight=args.model_weight,
        kelly_multiplier=args.kelly,
        min_edge=args.min_edge,
        max_stake_fraction=args.max_stake,
        max_slate_exposure=args.max_exposure,
        confidence_z=args.confidence,
        require_positive_worst_case=args.strict,
    )
    model = json.loads(Path(args.model).read_text()) if args.model else None

    if len(markets) == 1 and not args.slate:
        result = analyze(markets[0], (model or {}).get(markets[0].key) or model,
                         cfg, args.bankroll)
        if args.json:
            print(json.dumps(
                {
                    "market": result.market,
                    "fair_probs": result.fair_probs,
                    "warnings": result.warnings,
                    "bets": [
                        {**o.evaluation.as_dict(), "book": o.book} for o in result.bets
                    ],
                },
                indent=2,
            ))
        else:
            print(result.report())
        return 0

    slate = analyze_slate(markets, model, cfg, args.bankroll)
    if args.json:
        print(json.dumps(slate["recommendations"], indent=2))
        return 0
    for analysis in slate["analyses"]:
        print(analysis.report())
        print()
    recs = slate["recommendations"]
    if not recs:
        print("no bets clear the threshold across this slate")
        return 0
    print("== slate staking (jointly sized) ==")
    print(
        f"{'market':<38}{'selection':<20}{'book':<12}{'odds':>8}"
        f"{'edge':>8}{'solo':>8}{'slate':>8}{'stake':>10}"
    )
    for r in recs:
        print(
            f"{r['market'][:37]:<38}{r['outcome'][:19]:<20}{r['book'][:11]:<12}"
            f"{r['odds']:>8.3f}{r['edge_pp']:>7.2%} "
            f"{r['standalone_fraction']:>7.2%} {r['slate_fraction']:>7.2%} "
            f"{r['stake']:>9.2f}"
        )
    print(
        f"\ntotal exposure {slate['total_exposure']:.2%} of bankroll; "
        f"expected log-growth {slate['expected_growth']:+.4%} per slate"
    )
    if slate["arbs"]:
        print(f"{len(slate['arbs'])} arbitrage opportunity(ies) found -- see above")
    return 0


def cmd_kelly(args: argparse.Namespace) -> int:
    ev = evaluate(
        args.prob,
        args.odds,
        bankroll=args.bankroll,
        prob_stderr=args.stderr,
        kelly_multiplier=args.kelly,
        confidence_z=args.confidence,
        min_edge=args.min_edge,
        max_stake_fraction=args.max_stake,
    )
    d = ev.odds
    print(f"price          {d:.3f} decimal ({decimal_to_american(d):+.0f} American)")
    print(f"breakeven      {ev.breakeven_prob:.2%}")
    print(f"your estimate  {ev.fair_prob:.2%}")
    print(f"after haircut  {ev.used_prob:.2%}  ({args.confidence:g} std err of "
          f"{args.stderr:g} in log-odds)")
    print(f"edge           {ev.edge_pp:+.2%} ({ev.edge_cents:.1f} cents)")
    print(f"EV             {ev.ev_per_unit:+.2%} per unit staked")
    print(f"full Kelly     {ev.kelly_full:.2%} of bankroll")
    print(f"recommended    {ev.stake_fraction:.3%} of bankroll = {ev.stake:.2f}")
    print(f"verdict        {ev.verdict.upper()} -- {ev.reason}")
    if ev.verdict == "bet":
        ruin = risk_of_drawdown(ev.edge_pp, 1.0, args.kelly, 0.5)
        print(
            f"\nat {args.kelly:g} Kelly the chance of ever halving the bankroll "
            f"is about {ruin:.1%}"
        )
    return 0


def cmd_parlay(args: argparse.Namespace) -> int:
    legs = []
    for spec in args.leg:
        try:
            name, prob, odds = spec.rsplit(":", 2)
            legs.append(ParlayLeg(name.strip(), float(prob), parse_odds(odds)))
        except ValueError as exc:
            raise SystemExit(f"bad --leg {spec!r}; expected 'name:prob:odds'") from exc

    correlations = {}
    for spec in args.corr or []:
        try:
            pair, rho = spec.rsplit(":", 1)
            a, b = pair.split(",")
            correlations[(a.strip(), b.strip())] = float(rho)
        except ValueError as exc:
            raise SystemExit(f"bad --corr {spec!r}; expected 'legA,legB:rho'") from exc

    result = parlay(legs, args.offered, correlations or None)
    print(f"legs               {result.n_legs}  ({result.method})")
    print(f"independent prob   {result.independent_prob:.4%}")
    print(f"correlated prob    {result.fair_prob:.4%} "
          f"({result.correlation_gain:+.4%} from correlation)")
    print(f"fair price         {result.fair_odds:.2f} "
          f"({decimal_to_american(result.fair_odds):+.0f})")
    print(f"offered price      {result.offered_odds:.2f} "
          f"({decimal_to_american(result.offered_odds):+.0f})")
    print(f"hold               {result.hold:.2%}")
    print(f"EV                 {result.ev_per_unit:+.2%} per unit staked")
    print()
    if result.is_positive_ev:
        print("positive EV -- unusual for a parlay; double-check the leg "
              "probabilities and the correlations before trusting it")
    else:
        singles = 1.0
        for leg in legs:
            singles *= decimal_to_prob(leg.odds)
        print(f"negative EV. The same legs bet singly carry roughly "
              f"{(singles - result.independent_prob) / singles:.1%} combined hold "
              "against you, versus "
              f"{result.hold:.1%} in the parlay.")
    return 0


def cmd_clv(args: argparse.Namespace) -> int:
    taken, closing = parse_odds(args.taken), parse_odds(args.closing)
    pp = decimal_to_prob(closing) - decimal_to_prob(taken)
    print(f"taken    {taken:.3f} ({decimal_to_american(taken):+.0f}) "
          f"= {decimal_to_prob(taken):.2%}")
    print(f"closing  {closing:.3f} ({decimal_to_american(closing):+.0f}) "
          f"= {decimal_to_prob(closing):.2%}")
    print(f"CLV      {pp:+.2%} in probability points "
          f"({taken / closing - 1:+.2%} price improvement)")
    if pp > 0:
        print("\nyou beat the close. Sustained over a few hundred bets this is "
              "the strongest evidence of edge available.")
    else:
        print("\nyou were on the wrong side of the close. One bet means nothing; "
              "a pattern of it means the edge is not there.")
    return 0


def cmd_log(args: argparse.Namespace) -> int:
    log = BetLog(args.db)
    if args.log_command == "record":
        bet_id = log.record(
            market=args.market,
            outcome=args.outcome,
            book=args.book,
            odds=args.odds,
            stake=args.stake,
            fair_prob=args.fair_prob,
            tags=tuple(args.tags.split(",")) if args.tags else (),
        )
        print(f"recorded bet {bet_id}")
    elif args.log_command == "close":
        log.set_closing(args.id, args.closing_odds, args.closing_fair_prob)
        print(f"closing line saved for bet {args.id}")
    elif args.log_command == "grade":
        log.grade(args.id, args.result)
        print(f"bet {args.id} graded {args.result}")
    elif args.log_command == "pending":
        rows = log.pending()
        if not rows:
            print("nothing pending")
        for r in rows:
            print(f"{r['id']:>5}  {r['market']:<40} {r['outcome']:<18} "
                  f"{r['odds']:.3f}  {r['stake']:.2f}")
    elif args.log_command == "report":
        try:
            result = log.performance(args.bankroll)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(result.summary())
        cal = log.calibration()
        if cal.get("n"):
            print(
                f"\ncalibration: Brier {cal['brier']:.4f}, "
                f"ECE {cal['ece']:.3f}"
                + (
                    f", skill vs closing line {cal['skill_vs_reference']:+.4f}"
                    if "skill_vs_reference" in cal
                    else ""
                )
            )
            if "warning" in cal:
                print(f"warning: {cal['warning']}")
        missing = log.needs_closing_line()
        if missing:
            print(f"\n{len(missing)} bet(s) still missing a closing line -- "
                  "fill them in, it is the number that matters most")
    return 0


def cmd_form(args: argparse.Namespace) -> int:
    from .data.sources import load_match_log
    from .form import FormModel

    log = load_match_log(args.file)
    if args.home not in log.teams() or args.away not in log.teams():
        raise SystemExit(
            f"unknown team; the log contains: {', '.join(log.teams())}"
        )

    model = FormModel(window=args.last).fit(log)
    print(f"{len(log)} matches, {len(log.teams())} teams, "
          f"tracking: {', '.join(model.stat_models)}")
    print()

    print(f"-- recent form (last {args.last}) --")
    for team in (args.home, args.away):
        print("  " + model.team_form(team).line())
    if args.last != 10:
        print("\n-- recent form (last 10) --")
        for team in (args.home, args.away):
            print("  " + model.team_form(team, window=10).line())

    print("\n-- how much of that form is signal --")
    for stat, sm in sorted(model.stat_models.items()):
        fh = sm.form_attack.get(args.home, 1.0)
        fa = sm.form_attack.get(args.away, 1.0)
        print(
            f"  {stat:<10} form multiplier after shrinkage: "
            f"{args.home} x{fh:.3f}, {args.away} x{fa:.3f}"
        )
    print("  (values near 1.000 mean the recent run was inside normal noise "
          "and has been discounted accordingly)")

    projection = model.project(args.home, args.away, neutral=args.neutral,
                               use_form=not args.no_form)
    print()
    print(projection.report())

    if args.json:
        print()
        print(json.dumps(_jsonable(projection.markets()), indent=2))
    return 0


def _jsonable(value):
    """JSON keys must be strings; betting lines are naturally floats."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def cmd_template(args: argparse.Namespace) -> int:
    from .data.sources import write_match_log_template, write_template

    if args.log:
        path = write_match_log_template(args.path or "matches.csv")
        print(
            f"wrote {path}\nadd your results, then: "
            f'claudebet form {path} --home "Alianza Lima" --away "Sporting Cristal"'
        )
        return 0
    path = write_template(args.path or "market.json")
    print(f"wrote {path}\nedit it, then: claudebet analyze {path} --bankroll 1000")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    from .data.sources import OddsApiClient

    client = OddsApiClient(args.api_key)
    markets = client.odds(args.sport, regions=args.regions, markets=args.markets)
    payload = [
        {
            "key": m.key,
            "outcomes": list(m.outcomes),
            "prices": m.by_book(complete_only=False),
        }
        for m in markets
    ]
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(markets)} markets to {args.out}")
    print(f"api quota remaining: {client.last_quota.get('remaining')}")
    return 0


# -- wiring -------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="claudebet",
        description="Market-aware probability and staking for sports betting.",
    )
    p.add_argument("--version", action="version", version=f"claudebet {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("devig", help="strip the vig from a set of prices")
    d.add_argument("odds", nargs="+", help="decimal, +150, or 5/2")
    d.add_argument("--method", default="shin", choices=METHODS)
    d.add_argument("--all", action="store_true", help="compare every method")
    d.add_argument("--labels", nargs="+", help="outcome names")
    d.set_defaults(func=cmd_devig)

    a = sub.add_parser("analyze", help="analyse a market or slate from a file")
    a.add_argument("file", help="JSON or CSV of quotes")
    a.add_argument("--bankroll", type=float, default=0.0)
    a.add_argument("--method", default="shin", choices=METHODS)
    a.add_argument("--model", help="JSON file of {market: {outcome: prob}}")
    a.add_argument("--model-weight", type=float, default=0.0)
    a.add_argument("--kelly", type=float, default=0.25, help="fraction of Kelly")
    a.add_argument("--min-edge", type=float, default=0.015)
    a.add_argument("--max-stake", type=float, default=0.02)
    a.add_argument("--max-exposure", type=float, default=0.10)
    a.add_argument("--confidence", type=float, default=1.0, help="std errs to give back")
    a.add_argument("--strict", action="store_true",
                   help="require the edge to survive worst-case devigging")
    a.add_argument("--slate", action="store_true", help="force joint slate staking")
    a.add_argument("--json", action="store_true")
    a.set_defaults(func=cmd_analyze)

    k = sub.add_parser("kelly", help="size a single bet")
    k.add_argument("--prob", type=float, required=True, help="your fair probability")
    k.add_argument("--odds", required=True)
    k.add_argument("--bankroll", type=float, default=1000.0)
    k.add_argument("--stderr", type=float, default=0.10, help="log-odds std error")
    k.add_argument("--kelly", type=float, default=0.25)
    k.add_argument("--confidence", type=float, default=1.0)
    k.add_argument("--min-edge", type=float, default=0.01)
    k.add_argument("--max-stake", type=float, default=0.02)
    k.set_defaults(func=cmd_kelly)

    pl = sub.add_parser("parlay", help="price a parlay with correlation")
    pl.add_argument("--leg", action="append", required=True,
                    help="'name:fair_prob:odds', repeatable")
    pl.add_argument("--corr", action="append", help="'legA,legB:rho', repeatable")
    pl.add_argument("--offered", help="the book's parlay price")
    pl.set_defaults(func=cmd_parlay)

    c = sub.add_parser("clv", help="closing line value of one bet")
    c.add_argument("--taken", required=True)
    c.add_argument("--closing", required=True)
    c.set_defaults(func=cmd_clv)

    fm = sub.add_parser("form", help="analyse recent form and price every market")
    fm.add_argument("file", help="CSV match log (see: claudebet template --log)")
    fm.add_argument("--home", required=True)
    fm.add_argument("--away", required=True)
    fm.add_argument("--last", type=int, default=5, help="form window (default 5)")
    fm.add_argument("--neutral", action="store_true")
    fm.add_argument("--no-form", action="store_true",
                    help="season strength only, ignoring the recent window")
    fm.add_argument("--json", action="store_true", help="dump every market")
    fm.set_defaults(func=cmd_form)

    t = sub.add_parser("template", help="write an example input file")
    t.add_argument("path", nargs="?")
    t.add_argument("--log", action="store_true",
                   help="write a match-log CSV instead of a market file")
    t.set_defaults(func=cmd_template)

    f = sub.add_parser("fetch", help="pull live odds from the-odds-api.com")
    f.add_argument("--sport", required=True, help="e.g. americanfootball_nfl")
    f.add_argument("--regions", default="us,eu")
    f.add_argument("--markets", default="h2h")
    f.add_argument("--out", default="odds.json")
    f.add_argument("--api-key")
    f.set_defaults(func=cmd_fetch)

    lg = sub.add_parser("log", help="record and grade your bets")
    lg.add_argument("--db", default="bets.db")
    lsub = lg.add_subparsers(dest="log_command", required=True)

    lr = lsub.add_parser("record")
    lr.add_argument("--market", required=True)
    lr.add_argument("--outcome", required=True)
    lr.add_argument("--book", default="")
    lr.add_argument("--odds", required=True)
    lr.add_argument("--stake", type=float, required=True)
    lr.add_argument("--fair-prob", type=float)
    lr.add_argument("--tags", default="")

    lc = lsub.add_parser("close")
    lc.add_argument("id", type=int)
    lc.add_argument("closing_odds")
    lc.add_argument("--closing-fair-prob", type=float)

    lgr = lsub.add_parser("grade")
    lgr.add_argument("id", type=int)
    lgr.add_argument("result", choices=["win", "loss", "push", "void",
                                        "half-win", "half-loss"])

    lsub.add_parser("pending")
    lrep = lsub.add_parser("report")
    lrep.add_argument("--bankroll", type=float, default=1000.0)
    lg.set_defaults(func=cmd_log)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
