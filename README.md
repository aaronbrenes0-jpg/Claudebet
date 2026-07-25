# claudebet

A probability and staking engine for sports betting. Pure Python, no dependencies.

It answers three questions, in this order:

1. **What is the fair probability?** Strip the vig out of every book's prices, pool
   them by how sharp each book is, and blend in your own model by as much as its
   track record justifies — usually not much.
2. **Is this price better than that?** Compare each individual price against a
   consensus built *without* the book offering it, and take an estimation-error
   haircut before believing the edge.
3. **How much should I bet?** Fractional Kelly, sized across the whole slate at
   once, under per-bet and total-exposure caps.

## Quick start

```bash
git clone <this repo> && cd Claudebet
pip install -e .          # or just run with PYTHONPATH=.

claudebet template market.json
claudebet analyze market.json --bankroll 1000
```

```python
from claudebet import Market, analyze

market = Market.from_book_prices(
    "Arsenal v Chelsea | 1X2",
    {
        "pinnacle":  {"Arsenal": 2.30, "Draw": 3.45, "Chelsea": 3.30},
        "circa":     {"Arsenal": 2.29, "Draw": 3.44, "Chelsea": 3.28},
        "hardrock":  {"Arsenal": 2.45, "Draw": 3.14, "Chelsea": 3.00},
    },
)
print(analyze(market, bankroll=5000).report())
```

Prices can be decimal (`2.30`), American with an explicit sign (`"+130"`),
fractional (`"5/2"`), or a probability (`"43%"`).

For a tour of everything, run `python examples/walkthrough.py`.

## The parts

| Module | What it does |
|---|---|
| `odds` | Conversion between decimal, American, fractional, implied. Overround and hold. |
| `devig` | Five ways to remove the margin: multiplicative, additive, power, Shin, odds-ratio. |
| `market` | Consensus fair prices across books, line shopping, arbitrage, steam detection. |
| `blend` | Log-linear pooling of model and market, with an evidence-based weight. |
| `edge` | EV, Kelly, the uncertainty haircut, drawdown risk. |
| `portfolio` | Growth-optimal stakes across a whole slate, including mutually exclusive bets. |
| `correlation` | Parlay pricing under a Gaussian copula. |
| `calibration` | Brier, log loss, reliability, skill vs the market, Platt and isotonic recalibration. |
| `backtest` | Bankroll simulation, drawdown, bootstrapped confidence intervals, CLV. |
| `models` | Elo, Dixon-Coles scorelines, ridge power ratings. |
| `store` | SQLite bet log that records what you believed, not just what you bet. |

## Four ideas the code is built around

**Devigging is a modelling choice, not arithmetic.** Dividing by the overround
assumes the book spreads its margin proportionally. It does not — margin is
loaded onto longshots. On a -110/-110 market every method agrees; at +2000 the
spread between methods can exceed your entire edge. The default is Shin's
method, and `devig_spread()` tells you how much your answer depends on that
choice. If the methods disagree by more than your edge, you do not have an edge.

**Never benchmark a price against a consensus containing itself.** If you are
evaluating a price at DraftKings, the fair number has to come from everyone
*except* DraftKings. Leaving it in drags the benchmark toward the very number
you are testing and silently shrinks the edge. `analyze()` does this
leave-one-out automatically.

**Size for the error bar, not the point estimate.** Your probability is an
estimate, and Kelly is brutally sensitive to overestimating an edge. Every
recommendation walks the probability down by a standard error before staking,
where that error is built from how much the books disagree, how much the devig
methods disagree, and a floor for everything unmodelled. Then quarter Kelly, then
a hard per-bet cap. The defaults are strict enough to decline most of what looks
bettable — that is the intent, not a bug.

**Grade on closing line value, not profit.** At a 2% edge the standard deviation
over 500 bets is several times the edge itself, so profit stays ambiguous for
years. Beating the closing line converges far faster. `backtest.run()` refuses
to report an ROI without a confidence interval, and its verdict leads with CLV.

## Getting live prices

```bash
export ODDS_API_KEY=...
claudebet fetch --sport americanfootball_nfl --markets h2h --out odds.json
claudebet analyze odds.json --bankroll 1000
```

Uses [the-odds-api.com](https://the-odds-api.com). Check that your regions
actually return a sharp book — a consensus made only of recreational books is a
consensus of followers, and the whole method leans on having at least one price
that means something.

## Commands

```
claudebet devig 2.30 3.45 3.30 --all      compare every devig method
claudebet analyze market.json             fair prices, edges, stakes
claudebet kelly --prob 0.55 --odds 2.00   size one bet
claudebet parlay --leg "a:0.6:1.65" ...   price a parlay with correlation
claudebet clv --taken 2.10 --closing 1.95 did you beat the close
claudebet log record|close|grade|report   keep and grade a record
```

## Honest limits

- **It cannot manufacture an edge.** On a liquid market near kickoff there
  usually isn't one, and the engine will correctly tell you so. Real edges live
  in stale soft-book prices, in markets books price lazily, and in taking the
  best available number every single time.
- **The sharpness weights are priors, not fitted values.** They encode who takes
  size and who copies. Override them with `sharpness=` if your data disagrees.
- **Slates above ~17 independent bets** switch from exact enumeration to a
  quadratic approximation with an exact one-dimensional rescaling. It agrees with
  the exact solver to within a few percent; `Allocation.method` records which ran.
- **Correlation estimates for same-game parlays are yours to supply.** The copula
  is only as good as the rho you feed it, and a wrong rho on a parlay is
  expensive.
- **Books limit winners.** Beating a soft book reliably gets you restricted. This
  is a property of the business, not of the software.

Nothing here is financial advice, and no model makes betting a good way to make
money. Bet only what you can afford to lose.

## Tests

```bash
python -m pytest tests -q     # 196 tests
```
