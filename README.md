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

## Recent form: the last 5 or 10 games

Give it a results file and it prices every market on the slip — 1X2, over/under
at any goal line, BTTS, Asian handicaps, team totals, corners, cards, correct
score.

```bash
claudebet template matches.csv --log      # shows the column layout
claudebet form matches.csv --home "Alianza Lima" --away "Sporting Cristal" --last 5
```

The file is just one row per match. Whatever `home_*`/`away_*` column pairs you
put in it get modelled — goals, corners, cards, shots, offsides, anything:

```csv
date,home,away,home_goals,away_goals,home_corners,away_corners,home_cards,away_cards
2026-02-01,Alianza Lima,Sporting Cristal,2,1,7,4,2,3
```

**The thing to understand before trusting any of it.** Five games is almost no
data. A team averaging 7.0 corners over five matches has a standard error near
1.2 corners — and that is exactly the gap that decides a 10.5 line. Three
corrections keep this usable rather than dangerous:

- **Shrinkage toward the team's own season**, not the league average, by an
  amount empirical Bayes derives from your data. Before any form is believed at
  all, a chi-square test asks whether the spread across teams exceeds what
  sampling noise alone would produce. If it does not, every form multiplier
  comes back at exactly 1.000 and recent results change nothing.
- **Opponent and venue adjustment**, so a soft run of fixtures does not read as
  good form.
- **Measured dispersion.** Corners run at a variance-to-mean ratio near 1.5 and
  cards higher. Pricing them as Poisson understates both tails and makes the
  extreme lines look like value; the model measures the ratio and switches to a
  negative binomial when the data warrants it.

On the 5-versus-10 question, the honest answer from the model's own output: on
a synthetic league with a genuine ±35% form swing, a five-game window detects
nothing, and neither does ten — the swing only becomes separable from noise at
around twenty. Windows are a resolution/recency trade, and short ones frequently
land under the noise floor. `claudebet form` prints the shrinkage it applied so
you can see which side of that line you are on.

## The parts

| Module | What it does |
|---|---|
| `odds` | Conversion between decimal, American, fractional, implied. Overround and hold. |
| `devig` | Five ways to remove the margin: multiplicative, additive, power, Shin, odds-ratio. |
| `market` | Consensus fair prices across books, line shopping, arbitrage, steam detection. |
| `form` | Last-N form from a match log into probabilities for every market. |
| `counts` | Poisson and negative binomial for goals, corners, cards; over/under pricing. |
| `blend` | Log-linear pooling of model and market, with an evidence-based weight. |
| `edge` | EV, Kelly, the uncertainty haircut, drawdown risk. |
| `portfolio` | Growth-optimal stakes across a whole slate, including mutually exclusive bets. |
| `correlation` | Parlay pricing under a Gaussian copula. |
| `calibration` | Brier, log loss, reliability, skill vs the market, Platt and isotonic recalibration. |
| `backtest` | Bankroll simulation, drawdown, bootstrapped confidence intervals, CLV. |
| `models` | Elo, Dixon-Coles scorelines, ridge power ratings. |
| `store` | SQLite bet log that records what you believed, not just what you bet. |

## Putting the two halves together

Form gives you a probability; the market tells you whether it is worth
anything. Feed one into the other:

```python
from claudebet import AnalysisConfig, Market, analyze, weight_from_evidence
from claudebet.form import FormModel
from claudebet.data.sources import load_match_log

model = FormModel(window=10).fit(load_match_log("matches.csv"))
probs = model.project("Alianza Lima", "Sporting Cristal").markets()["1X2"]

market = Market.from_book_prices("Alianza v Cristal | 1X2", {
    "pinnacle":  {"home": 1.55, "draw": 4.20, "away": 6.00},
    "doradobet": {"home": 1.62, "draw": 4.00, "away": 5.50},
})

# Let the model's track record decide how far it may move the market price.
weight = weight_from_evidence(n_graded=600, calibration_score=0.015)
print(analyze(market, model_probs=probs,
              config=AnalysisConfig(model_weight=weight), bankroll=1000).report())
```

With no track record yet, `weight_from_evidence` returns a few percent and the
market does nearly all the work. That is the correct starting point: log your
bets with `claudebet log`, and the weight grows only as the record earns it.

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
claudebet form matches.csv --home A --away B --last 5
                                          recent form -> every market
claudebet kelly --prob 0.55 --odds 2.00   size one bet
claudebet parlay --leg "a:0.6:1.65" ...   price a parlay with correlation
claudebet clv --taken 2.10 --closing 1.95 did you beat the close
claudebet log record|close|grade|report   keep and grade a record
claudebet template market.json            example inputs (--log for a match log)
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
- **Form data needs volume.** The form layer is only as good as the match log
  behind it. A handful of games per team will correctly produce no form signal
  at all rather than a confident guess — that is the design, but it does mean
  you need a real results file to get value from it.
- **Books limit winners.** Beating a soft book reliably gets you restricted. This
  is a property of the business, not of the software.

Nothing here is financial advice, and no model makes betting a good way to make
money. Bet only what you can afford to lose.

## Tests

```bash
python -m pytest tests -q     # 269 tests
```
