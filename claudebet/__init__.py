"""claudebet -- market-aware probability and staking for sports betting.

The short version of how to use it::

    from claudebet import Market, analyze

    market = Market.from_book_prices(
        "KC@BUF moneyline",
        {
            "pinnacle":   {"KC": 2.05, "BUF": 1.87},
            "draftkings": {"KC": 2.20, "BUF": 1.76},
        },
    )
    print(analyze(market, bankroll=1000).report())

Everything else -- models, calibration, backtesting, portfolio staking -- exists
to make the number that comes out of that call trustworthy.

A word on what this can and cannot do. The engine finds prices that are out of
line with the rest of the market and sizes them so that a run of bad luck does
not end you. It cannot manufacture an edge where none exists, and on a liquid
market minutes before kickoff there usually is none. Real edges live in stale
soft-book prices, in markets the books price lazily, and in getting the best
number every single time. Expect to grind, expect to be limited by the books
you beat, and grade yourself on closing line value rather than on profit.
"""

from .backtest import SettledBet, clv_report
from .backtest import run as backtest
from .blend import blend, weight_from_evidence
from .calibration import IsotonicCalibrator, PlattCalibrator, brier_score, skill_score
from .correlation import ParlayLeg, parlay
from .counts import NegativeBinomial, Poisson, fit_counts
from .devig import devig, devig_spread
from .edge import evaluate, expected_roi, kelly_fraction
from .form import FormModel, MatchLog, MatchRecord
from .market import Market, Quote, consensus, movement
from .odds import american_to_decimal, decimal_to_american, parse_odds
from .pipeline import AnalysisConfig, analyze, analyze_slate
from .portfolio import Candidate, optimize
from .store import BetLog

__version__ = "0.1.0"

__all__ = [
    "AnalysisConfig",
    "BetLog",
    "Candidate",
    "FormModel",
    "IsotonicCalibrator",
    "Market",
    "MatchLog",
    "MatchRecord",
    "NegativeBinomial",
    "ParlayLeg",
    "PlattCalibrator",
    "Poisson",
    "Quote",
    "SettledBet",
    "american_to_decimal",
    "analyze",
    "analyze_slate",
    "backtest",
    "blend",
    "brier_score",
    "clv_report",
    "consensus",
    "decimal_to_american",
    "devig",
    "devig_spread",
    "evaluate",
    "expected_roi",
    "fit_counts",
    "kelly_fraction",
    "movement",
    "optimize",
    "parlay",
    "parse_odds",
    "skill_score",
    "weight_from_evidence",
]
