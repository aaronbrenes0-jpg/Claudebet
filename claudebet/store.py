"""A bet log, because a model you do not grade is a hobby.

Every bet is written down at the moment it is placed, with the probability you
believed at the time. Later you add the closing price and the result. The
stored ``fair_prob`` is what makes the record worth keeping: without it you can
measure whether you won, but not whether you were right, and those are
different questions with different answers over any sample you will ever have.

SQLite, single file, no server.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from .backtest import BacktestResult, SettledBet, run as run_backtest
from .odds import parse_odds

__all__ = ["BetLog"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bets (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    placed_at    TEXT NOT NULL,
    market       TEXT NOT NULL,
    outcome      TEXT NOT NULL,
    book         TEXT NOT NULL,
    odds         REAL NOT NULL,
    stake        REAL NOT NULL,
    fair_prob    REAL,
    edge_pp      REAL,
    model        TEXT,
    tags         TEXT DEFAULT '',
    closing_odds REAL,
    closing_fair_prob REAL,
    result       TEXT,
    settled_at   TEXT,
    note         TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_bets_result ON bets(result);
CREATE INDEX IF NOT EXISTS idx_bets_market ON bets(market);
"""


class BetLog:
    """Append-only-ish record of what you bet and how it turned out."""

    def __init__(self, path: str | Path = "bets.db") -> None:
        self.path = str(path)
        with closing(self._connect()) as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    # -- writing --------------------------------------------------------

    def record(
        self,
        market: str,
        outcome: str,
        book: str,
        odds: float,
        stake: float,
        fair_prob: float | None = None,
        edge_pp: float | None = None,
        model: str = "",
        tags: Sequence[str] = (),
        note: str = "",
        placed_at: datetime | None = None,
    ) -> int:
        """Write a bet at placement time. Returns its id."""
        stamp = (placed_at or datetime.now(timezone.utc)).isoformat()
        with closing(self._connect()) as conn:
            cur = conn.execute(
                """INSERT INTO bets
                   (placed_at, market, outcome, book, odds, stake, fair_prob,
                    edge_pp, model, tags, note)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    stamp,
                    market,
                    outcome,
                    book,
                    parse_odds(odds),
                    float(stake),
                    fair_prob,
                    edge_pp,
                    model,
                    ",".join(tags),
                    note,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def set_closing(
        self,
        bet_id: int,
        closing_odds: float,
        closing_fair_prob: float | None = None,
    ) -> None:
        """Record where the market closed. Do this for every bet -- it is the
        single most informative field in the table."""
        with closing(self._connect()) as conn:
            conn.execute(
                "UPDATE bets SET closing_odds = ?, closing_fair_prob = ? WHERE id = ?",
                (parse_odds(closing_odds), closing_fair_prob, bet_id),
            )
            conn.commit()

    def grade(self, bet_id: int, result: str, settled_at: datetime | None = None) -> None:
        stamp = (settled_at or datetime.now(timezone.utc)).isoformat()
        with closing(self._connect()) as conn:
            conn.execute(
                "UPDATE bets SET result = ?, settled_at = ? WHERE id = ?",
                (result.strip().lower(), stamp, bet_id),
            )
            conn.commit()

    # -- reading --------------------------------------------------------

    def _rows(self, where: str = "", params: Sequence = ()) -> list[sqlite3.Row]:
        with closing(self._connect()) as conn:
            return list(conn.execute(f"SELECT * FROM bets {where} ORDER BY placed_at", params))

    def pending(self) -> list[dict]:
        """Bets with no result yet."""
        return [dict(r) for r in self._rows("WHERE result IS NULL OR result = ''")]

    def needs_closing_line(self) -> list[dict]:
        return [dict(r) for r in self._rows("WHERE closing_odds IS NULL")]

    def settled(self) -> list[SettledBet]:
        out: list[SettledBet] = []
        for row in self._rows("WHERE result IS NOT NULL AND result != ''"):
            try:
                out.append(
                    SettledBet(
                        outcome=row["outcome"],
                        odds=row["odds"],
                        stake=row["stake"],
                        result=row["result"],
                        timestamp=datetime.fromisoformat(row["placed_at"]),
                        market=row["market"],
                        closing_odds=row["closing_odds"],
                        fair_prob=row["fair_prob"],
                        closing_fair_prob=row["closing_fair_prob"],
                        tags=tuple(t for t in (row["tags"] or "").split(",") if t),
                    )
                )
            except ValueError:
                continue  # unrecognised result string; skip rather than crash
        return out

    def performance(self, starting_bankroll: float = 1000.0) -> BacktestResult:
        """Grade the whole record. See :mod:`claudebet.backtest`."""
        bets = self.settled()
        if not bets:
            raise ValueError("no settled bets in the log yet")
        return run_backtest(bets, starting_bankroll=starting_bankroll)

    def calibration(self) -> dict:
        """How well the stored ``fair_prob`` values predicted reality, scored
        against the closing line where it is available."""
        from . import calibration as calib

        probs: list[float] = []
        outcomes: list[int] = []
        reference: list[float] = []
        for row in self._rows("WHERE result IS NOT NULL AND result != ''"):
            if row["fair_prob"] is None or row["result"] not in ("win", "loss"):
                continue
            probs.append(float(row["fair_prob"]))
            outcomes.append(1 if row["result"] == "win" else 0)
            if row["closing_fair_prob"] is not None:
                reference.append(float(row["closing_fair_prob"]))
            elif row["closing_odds"] is not None:
                reference.append(1.0 / float(row["closing_odds"]))
        if not probs:
            return {"n": 0, "note": "no graded bets with a stored fair probability"}
        ref = reference if len(reference) == len(probs) else None
        return calib.report(probs, outcomes, ref)

    def import_bets(self, bets: Iterable[SettledBet]) -> int:
        """Bulk-load an existing history."""
        count = 0
        for bet in bets:
            bet_id = self.record(
                market=bet.market,
                outcome=bet.outcome,
                book="",
                odds=bet.odds,
                stake=bet.stake,
                fair_prob=bet.fair_prob,
                tags=bet.tags,
                placed_at=bet.timestamp,
            )
            if bet.closing_odds is not None:
                self.set_closing(bet_id, bet.closing_odds, bet.closing_fair_prob)
            self.grade(bet_id, bet.result)
            count += 1
        return count
