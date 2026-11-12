"""Журнал сигналов и «память» финансиста (SQLite).

У каждого сигнала есть два учёта:
* реальный — только если владелец отметил «Поставил» (фактический коэффициент и сумма),
  именно он меняет банк;
* бумажный — результат системы по рекомендации (минимальный коэффициент и рекомендованная сумма),
  нужен, чтобы честно оценивать точность главного аналитика.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from .analysis.bankroll import RiskState
from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    event_id TEXT NOT NULL,
    league_code TEXT NOT NULL,
    league_name TEXT NOT NULL,
    home TEXT NOT NULL,
    away TEXT NOT NULL,
    kickoff TEXT NOT NULL,
    selection TEXT NOT NULL,
    odds_ref REAL NOT NULL,
    min_odds REAL NOT NULL,
    p_final REAL NOT NULL,
    p_market REAL NOT NULL,
    p_model REAL,
    edge REAL NOT NULL,
    stake REAL NOT NULL,
    confidence TEXT NOT NULL,
    reasoning TEXT NOT NULL,
    user_decision TEXT NOT NULL DEFAULT 'pending',
    actual_odds REAL,
    actual_stake REAL,
    decided_at TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    score TEXT,
    profit REAL,
    paper_profit REAL,
    settled_at TEXT,
    UNIQUE(event_id, selection)
);
CREATE TABLE IF NOT EXISTS llm_usage (
    day TEXT NOT NULL,
    analyses INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    web_searches INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (day)
);
"""

# Колонки, добавленные позже: для баз, созданных старой версией.
_MIGRATIONS = {
    "user_decision": "TEXT NOT NULL DEFAULT 'pending'",
    "actual_odds": "REAL",
    "actual_stake": "REAL",
    "decided_at": "TEXT",
    "paper_profit": "REAL",
}


@dataclass
class Signal:
    id: int
    created_at: str
    event_id: str
    league_code: str
    league_name: str
    home: str
    away: str
    kickoff: datetime
    selection: str
    odds_ref: float
    min_odds: float
    p_final: float
    p_market: float
    p_model: float | None
    edge: float
    stake: float
    confidence: str
    reasoning: str
    user_decision: str  # pending / placed / skipped / expired
    actual_odds: float | None
    actual_stake: float | None
    decided_at: str | None
    status: str  # pending / won / lost / void
    score: str | None
    profit: float | None
    paper_profit: float | None
    settled_at: str | None

    @property
    def placed(self) -> bool:
        return self.user_decision == "placed"


@dataclass
class Summary:
    count: int
    won: int
    staked: float
    profit: float
    avg_odds: float
    expected_wr: float
    max_drawdown: float

    @property
    def roi(self) -> float:
        return self.profit / self.staked if self.staked else 0.0


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _tz() -> ZoneInfo:
    return ZoneInfo(settings.timezone)


def local_today() -> str:
    return datetime.now(_tz()).date().isoformat()


def _local_day_start_utc() -> str:
    start = datetime.now(_tz()).replace(hour=0, minute=0, second=0, microsecond=0)
    return start.astimezone(timezone.utc).isoformat()


def month_start_utc() -> str:
    start = datetime.now(_tz()).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return start.astimezone(timezone.utc).isoformat()


def summarize(signals: list[Signal], real: bool) -> Summary | None:
    """Сводка по рассчитанным сигналам: real=True — только поставленные, иначе бумажный учёт всех."""
    rows = [s for s in signals if s.status in ("won", "lost") and (s.placed or not real)]
    if not rows:
        return None
    staked = sum((s.actual_stake if real else s.stake) or 0 for s in rows)
    profits = [(s.profit if real else s.paper_profit) or 0 for s in rows]
    odds = [(s.actual_odds if real else s.min_odds) or s.min_odds for s in rows]
    peak = cum = dd = 0.0
    for p in profits:
        cum += p
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return Summary(
        count=len(rows),
        won=sum(s.status == "won" for s in rows),
        staked=staked,
        profit=sum(profits),
        avg_odds=sum(odds) / len(odds),
        expected_wr=sum(s.p_final for s in rows) / len(rows),
        max_drawdown=dd,
    )


class DB:
    def __init__(self, path: str | None = None):
        path = path or settings.db_path
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        existing = {r["name"] for r in self.conn.execute("PRAGMA table_info(signals)")}
        for col, decl in _MIGRATIONS.items():
            if col not in existing:
                self.conn.execute(f"ALTER TABLE signals ADD COLUMN {col} {decl}")
        usage_cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(llm_usage)")}
        if "cost_usd" not in usage_cols:
            self.conn.execute("ALTER TABLE llm_usage ADD COLUMN cost_usd REAL NOT NULL DEFAULT 0")
        self.conn.commit()

    # ---------- key-value ----------
    def get(self, key: str, default=None):
        row = self.conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set(self, key: str, value) -> None:
        self.conn.execute(
            "INSERT INTO kv(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value, ensure_ascii=False)),
        )
        self.conn.commit()

    # ---------- банк ----------
    def bankroll(self) -> float:
        return float(self.get("bankroll", settings.start_bankroll))

    def set_bankroll(self, value: float) -> None:
        self.set("bankroll", round(value, 2))
        # ручная правка банка — новая точка отсчёта для месячной статистики
        self.set("bank_reset_at", _now().isoformat())

    def in_play(self) -> tuple[int, float]:
        """Сколько поставленных ставок ещё не рассчитано и на какую сумму."""
        row = self.conn.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(actual_stake), 0) s FROM signals "
            "WHERE user_decision = 'placed' AND status = 'pending'"
        ).fetchone()
        return row["n"], float(row["s"])

    def risk_state(self) -> RiskState:
        since = _local_day_start_utc()
        bets_today = self.conn.execute(
            "SELECT COUNT(*) FROM signals WHERE created_at >= ? AND user_decision IN ('pending','placed')",
            (since,),
        ).fetchone()[0]
        profit_today = self.conn.execute(
            "SELECT COALESCE(SUM(profit), 0) FROM signals WHERE user_decision = 'placed' AND settled_at >= ?",
            (since,),
        ).fetchone()[0]
        streak = 0
        reset_at = self.get("streak_reset_at", "")
        for row in self.conn.execute(
            "SELECT status, settled_at FROM signals WHERE user_decision = 'placed' "
            "AND status IN ('won','lost') ORDER BY settled_at DESC"
        ):
            if reset_at and row["settled_at"] <= reset_at:
                break
            if row["status"] == "lost":
                streak += 1
            else:
                break
        return RiskState(self.bankroll(), bets_today, float(profit_today), streak)

    def reset_streak(self) -> None:
        self.set("streak_reset_at", _now().isoformat())

    # ---------- сигналы ----------
    def add_signal(self, **fields) -> int | None:
        fields.setdefault("created_at", _now().isoformat())
        cols = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        try:
            cur = self.conn.execute(f"INSERT INTO signals({cols}) VALUES({marks})", tuple(fields.values()))
        except sqlite3.IntegrityError:
            return None  # такой сигнал уже есть
        self.conn.commit()
        return cur.lastrowid

    @staticmethod
    def _row(row: sqlite3.Row) -> Signal:
        d = dict(row)
        d["kickoff"] = datetime.fromisoformat(d["kickoff"])
        return Signal(**d)

    def signal(self, signal_id: int) -> Signal | None:
        row = self.conn.execute("SELECT * FROM signals WHERE id = ?", (signal_id,)).fetchone()
        return self._row(row) if row else None

    def has_signal(self, event_id: str) -> bool:
        return bool(self.conn.execute("SELECT 1 FROM signals WHERE event_id = ?", (event_id,)).fetchone())

    def pending(self) -> list[Signal]:
        rows = self.conn.execute("SELECT * FROM signals WHERE status = 'pending' ORDER BY kickoff")
        return [self._row(r) for r in rows]

    def recent(self, limit: int = 10) -> list[Signal]:
        rows = self.conn.execute(
            "SELECT * FROM signals WHERE status != 'pending' ORDER BY settled_at DESC LIMIT ?", (limit,)
        )
        return [self._row(r) for r in rows]

    def settled(self, since: str | None = None) -> list[Signal]:
        q = "SELECT * FROM signals WHERE status IN ('won','lost','void')"
        args: tuple = ()
        if since:
            q += " AND settled_at >= ?"
            args = (since,)
        return [self._row(r) for r in self.conn.execute(q + " ORDER BY settled_at", args)]

    def decide(self, signal_id: int, placed: bool, odds: float | None = None, stake: float | None = None) -> None:
        self.conn.execute(
            "UPDATE signals SET user_decision = ?, actual_odds = ?, actual_stake = ?, decided_at = ? WHERE id = ?",
            ("placed" if placed else "skipped", odds, stake, _now().isoformat(), signal_id),
        )
        self.conn.commit()

    def expire_undecided(self) -> int:
        """Матч начался, а отметки нет: считаем, что ставка не сделана (лайв не принимаем)."""
        cur = self.conn.execute(
            "UPDATE signals SET user_decision = 'expired' WHERE user_decision = 'pending' AND kickoff <= ?",
            (_now().isoformat(),),
        )
        self.conn.commit()
        return cur.rowcount

    def settle(self, signal_id: int, status: str, score: str, profit: float, paper_profit: float) -> None:
        self.conn.execute(
            "UPDATE signals SET status = ?, score = ?, profit = ?, paper_profit = ?, settled_at = ? WHERE id = ?",
            (status, score, round(profit, 2), round(paper_profit, 2), _now().isoformat(), signal_id),
        )
        self.conn.commit()
        if profit:
            self.set("bankroll", round(self.bankroll() + profit, 2))

    # ---------- расход LLM ----------
    def analyses_today(self) -> int:
        row = self.conn.execute("SELECT analyses FROM llm_usage WHERE day = ?", (local_today(),)).fetchone()
        return row["analyses"] if row else 0

    def add_llm_usage(self, analyses: int = 0, input_tokens: int = 0, output_tokens: int = 0,
                      web_searches: int = 0, cost_usd: float = 0.0) -> None:
        self.conn.execute(
            """INSERT INTO llm_usage(day, analyses, input_tokens, output_tokens, web_searches, cost_usd)
               VALUES(?, ?, ?, ?, ?, ?)
               ON CONFLICT(day) DO UPDATE SET
                 analyses = analyses + excluded.analyses,
                 input_tokens = input_tokens + excluded.input_tokens,
                 output_tokens = output_tokens + excluded.output_tokens,
                 web_searches = web_searches + excluded.web_searches,
                 cost_usd = cost_usd + excluded.cost_usd""",
            (local_today(), analyses, input_tokens, output_tokens, web_searches, cost_usd),
        )
        self.conn.commit()

    def llm_usage_since(self, days: int = 30) -> dict:
        since = (datetime.now(_tz()).date() - timedelta(days=days)).isoformat()
        row = self.conn.execute(
            """SELECT COALESCE(SUM(analyses),0) a, COALESCE(SUM(input_tokens),0) i,
                      COALESCE(SUM(output_tokens),0) o, COALESCE(SUM(web_searches),0) w,
                      COALESCE(SUM(cost_usd),0) c
               FROM llm_usage WHERE day >= ?""",
            (since,),
        ).fetchone()
        return {"analyses": row["a"], "input_tokens": row["i"], "output_tokens": row["o"],
                "web_searches": row["w"], "cost_usd": row["c"]}
