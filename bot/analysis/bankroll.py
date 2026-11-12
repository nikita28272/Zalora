"""Финансист: размер ставки и защитные лимиты.

Принципы:
* ставка по дробному критерию Келли (по умолчанию 1/4 Келли);
* не больше MAX_STAKE_PCT банка на одну ставку;
* не больше MAX_BETS_PER_DAY ставок в день;
* стоп на день, если дневной убыток превысил DAILY_LOSS_LIMIT_PCT;
* пауза после серии из STOP_AFTER_LOSSES проигрышей подряд.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings


def kelly(p: float, odds: float) -> float:
    """Доля банка по Келли. Отрицательное значение означает отсутствие перевеса."""
    b = odds - 1.0
    if b <= 0:
        return 0.0
    return (p * b - (1.0 - p)) / b


def edge(p: float, odds: float) -> float:
    """Математическое ожидание на 1 рубль ставки."""
    return p * odds - 1.0


def min_acceptable_odds(p: float, min_edge: float) -> float:
    """Минимальный коэффициент, при котором ставка ещё имеет перевес min_edge."""
    return (1.0 + min_edge) / p


@dataclass
class RiskState:
    bankroll: float
    bets_today: int
    profit_today: float
    losing_streak: int


@dataclass
class StakeDecision:
    allowed: bool
    stake: float
    stake_pct: float
    reason: str


def round_stake(amount: float) -> float:
    if amount >= 1000:
        return float(round(amount / 50) * 50)
    if amount >= 100:
        return float(round(amount / 10) * 10)
    return float(round(amount))


def decide_stake(p: float, odds: float, state: RiskState, cfg: Settings) -> StakeDecision:
    if state.bankroll <= 0:
        return StakeDecision(False, 0, 0, "Банк пуст. Задайте его командой /bank.")
    if state.bets_today >= cfg.max_bets_per_day:
        return StakeDecision(False, 0, 0, f"Лимит {cfg.max_bets_per_day} ставок в день исчерпан.")
    if state.profit_today <= -cfg.daily_loss_limit_pct * state.bankroll:
        return StakeDecision(False, 0, 0, "Дневной лимит убытка достигнут. Сегодня ставок больше нет.")
    if cfg.stop_after_losses and state.losing_streak >= cfg.stop_after_losses:
        return StakeDecision(
            False, 0, 0,
            f"{state.losing_streak} проигрышей подряд. Новые сигналы на паузе: сделайте перерыв "
            "и посмотрите /stats. Снять паузу: /resume.",
        )

    k = kelly(p, odds)
    if k <= 0:
        return StakeDecision(False, 0, 0, "Нет математического перевеса.")
    pct = min(k * cfg.kelly_fraction, cfg.max_stake_pct)
    stake = round_stake(state.bankroll * pct)
    if stake <= 0:
        return StakeDecision(False, 0, 0, "Ставка получается слишком маленькой.")
    return StakeDecision(True, stake, pct, f"{cfg.kelly_fraction:g} Келли, потолок {cfg.max_stake_pct:.0%} банка")
