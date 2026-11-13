"""Финансист: размер ставки и защитные лимиты.

Принципы:
* ставка по дробному критерию Келли (доля — по профилю риска) от свободного банка
  (банк минус открытые ставки); в НБА «по анализу» — от уверенности и вероятности;
* предохранитель MAX_STAKE_PCT на одну ставку (умеренный профиль — 5%);
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
    peak: float = 0.0  # максимум банка (0 — считать равным текущему)
    week_profit: float = 0.0  # результат ваших ставок за неделю
    in_play: float = 0.0  # сумма открытых ставок

    @property
    def free(self) -> float:
        """Свободные деньги: банк минус уже поставленное в открытых ставках."""
        return max(self.bankroll - self.in_play, 0.0)

    @property
    def drawdown(self) -> float:
        peak = max(self.peak, self.bankroll)
        return (peak - self.bankroll) / peak if peak > 0 else 0.0


@dataclass
class RiskMode:
    multiplier: float  # во сколько раз уменьшить ставку
    label: str  # «✅ обычный», «⚠️ осторожный» …
    stop: str | None  # причина остановки ставок


def risk_mode(state: RiskState, cfg: Settings) -> RiskMode:
    """Режим финансиста по просадке от пика банка и серии проигрышей."""
    dd = state.drawdown
    if dd >= cfg.dd_stop:
        return RiskMode(0.0, f"⛔ стоп (просадка {dd:.0%})",
                        f"банк просел на {dd:.0%} от пика — ставки остановлены. Разберите статистику; "
                        "продолжить с новой точкой отсчёта: /resume")
    if dd >= cfg.dd_defense:
        mult, label = 0.25, f"🛡 защита (просадка {dd:.0%})"
    elif dd >= cfg.dd_caution:
        mult, label = 0.5, f"⚠️ осторожный (просадка {dd:.0%})"
    else:
        mult, label = 1.0, "✅ обычный" + (f" (просадка {dd:.0%})" if dd >= 0.01 else "")
    if state.losing_streak >= 2:
        mult *= 0.75
        label += f", {state.losing_streak} проигрыша подряд"
    return RiskMode(mult, label, None)


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


def risk_block(state: RiskState, cfg: Settings) -> str | None:
    """Причина, по которой финансист сегодня не даёт ставить (лимиты), или None."""
    if state.bankroll <= 0:
        return "Банк пуст. Задайте его командой /bank."
    if state.bets_today >= cfg.max_bets_per_day:
        return f"Лимит {cfg.max_bets_per_day} ставок в день исчерпан."
    if state.profit_today <= -cfg.daily_loss_limit_pct * state.bankroll:
        return "Дневной лимит убытка достигнут. Сегодня ставок больше нет."
    if cfg.stop_after_losses and state.losing_streak >= cfg.stop_after_losses:
        return (f"{state.losing_streak} проигрышей подряд. Новые сигналы на паузе: сделайте перерыв "
                "и посмотрите /stats. Снять паузу: /resume.")
    if state.week_profit <= -cfg.weekly_loss_limit_pct * state.bankroll:
        return (f"недельный лимит убытка {cfg.weekly_loss_limit_pct:.0%} банка достигнут "
                f"({state.week_profit:+,.0f}). До понедельника ставок нет; снять: /resume.".replace(",", " "))
    mode = risk_mode(state, cfg)
    if mode.stop:
        return mode.stop
    if state.in_play >= cfg.max_exposure_pct * state.bankroll:
        return (f"в игре уже {state.in_play:,.0f} — это лимит одновременных ставок "
                f"({cfg.max_exposure_pct:.0%} банка). Дождитесь расчёта.".replace(",", " "))
    return None


def analysis_pct(p: float, confidence: str, cfg: Settings) -> float:
    """НБА «по анализу»: доля банка от уверенности босса и вероятности захода.
    База профиля × (1.5 при high) × коэффициент вероятности: 50% → ×0.67, 55% → ×1, 65% → ×1.67 (от ×0.5 до ×2)."""
    conf = 1.5 if confidence == "high" else 1.0
    prob = min(max((p - 0.40) / 0.15, 0.5), 2.0)
    return cfg.nba_stake_pct * conf * prob


def _scaled(state: RiskState, cfg: Settings, pct: float, match_in_play: float = 0.0) -> tuple[float, list[str]]:
    """Доля банка с учётом режима и лимитов денег в игре (всего и на один матч)."""
    notes = []
    if pct > cfg.max_stake_pct:
        pct = cfg.max_stake_pct
        notes.append(f"сработал предохранитель: не больше {cfg.max_stake_pct:.0%} банка на одну ставку")
    mode = risk_mode(state, cfg)
    if mode.multiplier < 1:
        notes.append(f"режим {mode.label}: ставка ×{mode.multiplier:g}")
    pct *= mode.multiplier
    bank = state.bankroll
    room = cfg.max_exposure_pct * bank - state.in_play
    if bank > 0 and pct * bank > room:
        pct = max(room, 0) / bank
        notes.append(f"урезал: всего в игре не больше {cfg.max_exposure_pct:.0%} банка")
    match_room = cfg.max_match_pct * bank - match_in_play
    if bank > 0 and pct * bank > match_room:
        pct = max(match_room, 0) / bank
        notes.append(f"урезал: на один матч не больше {cfg.max_match_pct:.0%} банка")
    return pct, notes


def flat_stake(state: RiskState, cfg: Settings, pct: float) -> StakeDecision:
    """Фиксированная доля банка (ставки «по анализу», без расчёта перевеса), с теми же лимитами."""
    block = risk_block(state, cfg)
    if block:
        return StakeDecision(False, 0, 0, block)
    pct, notes = _scaled(state, cfg, pct)
    stake = round_stake(state.free * pct)
    if stake <= 0:
        return StakeDecision(False, 0, 0, "Ставка получается слишком маленькой.")
    return StakeDecision(True, stake, pct, f"{pct:.1%} свободного банка" + "".join("; " + n for n in notes))


def decide_stake(p: float, odds: float, state: RiskState, cfg: Settings) -> StakeDecision:
    block = risk_block(state, cfg)
    if block:
        return StakeDecision(False, 0, 0, block)

    k = kelly(p, odds)
    if k <= 0:
        return StakeDecision(False, 0, 0, "Нет математического перевеса.")
    pct, notes = _scaled(state, cfg, k * cfg.kelly_fraction)
    stake = round_stake(state.free * pct)
    if stake <= 0:
        return StakeDecision(False, 0, 0, "Ставка получается слишком маленькой.")
    return StakeDecision(True, stake, pct, f"{cfg.kelly_fraction:g} Келли по шансу и цене"
                         + "".join("; " + n for n in notes))


@dataclass
class BetPlan:
    """Разбор финансиста для конкретного кэфа владельца."""
    p: float  # шанс захода (оценка босса)
    odds: float
    ev: float  # перевес: p × кэф − 1
    stake: float  # рекомендованная сумма (0 — не ставить)
    pct: float  # доля банка
    verdict: str  # «ставить» / «не ставить» / «на свой риск»
    reason: str
    need_odds: float  # кэф, с которого появляется перевес MIN_EDGE
    max_own: float  # потолок, если владелец всё же ставит без рекомендации

    @property
    def win(self) -> float:
        return round(self.stake * (self.odds - 1), 2)


def plan_bet(p: float, odds: float, state: RiskState, cfg: Settings,
             by_analysis: bool = False, confidence: str = "medium", match_in_play: float = 0.0) -> BetPlan:
    """Сколько ставить по кэфу владельца.

    Обычный режим: 1/4 Келли от перевеса при ВАШЕМ кэфе, не больше MAX_STAKE_PCT; перевес меньше
    MIN_EDGE — не ставить. НБА «по анализу»: фиксированная доля по уверенности босса.
    Лимиты (дневной убыток, серия проигрышей, число ставок) действуют всегда.
    """
    ev = edge(p, odds)
    need = min_acceptable_odds(p, cfg.min_edge)
    own = round_stake(state.bankroll * cfg.paper_stake_pct * risk_mode(state, cfg).multiplier)
    block = risk_block(state, cfg)
    if block:
        return BetPlan(p, odds, ev, 0, 0, "не ставить", block, need, 0)
    if by_analysis:
        if confidence not in ("medium", "high"):
            return BetPlan(p, odds, ev, 0, 0, "на свой риск",
                           "разбор слабый (низкая уверенность)", need, own)
        pct = analysis_pct(p, confidence, cfg)
        reason = (f"по анализу: уверенность {'высокая' if confidence == 'high' else 'средняя'}, "
                  f"шанс {p:.0%}")
        if ev < 0:
            pct /= 2  # по нашей же оценке кэф мал — половина обычной ставки
            reason += f"; кэф ниже нашей оценки (в плюс от {1 / p:.2f}) — ставку режу вдвое"
        pct, notes = _scaled(state, cfg, pct, match_in_play)
        reason += "".join("; " + n for n in notes)
        stake = round_stake(state.free * pct)
        return BetPlan(p, odds, ev, stake, pct, "ставить" if stake > 0 else "не ставить", reason, need, own)
    if ev < cfg.min_edge:
        why = (f"перевес {ev:+.1%} меньше порога {cfg.min_edge:.0%}" if ev >= 0
               else f"при этом кэфе ставка в минус ({ev:+.1%})")
        return BetPlan(p, odds, ev, 0, 0, "не ставить", why + f"; нужен кэф от {need:.2f}", need, own)
    if not cfg.min_odds <= odds <= cfg.max_odds:
        return BetPlan(p, odds, ev, 0, 0, "не ставить",
                       f"кэф вне рабочего диапазона {cfg.min_odds}–{cfg.max_odds}", need, own)
    k = kelly(p, odds)
    pct, notes = _scaled(state, cfg, k * cfg.kelly_fraction, match_in_play)
    stake = round_stake(state.free * pct)
    return BetPlan(p, odds, ev, stake, pct, "ставить" if stake > 0 else "не ставить",
                   f"шанс {p:.0%} и перевес {ev:+.1%} → {cfg.kelly_fraction:g} Келли"
                   + "".join("; " + n for n in notes), need, own)
