"""Главный конвейер.

Порядок (иерархия):
  1. Статистик (код) и рынок дают цифры по каждому исходу.
  2. Главный ставит задачи подчинённым: новости и зарубежные аналитики (параллельно),
     затем адвокат дьявола. Каждый обязан высказаться по всем исходам вида спорта
     (футбол: 1X2 и тотал; хоккей: ещё победа с ОТ и фора).
  3. Финансист докладывает состояние банка и лимиты.
  4. Только после ВСЕХ докладов главный выносит вердикт. Если кто-то не доложил даже
     после повторного задания, вердикт не выносится.
  5. Владелец отмечает в Telegram, поставил ли он. Финансист учитывает только реальные ставки.
  6. После матча аналитик проверяет исход, финансист меняет банк, главный докладывает владельцу.

Работаем только в прематче.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable
from zoneinfo import ZoneInfo

from .agents import team
from .agents.llm import AgentError, AgentRun
from .agents.schemas import ChiefOutcome, ChiefVerdict, DevilReport, ExpertsReport, NewsReport
from .analysis.bankroll import (StakeDecision, analysis_pct, decide_stake, edge, flat_stake,
                                min_acceptable_odds, risk_mode, round_stake)
from .analysis.market import MarketView, market_view
from .analysis.altlines import alt_market, safe_options, safe_text
from .analysis.basketball import NbaModel
from .analysis.nba_factors import Factors, nba_factors
from .analysis.poisson import ModelOutput, PoissonModel, sport_probs
from .config import LEAGUES_BY_ODDS_KEY, RISK_NAMES_RU, League, settings
from .data import news
from .data.nba import NbaClient, standings_from_results
from .data.nba import season_start as nba_season_start
from .data.nhl import NhlClient
from .data.nhl import season_start as nhl_season_start
from .data.odds import Event, OddsClient, Quota, ScanReport
from .data.stats import Result, StandingRow, StatsClient, football_season_start
from .data.teams import best_match
from .db import DB, PAPER_UNIT, Signal
from .sports import FOOTBALL, SPORTS, Sport, group_of, line_for, sel_ru, sel_short

log = logging.getLogger(__name__)

# Совместимость со старым кодом и тестами: подписи футбольных исходов
SEL_RU = {o: sel_ru(FOOTBALL, o, 2.5 if o in ("over", "under") else None) for o in FOOTBALL.outcomes}
SEL_SHORT = {o: sel_short(FOOTBALL, o, 2.5 if o in ("over", "under") else None) for o in FOOTBALL.outcomes}


def settle_outcome(sport_key: str, selection: str, line: float | None, r: Result) -> str:
    """Итог ставки по результату матча: won / lost / void (возврат при целой линии)."""
    fh, fa = r.final  # итог с овертаймом (в футболе совпадает с основным временем)
    rh, ra = r.home_goals, r.away_goals  # основное время
    if selection == "home":
        return "won" if rh > ra else "lost"
    if selection == "draw":
        return "won" if rh == ra else "lost"
    if selection == "away":
        return "won" if rh < ra else "lost"
    if selection == "dc_home":  # двойной шанс 1X (основное время)
        return "won" if rh >= ra else "lost"
    if selection == "dc_away":  # X2
        return "won" if rh <= ra else "lost"
    if selection == "ml_home":
        return "won" if fh > fa else "lost"
    if selection == "ml_away":
        return "won" if fa > fh else "lost"
    if selection in ("over", "under"):
        total = fh + fa
        ln = line if line is not None else SPORTS[sport_key].default_total
        if total == ln:
            return "void"
        return "won" if (total > ln) == (selection == "over") else "lost"
    if selection in ("hcp_home", "hcp_away"):
        diff = (fh - fa) if selection == "hcp_home" else (fa - fh)
        ln = line if line is not None else -1.5
        margin = diff + ln
        if margin == 0:
            return "void"
        return "won" if margin > 0 else "lost"
    raise ValueError(f"Неизвестный исход {selection}")


def same_scenario(ctx: "MatchContext", a: str, b: str, lines: dict | None = None) -> str:
    """Почему два исхода одного матча — по сути одна ставка (или пусто, если независимы)."""
    pair = {a, b}
    for side in ("home", "away"):
        if pair <= {side, f"ml_{side}", f"hcp_{side}", f"dc_{side}"}:
            return "победа и фора за одну команду"
    hcp = next((s for s in pair if s.startswith("hcp_")), None)
    total = next((s for s in pair if s in ("over", "under")), None)
    if hcp and total:
        line = (lines or {}).get(hcp)
        line = ctx.line(hcp) if line is None else line
        if line is not None and line > 0 and total == "under":
            return "фора андердога и «меньше» — оба про вязкую игру с малым счётом"
        if line is not None and line < 0 and total == "over":
            return "фора фаворита и «больше» — оба про результативный разгром"
    return ""


def days_until_reset(today, reset_day: int) -> int:
    """Сколько дней (включая сегодня) осталось до обновления месячного лимита кредитов."""
    from datetime import date

    reset_day = min(max(reset_day, 1), 28)
    if today.day < reset_day:
        nxt = date(today.year, today.month, reset_day)
    else:
        y, m = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
        nxt = date(y, m, reset_day)
    return max((nxt - today).days, 1)


def score_text(r: Result) -> str:
    fh, fa = r.final
    if r.decided in ("OT", "SO"):
        if (r.home_goals, r.away_goals) == (fh, fa):  # счёт основного времени неизвестен
            return f"{fh}:{fa} {'ОТ' if r.decided == 'OT' else 'Б'}"
        return f"{fh}:{fa} {'ОТ' if r.decided == 'OT' else 'Б'} (осн. время {r.home_goals}:{r.away_goals})"
    return f"{fh}:{fa}"


def local(dt: datetime) -> datetime:
    return dt.astimezone(ZoneInfo(settings.timezone))


def is_prematch(kickoff: datetime) -> bool:
    return kickoff - datetime.now(timezone.utc) >= timedelta(minutes=settings.min_minutes_before_kickoff)


@dataclass
class MatchContext:
    idx: int
    event: Event
    market: MarketView
    model: ModelOutput | None = None
    fd_home: str | None = None
    fd_away: str | None = None
    home_row: StandingRow | None = None
    away_row: StandingRow | None = None
    home_form: str = ""
    away_form: str = ""
    home_last: list[str] = field(default_factory=list)
    away_last: list[str] = field(default_factory=list)
    factors: Factors | None = None  # НБА: перекос графика (считается при загрузке матчей)
    factors_text: str = ""  # то же текстом для досье (сохраняется в снимке)

    @property
    def title(self) -> str:
        return f"{self.event.home_team} — {self.event.away_team}"

    @property
    def sport(self) -> Sport:
        return SPORTS[self.event.league.sport]

    def line(self, sel: str) -> float | None:
        return line_for(sel, self.market.lines)

    def short(self, sel: str) -> str:
        return sel_short(self.sport, sel, self.line(sel))

    def ru(self, sel: str) -> str:
        return sel_ru(self.sport, sel, self.line(sel))

    def p_quant(self, sel: str) -> float | None:
        """Оценка до агентов: рынок, сдвинутый в сторону модели на долю MODEL_WEIGHT.

        Линия топ-лиг точнее модели на 20 матчах, поэтому модель лишь корректирует рынок.
        """
        pm = self.market.fair.get(sel)
        if pm is None:
            return None
        if self.model is None or sel not in self.model.probs:
            return pm
        return pm + settings.model_weight * (self.model.probs[sel] - pm)

    def model_gap(self, sel: str) -> float | None:
        """На сколько модель расходится с рынком по исходу."""
        pm = self.market.fair.get(sel)
        if pm is None or self.model is None or sel not in self.model.probs:
            return None
        return self.model.probs[sel] - pm

    def best_chance(self) -> tuple[str, float] | None:
        """Самый перспективный исход матча для разбора: (исход, перевес по модели+рынку при среднем кэфе).

        Те же фильтры, что у кандидатов (реальный кэф, рабочий диапазон, модель не дальше
        MAX_MODEL_GAP от рынка), но без порога MIN_EDGE: нужен рейтинг, а не сигнал.
        """
        best: tuple[str, float] | None = None
        for sel, odds in self.market.avg_odds.items():
            if sel in self.market.estimated or not (settings.min_odds <= odds <= settings.max_odds):
                continue
            gap = self.model_gap(sel)
            if gap is not None and abs(gap) > settings.max_model_gap:
                continue
            p = self.p_quant(sel)
            if p is None:
                continue
            ev = edge(p, odds)
            if best is None or ev > best[1]:
                best = (sel, ev)
        return best

    def quant_candidates(self) -> list[tuple[str, float]]:
        """Исходы с перевесом не меньше MIN_EDGE по модели+рынку, лучший первым."""
        if self.model is None:
            return []
        out = []
        for sel, odds in self.market.avg_odds.items():
            if sel in self.market.estimated:  # оценочный кэф — не повод для кандидата
                continue
            gap = self.model_gap(sel)
            if gap is not None and abs(gap) > settings.max_model_gap:
                continue  # модель слишком далеко от рынка — скорее ошибка модели, чем находка
            if not (settings.min_odds <= odds <= settings.max_odds):
                continue
            p = self.p_quant(sel)
            if p is None:
                continue
            ev = edge(p, odds)
            if ev >= settings.min_edge:  # мелкий «перевес» — почти всегда шум модели
                out.append((sel, ev))
        return sorted(out, key=lambda x: x[1], reverse=True)

    def brief(self) -> str:
        text = (
            f"{self.sport.emoji} {self.sport.name_ru}. {self.event.league.name_ru}: "
            f"{self.event.home_team} (хозяева) — {self.event.away_team} (гости)\n"
            f"Начало: {local(self.event.kickoff):%d.%m.%Y %H:%M} ({settings.timezone}). Ставка — только прематч."
        )
        if self.sport.key == "hockey":
            text += ("\nХоккей: «П1/П2 ОТ» — победа с учётом овертайма и буллитов; «осн» — исход основного времени;"
                     " тотал и фора — по итоговому счёту с овертаймом (победа в буллитах = +1 гол).")
        if self.sport.key == "basketball":
            text += "\nБаскетбол: П1/П2, фора и тотал очков — с учётом овертайма (ничьих не бывает)."
        if self.sport.key in ("hockey", "basketball"):
            text += f"\nДаты ниже — по поясу {settings.timezone}: вечерняя игра в США показана следующим числом."
        return text

    def numbers(self) -> str:
        m = self.market
        lines = [f"ДОКЛАД РЫНКА ({m.n_books} букмекеров, источник честной линии: {m.source}):"]
        for sel in self.sport.outcomes:
            if sel in m.fair:
                est = " (оценка по линии П1/П2 ОТ: 1X2 основного времени нет в линии)" if sel in m.estimated else ""
                lines.append(
                    f"  {sel} = {self.short(sel)}: вероятность {m.fair[sel]:.1%}, средний кэф "
                    f"{m.avg_odds.get(sel, 0):.2f}, лучший {m.best_odds.get(sel, 0):.2f}{est}"
                )
        if self.model:
            mo = self.model
            if self.sport.key == "basketball":
                lines.append(
                    f"ДОКЛАД СТАТИСТИКА (рейтинги нападения и защиты в очках, текущий + прошлый сезон): ожидаемые "
                    f"очки {mo.lambda_home:.1f} : {mo.lambda_away:.1f}, разница {mo.lambda_home - mo.lambda_away:+.1f},"
                    f" сумма {mo.lambda_home + mo.lambda_away:.1f} (матчей в выборке: {mo.home_games} и "
                    f"{mo.away_games}; прошлый сезон с весом ×0.5)"
                )
            else:
                lines.append(
                    f"ДОКЛАД СТАТИСТИКА (модель Пуассона, текущий + прошлый сезон): ожидаемые голы {mo.lambda_home:.2f}"
                    f" : {mo.lambda_away:.2f} (матчей в выборке с прошлым сезоном: {mo.home_games} и {mo.away_games};"
                    f" прошлый сезон с весом ×0.7 и затуханием)"
                )
            if mo.note:
                lines.append(f"  Особо: {mo.note}")
        if self.factors_text:
            lines.append(self.factors_text)
        if self.factors_text and self.model:
            lines.append("ВЕРОЯТНОСТИ ПО МОДЕЛИ (рейтинги и усталость, без травм):")
            for sel in self.sport.outcomes:
                if sel in mo.probs:
                    lines.append(f"  {self.short(sel)}: {mo.probs[sel]:.1%}")
        else:
            lines.append("ДОКЛАД СТАТИСТИКА: недостаточно данных (например, еврокубок, начало сезона "
                         "или источник статистики недоступен).")
        for name, row, form, last in (
            (self.event.home_team, self.home_row, self.home_form, self.home_last),
            (self.event.away_team, self.away_row, self.away_form, self.away_last),
        ):
            lines.append(f"{name}:")
            if row:
                if self.sport.key == "basketball":
                    lines.append(f"  место {row.position} в лиге, {row.form}, разница очков {row.goal_diff:+d}")
                else:
                    unit = "шайб" if self.sport.key == "hockey" else "мячей"
                    lines.append(
                        f"  место {row.position}, очков {row.points} за {row.played} матчей, разница {unit} "
                        f"{row.goal_diff:+d}" + (f", {row.form}" if row.form and self.sport.key == "hockey" else "")
                    )
            if form:
                lines.append(f"  форма (старые → новые): {form}")
            for r in last:
                lines.append(f"  {r}")
        return "\n".join(lines)


@dataclass
class OutcomeRow:
    selection: str
    p_market: float | None
    p_model: float | None
    p_chief: float | None  # как оценил главный
    p_final: float | None  # после нормировки и ограничения отклонения от рынка
    odds: float | None
    edge: float | None


@dataclass
class Pick:
    """Один из двух лучших исходов босса по матчу."""
    selection: str
    p: float  # вероятность босса после ограничения ±8 п.п. от рынка
    p_market: float
    odds: float  # средний кэф из досье
    edge: float
    min_odds: float
    positive: bool  # перевес не меньше MIN_EDGE
    stake: float  # рекомендованная (или бумажная) сумма
    note: str  # слово финансиста
    hot: bool = False  # 🔥
    line: float | None = None  # линия исхода (у запасной линии — своя)
    estimated: bool = False  # кэф оценочный (запасная линия / двойной шанс)
    rank: int = 0
    signal_id: int | None = None


@dataclass
class TopPicks:
    ctx: "MatchContext"
    num: int
    picks: list[Pick]
    replaced: bool = False
    confidence: str = "medium"
    link_reason: str = ""  # 🔗 оба исхода — один сценарий


@dataclass
class AnalysisResult:
    ctx: MatchContext
    news: NewsReport | None = None
    experts: ExpertsReport | None = None
    devil: DevilReport | None = None
    verdict: ChiefVerdict | None = None
    candidate: str | None = None
    table: dict[str, OutcomeRow] = field(default_factory=dict)
    p_final: float | None = None
    odds_ref: float | None = None
    final_edge: float | None = None
    min_odds: float | None = None
    stake: StakeDecision | None = None
    signal_id: int | None = None
    skip_reason: str | None = None
    errors: list[str] = field(default_factory=list)
    cost_usd: float = 0.0


def _normalize(probs: dict[str, float], group: tuple[str, ...]) -> None:
    total = sum(probs[s] for s in group if s in probs)
    if total > 0:
        for s in group:
            if s in probs:
                probs[s] /= total


class Engine:
    def __init__(self, db: DB):
        self.db = db
        self.odds = OddsClient()
        self.stats = StatsClient()
        self.nhl = NhlClient()
        self.nba = NbaClient()
        self.matches: list[MatchContext] = []
        self._events: tuple[float, list[Event]] = (0.0, [])
        self.last_report: ScanReport | None = None
        self._headlines: tuple[float, list[news.Headline]] = (0.0, [])
        self._lock = asyncio.Lock()

    # ---------- данные ----------
    async def _league_data(self, league: League) -> tuple[PoissonModel | NbaModel | None, list[StandingRow]]:
        if league.sport == "basketball":
            return await self._nba_data()
        model = None
        rows: list[StandingRow] = []
        hockey = league.sport == "hockey"
        source = "NHL API" if hockey else "football-data"
        try:
            today = datetime.now(timezone.utc).date()
            if hockey:
                results = await self.nhl.results_with_previous()
                start = nhl_season_start(today)
            else:
                results = await self.stats.results_with_previous(league.fd_code)
                start = football_season_start(today)
            if results:
                model = PoissonModel(results, season_start=datetime(start.year, start.month, start.day,
                                                                    tzinfo=timezone.utc))
        except Exception as exc:  # noqa: BLE001 — источник может быть недоступен
            log.warning("%s: результаты %s недоступны: %s", source, league.fd_code, type(exc).__name__)
        try:
            rows = await (self.nhl.standings() if hockey else self.stats.standings(league.fd_code))
        except Exception as exc:  # noqa: BLE001
            log.warning("%s: таблица %s недоступна: %s", source, league.fd_code, type(exc).__name__)
        return model, rows

    async def _nba_data(self) -> tuple[NbaModel | None, list[StandingRow]]:
        """НБА: результаты ESPN -> модель рейтингов и таблица (её считаем сами по результатам)."""
        try:
            results = await self.nba.results_with_previous()
        except Exception as exc:  # noqa: BLE001
            log.warning("ESPN NBA: результаты недоступны: %s", type(exc).__name__)
            return None, []
        if not results:
            return None, []
        start = nba_season_start(datetime.now(timezone.utc).date())
        season_start = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
        model = NbaModel(results, season_start=season_start)
        return model, standings_from_results(model.current_season())

    async def _upcoming(self, max_age_minutes: int | None = None, force: bool = False) -> list[Event]:
        """Линии с кешем: повторные /matches и досье не тратят кредиты The Odds API.
        force — владелец сам попросил загрузить отложенные лиги: дневной бюджет не применяем
        (сохранённые линии других лиг всё равно берутся бесплатно)."""
        max_age = settings.odds_cache_minutes if max_age_minutes is None else max_age_minutes
        ts, cached = self._events
        if not force and ts and time.monotonic() - ts < max_age * 60:
            if self.last_report:
                self.last_report.cached = True
            return cached
        fresh, report = await self.odds.upcoming(allowance=None if force else self.scan_allowance(),
                                                 gate=self._odds_gate,
                                                 store=self.db, max_age_minutes=max_age)
        self._events = (time.monotonic(), fresh)
        if report.spent > 0 or self.db.get("odds_fetched_at") is None:
            self.db.set("odds_fetched_at", datetime.now(timezone.utc).isoformat())
        self.last_report = report
        self._save_quota()
        today = local(datetime.now(timezone.utc)).date().isoformat()
        self.db.set(f"odds_spent:{today}", int(self.db.get(f"odds_spent:{today}", 0)) + report.spent)
        return fresh

    async def _odds_gate(self, league: League, free_events: list[dict]) -> bool:
        """Нужны ли платные линии лиги. НБА «по анализу»: только если хотя бы у одного матча
        заметный перекос графика (считается бесплатно по результатам ESPN)."""
        if league.sport != "basketball" or settings.nba_mode != "analysis" or not settings.nba_odds_gate:
            return True
        results = await self.nba.results_with_previous()
        if not results:
            return True  # статистика недоступна — отбор не посчитать, лучше загрузить линии
        teams = {r.home for r in results} | {r.away for r in results}
        # модель ещё может добавить +1 («сильнее та же сторона»), поэтому порог с запасом
        threshold = settings.nba_min_factor - 1.0
        for ev in free_events:
            try:
                kickoff = datetime.fromisoformat(str(ev["commence_time"]).replace("Z", "+00:00"))
                home = best_match(ev["home_team"], teams)
                away = best_match(ev["away_team"], teams)
            except (KeyError, TypeError, ValueError):
                return True
            if not home or not away:
                return True  # не узнали команду — не рискуем пропустить матч
            if is_prematch(kickoff) and nba_factors(results, home, away, kickoff).score >= threshold:
                return True
        return False

    def daily_max_age(self, now: datetime | None = None) -> int:
        """Допустимый возраст линий для ежедневного списка (минуты).

        Если сегодня линии уже загружались (не раньше DAILY_REUSE_HOURS назад), берём их из кеша:
        повторная загрузка в SCAN_TIME потратила бы кредиты на почти те же данные."""
        now = now or datetime.now(timezone.utc)
        raw = self.db.get("odds_fetched_at")  # в базе: переживает перезапуск бота
        at = datetime.fromisoformat(raw) if raw else None
        hours = settings.daily_reuse_hours
        if hours > 0 and at is not None and local(at).date() == local(now).date() and now - at <= timedelta(hours=hours):
            return hours * 60
        return settings.odds_scan_max_age_minutes

    def scan_allowance(self) -> float | None:
        """Сколько кредитов можно потратить сейчас, чтобы остатка хватило до обновления лимита.

        Дневной бюджет = (остаток + потрачено сегодня − запас) / дней до обновления;
        разрешено сейчас = дневной бюджет − уже потрачено сегодня.
        """
        q = self.quota()
        if q is None:
            return None  # остаток неизвестен — не ограничиваем (узнаем после первого запроса)
        today = local(datetime.now(timezone.utc)).date()
        spent_today = int(self.db.get(f"odds_spent:{today.isoformat()}", 0))
        days_left = days_until_reset(today, settings.odds_reset_day)
        daily = max(q.remaining + spent_today - settings.odds_reserve, 0) / days_left
        return max(daily - spent_today, 0.0)

    # ---------- кредиты The Odds API ----------
    def _save_quota(self) -> None:
        q = self.odds.quota
        if q:
            self.db.set("odds_quota", {"remaining": q.remaining, "used": q.used, "checked_at": q.checked_at})

    def quota(self) -> Quota | None:
        """Последний известный остаток (переживает перезапуск бота)."""
        if self.odds.quota:
            return self.odds.quota
        raw = self.db.get("odds_quota")
        return Quota(**raw) if raw else None

    async def refresh_quota(self) -> Quota | None:
        await self.odds.check_quota()
        self._save_quota()
        return self.quota()

    async def load_matches(self, max_age_minutes: int | None = None, force: bool = False) -> list[MatchContext]:
        events = [e for e in await self._upcoming(max_age_minutes, force) if is_prematch(e.kickoff)]
        leagues = {e.league.fd_code: e.league for e in events}
        league_data = {code: await self._league_data(lg) for code, lg in sorted(leagues.items())}

        matches: list[MatchContext] = []
        for event in events:
            mv = market_view(event)
            if mv is None:
                continue
            ctx = MatchContext(idx=len(matches) + 1, event=event, market=mv)
            model, rows = league_data.get(event.league.fd_code, (None, []))
            if model:
                ctx.fd_home = best_match(event.home_team, model.teams.keys())
                ctx.fd_away = best_match(event.away_team, model.teams.keys())
                if ctx.fd_home and ctx.fd_away and ctx.fd_home != ctx.fd_away:
                    if isinstance(model, NbaModel):  # НБА: учитываем back-to-back к дате матча
                        ctx.model = model.predict(ctx.fd_home, ctx.fd_away, event.kickoff)
                    else:
                        ctx.model = model.predict(ctx.fd_home, ctx.fd_away)
                    if ctx.model:  # вероятности под рынки и линии этого вида спорта
                        ctx.model.probs = sport_probs(event.league.sport, ctx.model.lambda_home,
                                                      ctx.model.lambda_away, mv.lines, ctx.model.draw_boost,
                                                      ctx.model.sds)
                    if isinstance(model, NbaModel):
                        margin = (ctx.model.lambda_home - ctx.model.lambda_away) if ctx.model else None
                        ctx.factors = nba_factors(model.results, ctx.fd_home, ctx.fd_away, event.kickoff, margin)
                        ctx.factors_text = ctx.factors.text()
                    ctx.home_form, ctx.away_form = model.form(ctx.fd_home), model.form(ctx.fd_away)
                    ctx.home_last, ctx.away_last = model.last_results(ctx.fd_home), model.last_results(ctx.fd_away)
            if rows:
                by_name = {r.team: r for r in rows}
                h = best_match(event.home_team, by_name.keys())
                a = best_match(event.away_team, by_name.keys())
                ctx.home_row = by_name.get(h) if h else None
                ctx.away_row = by_name.get(a) if a else None
            matches.append(ctx)
        self.matches = matches
        return matches

    async def headlines(self) -> list[news.Headline]:
        ts, cached = self._headlines
        if time.monotonic() - ts < 1800 and cached:
            return cached
        fresh = await news.fetch_all()
        self._headlines = (time.monotonic(), fresh)
        return fresh

    # ---------- доклады ----------
    def finance_report(self) -> str:
        st = self.db.risk_state()
        n_play, s_play = self.db.in_play()
        mode = risk_mode(st, settings)
        return (
            "ДОКЛАД ФИНАНСИСТА:\n"
            f"  банк {st.bankroll:.0f} (пик {max(st.peak, st.bankroll):.0f}, просадка {st.drawdown:.0%}), "
            f"в игре {n_play} ставок на {s_play:.0f}\n"
            f"  режим: {mode.label}, ставки ×{mode.multiplier:g}"
            + (f" — СТАВКИ ОСТАНОВЛЕНЫ: {mode.stop}" if mode.stop else "") + "\n"
            f"  неделя {st.week_profit:+.0f} (лимит убытка {settings.weekly_loss_limit_pct:.0%} банка)\n"
            f"  ставок сегодня {st.bets_today}"
            + (f" из {settings.max_bets_per_day}" if settings.max_bets_per_day else " (без лимита)") + ", "
            f"результат дня {st.profit_today:+.0f} (лимит убытка {settings.daily_loss_limit_pct:.0%} банка)\n"
            f"  проигрышей подряд {st.losing_streak} (пауза после {settings.stop_after_losses})\n"
            f"  профиль риска: {RISK_NAMES_RU.get(settings.risk_profile, settings.risk_profile)} "
            f"({settings.kelly_fraction:g} Келли от свободного банка {st.free:.0f}); "
            f"минимальный перевес {settings.min_edge:.0%}, коэффициенты {settings.min_odds}–{settings.max_odds}"
        )

    def _dossier(self, ctx: MatchContext, res: AnalysisResult, for_chief: bool) -> str:
        parts = [f"МАТЧ\n{ctx.brief()}", ctx.numbers()]
        if res.news:
            parts.append("ДОКЛАД АГЕНТА НОВОСТЕЙ:\n" + res.news.model_dump_json(indent=1))
        if res.experts:
            parts.append("ДОКЛАД ПО ЗАРУБЕЖНЫМ АНАЛИТИКАМ:\n" + res.experts.model_dump_json(indent=1))
        elif not settings.web_search_enabled:
            parts.append("ДОКЛАД ПО ЗАРУБЕЖНЫМ АНАЛИТИКАМ: отключён (нет веб-поиска).")
        if res.candidate:
            sel = res.candidate
            p = ctx.p_quant(sel) or 0
            parts.append(
                f"КАНДИДАТ ОТ КОЛИЧЕСТВЕННОГО АНАЛИЗА: {ctx.ru(sel)}, средний кэф {ctx.market.avg_odds[sel]:.2f}, "
                f"оценка вероятности (рынок+модель) {p:.1%}, ожидание {edge(p, ctx.market.avg_odds[sel]):+.1%}"
            )
        else:
            parts.append("КАНДИДАТ ОТ КОЛИЧЕСТВЕННОГО АНАЛИЗА: нет (модель не нашла перевеса или данных мало).")
        if for_chief:
            if res.devil:
                parts.append("ДОКЛАД АДВОКАТА ДЬЯВОЛА:\n" + res.devil.model_dump_json(indent=1))
            parts.append(self.finance_report())
        return "\n\n".join(parts)

    def _track(self, res: AnalysisResult, run: AgentRun) -> None:
        res.cost_usd += run.cost_usd
        self.db.add_llm_usage(input_tokens=run.input_tokens, output_tokens=run.output_tokens,
                              web_searches=run.web_searches, cost_usd=run.cost_usd)

    async def _order(self, res: AnalysisResult, who: str, task: Callable[[], Awaitable[AgentRun]]):
        """Поручение подчинённому: при сбое главный повторяет задание один раз."""
        for attempt in (1, 2):
            try:
                run = await task()
                self._track(res, run)
                return run.result
            except AgentError as exc:
                log.warning("%s: попытка %d не удалась: %s", who, attempt, exc)
                if attempt == 2:
                    res.errors.append(f"{who}: {exc}")
        return None

    async def analyze(self, ctx: MatchContext) -> AnalysisResult:
        res = AnalysisResult(ctx=ctx)
        cands = ctx.quant_candidates()
        res.candidate = cands[0][0] if cands else None
        self.db.add_llm_usage(analyses=1)

        heads = news.for_match(await self.headlines(), ctx.event.home_team, ctx.event.away_team)
        heads_text = "\n".join(f"- [{h.source}] {h.title}: {h.summary} ({h.link})" for h in heads)

        # 1. Новости и зарубежные аналитики — параллельно
        sp, ln = ctx.sport, ctx.market.lines
        orders = [self._order(res, "Агент новостей", lambda: team.news_agent(ctx.brief(), heads_text, sp, ln))]
        if settings.web_search_enabled:
            orders.append(self._order(res, "Зарубежные аналитики", lambda: team.experts_agent(ctx.brief(), sp, ln)))
        reports = await asyncio.gather(*orders)
        res.news = reports[0]
        res.experts = reports[1] if len(reports) > 1 else None

        # 2. Адвокат дьявола видит всё собранное
        res.devil = await self._order(
            res, "Адвокат дьявола", lambda: team.devil_agent(self._dossier(ctx, res, for_chief=False), sp, ln)
        )

        # 3. Босс не выносит вердикт, пока не выслушал всех
        missing = [name for name, rep in (
            ("агент новостей", res.news),
            ("зарубежные аналитики", res.experts if settings.web_search_enabled else True),
            ("адвокат дьявола", res.devil),
        ) if rep is None]
        if missing:
            res.skip_reason = "Совет не собран: не доложили " + ", ".join(missing) + ". Вердикт не выносится."
            return res

        res.verdict = await self._order(
            res, "Главный аналитик", lambda: team.chief_agent(self._dossier(ctx, res, for_chief=True), sp, ln)
        )
        if res.verdict is None:
            res.skip_reason = "Главный аналитик не смог вынести вердикт."
            return res

        self._finalize(res)
        return res

    def _build_table(self, res: AnalysisResult) -> None:
        ctx, v = res.ctx, res.verdict
        assert v is not None
        chief = {o.selection: o.probability for o in v.outcomes}
        final = dict(chief)
        for group in ctx.sport.groups:
            _normalize(final, group)
        for sel in ctx.sport.outcomes:
            pm = ctx.market.fair.get(sel)
            odds = ctx.market.avg_odds.get(sel)
            pf = final.get(sel)
            if pf is not None and pm is not None:
                # защита от самоуверенности: не дальше MAX_PROB_DEVIATION от рынка
                dev = (settings.nba_max_prob_deviation if ctx.sport.key == "basketball"
                       else settings.max_prob_deviation)
                pf = min(max(pf, pm - dev), pm + dev)
                pf = min(max(pf, 0.01), 0.99)
            res.table[sel] = OutcomeRow(
                selection=sel,
                p_market=pm,
                p_model=ctx.model.probs.get(sel) if ctx.model else None,
                p_chief=chief.get(sel),
                p_final=pf if pm is not None else None,
                odds=odds,
                edge=edge(pf, odds) if (pf is not None and odds and pm is not None) else None,
            )

    def _finalize(self, res: AnalysisResult) -> None:
        ctx, v = res.ctx, res.verdict
        assert v is not None
        self._build_table(res)
        row = res.table[v.selection]
        if row.p_final is None or row.odds is None or row.p_market is None:
            res.skip_reason = "По выбранному исходу нет линии букмекеров."
            return
        p, odds = row.p_final, row.odds
        res.p_final, res.odds_ref, res.final_edge = p, odds, row.edge
        res.min_odds = min_acceptable_odds(p, settings.min_edge)

        if v.decision != "bet":
            res.skip_reason = "Главный аналитик рекомендует пропустить матч."
            return
        if not is_prematch(ctx.event.kickoff):
            res.skip_reason = "До начала матча слишком мало времени: работаем только в прематче."
            return
        if not (settings.min_odds <= odds <= settings.max_odds):
            res.skip_reason = f"Коэффициент {odds:.2f} вне рабочего диапазона {settings.min_odds}–{settings.max_odds}."
            return
        if (row.edge or 0) < settings.min_edge:
            res.skip_reason = (
                f"Перевес {row.edge:+.1%} меньше порога {settings.min_edge:.0%}: ставка не окупает маржу."
            )
            return
        if self.db.has_signal(ctx.event.event_id):
            res.skip_reason = "Сигнал на этот матч уже выдавался."
            return
        res.stake = decide_stake(p, odds, self.db.risk_state(), settings)
        if not res.stake.allowed:
            res.skip_reason = "Финансист: " + res.stake.reason
            return
        res.signal_id = self.db.add_signal(
            event_id=ctx.event.event_id,
            league_code=ctx.event.league.fd_code,
            league_name=ctx.event.league.name_ru,
            home=ctx.event.home_team,
            away=ctx.event.away_team,
            kickoff=ctx.event.kickoff.isoformat(),
            selection=v.selection,
            odds_ref=odds,
            min_odds=round(res.min_odds, 2),
            p_final=round(p, 4),
            p_market=round(row.p_market, 4),
            p_model=round(row.p_model, 4) if row.p_model is not None else None,
            edge=round(row.edge or 0, 4),
            stake=res.stake.stake,
            confidence=v.confidence,
            reasoning=v.reasoning,
            sport=ctx.sport.key,
            line=ctx.line(v.selection),
        )

    # ---------- автоматический скан ----------
    async def scan(self) -> tuple[list[MatchContext], list[AnalysisResult]]:
        async with self._lock:
            matches = await self.load_matches(max_age_minutes=settings.odds_scan_max_age_minutes)
            budget = min(
                settings.max_llm_matches_per_scan,
                settings.max_llm_analyses_per_day - self.db.analyses_today(),
            )
            ranked = sorted(
                (
                    (ctx, cands[0][1])
                    for ctx in matches
                    if (cands := ctx.quant_candidates()) and not self.db.has_signal(ctx.event.event_id)
                ),
                key=lambda x: x[1],
                reverse=True,
            )
            results = []
            for ctx, _ in ranked[: max(budget, 0)]:
                results.append(await self.analyze(ctx))
            return matches, results

    async def analyze_by_index(self, idx: int) -> AnalysisResult:
        async with self._lock:
            if not self.matches:
                await self.load_matches()
            ctx = next((m for m in self.matches if m.idx == idx), None)
            if ctx is None:
                raise ValueError("Нет матча с таким номером. Обновите список: /matches")
            if not is_prematch(ctx.event.kickoff):
                raise ValueError("Матч скоро начнётся или уже идёт. Работаем только в прематче.")
            if self.db.analyses_today() >= settings.max_llm_analyses_per_day:
                raise ValueError(
                    f"Дневной лимит разборов ({settings.max_llm_analyses_per_day}) исчерпан. "
                    "Он защищает бюджет на Claude API; поменять можно в MAX_LLM_ANALYSES_PER_DAY."
                )
            return await self.analyze(ctx)

    # ---------- режим «босс в чате» ----------
    @staticmethod
    def _snapshot(ctx: MatchContext) -> dict:
        m, mo = ctx.market, ctx.model
        row = lambda r: None if r is None else {  # noqa: E731
            "position": r.position, "team": r.team, "played": r.played, "points": r.points,
            "goal_diff": r.goal_diff, "form": r.form,
        }
        return {
            "event_id": ctx.event.event_id,
            "league": ctx.event.league.odds_key,
            "kickoff": ctx.event.kickoff.isoformat(),
            "home": ctx.event.home_team,
            "away": ctx.event.away_team,
            "market": {"fair": m.fair, "avg_odds": m.avg_odds, "best_odds": m.best_odds,
                       "n_books": m.n_books, "margin": m.avg_margin_1x2, "source": m.source,
                       "lines": m.lines, "estimated": m.estimated},
            "model": None if mo is None else {
                "lh": mo.lambda_home, "la": mo.lambda_away, "probs": mo.probs,
                "hg": mo.home_games, "ag": mo.away_games, "db": mo.draw_boost, "note": mo.note,
            },
            "home_row": row(ctx.home_row), "away_row": row(ctx.away_row),
            "home_form": ctx.home_form, "away_form": ctx.away_form,
            "home_last": ctx.home_last, "away_last": ctx.away_last,
            "factors_text": ctx.factors_text,
        }

    @staticmethod
    def _from_snapshot(snap: dict, num: int) -> MatchContext:
        event = Event(snap["event_id"], LEAGUES_BY_ODDS_KEY[snap["league"]],
                      datetime.fromisoformat(snap["kickoff"]), snap["home"], snap["away"])
        mk = snap["market"]
        market = MarketView(mk["fair"], mk["avg_odds"], mk["best_odds"], mk["n_books"], mk["margin"], mk["source"],
                            mk.get("lines", {}), mk.get("estimated", []))
        mo = snap["model"]
        model = None if mo is None else ModelOutput(mo["lh"], mo["la"], mo["probs"], mo["hg"], mo["ag"],
                                                    mo.get("db", 1.0), note=mo.get("note", ""))
        row = lambda r: None if r is None else StandingRow(**r)  # noqa: E731
        return MatchContext(
            idx=num, event=event, market=market, model=model,
            home_row=row(snap["home_row"]), away_row=row(snap["away_row"]),
            home_form=snap["home_form"], away_form=snap["away_form"],
            home_last=snap["home_last"], away_last=snap["away_last"],
            factors_text=snap.get("factors_text", ""),
        )

    def save_candidate(self, ctx: MatchContext) -> int:
        """Сохраняет снимок матча для досье и возвращает его номер (тот, что идёт в /verdict)."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=3)
        for key, snap in self.db.items("cand:"):
            if datetime.fromisoformat(snap["kickoff"]) < cutoff:
                self.db.delete(key)
                self.db.delete(f"cand_event:{snap['event_id']}")
        num = self.db.get(f"cand_event:{ctx.event.event_id}")
        if num is None:
            num = int(self.db.get("cand_seq", 0)) + 1
            self.db.set("cand_seq", num)
            self.db.set(f"cand_event:{ctx.event.event_id}", num)
        self.db.set(f"cand:{num}", self._snapshot(ctx))  # свежая линия при каждом досье
        return num

    def load_candidate(self, num: int) -> MatchContext | None:
        snap = self.db.get(f"cand:{num}")
        return self._from_snapshot(snap, num) if snap else None

    def dossier(self, ctx: MatchContext, num: int) -> str:
        codes = ", ".join(f"{o} ({ctx.short(o)})" for o in ctx.sport.outcomes if o in ctx.market.fair)
        cands = ctx.quant_candidates()
        if cands:
            sel, ev = cands[0]
            cand = f"КАНДИДАТ ОТ МОДЕЛИ: {sel} = {ctx.ru(sel)}, средний кэф {ctx.market.avg_odds[sel]:.2f}, ожидание {ev:+.1%}"
        else:
            cand = "КАНДИДАТ ОТ МОДЕЛИ: нет явного перевеса."
        if ctx.sport.key == "basketball" and settings.nba_mode == "analysis":
            return "\n\n".join(x for x in [
                f"ДОСЬЕ #{num} ДЛЯ БОССА",
                ctx.brief(),
                ctx.numbers(),
                safe_text(ctx),
                self.finance_report(),
                "РЕЖИМ НБА: ПО АНАЛИЗУ (не по перевесу над линией).",
                "ЗАДАНИЕ БОССУ (Claude), подробно — в BOSS.md, раздел «НБА: работа по анализу»:\n"
                "1. Параллельно запусти субагентов «травмы и составы», «контекст: мораль, мотивация, график, "
                "перелёты», «статистика матчапа», «эксперты (зарубежные и РУ)», затем «адвоката дьявола».\n"
                f"2. Каждый докладывает по всем исходам: {codes}.\n"
                "3. Проверь факторы графика бота, оцени вероятность каждого исхода по фактам.\n"
                f"4. Выбери ДВА исхода с самым сильным обоснованием и самым высоким шансом захода среди кэфов "
                f"{settings.min_odds:.2f}–{settings.max_odds:.2f} (в том числе код@линия). Уверенность: high — 3+ "
                "независимых фактора за и нет вопросов по составу; medium — 2 фактора; low — иначе.\n"
                f"/verdict {num} <исход1> <вероятность1> <исход2> <вероятность2> <low|medium|high>\n"
                f"Коды исходов: {'|'.join(o for o in ctx.sport.outcomes if o in ctx.market.fair)}.",
            ] if x)
        return "\n\n".join(x for x in [
            f"ДОСЬЕ #{num} ДЛЯ БОССА",
            ctx.brief(),
            ctx.numbers(),
            safe_text(ctx),
            self.finance_report(),
            cand,
            "ЗАДАНИЕ БОССУ (Claude), подробно — в BOSS.md репозитория nikita28272/Zalora:\n"
            "1. Параллельно запусти субагентов «новости» (веб-поиск: травмы, составы, мотивация), "
            "«зарубежные аналитики» (веб-поиск: прогнозы экспертов, доля за каждый исход), "
            "«РУ-аналитики» (российские прогнозисты, публика, кэфы российских контор) и "
            "«продвинутая статистика» (xG, качество моментов, вратари, везение), "
            "затем «адвоката дьявола» (сильнейший аргумент против каждого исхода).\n"
            f"2. Каждый докладывает по всем исходам: {codes}.\n"
            + ("   Хоккей: обязательно выясни стартовых вратарей.\n" if ctx.sport.key == "hockey" else "")
            + ("   Баскетбол: обязательно отчёт о травмах (injury report: out / questionable), отдых звёзд, "
               "back-to-back.\n" if ctx.sport.key == "basketball" else "")
            +
            f"3. Оцени вероятность каждого исхода; от рынка дальше {settings.max_prob_deviation:.0%} не уходи "
            "без веской причины.\n"
            f"4. Выбери ДВА лучших исхода с кэфом {settings.min_odds:.2f}–{settings.max_odds:.2f} и самым высоким "
            f"шансом захода (код@линия), среди них — с перевесом от {settings.min_edge:.0%}; если таких нет — "
            "с наибольшим перевесом из оставшихся. Последней строкой — команда для бота:\n"
            f"/verdict {num} <исход1> <вероятность1> <исход2> <вероятность2> <low|medium|high>\n"
            f"Коды исходов: {'|'.join(o for o in ctx.sport.outcomes if o in ctx.market.fair)}.",
        ] if x)

    async def candidates(self) -> tuple[list[MatchContext], list[tuple[int, MatchContext]]]:
        async with self._lock:
            matches = await self.load_matches(max_age_minutes=settings.odds_scan_max_age_minutes)
            ranked = sorted(
                (
                    (ctx, cands[0][1])
                    for ctx in matches
                    if (cands := ctx.quant_candidates()) and not self.db.has_signal(ctx.event.event_id)
                ),
                key=lambda x: x[1],
                reverse=True,
            )
            out = [(self.save_candidate(ctx), ctx) for ctx, _ in ranked[: settings.manual_candidates_per_scan]]
            return matches, out

    async def dossier_by_index(self, idx: int) -> tuple[int, MatchContext]:
        async with self._lock:
            if not self.matches:
                await self.load_matches()
            ctx = next((m for m in self.matches if m.idx == idx), None)
            if ctx is None:
                raise ValueError("Нет матча с таким номером. Обновите список: /matches")
            if not is_prematch(ctx.event.kickoff):
                raise ValueError("Матч скоро начнётся или уже идёт. Работаем только в прематче.")
            return self.save_candidate(ctx), ctx

    def manual_top(self, num: int, picks: list, confidence: str = "medium",
                   hot: set[str] | None = None, link: bool = False) -> TopPicks:
        """Вердикт босса из чата: два лучших исхода матча -> финансист -> два прогноза в журнал.

        Прогнозы записываются всегда, даже без перевеса: так статистика честно покажет, как работают
        и ставки с перевесом, и «лучшие из оставшихся». Исходы с перевесом идут первыми.
        Повторный /verdict по тому же матчу заменяет прежние прогнозы, если по ним нет вашей ставки
        («Не ставил» не мешает замене).
        """
        ctx = self.load_candidate(num)
        if ctx is None:
            raise ValueError(f"Досье #{num} не найдено (или устарело). Запросите новое: «🔍 Анализ».")
        if ctx.event.kickoff - datetime.now(timezone.utc) < timedelta(minutes=settings.verdict_cutoff_minutes):
            raise ValueError("Матч вот-вот начнётся или уже идёт: работаем только в прематче, прогноз не записываю.")
        if not 1 <= len(picks) <= 2:
            raise ValueError("Нужно один или два исхода.")
        # исход: (код, вероятность) или (код, вероятность, линия) — запасная линия / двойной шанс
        picks = [(p[0], p[1], p[2] if len(p) > 2 else None) for p in picks]
        if len({sel for sel, _, _ in picks}) != len(picks):
            raise ValueError("Исходы в вердикте повторяются.")
        fair = ctx.market.fair
        alt = {}  # код -> (честная вероятность, оценочный кэф, линия)
        main = []
        for sel, p, line in picks:
            if sel in ("dc_home", "dc_away") or (line is not None and line != ctx.line(sel)):
                got = alt_market(ctx, sel, line)
                if got is None:
                    raise ValueError(f"Для исхода «{sel}» с линией {line} вероятность посчитать нельзя.")
                alt[sel] = (got[0], got[1], line)
            else:
                main.append((sel, p))
        for sel, _ in main:
            if sel not in ctx.sport.outcomes:
                raise ValueError(f"Для этого вида спорта нет исхода «{sel}».")
            if sel not in fair or not ctx.market.avg_odds.get(sel):
                raise ValueError(f"По исходу {ctx.short(sel)} нет линии букмекеров.")

        old = self.db.boss_picks(ctx.event.event_id)
        placed = [s for s in old if s.placed]
        if placed:
            names = ", ".join(ctx.short(s.selection) for s in placed)
            raise ValueError(
                f"По этому матчу у вас уже есть ставка ({names}): новый вердикт не принимаю, "
                "чтобы не потерять её учёт."
            )
        if any(s.status != "pending" for s in old):
            raise ValueError("Прогноз по этому матчу уже рассчитан: новый вердикт не принимаю.")

        # вероятности босса по всем исходам: выбранные — как сказал босс, остальные в том же рынке
        # делят остаток пропорционально рынку
        probs = {o: fair.get(o, 0.5) for o in ctx.sport.outcomes}
        chosen = dict(main)
        for group in ctx.sport.groups:
            fixed = {o: chosen[o] for o in group if o in chosen}
            if not fixed:
                continue
            left = 1 - sum(fixed.values())
            others = [o for o in group if o not in fixed]
            if left <= 0 or (not others and abs(left) > 0.02):
                raise ValueError("Вероятности исходов одного рынка в сумме больше 100%.")
            rest = sum(fair.get(o, 0) for o in others)
            for o in group:
                probs[o] = fixed[o] if o in fixed else (fair.get(o, 0) * left / rest if rest else 0)
        res = AnalysisResult(ctx=ctx)
        res.verdict = ChiefVerdict(
            outcomes=[ChiefOutcome(selection=o, probability=probs[o], comment="") for o in ctx.sport.outcomes],
            decision="bet", selection=main[0][0] if main else ctx.sport.outcomes[0], confidence=confidence,
            reasoning=f"Полный разбор с субагентами — в чате с боссом (досье #{num}).", pros=[], cons=[],
        )
        self._build_table(res)

        bank = self.db.bankroll()
        unit = round_stake(bank * settings.paper_stake_pct)
        state = self.db.risk_state()
        made: list[Pick] = []
        by_analysis = ctx.sport.key == "basketball" and settings.nba_mode == "analysis"
        dev = settings.nba_max_prob_deviation if ctx.sport.key == "basketball" else settings.max_prob_deviation
        for sel, boss_p, _ in picks:
            if sel in alt:  # запасная линия: рынок — пересчёт честной линии, кэф оценочный
                p_mkt, odds, line = alt[sel]
                p = min(max(boss_p, p_mkt - dev), p_mkt + dev, 0.99)
                p_market, ev, estimated = p_mkt, p * odds - 1, True
            else:
                row = res.table[sel]
                p, odds, p_market, ev = row.p_final, row.odds, row.p_market, row.edge or 0.0
                line, estimated = ctx.line(sel), False
            in_range = settings.min_odds <= odds <= settings.max_odds
            if by_analysis:
                # НБА: ставка по силе разбора (уверенность босса), а не по перевесу над линией
                positive = confidence in ("medium", "high") and in_range
                if positive:
                    sd = flat_stake(state, settings, analysis_pct(p, confidence, settings))
                    stake, note = (sd.stake, f"по анализу: {sd.reason}") if sd.allowed else (
                        unit, "финансист: " + sd.reason)
                else:
                    stake = unit
                    note = ("разбор слабый: если ставить, то не больше "
                            f"{settings.paper_stake_pct:.1%} банка" if in_range
                            else f"кэф вне диапазона {settings.min_odds}–{settings.max_odds}")
                made.append(Pick(sel, p, p_market, odds, ev, round(1 / p, 2), positive, stake, note,
                                 line=line, estimated=estimated))
                continue
            positive = ev >= settings.min_edge and in_range
            if positive:
                sd = decide_stake(p, odds, state, settings)
                if sd.allowed:
                    stake, note = sd.stake, f"рекомендую {sd.stake_pct:.1%} банка"
                else:
                    stake, note = unit, "финансист: " + sd.reason
            else:
                stake = unit
                note = ("перевеса нет: если ставить, то не больше "
                        f"{settings.paper_stake_pct:.1%} банка, на свой риск")
                if not settings.min_odds <= odds <= settings.max_odds and ev >= settings.min_edge:
                    note = f"кэф вне диапазона {settings.min_odds}–{settings.max_odds}; " + note
            # для ставок с перевесом — минимальный кэф с перевесом от MIN_EDGE, для прочих — «в ноль»
            min_odds = min_acceptable_odds(p, settings.min_edge) if ev >= settings.min_edge else 1 / p
            made.append(Pick(sel, p, p_market, odds, ev, round(min_odds, 2), positive, stake, note,
                             line=line, estimated=estimated))
        hot = hot or set()
        for pk in made:  # 🔥: отметил босс, или сильный исход по правилу
            auto = (confidence == "high") if by_analysis else False
            pk.hot = pk.selection in hot or (pk.positive and (auto or pk.edge >= settings.hot_min_edge))
        made.sort(key=lambda x: not x.positive)  # с перевесом — первыми

        self.db.delete_signals([s.id for s in old])
        for rank, pk in enumerate(made, 1):
            row = res.table.get(pk.selection)
            pk.rank = rank
            pk.signal_id = self.db.add_signal(
                event_id=ctx.event.event_id,
                league_code=ctx.event.league.fd_code,
                league_name=ctx.event.league.name_ru,
                home=ctx.event.home_team,
                away=ctx.event.away_team,
                kickoff=ctx.event.kickoff.isoformat(),
                selection=pk.selection,
                odds_ref=pk.odds,
                min_odds=pk.min_odds,
                p_final=round(pk.p, 4),
                p_market=round(pk.p_market, 4),
                p_model=round(row.p_model, 4) if row and row.p_model is not None and not pk.estimated else None,
                edge=round(pk.edge, 4),
                stake=pk.stake,
                confidence=confidence,
                reasoning=f"Топ-{rank} босса по досье #{num}" + (" (кэф оценочный)" if pk.estimated else ""),
                sport=ctx.sport.key,
                line=pk.line,
                source="boss",
                pick_rank=rank,
                hot=1 if pk.hot else 0,
            )
        link_reason = ""
        if len(made) == 2:
            link_reason = same_scenario(ctx, made[0].selection, made[1].selection,
                                        {pk.selection: pk.line for pk in made}) or (
                "босс: исходы зависят от одного сценария матча" if link else "")
            if link_reason and made[0].signal_id and made[1].signal_id:
                self.db.link(made[0].signal_id, made[1].signal_id)
        return TopPicks(ctx=ctx, num=num, picks=made, replaced=bool(old), confidence=confidence,
                        link_reason=link_reason)

    # ---------- отбор матчей для разбора ----------
    def analyzed_events(self) -> set[str]:
        """Матчи, по которым уже есть прогноз босса (их второй раз разбирать не нужно)."""
        return {s.event_id for s in self.db.pending() if s.source == "boss"}

    def top_for_analysis(self, matches: list[MatchContext] | None = None) -> list[tuple[MatchContext, str, float]]:
        """До TOP_MATCHES матчей, где перевес вероятнее всего: лучший исход по модели+рынку не хуже
        TOP_MIN_EDGE. Уже разобранные и слишком близкие к началу пропускаем."""
        done = self.analyzed_events()
        scored = []
        for m in (self.matches if matches is None else matches):
            if m.event.event_id in done or not is_prematch(m.event.kickoff):
                continue
            if m.sport.key == "basketball" and settings.nba_mode == "analysis":
                continue  # НБА отбирается по факторам графика, а не по перевесу (см. nba_top)
            chance = m.best_chance()
            if chance and chance[1] >= settings.top_min_edge:
                scored.append((m, chance[0], chance[1]))
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:settings.top_matches]

    def daily_picks(self, matches: list[MatchContext] | None = None,
                    now: datetime | None = None) -> list[tuple[MatchContext, str]]:
        """Матчи дня к разбору: DAILY_HOCKEY хоккея, DAILY_NBA баскетбола, DAILY_FOOTBALL футбола.

        Хоккей и футбол — где лучший исход по модели и рынку ближе всего к перевесу; НБА — с самым
        сильным перекосом графика. Сначала матчи ближайших DAILY_WINDOW_HOURS часов (чтобы завтрашние
        не дублировались), если их не хватает — дальше. Уже разобранные и начавшиеся пропускаем.
        """
        now = now or datetime.now(timezone.utc)
        done = self.analyzed_events()
        pool = [m for m in (self.matches if matches is None else matches)
                if m.event.event_id not in done and is_prematch(m.event.kickoff)]
        soon = now + timedelta(hours=settings.daily_window_hours)
        quota = {"hockey": settings.daily_hockey, "basketball": settings.daily_nba,
                 "football": settings.daily_football}
        out: list[tuple[MatchContext, str]] = []
        for sport in ("hockey", "basketball", "football"):
            cands = [m for m in pool if m.sport.key == sport]
            scored = [(self._daily_score(m), m) for m in cands]
            scored.sort(key=lambda x: (x[1].event.kickoff > soon, -x[0][0]))
            out += [(m, why) for (_, why), m in scored[:quota[sport]]]
        return out

    @staticmethod
    def _daily_score(m: MatchContext) -> tuple[float, str]:
        score, why = Engine._daily_reason(m)
        opts = safe_options(m)
        if opts:
            o = opts[0]
            why += f"; надёжный вариант: {o.label} ~{o.p:.0%}, кэф {'≈' if o.estimated else ''}{o.odds:.2f}"
        return score, why

    @staticmethod
    def _daily_reason(m: MatchContext) -> tuple[float, str]:
        if m.sport.key == "basketball" and settings.nba_mode == "analysis":
            f = m.factors
            if f is None:
                return -1.0, "статистики графика нет"
            if f.side == "—":
                return f.score, "перекоса графика нет — лучший из имеющихся"
            return f.score, (f"график в пользу {f.side_gen} ({f.score:.1f})"
                             + (": " + "; ".join(f.reasons) if f.reasons else ""))
        chance = m.best_chance()
        if chance is None:
            return -1.0, "оценки нет"
        sel, ev = chance
        mark = " 🔎" if ev >= settings.min_edge else ""
        return ev, f"{m.short(sel)} {m.market.avg_odds[sel]:.2f}, оценка {ev:+.1%}{mark}"

    def nba_top(self, matches: list[MatchContext] | None = None) -> list[MatchContext]:
        """НБА «по анализу»: до NBA_TOP матчей с самым сильным перекосом графика и усталости
        (back-to-back против отдыха, перелёты, длинный выезд, высота), не слабее NBA_MIN_FACTOR.
        Остальные матчи НБА агентам не отдаём: там у анализа нет зацепки."""
        if settings.nba_mode != "analysis":
            return []
        done = self.analyzed_events()
        picked = [
            m for m in (self.matches if matches is None else matches)
            if m.sport.key == "basketball" and m.factors is not None and m.event.event_id not in done
            and is_prematch(m.event.kickoff) and m.factors.score >= settings.nba_min_factor
        ]
        picked.sort(key=lambda m: m.factors.score, reverse=True)
        return picked[:settings.nba_top]

    # ---------- поздняя проверка ----------
    def num_for_event(self, event_id: str) -> int | None:
        return self.db.get(f"cand_event:{event_id}")

    def late_check(self, num: int) -> str:
        """Короткое задание боссу перед матчем: подтвердить составы/вратарей и оставить или заменить прогноз."""
        ctx = self.load_candidate(num)
        if ctx is None:
            raise ValueError(f"Досье #{num} не найдено (или устарело).")
        picks = [s for s in self.db.boss_picks(ctx.event.event_id) if s.status == "pending"]
        if not picks:
            raise ValueError("По этому матчу нет прогноза босса: сначала досье и /verdict.")
        m = ctx.market
        lines = [
            f"ПРОВЕРКА #{num} ДЛЯ БОССА",
            f"{ctx.sport.emoji} {ctx.sport.name_ru}. {ctx.event.league.name_ru}: {ctx.event.home_team} (хозяева) — "
            f"{ctx.event.away_team} (гости)",
            f"Начало: {local(ctx.event.kickoff):%d.%m.%Y %H:%M} ({settings.timezone}).",
            "",
            "ТЕКУЩИЙ ВЕРДИКТ:",
        ]
        mark = {"pending": "не отмечен", "placed": "ВЛАДЕЛЕЦ УЖЕ ПОСТАВИЛ", "skipped": "не ставил",
                "expired": "без отметки"}
        for s in picks:
            lines.append(
                f"  {s.pick_rank or '-'}. {s.selection} = {ctx.short(s.selection)}: босс {s.p_final:.1%}, "
                f"кэф {s.odds_ref:.2f}, перевес {s.edge:+.1%} ({mark.get(s.user_decision, s.user_decision)})"
            )
        lines += ["", "ЛИНИЯ НА МОМЕНТ ДОСЬЕ (вероятность без маржи / средний кэф):"]
        for o in ctx.sport.outcomes:
            if o in m.fair:
                est = " (оценка)" if o in m.estimated else ""
                lines.append(f"  {o} = {ctx.short(o)}: {m.fair[o]:.1%} / {m.avg_odds[o]:.2f}{est}")
        lineup = {"hockey": "стартовые вратари обеих команд",
                  "basketball": "официальный отчёт о травмах (кто out / questionable) и стартовые пятёрки"}.get(
            ctx.sport.key, "официальные стартовые составы обеих команд")
        lines += [
            "",
            "ЗАДАНИЕ БОССУ (Claude), подробно — в BOSS.md, раздел «Поздняя проверка»:",
            f"1. Запусти агента «поздняя проверка»: {lineup}, свежие травмы и снятия, "
            "движение линии с момента досье.",
            "2. Если что-то меняет оценку — дай новый /verdict (он заменит прежний, если владелец ещё не отметил ставку).",
            "3. Если ничего не меняется — так и скажи: «оставляем», без команды.",
            f"Формат: /verdict {num} <исход1> <вероятность1> <исход2> <вероятность2> <low|medium|high>",
        ]
        return "\n".join(lines)

    def due_late_checks(self, now: datetime | None = None) -> list[int]:
        """Матчи, по которым пора делать позднюю проверку (напоминание — один раз): прогнозы босса
        без отметки или с вашей ставкой («Сделал»). Если по матчу всё отмечено «Не ставил» — молчим."""
        now = now or datetime.now(timezone.utc)
        due: list[int] = []
        for s in self.db.pending():
            if s.source != "boss" or s.user_decision not in ("pending", "placed"):
                continue
            lead = {"hockey": settings.late_check_hockey_minutes,
                    "basketball": settings.late_check_basketball_minutes}.get(
                s.sport, settings.late_check_football_minutes)
            if not now < s.kickoff <= now + timedelta(minutes=lead):
                continue
            num = self.num_for_event(s.event_id)
            if num is None or num in due or self.db.get(f"late_sent:{s.event_id}"):
                continue
            due.append(num)
        return due

    def mark_late_sent(self, num: int) -> None:
        ctx = self.load_candidate(num)
        if ctx is not None:
            self.db.set(f"late_sent:{ctx.event.event_id}", True)

    # ---------- ваши ставки из досье ----------
    def _dossier_ctx(self, num: int) -> MatchContext:
        ctx = self.load_candidate(num)
        if ctx is None:
            raise ValueError(f"Досье #{num} не найдено (или устарело). Запросите новое: «🔍 Анализ».")
        if ctx.event.kickoff <= datetime.now(timezone.utc):
            raise ValueError("Матч уже начался: работаем только в прематче, отметку не принимаю.")
        return ctx

    def record_user_bet(self, num: int, selection: str, odds: float, stake: float | None) -> Signal:
        """«✅ Ставил» под досье: финансист записывает реальную ставку.

        Если на этот исход уже есть прогноз босса (/verdict) — отмечаем его как поставленный.
        Иначе заводим отдельную запись «ставка владельца»: банк она меняет,
        но в бумажную статистику прогнозов босса не попадает.
        """
        ctx = self._dossier_ctx(num)
        group_of(ctx.sport, selection)  # проверка, что исход подходит виду спорта
        boss = self.db.find_signal(ctx.event.event_id, selection, "boss")
        if boss:
            self.db.decide(boss.id, placed=True, odds=odds, stake=stake or boss.stake)
            return self.db.signal(boss.id)
        own = self.db.find_signal(ctx.event.event_id, selection, "user")
        stake = stake or (own.stake if own else round_stake(self.db.bankroll() * 0.01))
        if own is None:
            pm = ctx.market.fair.get(selection)
            sid = self.db.add_signal(
                event_id=ctx.event.event_id,
                league_code=ctx.event.league.fd_code,
                league_name=ctx.event.league.name_ru,
                home=ctx.event.home_team,
                away=ctx.event.away_team,
                kickoff=ctx.event.kickoff.isoformat(),
                selection=selection,
                odds_ref=ctx.market.avg_odds.get(selection) or odds,
                min_odds=odds,
                p_final=round(pm or 1 / odds, 4),
                p_market=round(pm or 1 / odds, 4),
                p_model=round(ctx.model.probs[selection], 4) if ctx.model and selection in ctx.model.probs else None,
                edge=round((pm * odds - 1) if pm else 0.0, 4),
                stake=stake,
                confidence="user",
                reasoning="Ставка владельца по досье",
                sport=ctx.sport.key,
                line=ctx.line(selection),
                source="user",
            )
            own = self.db.signal(sid)
        self.db.decide(own.id, placed=True, odds=odds, stake=stake)
        return self.db.signal(own.id)

    def skip_dossier(self, num: int) -> MatchContext:
        """«❌ Не ставил» под досье: прогнозы босса на этот матч идут только в статистику."""
        ctx = self._dossier_ctx(num)
        for row in self.db.conn.execute(
            "SELECT id FROM signals WHERE event_id = ? AND source = 'boss' AND user_decision = 'pending'",
            (ctx.event.event_id,),
        ):
            self.db.decide(row["id"], placed=False)
        self.db.set(f"dossier_skip:{num}", True)
        return ctx

    # ---------- проверка исходов (аналитик) и учёт (финансист) ----------
    async def settle(self) -> list[Signal]:
        self.db.expire_undecided()
        now = datetime.now(timezone.utc)
        due = [s for s in self.db.pending() if s.kickoff < now - timedelta(hours=2, minutes=45)]
        if not due:
            return []
        date_from = min(s.kickoff for s in due).date() - timedelta(days=1)
        date_to = now.date()
        finished: list[Result] = []
        football = sorted({s.league_code for s in due if s.sport == "football"})
        if football:
            start = date_from
            while start <= date_to:  # football-data принимает интервал не больше 10 дней
                end = min(start + timedelta(days=9), date_to)
                finished.extend(await self.stats.finished_between(football, start, end))
                start = end + timedelta(days=1)
        if any(s.sport == "hockey" for s in due):
            try:
                finished.extend(await self.nhl.finished_between(date_from, date_to))
            except Exception as exc:  # noqa: BLE001
                log.warning("NHL API: результаты недоступны: %s", type(exc).__name__)
        if any(s.sport == "basketball" for s in due):
            try:
                finished.extend(await self.nba.finished_between(date_from, date_to))
            except Exception as exc:  # noqa: BLE001
                log.warning("ESPN NBA: результаты недоступны: %s", type(exc).__name__)

        competition = {"hockey": "NHL", "basketball": "NBA"}
        settled = []
        for s in due:
            want = competition.get(s.sport)
            pool = [r for r in finished if (r.competition == want if want else r.competition not in ("NHL", "NBA"))]
            same_day = [r for r in pool if abs((r.utc_date - s.kickoff).total_seconds()) < 36 * 3600]
            res = next(
                (r for r in same_day if best_match(s.home, [r.home]) and best_match(s.away, [r.away])), None
            )
            if res is None:
                if now - s.kickoff > timedelta(days=5):
                    self.db.settle(s.id, "void", "нет данных", 0.0, 0.0)
                    settled.append(self.db.signal(s.id))
                continue
            outcome = settle_outcome(s.sport, s.selection, s.line, res)
            if outcome == "void":
                paper = real = 0.0
            else:
                won = outcome == "won"
                paper = PAPER_UNIT * (s.paper_odds - 1) if won else -PAPER_UNIT
                real = 0.0
                if s.placed and s.actual_stake:
                    real = s.actual_stake * ((s.actual_odds or s.min_odds) - 1) if won else -s.actual_stake
            self.db.settle(s.id, outcome, score_text(res), real, paper)
            settled.append(self.db.signal(s.id))
        return [s for s in settled if s is not None]
