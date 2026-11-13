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
from .analysis.bankroll import StakeDecision, decide_stake, edge, min_acceptable_odds
from .analysis.market import MarketView, market_view
from .analysis.poisson import ModelOutput, PoissonModel, sport_probs
from .config import LEAGUES_BY_ODDS_KEY, League, settings
from .data import news
from .data.nhl import NhlClient
from .data.odds import Event, OddsClient, Quota
from .data.stats import Result, StandingRow, StatsClient
from .data.teams import best_match
from .db import DB, Signal
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
    if selection == "ml_home":
        return "won" if fh > fa else "lost"
    if selection == "ml_away":
        return "won" if fa > fh else "lost"
    if selection in ("over", "under"):
        total = fh + fa
        ln = line if line is not None else (2.5 if sport_key == "football" else 5.5)
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


def score_text(r: Result) -> str:
    fh, fa = r.final
    if r.decided in ("OT", "SO"):
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
        """Средняя вероятность рынка и модели — «количественная» оценка до агентов."""
        pm = self.market.fair.get(sel)
        if pm is None:
            return None
        if self.model is None or sel not in self.model.probs:
            return pm
        return (pm + self.model.probs[sel]) / 2

    def quant_candidates(self) -> list[tuple[str, float]]:
        """Исходы с перевесом не меньше MIN_EDGE по модели+рынку, лучший первым."""
        if self.model is None:
            return []
        out = []
        for sel, odds in self.market.avg_odds.items():
            if sel in self.market.estimated:  # оценочный кэф — не повод для кандидата
                continue
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
            lines.append(
                f"ДОКЛАД СТАТИСТИКА (модель Пуассона, сезон): ожидаемые голы {mo.lambda_home:.2f} : "
                f"{mo.lambda_away:.2f} (матчей в выборке: {mo.home_games} и {mo.away_games})"
            )
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
                unit = "шайб" if self.sport.key == "hockey" else "мячей"
                lines.append(
                    f"  место {row.position}, очков {row.points} за {row.played} матчей, разница {unit} {row.goal_diff:+d}"
                    + (f", {row.form}" if row.form and self.sport.key == "hockey" else "")
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
        self.matches: list[MatchContext] = []
        self._events: tuple[float, list[Event]] = (0.0, [])
        self._headlines: tuple[float, list[news.Headline]] = (0.0, [])
        self._lock = asyncio.Lock()

    # ---------- данные ----------
    async def _league_data(self, league: League) -> tuple[PoissonModel | None, list[StandingRow]]:
        model = None
        rows: list[StandingRow] = []
        hockey = league.sport == "hockey"
        source = "NHL API" if hockey else "football-data"
        try:
            results = await (self.nhl.season_results() if hockey else self.stats.season_results(league.fd_code))
            if results:
                model = PoissonModel(results)
        except Exception as exc:  # noqa: BLE001 — источник может быть недоступен
            log.warning("%s: результаты %s недоступны: %s", source, league.fd_code, type(exc).__name__)
        try:
            rows = await (self.nhl.standings() if hockey else self.stats.standings(league.fd_code))
        except Exception as exc:  # noqa: BLE001
            log.warning("%s: таблица %s недоступна: %s", source, league.fd_code, type(exc).__name__)
        return model, rows

    async def _upcoming(self) -> list[Event]:
        """Линии с кешем: повторные /matches и /scan не тратят кредиты The Odds API."""
        ts, cached = self._events
        if cached and time.monotonic() - ts < settings.odds_cache_minutes * 60:
            return cached
        fresh = await self.odds.upcoming()
        self._events = (time.monotonic(), fresh)
        self._save_quota()
        return fresh

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

    async def load_matches(self) -> list[MatchContext]:
        events = [e for e in await self._upcoming() if is_prematch(e.kickoff)]
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
                    ctx.model = model.predict(ctx.fd_home, ctx.fd_away)
                    if ctx.model:  # вероятности под рынки и линии этого вида спорта
                        ctx.model.probs = sport_probs(event.league.sport, ctx.model.lambda_home,
                                                      ctx.model.lambda_away, mv.lines, ctx.model.draw_boost)
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
        return (
            "ДОКЛАД ФИНАНСИСТА:\n"
            f"  банк {st.bankroll:.0f}, в игре {n_play} ставок на {s_play:.0f}\n"
            f"  сигналов сегодня {st.bets_today} из {settings.max_bets_per_day}, "
            f"результат дня {st.profit_today:+.0f} (лимит убытка {settings.daily_loss_limit_pct:.0%} банка)\n"
            f"  проигрышей подряд {st.losing_streak} (пауза после {settings.stop_after_losses})\n"
            f"  размер ставки: {settings.kelly_fraction:g} Келли, не больше {settings.max_stake_pct:.0%} банка; "
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
                pf = min(max(pf, pm - settings.max_prob_deviation), pm + settings.max_prob_deviation)
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
            matches = await self.load_matches()
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
                "hg": mo.home_games, "ag": mo.away_games, "db": mo.draw_boost,
            },
            "home_row": row(ctx.home_row), "away_row": row(ctx.away_row),
            "home_form": ctx.home_form, "away_form": ctx.away_form,
            "home_last": ctx.home_last, "away_last": ctx.away_last,
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
                                                    mo.get("db", 1.0))
        row = lambda r: None if r is None else StandingRow(**r)  # noqa: E731
        return MatchContext(
            idx=num, event=event, market=market, model=model,
            home_row=row(snap["home_row"]), away_row=row(snap["away_row"]),
            home_form=snap["home_form"], away_form=snap["away_form"],
            home_last=snap["home_last"], away_last=snap["away_last"],
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
        return "\n\n".join([
            f"ДОСЬЕ #{num} ДЛЯ БОССА",
            ctx.brief(),
            ctx.numbers(),
            self.finance_report(),
            cand,
            "ЗАДАНИЕ БОССУ (Claude), подробно — в BOSS.md репозитория nikita28272/Zalora:\n"
            "1. Параллельно запусти субагентов «новости» (веб-поиск: травмы, составы, мотивация) "
            "и «зарубежные аналитики» (веб-поиск: прогнозы экспертов, доля за каждый исход), "
            "затем «адвоката дьявола» (сильнейший аргумент против каждого исхода).\n"
            f"2. Каждый докладывает по всем исходам: {codes}.\n"
            + ("   Хоккей: обязательно выясни стартовых вратарей.\n" if ctx.sport.key == "hockey" else "")
            +
            f"3. Оцени вероятность каждого исхода; от рынка дальше {settings.max_prob_deviation:.0%} не уходи "
            "без веской причины.\n"
            f"4. Если есть ставка с перевесом от {settings.min_edge:.0%}, последней строкой дай команду для бота:\n"
            f"/verdict {num} <{'|'.join(ctx.sport.outcomes)}> <вероятность 0-1> <low|medium|high>\n"
            "Если ставки нет — так и скажи, без команды.",
        ])

    async def candidates(self) -> tuple[list[MatchContext], list[tuple[int, MatchContext]]]:
        async with self._lock:
            matches = await self.load_matches()
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

    def manual_verdict(self, num: int, selection: str, probability: float, confidence: str) -> AnalysisResult:
        """Вердикт босса из чата -> проверки и расчёт финансиста -> сигнал."""
        ctx = self.load_candidate(num)
        if ctx is None:
            raise ValueError(f"Досье #{num} не найдено (или устарело). Запросите новое.")
        fair = ctx.market.fair
        if selection not in fair:
            raise ValueError("По этому исходу нет линии букмекеров.")
        try:
            group = group_of(ctx.sport, selection)
        except ValueError:
            raise ValueError(f"Для этого вида спорта нет исхода «{selection}».") from None
        probs = {o: fair.get(o, 0.5) for o in ctx.sport.outcomes}
        rest = sum(fair[o] for o in group if o != selection)
        for o in group:
            probs[o] = probability if o == selection else fair[o] * (1 - probability) / rest
        res = AnalysisResult(ctx=ctx)
        cands = ctx.quant_candidates()
        res.candidate = cands[0][0] if cands else None
        res.verdict = ChiefVerdict(
            outcomes=[ChiefOutcome(selection=o, probability=probs[o], comment="") for o in ctx.sport.outcomes],
            decision="bet",
            selection=selection,
            confidence=confidence,
            reasoning=f"Полный разбор с субагентами — в чате с боссом (досье #{num}).",
            pros=[],
            cons=[],
        )
        self._finalize(res)
        return res

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

        settled = []
        for s in due:
            pool = [r for r in finished if (r.competition == "NHL") == (s.sport == "hockey")]
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
                paper = s.stake * (s.min_odds - 1) if won else -s.stake
                real = 0.0
                if s.placed and s.actual_stake:
                    real = s.actual_stake * ((s.actual_odds or s.min_odds) - 1) if won else -s.actual_stake
            self.db.settle(s.id, outcome, score_text(res), real, paper)
            settled.append(self.db.signal(s.id))
        return [s for s in settled if s is not None]
