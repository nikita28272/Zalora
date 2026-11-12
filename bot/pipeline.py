"""Главный конвейер.

Порядок (иерархия):
  1. Статистик (код) и рынок дают цифры по каждому исходу.
  2. Главный ставит задачи подчинённым: новости и зарубежные аналитики (параллельно),
     затем адвокат дьявола. Каждый обязан высказаться по всем пяти исходам.
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
from .agents.schemas import ALL_OUTCOMES, ChiefVerdict, DevilReport, ExpertsReport, NewsReport
from .analysis.bankroll import StakeDecision, decide_stake, edge, min_acceptable_odds
from .analysis.market import MarketView, market_view
from .analysis.poisson import ModelOutput, PoissonModel
from .config import LEAGUES, settings
from .data import news
from .data.odds import Event, OddsClient
from .data.stats import StandingRow, StatsClient
from .data.teams import best_match
from .db import DB, Signal

log = logging.getLogger(__name__)

SEL_RU = {
    "home": "П1 (победа хозяев)",
    "draw": "X (ничья)",
    "away": "П2 (победа гостей)",
    "over": "ТБ 2.5 (больше 2.5 голов)",
    "under": "ТМ 2.5 (меньше 2.5 голов)",
}
SEL_SHORT = {"home": "П1", "draw": "X", "away": "П2", "over": "ТБ2.5", "under": "ТМ2.5"}


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

    def p_quant(self, sel: str) -> float | None:
        """Средняя вероятность рынка и модели — «количественная» оценка до агентов."""
        pm = self.market.fair.get(sel)
        if pm is None:
            return None
        if self.model is None:
            return pm
        return (pm + self.model.probs[sel]) / 2

    def quant_candidates(self) -> list[tuple[str, float]]:
        """Исходы с положительным ожиданием по модели+рынку, лучший первым."""
        if self.model is None:
            return []
        out = []
        for sel, odds in self.market.avg_odds.items():
            if not (settings.min_odds <= odds <= settings.max_odds):
                continue
            p = self.p_quant(sel)
            if p is None:
                continue
            ev = edge(p, odds)
            if ev > 0:
                out.append((sel, ev))
        return sorted(out, key=lambda x: x[1], reverse=True)

    def brief(self) -> str:
        return (
            f"{self.event.league.name_ru}: {self.event.home_team} (хозяева) — {self.event.away_team} (гости)\n"
            f"Начало: {local(self.event.kickoff):%d.%m.%Y %H:%M} ({settings.timezone}). Ставка — только прематч."
        )

    def numbers(self) -> str:
        m = self.market
        lines = [f"ДОКЛАД РЫНКА ({m.n_books} букмекеров, источник честной линии: {m.source}):"]
        for sel in ALL_OUTCOMES:
            if sel in m.fair:
                lines.append(
                    f"  {SEL_SHORT[sel]}: вероятность {m.fair[sel]:.1%}, средний кэф {m.avg_odds.get(sel, 0):.2f},"
                    f" лучший {m.best_odds.get(sel, 0):.2f}"
                )
        if self.model:
            mo = self.model
            lines.append(
                f"ДОКЛАД СТАТИСТИКА (модель Пуассона, сезон): ожидаемые голы {mo.lambda_home:.2f} : "
                f"{mo.lambda_away:.2f} (матчей в выборке: {mo.home_games} и {mo.away_games})"
            )
            for sel in ALL_OUTCOMES:
                lines.append(f"  {SEL_SHORT[sel]}: {mo.probs[sel]:.1%}")
        else:
            lines.append("ДОКЛАД СТАТИСТИКА: недостаточно данных (например, еврокубок или начало сезона).")
        for name, row, form, last in (
            (self.event.home_team, self.home_row, self.home_form, self.home_last),
            (self.event.away_team, self.away_row, self.away_form, self.away_last),
        ):
            lines.append(f"{name}:")
            if row:
                lines.append(
                    f"  место {row.position}, очков {row.points} за {row.played} матчей, разница мячей {row.goal_diff:+d}"
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
        self.matches: list[MatchContext] = []
        self._headlines: tuple[float, list[news.Headline]] = (0.0, [])
        self._lock = asyncio.Lock()

    # ---------- данные ----------
    async def _league_data(self, code: str) -> tuple[PoissonModel | None, list[StandingRow]]:
        model = None
        rows: list[StandingRow] = []
        try:
            results = await self.stats.season_results(code)
            if results:
                model = PoissonModel(results)
        except Exception as exc:  # noqa: BLE001 — источник может быть недоступен
            log.warning("football-data: результаты %s недоступны: %s", code, type(exc).__name__)
        try:
            rows = await self.stats.standings(code)
        except Exception as exc:  # noqa: BLE001
            log.warning("football-data: таблица %s недоступна: %s", code, type(exc).__name__)
        return model, rows

    async def load_matches(self) -> list[MatchContext]:
        events = [e for e in await self.odds.upcoming() if is_prematch(e.kickoff)]
        codes = sorted({e.league.fd_code for e in events})
        league_data = {code: await self._league_data(code) for code in codes}

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
                f"КАНДИДАТ ОТ КОЛИЧЕСТВЕННОГО АНАЛИЗА: {SEL_RU[sel]}, средний кэф {ctx.market.avg_odds[sel]:.2f}, "
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
                              web_searches=run.web_searches)

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
        orders = [self._order(res, "Агент новостей", lambda: team.news_agent(ctx.brief(), heads_text))]
        if settings.web_search_enabled:
            orders.append(self._order(res, "Зарубежные аналитики", lambda: team.experts_agent(ctx.brief())))
        reports = await asyncio.gather(*orders)
        res.news = reports[0]
        res.experts = reports[1] if len(reports) > 1 else None

        # 2. Адвокат дьявола видит всё собранное
        res.devil = await self._order(
            res, "Адвокат дьявола", lambda: team.devil_agent(self._dossier(ctx, res, for_chief=False))
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
            res, "Главный аналитик", lambda: team.chief_agent(self._dossier(ctx, res, for_chief=True))
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
        _normalize(final, ("home", "draw", "away"))
        _normalize(final, ("over", "under"))
        for sel in ALL_OUTCOMES:
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
                p_model=ctx.model.probs[sel] if ctx.model else None,
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

    # ---------- проверка исходов (аналитик) и учёт (финансист) ----------
    async def settle(self) -> list[Signal]:
        self.db.expire_undecided()
        now = datetime.now(timezone.utc)
        due = [s for s in self.db.pending() if s.kickoff < now - timedelta(hours=2, minutes=15)]
        if not due:
            return []
        codes = sorted({s.league_code for s in due} & {lg.fd_code for lg in LEAGUES})
        date_from = min(s.kickoff for s in due).date() - timedelta(days=1)
        date_to = now.date()
        finished = []
        start = date_from
        while start <= date_to:  # API принимает интервал не больше 10 дней
            end = min(start + timedelta(days=9), date_to)
            finished.extend(await self.stats.finished_between(codes, start, end))
            start = end + timedelta(days=1)

        settled = []
        for s in due:
            same_day = [r for r in finished if abs((r.utc_date - s.kickoff).total_seconds()) < 36 * 3600]
            res = next(
                (r for r in same_day if best_match(s.home, [r.home]) and best_match(s.away, [r.away])), None
            )
            if res is None:
                if now - s.kickoff > timedelta(days=5):
                    self.db.settle(s.id, "void", "нет данных", 0.0, 0.0)
                    settled.append(self.db.signal(s.id))
                continue
            hg, ag = res.home_goals, res.away_goals
            won = {
                "home": hg > ag,
                "draw": hg == ag,
                "away": hg < ag,
                "over": hg + ag > 2.5,
                "under": hg + ag < 2.5,
            }[s.selection]
            paper = s.stake * (s.min_odds - 1) if won else -s.stake
            real = 0.0
            if s.placed and s.actual_stake:
                real = s.actual_stake * ((s.actual_odds or s.min_odds) - 1) if won else -s.actual_stake
            self.db.settle(s.id, "won" if won else "lost", f"{hg}:{ag}", real, paper)
            settled.append(self.db.signal(s.id))
        return [s for s in settled if s is not None]
