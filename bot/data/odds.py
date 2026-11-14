"""Коэффициенты букмекеров через The Odds API (https://the-odds-api.com).

Бесплатный тариф: 500 кредитов в месяц. Стоимость запроса по лиге = число рынков
(регион один: eu). Футбол: h2h + totals = 2 кредита на лигу. НХЛ: до 4 рынков.
Полный скан всех лиг ~16 кредитов, поэтому линии кешируются (ODDS_CACHE_MINUTES).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import httpx
from zoneinfo import ZoneInfo

from ..config import LEAGUES, League, settings
from ..sports import SPORTS

log = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"


@dataclass
class BookOdds:
    """Котировки одного букмекера: исход -> {линия: коэффициент}. Для рынков без линии ключ None."""

    bookmaker: str
    quotes: dict[str, dict[float | None, float]] = field(default_factory=dict)

    def put(self, selection: str, price: float, point: float | None = None) -> None:
        self.quotes.setdefault(selection, {})[point] = float(price)

    def price(self, selection: str, point: float | None = None) -> float | None:
        return self.quotes.get(selection, {}).get(point)

    def points(self, selection: str) -> list[float]:
        return [p for p in self.quotes.get(selection, {}) if p is not None]

    @classmethod
    def football(cls, bookmaker: str, home: float | None = None, draw: float | None = None,
                 away: float | None = None, over25: float | None = None, under25: float | None = None) -> "BookOdds":
        """Футбольная линия одной строкой: 1X2 и тотал 2.5 (удобно для тестов)."""
        book = cls(bookmaker)
        for sel, price, point in (("home", home, None), ("draw", draw, None), ("away", away, None),
                                  ("over", over25, 2.5), ("under", under25, 2.5)):
            if price:
                book.put(sel, price, point)
        return book


@dataclass
class Event:
    event_id: str
    league: League
    kickoff: datetime
    home_team: str
    away_team: str
    books: list[BookOdds] = field(default_factory=list)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _point(o: dict) -> float | None:
    p = o.get("point")
    return float(p) if p is not None else None


def parse_event(raw: dict, league: League) -> Event:
    with_ot = league.sport in ("hockey", "basketball")  # h2h — победа с учётом овертайма
    event = Event(
        event_id=raw["id"],
        league=league,
        kickoff=datetime.fromisoformat(raw["commence_time"].replace("Z", "+00:00")),
        home_team=raw["home_team"],
        away_team=raw["away_team"],
    )
    for bm in raw.get("bookmakers", []):
        book = BookOdds(bookmaker=bm.get("key", "?"))
        for market in bm.get("markets", []):
            key = market.get("key")
            for o in market.get("outcomes", []):
                name, price = o.get("name", ""), o.get("price")
                if price is None:
                    continue
                side = (
                    "home" if name == event.home_team
                    else "away" if name == event.away_team
                    else "draw" if name.lower() == "draw"
                    else None
                )
                if key == "h2h":
                    if with_ot and side in ("home", "away"):
                        book.put("ml_" + side, price)  # в хоккее и баскетболе h2h — победа с учётом ОТ
                    elif not with_ot and side:
                        book.put(side, price)
                elif key == "h2h_3_way" and side:
                    book.put(side, price)  # исход основного времени
                elif key == "totals":
                    if name.lower() == "over":
                        book.put("over", price, _point(o))
                    elif name.lower() == "under":
                        book.put("under", price, _point(o))
                elif key == "spreads" and side in ("home", "away"):
                    book.put("hcp_" + side, price, _point(o))
        event.books.append(book)
    return event


@dataclass
class Quota:
    remaining: int
    used: int
    checked_at: str  # ISO-время проверки (UTC)

    @property
    def total(self) -> int:
        return self.remaining + self.used


def _int_header(resp: httpx.Response, name: str) -> int | None:
    raw = resp.headers.get(name)
    try:
        return int(float(raw)) if raw is not None else None
    except ValueError:
        return None


@dataclass
class ScanReport:
    """Сколько кредитов ушло на обновление линий и что пропущено."""

    spent: int = 0
    paid: list[str] = field(default_factory=list)  # лиги, за линии которых заплатили
    empty: list[str] = field(default_factory=list)  # лиги без матчей в окне — пропущены бесплатно
    budget_skipped: list[str] = field(default_factory=list)  # отложены, чтобы уложиться в дневной бюджет
    failed: list[str] = field(default_factory=list)
    gated: list[str] = field(default_factory=list)  # матчи есть, но разбирать нечего — линии не загружались
    reused: list[str] = field(default_factory=list)  # взяты из сохранённых (на диске) — кредиты не тратились
    # матчи лиг без платных линий (из бесплатного расписания): (лига, хозяева, гости, начало UTC ISO)
    schedule_only: list[tuple[str, str, str, str]] = field(default_factory=list)
    allowance: float | None = None  # сколько разрешено потратить на этот скан
    cached: bool = False  # линии взяты из кеша, кредиты не тратились
    fetched_at: str = ""  # когда линии были получены (UTC ISO)


def market_cost(markets: str) -> int:
    """Стоимость запроса по одной лиге: число рынков (регион у нас один)."""
    return len([m for m in markets.split(",") if m.strip()])


class OddsClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.odds_api_key
        self.quota: Quota | None = None
        # лига -> набор рынков, который API точно принимает (запоминаем после HTTP 422)
        self.markets_ok: dict[str, str] = {}

    @property
    def requests_remaining(self) -> int | None:
        return self.quota.remaining if self.quota else None

    def _remember(self, resp: httpx.Response) -> None:
        remaining, used = _int_header(resp, "x-requests-remaining"), _int_header(resp, "x-requests-used")
        if remaining is not None:
            self.quota = Quota(remaining, used or 0, datetime.now(timezone.utc).isoformat())

    async def check_quota(self) -> Quota | None:
        """Остаток кредитов. Запрос списка видов спорта у The Odds API кредиты не тратит."""
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(f"{BASE_URL}/sports", params={"apiKey": self.api_key})
        resp.raise_for_status()
        self._remember(resp)
        return self.quota

    def _window(self, hours: int) -> dict:
        now = datetime.now(timezone.utc)
        return {"commenceTimeFrom": _iso(now), "commenceTimeTo": _iso(now + timedelta(hours=hours))}

    async def has_events(self, client: httpx.AsyncClient, league: League, hours: int) -> bool | None:
        """Есть ли матчи лиги в окне. None — не удалось узнать (тогда лучше заплатить, чем пропустить матчи)."""
        evs = await self.free_events(client, league, hours)
        return None if evs is None else len(evs) > 0

    async def free_events(self, client: httpx.AsyncClient, league: League, hours: int) -> list[dict] | None:
        """Матчи лиги в окне без коэффициентов (команды и время) — этот запрос кредиты не тратит.
        None — не удалось узнать."""
        try:
            resp = await client.get(
                f"{BASE_URL}/sports/{league.odds_key}/events",
                params={"apiKey": self.api_key, "dateFormat": "iso", **self._window(hours)},
            )
        except httpx.RequestError:
            return None
        if resp.status_code == 404:
            return []
        if resp.status_code != 200:
            return None
        self._remember(resp)
        try:
            data = resp.json()
        except ValueError:
            return None
        return data if isinstance(data, list) else None

    def markets_for(self, league: League) -> str:
        return self.markets_ok.get(league.odds_key) or SPORTS[league.sport].odds_markets

    async def _get(self, client: httpx.AsyncClient, league: League, markets: str, hours: int) -> httpx.Response:
        params = {
            "apiKey": self.api_key,
            "regions": "eu",
            "markets": markets,
            "oddsFormat": "decimal",
            "dateFormat": "iso",
            **self._window(hours),
        }
        return await client.get(f"{BASE_URL}/sports/{league.odds_key}/odds", params=params)

    async def fetch_league(self, client: httpx.AsyncClient, league: League, hours: int) -> tuple[list[Event], int]:
        """Линии лиги и сколько кредитов это стоило."""
        sport = SPORTS[league.sport]
        markets = self.markets_for(league)
        resp = await self._get(client, league, markets, hours)
        if resp.status_code == 422 and sport.odds_markets_fallback and markets != sport.odds_markets_fallback:
            # API не принимает один из рынков (например, 1X2 основного времени) — запоминаем и берём без него
            log.info("Odds API: %s не принимает часть рынков (HTTP 422), дальше беру %s",
                     league.odds_key, sport.odds_markets_fallback)
            markets = sport.odds_markets_fallback
            self.markets_ok[league.odds_key] = markets
            resp = await self._get(client, league, markets, hours)
        if resp.status_code == 404:
            return [], 0  # лига не в сезоне
        resp.raise_for_status()
        before = self.quota.used if self.quota else None
        self._remember(resp)
        cost = _int_header(resp, "x-requests-last")
        if cost is None:
            after = self.quota.used if self.quota else None
            cost = (after - before) if (before is not None and after is not None) else market_cost(markets)
        self.last_raw = resp.json()
        return [parse_event(raw, league) for raw in self.last_raw], max(cost, 0)

    async def upcoming(self, hours: int | None = None, allowance: float | None = None,
                       gate=None, store=None, max_age_minutes: int | None = None) -> tuple[list[Event], ScanReport]:
        """Линии по лигам в порядке приоритета (порядок ENABLED_LEAGUES).

        1. Бесплатно проверяем, есть ли у лиги матчи в окне; пустые лиги пропускаем.
        2. gate(лига, матчи) — бесплатный фильтр: False — разбирать нечего, линии не нужны
           (НБА «по анализу» без перекоса графика).
        3. store (база бота) — линии каждой лиги сохраняются на диск: если они моложе max_age_minutes,
           берём сохранённые бесплатно (в том числе после перезапуска бота).
        4. Платим только за лиги с матчами, пока укладываемся в бюджет скана (allowance).
           Первую лигу с матчами берём всегда, если на неё хватает остатка.
        """
        hours = hours or settings.lookahead_hours
        report = ScanReport(allowance=allowance, fetched_at=datetime.now(timezone.utc).isoformat())
        events: list[Event] = []
        now = datetime.now(timezone.utc)
        tz = ZoneInfo(settings.timezone)
        async with httpx.AsyncClient(timeout=30) as client:
            for league in LEAGUES:
                saved = store.get(f"odds_raw:{league.odds_key}") if store is not None else None
                # первая загрузка лиги за день разрешена всегда (кроме неприкосновенного запаса);
                # дневной бюджет ограничивает только повторные обновления той же лиги
                fetched_today = bool(saved) and (
                    datetime.fromisoformat(saved["at"]).astimezone(tz).date() == now.astimezone(tz).date())
                if saved and max_age_minutes is not None:
                    age = now - datetime.fromisoformat(saved["at"])
                    if age <= timedelta(minutes=max_age_minutes):
                        events.extend(parse_event(raw, league) for raw in saved["data"])
                        report.reused.append(league.name_ru)
                        continue
                free = await self.free_events(client, league, hours)
                if free == []:
                    report.empty.append(league.name_ru)
                    continue
                if gate is not None and free:
                    try:
                        need = await gate(league, free)
                    except Exception as exc:  # noqa: BLE001 — фильтр не сработал: лучше заплатить
                        log.warning("фильтр %s не сработал: %s", league.odds_key, type(exc).__name__)
                        need = True
                    if not need:
                        report.gated.append(league.name_ru)
                        report.schedule_only += [
                            (league.name_ru, str(e.get("home_team", "?")), str(e.get("away_team", "?")),
                             str(e.get("commence_time", ""))) for e in free
                        ]
                        continue
                cost = market_cost(self.markets_for(league))
                remaining = self.quota.remaining if self.quota else None
                if remaining is not None and cost > remaining:
                    report.budget_skipped.append(league.name_ru)
                    continue
                first_today = store is not None and not fetched_today
                if first_today:
                    if remaining is not None and remaining - cost < settings.odds_reserve:
                        report.budget_skipped.append(league.name_ru)
                        continue
                elif allowance is not None and report.spent > 0 and report.spent + cost > allowance:
                    report.budget_skipped.append(league.name_ru)
                    continue
                try:
                    got, spent = await self.fetch_league(client, league, hours)
                    events.extend(got)
                    if store is not None:
                        store.set(f"odds_raw:{league.odds_key}", {"at": now.isoformat(), "data": self.last_raw})
                    report.spent += spent
                    report.paid.append(league.name_ru)
                except httpx.HTTPStatusError as exc:
                    # не логируем URL целиком: в нём ключ API
                    log.warning("Odds API: %s вернул HTTP %s", league.odds_key, exc.response.status_code)
                    report.failed.append(league.name_ru)
                except httpx.RequestError as exc:
                    log.warning("Odds API: сетевая ошибка для %s: %s", league.odds_key, type(exc).__name__)
                    report.failed.append(league.name_ru)
        events.sort(key=lambda e: e.kickoff)
        log.info("Odds API: %d матчей, потрачено %d, осталось %s, без матчей: %d, отложено: %d",
                 len(events), report.spent, self.requests_remaining, len(report.empty), len(report.budget_skipped))
        return events, report
