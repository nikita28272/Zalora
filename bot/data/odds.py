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
    hockey = league.sport == "hockey"
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
                    if hockey and side in ("home", "away"):
                        book.put("ml_" + side, price)  # в хоккее h2h — победа с учётом ОТ
                    elif not hockey and side:
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


class OddsClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.odds_api_key
        self.quota: Quota | None = None

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

    async def _get(self, client: httpx.AsyncClient, league: League, markets: str, hours: int) -> httpx.Response:
        now = datetime.now(timezone.utc)
        params = {
            "apiKey": self.api_key,
            "regions": "eu",
            "markets": markets,
            "oddsFormat": "decimal",
            "dateFormat": "iso",
            "commenceTimeFrom": _iso(now),
            "commenceTimeTo": _iso(now + timedelta(hours=hours)),
        }
        return await client.get(f"{BASE_URL}/sports/{league.odds_key}/odds", params=params)

    async def fetch_league(self, client: httpx.AsyncClient, league: League, hours: int) -> list[Event]:
        sport = SPORTS[league.sport]
        resp = await self._get(client, league, sport.odds_markets, hours)
        if resp.status_code == 422 and sport.odds_markets_fallback:
            # API не принимает один из рынков (например, 1X2 основного времени) — берём без него
            log.info("Odds API: %s без части рынков (HTTP 422), повторяю с %s",
                     league.odds_key, sport.odds_markets_fallback)
            resp = await self._get(client, league, sport.odds_markets_fallback, hours)
        if resp.status_code == 404:
            return []  # лига не в сезоне
        resp.raise_for_status()
        self._remember(resp)
        return [parse_event(raw, league) for raw in resp.json()]

    async def upcoming(self, hours: int | None = None) -> list[Event]:
        hours = hours or settings.lookahead_hours
        events: list[Event] = []
        async with httpx.AsyncClient(timeout=30) as client:
            for league in LEAGUES:
                try:
                    events.extend(await self.fetch_league(client, league, hours))
                except httpx.HTTPStatusError as exc:
                    # не логируем URL целиком: в нём ключ API
                    log.warning("Odds API: %s вернул HTTP %s", league.odds_key, exc.response.status_code)
                except httpx.RequestError as exc:
                    log.warning("Odds API: сетевая ошибка для %s: %s", league.odds_key, type(exc).__name__)
        events.sort(key=lambda e: e.kickoff)
        log.info("Odds API: %d матчей, осталось кредитов: %s", len(events), self.requests_remaining)
        return events
