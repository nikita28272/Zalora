"""Коэффициенты букмекеров через The Odds API (https://the-odds-api.com).

Бесплатный тариф: 500 кредитов в месяц. Один запрос по лиге с рынками
h2h (1X2) и totals в регионе eu стоит 2 кредита. Полный скан шести лиг
стоит 12 кредитов, то есть одного скана в день хватает на месяц с запасом.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import httpx

from ..config import LEAGUES, League, settings

log = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"


@dataclass
class BookOdds:
    bookmaker: str
    home: float | None = None
    draw: float | None = None
    away: float | None = None
    over25: float | None = None
    under25: float | None = None


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


def _parse_event(raw: dict, league: League) -> Event:
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
            if market.get("key") == "h2h":
                for o in market.get("outcomes", []):
                    if o["name"] == event.home_team:
                        book.home = o["price"]
                    elif o["name"] == event.away_team:
                        book.away = o["price"]
                    elif o["name"].lower() == "draw":
                        book.draw = o["price"]
            elif market.get("key") == "totals":
                for o in market.get("outcomes", []):
                    if abs(float(o.get("point", 0)) - 2.5) > 1e-9:
                        continue
                    if o["name"].lower() == "over":
                        book.over25 = o["price"]
                    elif o["name"].lower() == "under":
                        book.under25 = o["price"]
        event.books.append(book)
    return event


class OddsClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.odds_api_key
        self.requests_remaining: str | None = None

    async def fetch_league(self, client: httpx.AsyncClient, league: League, hours: int) -> list[Event]:
        now = datetime.now(timezone.utc)
        params = {
            "apiKey": self.api_key,
            "regions": "eu",
            "markets": "h2h,totals",
            "oddsFormat": "decimal",
            "dateFormat": "iso",
            "commenceTimeFrom": _iso(now),
            "commenceTimeTo": _iso(now + timedelta(hours=hours)),
        }
        resp = await client.get(f"{BASE_URL}/sports/{league.odds_key}/odds", params=params)
        if resp.status_code == 404:
            return []  # лига не в сезоне
        resp.raise_for_status()
        self.requests_remaining = resp.headers.get("x-requests-remaining", self.requests_remaining)
        return [_parse_event(raw, league) for raw in resp.json()]

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
