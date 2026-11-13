"""Экономия кредитов The Odds API: бесплатная разведка, память о рынках, дневной бюджет."""

import asyncio
from datetime import date, datetime, timedelta, timezone

import httpx
import pytest

from bot.config import LEAGUES_BY_ODDS_KEY
from bot.data import odds as odds_mod
from bot.db import DB
from bot.pipeline import Engine, days_until_reset
from bot.tg import formatting as fmt

PL = LEAGUES_BY_ODDS_KEY["soccer_epl"]
CL = LEAGUES_BY_ODDS_KEY["soccer_uefa_champs_league"]
NHL = LEAGUES_BY_ODDS_KEY["icehockey_nhl"]
SOON = (datetime.now(timezone.utc) + timedelta(hours=20)).strftime("%Y-%m-%dT%H:%M:%SZ")


class FakeOddsApi:
    """Подделка The Odds API: считает платные запросы и списывает кредиты."""

    def __init__(self, remaining=500, has_3way=False):
        self.remaining, self.used = remaining, 0
        self.paid_calls, self.free_calls, self.rejected = [], [], 0
        self.has_3way = has_3way

    def _headers(self, last):
        return {"x-requests-remaining": str(self.remaining), "x-requests-used": str(self.used),
                "x-requests-last": str(last)}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        sport = path.split("/")[3] if path.count("/") >= 4 else ""
        if path.endswith("/events"):
            self.free_calls.append(sport)
            events = [] if sport == CL.odds_key else [{"id": f"{sport}-1"}]
            return httpx.Response(200, json=events, headers=self._headers(0))
        if path.endswith("/odds"):
            markets = request.url.params["markets"]
            if "h2h_3_way" in markets and not self.has_3way:
                self.rejected += 1
                return httpx.Response(422, json={"message": "invalid market"})
            cost = len(markets.split(","))
            self.remaining -= cost
            self.used += cost
            self.paid_calls.append((sport, markets))
            home, away = ("Arsenal", "Chelsea") if sport == PL.odds_key else ("Toronto Maple Leafs", "Boston Bruins")
            body = [{"id": f"{sport}-1", "commence_time": SOON, "home_team": home, "away_team": away, "bookmakers": [
                {"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [
                    {"name": home, "price": 2.0}, {"name": away, "price": 3.6}, {"name": "Draw", "price": 3.5}]}]}]}]
            return httpx.Response(200, json=body, headers=self._headers(cost))
        return httpx.Response(200, json=[], headers=self._headers(0))


@pytest.fixture
def api(monkeypatch):
    fake = FakeOddsApi()
    real = httpx.AsyncClient
    monkeypatch.setattr(odds_mod.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(fake), **kw))
    monkeypatch.setattr(odds_mod, "LEAGUES", (PL, NHL, CL))
    return fake


def test_free_precheck_and_market_memory(api):
    client = odds_mod.OddsClient("key")
    events, report = asyncio.run(client.upcoming(hours=48))
    assert report.empty == [CL.name_ru]  # без матчей — пропущена бесплатно
    assert [s for s, _ in api.paid_calls] == [PL.odds_key, NHL.odds_key]
    assert report.spent == 2 + 3 and api.rejected == 1  # 1X2 основного времени не принят, взяли без него
    assert len(events) == 2

    asyncio.run(client.upcoming(hours=48))
    assert api.rejected == 1  # второй раз неподдерживаемый рынок уже не пробуем


def test_budget_skips_low_priority_leagues(api):
    client = odds_mod.OddsClient("key")
    _, report = asyncio.run(client.upcoming(hours=48, allowance=3))
    assert report.paid == [PL.name_ru] and report.budget_skipped == [NHL.name_ru]
    assert "Отложено" in fmt.scan_report_line(report)


def test_engine_cache_allowance_and_report(api):
    db = DB(":memory:")
    engine = Engine(db)

    async def no_stats(league):  # статистика здесь не нужна
        return None, []

    engine._league_data = no_stats
    asyncio.run(engine.load_matches())
    first = api.remaining
    assert engine.last_report.spent == 5 and not engine.last_report.cached

    asyncio.run(engine.load_matches())  # из кеша: ни одного платного запроса
    assert api.remaining == first and engine.last_report.cached
    assert "из кеша" in fmt.scan_report_line(engine.last_report)

    allowance = engine.scan_allowance()
    assert allowance is not None and allowance >= 0


def test_days_until_reset():
    assert days_until_reset(date(2026, 11, 13), 1) == 18  # до 1 декабря
    assert days_until_reset(date(2026, 11, 13), 20) == 7
    assert days_until_reset(date(2026, 12, 31), 1) == 1
    assert days_until_reset(date(2026, 11, 20), 20) == 30  # в день обновления — до следующего
