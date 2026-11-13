"""Проверка КХЛ: разбор ответов API-Sports на подменённом сервере."""

import asyncio

import httpx

from bot import khl_check


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _ok(payload, paging=None):
    return httpx.Response(200, json={"errors": [], "response": payload, "paging": paging or {"total": 1}})


def test_full_check_reports_seasons_games_and_russian_books():
    calls = []

    def handler(req: httpx.Request):
        calls.append(req.url.path)
        path, q = req.url.path, req.url.params
        if path == "/status":
            return _ok({"subscription": {"plan": "Free"}, "requests": {"current": 3, "limit_day": 100}})
        if path == "/leagues":
            return _ok([{"id": 35, "name": "KHL", "country": {"name": "Russia"}, "seasons": [
                {"season": 2025, "current": False},
                {"season": 2026, "current": True, "start": "2026-09-01", "end": "2027-04-30",
                 "coverage": {"standings": True, "odds": True, "games": {"statistics": True}}}]}])
        if path == "/games":
            if q["season"] == "2026":  # бесплатный тариф не открывает текущий сезон
                return httpx.Response(200, json={"errors": {"plan": "Free plans do not have access to this season"},
                                                 "response": []})
            return _ok([
                {"date": "2026-03-01T16:30:00+00:00", "status": {"short": "FT"}, "teams": {}},
                {"date": "2026-03-05T16:30:00+00:00", "status": {"short": "NS"},
                 "teams": {"home": {"name": "CSKA"}, "away": {"name": "SKA"}}},
            ])
        if path == "/standings":
            return _ok([[{"team": {"name": "Lokomotiv"}}, {"team": {"name": "CSKA"}}]])
        if path == "/odds":
            return _ok([{"bookmakers": [
                {"name": "Marathonbet", "bets": [{"name": "Home/Away"}]},
                {"name": "Bet365", "bets": [{"name": "3Way Result"}]},
            ]}])
        return _ok([])

    lines = []

    async def go():
        async with _client(handler) as c:
            return await khl_check.run(c, out=lines.append)

    summary = asyncio.run(go())
    text = "\n".join(lines)
    assert summary["key_ok"] and summary["league"] == 35
    assert summary["season"] == 2025 and summary["games"] == 2 and summary["upcoming"] == 1
    assert "закрыт" in text and "CSKA — SKA" in text
    assert summary["ru_books"] == ["Marathonbet"] and "Marathonbet 🇷🇺" in text
    assert "Футбол (API-Football)" in text


def test_bad_key_stops_early():
    def handler(req):
        return httpx.Response(200, json={"errors": {"token": "Error/Missing application key"}, "response": []})

    lines = []

    async def go():
        async with _client(handler) as c:
            return await khl_check.run(c, out=lines.append)

    assert asyncio.run(go()) == {"key_ok": False}
    assert "Ключ не принят" in lines[0]
