"""Статистика НБА из открытых данных ESPN (site.api.espn.com, ключ не нужен).

Используется табло /scoreboard: за день (?dates=20261112) или за период
(?dates=20261101-20261130&limit=1000). Период экономит запросы; если ESPN вдруг
не отдаст матчи за период, бот добирает их по дням.

Из табло берём итоговый счёт (с овертаймом), счёт основного времени (сумма четвертей)
и тип матча (регулярный сезон, плей-ин, плей-офф; предсезонку и Матч всех звёзд пропускаем).
Таблицу строим сами по результатам: победы-поражения, разница очков, последние 10.

Разбор написан с запасом: если формат изменится, бот не упадёт, а будет работать
по баскетболу без модели (только рынок).
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, datetime, timedelta, timezone

import httpx

from .stats import Result, StandingRow

log = logging.getLogger(__name__)

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
SEASON_TYPES = {2, 3, 5}  # регулярный сезон, плей-офф, плей-ин
CHUNK_DAYS = 31


def season_start(today: date) -> date:
    """Сезон НБА стартует во второй половине октября: в январе–сентябре это октябрь прошлого года."""
    year = today.year if today.month >= 10 else today.year - 1
    return date(year, 10, 1)


def _int(x) -> int | None:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def parse_event(ev: dict) -> Result | None:
    """Завершённый матч ESPN -> Result (home_goals/away_goals — основное время, final_* — с ОТ)."""
    try:
        season_type = (ev.get("season") or {}).get("type")
        if season_type is not None and int(season_type) not in SEASON_TYPES:
            return None
        comp = ev["competitions"][0]
        status = comp.get("status") or ev.get("status") or {}
        if not (status.get("type") or {}).get("completed"):
            return None
        sides = {c.get("homeAway"): c for c in comp["competitors"]}
        home, away = sides["home"], sides["away"]
        fh, fa = _int(home.get("score")), _int(away.get("score"))
        if fh is None or fa is None:
            return None
        period = _int(status.get("period")) or 4
        decided = "OT" if period > 4 else "REG"
        rh, ra = fh, fa
        if decided == "OT":
            qh = [_int(q.get("value")) for q in home.get("linescores") or []][:4]
            qa = [_int(q.get("value")) for q in away.get("linescores") or []][:4]
            if len(qh) == 4 and len(qa) == 4 and None not in qh and None not in qa:
                rh, ra = sum(qh), sum(qa)
        when = datetime.fromisoformat(str(ev["date"]).replace("Z", "+00:00"))
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return Result(
            match_id=int(ev["id"]),
            competition="NBA",
            utc_date=when,
            home=str(home["team"].get("displayName") or home["team"].get("name")),
            away=str(away["team"].get("displayName") or away["team"].get("name")),
            home_goals=rh,
            away_goals=ra,
            final_home=fh,
            final_away=fa,
            decided=decided,
        )
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def standings_from_results(results: list[Result]) -> list[StandingRow]:
    """Таблица по сыгранным матчам сезона: место по проценту побед."""
    stats: dict[str, dict] = {}
    for r in sorted(results, key=lambda r: r.utc_date):
        fh, fa = r.final
        for team, gf, ga in ((r.home, fh, fa), (r.away, fa, fh)):
            s = stats.setdefault(team, {"w": 0, "l": 0, "diff": 0, "last": []})
            won = gf > ga
            s["w" if won else "l"] += 1
            s["diff"] += gf - ga
            s["last"].append(won)
    order = sorted(stats.items(), key=lambda kv: (-(kv[1]["w"] / max(kv[1]["w"] + kv[1]["l"], 1)), -kv[1]["diff"]))
    rows = []
    for pos, (team, s) in enumerate(order, start=1):
        last = s["last"][-10:]
        rows.append(StandingRow(
            position=pos,
            team=team,
            played=s["w"] + s["l"],
            points=s["w"],
            goal_diff=s["diff"],
            form=f"{s['w']}-{s['l']}, последние 10: {sum(last)}-{len(last) - sum(last)}",
        ))
    return rows


CDN_SCHEDULE = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
CDN_TYPES = ("002", "004", "005", "006")  # регулярка, плей-офф, плей-ин, финал Кубка НБА


def parse_cdn_game(g: dict) -> Result | None:
    """Матч из официального расписания НБА (запасной источник: только текущий сезон, без счёта четвертей)."""
    try:
        gid = str(g["gameId"])
        if int(g.get("gameStatus", 0)) != 3 or not gid.startswith(CDN_TYPES):
            return None
        h, a = g["homeTeam"], g["awayTeam"]
        fh, fa = int(h["score"]), int(a["score"])
        decided = "OT" if "OT" in str(g.get("gameStatusText", "")).upper() else "REG"
        return Result(
            match_id=int(gid),
            competition="NBA",
            utc_date=datetime.fromisoformat(str(g["gameDateTimeUTC"]).replace("Z", "+00:00")),
            home=f"{h['teamCity']} {h['teamName']}",
            away=f"{a['teamCity']} {a['teamName']}",
            home_goals=fh, away_goals=fa, final_home=fh, final_away=fa, decided=decided,
        )
    except (KeyError, TypeError, ValueError):
        return None


class NbaClient:
    def __init__(self) -> None:
        self._cache: dict[str, tuple[float, object]] = {}
        self._lock = asyncio.Lock()

    async def _fetch(self, url: str, params: dict | None, key: str, ttl: float) -> dict:
        cached = self._cache.get(key)
        if cached and time.monotonic() - cached[0] < ttl:
            return cached[1]  # type: ignore[return-value]
        async with self._lock:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True,
                                         headers={"User-Agent": "Mozilla/5.0"}) as client:
                resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        self._cache[key] = (time.monotonic(), data)
        return data

    async def _day(self, day: date, ttl: float) -> list[dict]:
        try:
            data = await self._fetch(BASE_URL, {"dates": f"{day:%Y%m%d}"}, f"d{day:%Y%m%d}", ttl)
        except httpx.HTTPError as exc:
            log.warning("ESPN NBA: день %s недоступен: %s", day, _why(exc))
            return []
        return data.get("events") or []

    async def _chunk(self, start: date, end: date, ttl: float) -> list[Result]:
        """Матчи за период одним запросом; если ESPN его не принял или вернул пусто — по дням."""
        events: list[dict] = []
        try:
            data = await self._fetch(BASE_URL, {"dates": f"{start:%Y%m%d}-{end:%Y%m%d}", "limit": 1000},
                                     f"r{start:%Y%m%d}-{end:%Y%m%d}", ttl)
            events = data.get("events") or []
        except httpx.HTTPError as exc:
            log.info("ESPN NBA: период %s–%s одним запросом не принят (%s), беру по дням", start, end, _why(exc))
        if not events and (end - start).days >= 1:
            day = start
            while day <= end:
                events += await self._day(day, ttl)
                day += timedelta(days=1)
        return [r for r in (parse_event(e) for e in events) if r]

    async def _cdn_season(self) -> list[Result]:
        try:
            data = await self._fetch(CDN_SCHEDULE, None, "cdn", 1800)
        except httpx.HTTPError as exc:
            log.warning("NBA CDN: расписание недоступно: %s", _why(exc))
            return []
        games = [g for d in (data.get("leagueSchedule") or {}).get("gameDates") or [] for g in d.get("games") or []]
        return [r for r in (parse_cdn_game(g) for g in games) if r]

    async def games_between(self, start: date, end: date) -> list[Result]:
        out: dict[int, Result] = {}
        today = datetime.now(timezone.utc).date()
        day = start
        while day <= end:
            chunk_end = min(day + timedelta(days=CHUNK_DAYS - 1), end)
            # прошлые периоды не меняются — кешируем надолго, свежий — на 10 минут
            ttl = 600 if chunk_end + timedelta(days=2) >= today else 7 * 24 * 3600
            for r in await self._chunk(day, chunk_end, ttl):
                out[r.match_id] = r
            day = chunk_end + timedelta(days=1)
        if not out and end >= season_start(today):
            # ESPN недоступен — текущий сезон берём из официального расписания НБА
            for r in await self._cdn_season():
                if start <= r.utc_date.date() <= end + timedelta(days=1):
                    out[r.match_id] = r
        return sorted(out.values(), key=lambda r: r.utc_date)

    async def season_results(self) -> list[Result]:
        today = datetime.now(timezone.utc).date()
        return await self.games_between(season_start(today), today)

    async def results_with_previous(self) -> list[Result]:
        """Текущий сезон плюс прошлый (октябрь–июнь, включая плей-офф)."""
        today = datetime.now(timezone.utc).date()
        start = season_start(today)
        previous = await self.games_between(date(start.year - 1, 10, 1), date(start.year, 6, 30))
        current = await self.games_between(start, today)
        seen = {(r.utc_date.date(), r.home) for r in current}
        return [r for r in previous if (r.utc_date.date(), r.home) not in seen] + current

    async def standings(self) -> list[StandingRow]:
        return standings_from_results(await self.season_results())

    async def finished_between(self, date_from: date, date_to: date) -> list[Result]:
        return await self.games_between(date_from, date_to)


def _why(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return type(exc).__name__
