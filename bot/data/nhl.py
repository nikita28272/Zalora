"""Статистика НХЛ из открытого API лиги (api-web.nhle.com, ключ не нужен).

Используются:
* /v1/schedule/{дата} — неделя матчей со счётом и типом развязки (REG / OT / SO);
* /v1/standings/now — таблица.

Разбор написан с запасом: если формат ответа изменится, бот не упадёт,
а просто будет работать по хоккею без статистической модели (только рынок).
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, datetime, timedelta, timezone

import httpx

from .stats import Result, StandingRow

log = logging.getLogger(__name__)

BASE_URL = "https://api-web.nhle.com/v1"
FINISHED = {"OFF", "FINAL"}
GAME_TYPES = {2, 3}  # регулярный сезон и плей-офф


def _text(obj) -> str:
    if isinstance(obj, dict):
        return str(obj.get("default") or next(iter(obj.values()), ""))
    return str(obj or "")


def team_name(team: dict) -> str:
    """«Toronto» + «Maple Leafs» -> «Toronto Maple Leafs» (как в линиях букмекеров)."""
    place, common = _text(team.get("placeName")), _text(team.get("commonName"))
    if place and common:
        return f"{place} {common}"
    return (_text(team.get("name")) or _text(team.get("teamName")) or common or place
            or str(team.get("abbrev", "?")))


def parse_game(g: dict) -> Result | None:
    """Завершённый матч -> Result. Голы основного времени: при ОТ/буллитах ничья по меньшему счёту."""
    try:
        if g.get("gameState") not in FINISHED or g.get("gameType", 2) not in GAME_TYPES:
            return None
        home, away = g["homeTeam"], g["awayTeam"]
        fh, fa = int(home["score"]), int(away["score"])
        decided = str((g.get("gameOutcome") or {}).get("lastPeriodType")
                      or (g.get("periodDescriptor") or {}).get("periodType") or "REG").upper()
        if decided not in ("REG", "OT", "SO"):
            decided = "REG"
        if decided in ("OT", "SO"):
            rh = ra = min(fh, fa)
        else:
            rh, ra = fh, fa
        return Result(
            match_id=int(g["id"]),
            competition="NHL",
            utc_date=datetime.fromisoformat(str(g["startTimeUTC"]).replace("Z", "+00:00")),
            home=team_name(home),
            away=team_name(away),
            home_goals=rh,
            away_goals=ra,
            final_home=fh,
            final_away=fa,
            decided=decided,
        )
    except (KeyError, TypeError, ValueError):
        return None


def season_start(today: date) -> date:
    """Сезон НХЛ стартует в октябре: в январе–сентябре это октябрь прошлого года."""
    year = today.year if today.month >= 9 else today.year - 1
    return date(year, 10, 1)


class NhlClient:
    def __init__(self) -> None:
        self._cache: dict[str, tuple[float, object]] = {}
        self._lock = asyncio.Lock()

    async def _get(self, path: str, ttl: float) -> dict:
        cached = self._cache.get(path)
        if cached and time.monotonic() - cached[0] < ttl:
            return cached[1]  # type: ignore[return-value]
        async with self._lock:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                resp = await client.get(f"{BASE_URL}{path}")
            resp.raise_for_status()
            data = resp.json()
        self._cache[path] = (time.monotonic(), data)
        return data

    async def games_between(self, start: date, end: date) -> list[Result]:
        """Завершённые матчи за период. Запрос /schedule/{дата} возвращает неделю."""
        out: dict[int, Result] = {}
        day = start
        today = datetime.now(timezone.utc).date()
        while day <= end:
            # прошлые недели не меняются — кешируем надолго, текущую — на 10 минут
            ttl = 600 if day + timedelta(days=7) >= today else 7 * 24 * 3600
            try:
                data = await self._get(f"/schedule/{day.isoformat()}", ttl)
            except httpx.HTTPError as exc:
                log.warning("NHL API: неделя %s недоступна: %s", day, type(exc).__name__)
                day += timedelta(days=7)
                continue
            for week_day in data.get("gameWeek", []):
                for g in week_day.get("games", []):
                    r = parse_game(g)
                    if r and start <= r.utc_date.date() <= end + timedelta(days=1):
                        out[r.match_id] = r
            day += timedelta(days=7)
        return sorted(out.values(), key=lambda r: r.utc_date)

    async def season_results(self) -> list[Result]:
        today = datetime.now(timezone.utc).date()
        return await self.games_between(season_start(today), today)

    async def results_with_previous(self) -> list[Result]:
        """Текущий сезон плюс прошлый (октябрь–июнь, включая плей-офф)."""
        today = datetime.now(timezone.utc).date()
        start = season_start(today)
        prev_start = date(start.year - 1, 10, 1)
        previous = await self.games_between(prev_start, date(start.year, 6, 30))
        current = await self.games_between(start, today)
        seen = {r.match_id for r in current}
        return [r for r in previous if r.match_id not in seen] + current

    async def standings(self) -> list[StandingRow]:
        data = await self._get("/standings/now", 3 * 3600)
        rows = []
        for i, t in enumerate(data.get("standings", []), start=1):
            try:
                name = _text(t.get("teamName")) or f"{_text(t.get('placeName'))} {_text(t.get('teamCommonName'))}"
                l10 = f"{t.get('l10Wins', '?')}-{t.get('l10Losses', '?')}-{t.get('l10OtLosses', '?')}"
                rows.append(StandingRow(
                    position=int(t.get("leagueSequence") or i),
                    team=name.strip(),
                    played=int(t.get("gamesPlayed", 0)),
                    points=int(t.get("points", 0)),
                    goal_diff=int(t.get("goalDifferential", 0)),
                    form=f"последние 10: {l10}",
                ))
            except (TypeError, ValueError):
                continue
        return sorted(rows, key=lambda r: r.position)

    async def finished_between(self, date_from: date, date_to: date) -> list[Result]:
        return await self.games_between(date_from - timedelta(days=6), date_to)
