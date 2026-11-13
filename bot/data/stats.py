"""Статистика матчей через football-data.org (бесплатный тариф: 10 запросов в минуту).

Нужна для трёх вещей:
* результаты сезона, на которых строится статистическая модель;
* турнирная таблица и форма команд;
* итоговые счета, чтобы закрывать сигналы в журнале.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone

import httpx

from ..config import settings

log = logging.getLogger(__name__)

BASE_URL = "https://api.football-data.org/v4"
MIN_INTERVAL = 6.5  # секунд между запросами (лимит 10/мин)


@dataclass
class Result:
    match_id: int
    competition: str
    utc_date: datetime
    home: str
    away: str
    home_goals: int  # голы в основное время (по ним учится модель)
    away_goals: int
    final_home: int | None = None  # итоговый счёт с овертаймом/буллитами (хоккей); None = как основное
    final_away: int | None = None
    decided: str = "REG"  # REG / OT / SO

    @property
    def final(self) -> tuple[int, int]:
        if self.final_home is None or self.final_away is None:
            return self.home_goals, self.away_goals
        return self.final_home, self.final_away


@dataclass
class StandingRow:
    position: int
    team: str
    played: int
    points: int
    goal_diff: int
    form: str | None


class StatsClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.football_data_key
        self._last_call = 0.0
        self._lock = asyncio.Lock()
        self._cache: dict[str, tuple[float, object]] = {}

    async def _get(self, path: str, params: dict | None = None, ttl: float = 6 * 3600) -> dict:
        key = path + repr(sorted((params or {}).items()))
        cached = self._cache.get(key)
        if cached and time.monotonic() - cached[0] < ttl:
            return cached[1]  # type: ignore[return-value]

        async with self._lock:
            for attempt in range(2):
                wait = MIN_INTERVAL - (time.monotonic() - self._last_call)
                if wait > 0:
                    await asyncio.sleep(wait)
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.get(
                        f"{BASE_URL}{path}", params=params, headers={"X-Auth-Token": self.api_key}
                    )
                self._last_call = time.monotonic()
                if resp.status_code == 429 and attempt == 0:
                    log.warning("football-data: лимит запросов, жду минуту")
                    await asyncio.sleep(61)
                    continue
                resp.raise_for_status()
                data = resp.json()
                self._cache[key] = (time.monotonic(), data)
                return data
        raise RuntimeError("football-data: запрос не удался")

    @staticmethod
    def _parse_results(data: dict) -> list[Result]:
        out = []
        for m in data.get("matches", []):
            ft = (m.get("score") or {}).get("fullTime") or {}
            if ft.get("home") is None or ft.get("away") is None:
                continue
            out.append(
                Result(
                    match_id=m["id"],
                    competition=(m.get("competition") or {}).get("code", ""),
                    utc_date=datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00")),
                    home=m["homeTeam"]["name"],
                    away=m["awayTeam"]["name"],
                    home_goals=int(ft["home"]),
                    away_goals=int(ft["away"]),
                )
            )
        return out

    async def season_results(self, competition: str, season: int | None = None) -> list[Result]:
        """Сыгранные матчи сезона. season — год начала (2025 = сезон 2025/26), None — текущий."""
        params = {"status": "FINISHED"}
        if season is not None:
            params["season"] = str(season)
        # прошлые сезоны не меняются — кешируем на сутки
        data = await self._get(f"/competitions/{competition}/matches", params,
                               ttl=24 * 3600 if season is not None else 6 * 3600)
        results = self._parse_results(data)
        for r in results:
            r.competition = r.competition or competition
        return results

    async def results_with_previous(self, competition: str) -> list[Result]:
        """Текущий сезон плюс прошлый (если бесплатный тариф его отдаёт)."""
        current = await self.season_results(competition)
        try:
            previous = await self.season_results(competition, football_season_start(utcnow().date()).year - 1)
        except Exception as exc:  # noqa: BLE001 — прошлый сезон необязателен
            log.info("football-data: прошлый сезон %s недоступен: %s", competition, type(exc).__name__)
            previous = []
        seen = {r.match_id for r in current}
        return [r for r in previous if r.match_id not in seen] + current

    async def standings(self, competition: str) -> list[StandingRow]:
        data = await self._get(f"/competitions/{competition}/standings")
        for block in data.get("standings", []):
            if block.get("type") == "TOTAL":
                return [
                    StandingRow(
                        position=row["position"],
                        team=row["team"]["name"],
                        played=row["playedGames"],
                        points=row["points"],
                        goal_diff=row["goalDifference"],
                        form=row.get("form"),
                    )
                    for row in block.get("table", [])
                ]
        return []

    async def finished_between(self, competitions: list[str], date_from: date, date_to: date) -> list[Result]:
        params = {
            "competitions": ",".join(competitions),
            "dateFrom": date_from.isoformat(),
            "dateTo": date_to.isoformat(),
            "status": "FINISHED",
        }
        data = await self._get("/matches", params, ttl=600)
        return self._parse_results(data)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def football_season_start(today: date) -> date:
    """Европейский футбольный сезон начинается летом: с июля по июнь."""
    return date(today.year if today.month >= 7 else today.year - 1, 7, 1)
