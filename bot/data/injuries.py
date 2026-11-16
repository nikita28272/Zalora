"""Отчёты о травмах ESPN (site.api.espn.com, бесплатно, без ключа) для НБА и НХЛ.

Формат ответа /injuries: {"injuries": [{"displayName": "<команда>", "injuries": [
    {"status": "Out", "athlete": {"displayName": "...", "position": {"abbreviation": "C"}},
     "shortComment": "...", "details": {"type": "Knee", "returnDate": "2026-12-01"}}, ...]}, ...]}
Разбор с запасом: если формат изменится или сайт не ответит — травм просто нет, бот работает как раньше.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

URLS = {
    "basketball": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries",
    "hockey": "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries",
}
CACHE_SECONDS = 30 * 60
STATUS_RU = {"out": "не играет", "injured reserve": "в списке травмированных (не играет)",
             "ir": "в списке травмированных (не играет)", "ltir": "долгосрочно травмирован (не играет)", "doubtful": "вряд ли сыграет", "questionable": "под вопросом",
             "day-to-day": "решат в день игры", "probable": "скорее сыграет", "suspension": "дисквалифицирован"}


@dataclass
class Injury:
    name: str
    status: str  # как у ESPN: Out / Day-To-Day / Questionable …
    detail: str = ""

    @property
    def out(self) -> bool:
        """Точно не играет: Out, список травмированных (Injured Reserve / IR, LTIR), дисквалификация."""
        s = self.status.lower()
        return (s.startswith("out") or "suspen" in s or "injured reserve" in s
                or s in ("ir", "ltir", "il") or s.startswith(("ir-", "ir ", "ltir")))

    @property
    def status_ru(self) -> str:
        s = self.status.lower()
        for key, ru in STATUS_RU.items():
            if s.startswith(key):
                return ru + (" до конца сезона" if "season" in s else "")
        return self.status

    def as_list(self) -> list[str]:
        return [self.name, self.status, self.detail]

    @classmethod
    def from_list(cls, x: list[str]) -> "Injury":
        return cls(*x[:3])


def parse_injuries(data: dict) -> dict[str, list[Injury]]:
    """Ответ ESPN -> {команда: [травмы]}; битые записи пропускаем."""
    out: dict[str, list[Injury]] = {}
    for team in (data or {}).get("injuries") or []:
        name = team.get("displayName") or (team.get("team") or {}).get("displayName")
        if not name:
            continue
        items = []
        for it in team.get("injuries") or []:
            ath = it.get("athlete") or {}
            player = ath.get("displayName") or ath.get("fullName")
            status = it.get("status") or (it.get("type") or {}).get("description") or ""
            if not player or not status:
                continue
            det = it.get("details") or {}
            detail = det.get("type") or it.get("shortComment") or ""
            items.append(Injury(player, str(status).strip(), str(detail)[:40]))
        out[name] = items
    return out


class InjuryClient:
    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout
        self._cache: dict[str, tuple[float, dict[str, list[Injury]]]] = {}

    async def fetch(self, sport: str) -> dict[str, list[Injury]]:
        url = URLS.get(sport)
        if not url:
            return {}
        ts, cached = self._cache.get(sport, (0.0, {}))
        if time.time() - ts < CACHE_SECONDS:
            return cached
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                result = parse_injuries(resp.json())
        except Exception as exc:  # noqa: BLE001
            log.warning("ESPN injuries (%s) недоступны: %s", sport, type(exc).__name__)
            return cached
        self._cache[sport] = (time.time(), result)
        return result
