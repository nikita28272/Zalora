"""Стартовые вратари НХЛ со страницы DailyFaceoff (dailyfaceoff.com/starting-goalies/ДАТА, бесплатно, без ключа).

На странице — данные Next.js (<script id="__NEXT_DATA__">) со списком матчей дня, в каждом:
  homeTeamName / awayTeamName, homeGoalieName / awayGoalieName,
  homeNewsStrengthName / awayNewsStrengthName — «Confirmed» (подтверждён), «Likely» (скорее всего),
  «Unconfirmed» (не подтверждён).
Дата страницы — по времени Нью-Йорка (как расписание НХЛ).
Разбор с запасом: если формат изменится или сайт не ответит — вратарей просто нет, сторож работает
только по травмам, как раньше.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import date

import httpx

log = logging.getLogger(__name__)

URL = "https://www.dailyfaceoff.com/starting-goalies/{day}"
CACHE_SECONDS = 10 * 60
_NEXT = re.compile(r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)
STRENGTH_RU = {"confirmed": "подтверждён", "likely": "скорее всего", "unconfirmed": "не подтверждён"}


@dataclass
class GoalieStart:
    team: str
    name: str
    strength: str  # confirmed / likely / unconfirmed

    @property
    def confirmed(self) -> bool:
        return self.strength == "confirmed"

    @property
    def strength_ru(self) -> str:
        return STRENGTH_RU.get(self.strength, self.strength)


def _strength(raw) -> str:
    s = str(raw or "").strip().lower()
    if s.startswith("confirm"):
        return "confirmed"
    if s.startswith(("likely", "expected", "projected")):
        return "likely"
    return "unconfirmed"


def _games(obj):
    """Все словари с вратарями хозяев и гостей, где бы они ни лежали в данных страницы."""
    if isinstance(obj, dict):
        if "homeGoalieName" in obj and "awayGoalieName" in obj:
            yield obj
            return
        for v in obj.values():
            yield from _games(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _games(v)


def parse_goalies(html: str) -> dict[str, GoalieStart]:
    """Страница -> {команда: вратарь}; битые записи пропускаем."""
    m = _NEXT.search(html or "")
    if not m:
        return {}
    try:
        data = json.loads(m.group(1))
    except ValueError:
        return {}
    out: dict[str, GoalieStart] = {}
    for g in _games(data):
        for side in ("home", "away"):
            team, name = g.get(f"{side}TeamName"), g.get(f"{side}GoalieName")
            if team and name:
                out[str(team)] = GoalieStart(str(team), str(name).strip(), _strength(g.get(f"{side}NewsStrengthName")))
    return out


class GoalieClient:
    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout
        self._cache: dict[str, tuple[float, dict[str, GoalieStart]]] = {}

    async def fetch(self, day: date, max_age: float = CACHE_SECONDS) -> dict[str, GoalieStart]:
        key = day.isoformat()
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < max_age:
            return hit[1]
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True,
                                         headers={"User-Agent": "Mozilla/5.0 (compatible; signals-bot)"}) as c:
                r = await c.get(URL.format(day=key))
                r.raise_for_status()
            got = parse_goalies(r.text)
        except Exception as exc:  # noqa: BLE001
            log.warning("starting goalies %s: %s", key, exc)
            got = {}
        if not got and hit:  # сайт не ответил — держим прошлые данные, чтобы не прислать ложных новостей
            return hit[1]
        self._cache[key] = (time.monotonic(), got)
        return got
