"""Проверка ключей и источников данных: python -m bot.check

Ничего не тратит: Claude проверяется запросом списка моделей (бесплатно),
The Odds API — списком видов спорта (кредиты не списываются).
Ключи в вывод не попадают.
"""

from __future__ import annotations

import asyncio
import sys

import anthropic
import httpx

from .config import LEAGUES, settings
from .data import news
from .data.nhl import NhlClient

OK, FAIL = "✅", "❌"


def _status(resp: httpx.Response) -> str:
    return f"HTTP {resp.status_code}"


async def check_telegram(client: httpx.AsyncClient) -> tuple[bool, str]:
    if not settings.telegram_token:
        return False, "TELEGRAM_BOT_TOKEN не задан"
    resp = await client.get(f"https://api.telegram.org/bot{settings.telegram_token}/getMe")
    if resp.status_code != 200:
        return False, f"токен не принят ({_status(resp)})"
    name = resp.json().get("result", {}).get("username", "?")
    owner = f", владелец {settings.owner_id}" if settings.owner_id else ", TELEGRAM_OWNER_ID пока не задан"
    return True, f"бот @{name}{owner}"


async def check_odds(client: httpx.AsyncClient) -> tuple[bool, str]:
    if not settings.odds_api_key:
        return False, "ODDS_API_KEY не задан"
    resp = await client.get("https://api.the-odds-api.com/v4/sports", params={"apiKey": settings.odds_api_key})
    if resp.status_code != 200:
        return False, f"ключ не принят ({_status(resp)})"
    return True, f"осталось кредитов: {resp.headers.get('x-requests-remaining', '?')}"


async def check_football_data(client: httpx.AsyncClient) -> tuple[bool, str]:
    if not settings.football_data_key:
        return False, "FOOTBALL_DATA_API_KEY не задан"
    resp = await client.get(
        "https://api.football-data.org/v4/competitions/PL",
        headers={"X-Auth-Token": settings.football_data_key},
    )
    if resp.status_code != 200:
        return False, f"ключ не принят ({_status(resp)})"
    return True, "доступ к лигам есть"


async def check_claude() -> tuple[bool, str]:
    if not settings.api_mode:
        return True, "не нужен: режим «босс в чате» (AGENTS_MODE=manual)"
    client = anthropic.AsyncAnthropic(max_retries=1, timeout=30)
    try:
        for model in {settings.llm_model_chief, settings.llm_model_agents}:
            await client.models.retrieve(model)
    except anthropic.AuthenticationError:
        return False, "ключ не принят (неверный или удалён)"
    except anthropic.PermissionDeniedError:
        return False, "у ключа нет доступа (проверьте аккаунт и регион)"
    except anthropic.NotFoundError:
        return False, "модель недоступна для этого ключа"
    except anthropic.APIConnectionError:
        return False, "нет связи с api.anthropic.com"
    except anthropic.APIStatusError as exc:
        return False, f"ошибка API (HTTP {exc.status_code})"
    return True, f"доступ есть: босс {settings.llm_model_chief}, помощники {settings.llm_model_agents}"


async def check_nhl() -> tuple[bool, str]:
    if not any(lg.sport == "hockey" for lg in LEAGUES):
        return True, "НХЛ выключена (ENABLED_LEAGUES)"
    client = NhlClient()
    rows = await client.standings()
    games = await client.season_results()
    if not rows or not games:
        return False, f"данные не распознаны: команд {len(rows)}, матчей {len(games)} (хоккей пойдёт без модели)"
    ot = sum(g.decided != "REG" for g in games)
    return True, f"команд {len(rows)}, сыграно матчей {len(games)}, из них с ОТ/буллитами {ot}"


async def check_rss() -> tuple[bool, str]:
    heads = await news.fetch_all()
    return (len(heads) > 0), f"заголовков: {len(heads)}"


async def main() -> int:
    results: list[tuple[str, bool, str]] = []
    async with httpx.AsyncClient(timeout=20) as client:
        for title, coro in (
            ("Telegram", check_telegram(client)),
            ("Линии (The Odds API)", check_odds(client)),
            ("Статистика (football-data.org)", check_football_data(client)),
        ):
            try:
                ok, msg = await coro
            except httpx.RequestError as exc:
                ok, msg = False, f"нет связи ({type(exc).__name__})"
            results.append((title, ok, msg))
    for title, coro in (("НХЛ (статистика)", check_nhl()), ("Claude API", check_claude()),
                        ("Новости (RSS)", check_rss())):
        try:
            ok, msg = await coro
        except Exception as exc:  # noqa: BLE001
            ok, msg = False, f"ошибка ({type(exc).__name__})"
        results.append((title, ok, msg))

    print("\nПроверка бота:")
    for title, ok, msg in results:
        print(f"{OK if ok else FAIL} {title}: {msg}")
    critical = [r for r in results if not r[1] and r[0] not in ("Новости (RSS)", "НХЛ (статистика)")]
    if critical:
        print("\nИсправьте ключ командой: bash scripts/set.sh НАЗВАНИЕ ЗНАЧЕНИЕ, затем: bash scripts/check.sh")
        return 1
    print("\nВсё в порядке. Напишите боту /matches")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
