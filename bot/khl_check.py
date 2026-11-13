"""Проверка КХЛ в API-Sports: python -m bot.khl_check

Отвечает на вопросы перед подключением КХЛ:
  * есть ли КХЛ и какие сезоны доступны на бесплатном тарифе;
  * отдаёт ли API матчи и таблицу текущего сезона;
  * есть ли коэффициенты и каких контор (особенно российских);
  * бонус: доступен ли текущий сезон футбола (API-Football, тот же ключ).

Тратит около 8 запросов из 100 бесплатных в сутки. Ключ в вывод не попадает.
"""

from __future__ import annotations

import asyncio
import os
from collections import Counter
import httpx
from dotenv import load_dotenv

HOCKEY = "https://v1.hockey.api-sports.io"
FOOTBALL = "https://v3.football.api-sports.io"
CHAMPIONSHIP = 40  # английский Чемпионшип в API-Football

# Российские легальные конторы и их частые написания в API
RU_BOOKS = ("fonbet", "winline", "betboom", "pari", "liga stavok", "ligastavok", "marathon", "olimp",
            "betcity", "leon", "melbet", "tennisi", "zenit", "baltbet", "bettery", "sportbet")


def _errors(data: dict) -> str:
    """API-Sports отвечает HTTP 200 даже при ошибке: смотрим поле errors."""
    err = data.get("errors")
    if not err:
        return ""
    if isinstance(err, dict):
        return "; ".join(f"{k}: {v}" for k, v in err.items())
    return "; ".join(str(e) for e in err)


async def _get(client: httpx.AsyncClient, url: str, **params) -> tuple[dict, str]:
    try:
        resp = await client.get(url, params=params or None)
    except httpx.HTTPError as exc:
        return {}, f"нет связи ({type(exc).__name__})"
    if resp.status_code != 200:
        return {}, f"HTTP {resp.status_code}"
    data = resp.json()
    return data, _errors(data)


def _is_ru_book(name: str) -> bool:
    low = name.lower()
    return any(b in low for b in RU_BOOKS)


async def run(client: httpx.AsyncClient, out=print) -> dict:
    """Все проверки; возвращает сводку (для тестов)."""
    summary: dict = {}

    data, err = await _get(client, f"{HOCKEY}/status")
    if err:
        out(f"❌ Ключ не принят: {err}")
        summary["key_ok"] = False
        return summary
    resp = data.get("response") or {}
    sub, req = resp.get("subscription") or {}, resp.get("requests") or {}
    out(f"✅ Ключ принят. Тариф: {sub.get('plan', '?')}, запросов сегодня: "
        f"{req.get('current', '?')} из {req.get('limit_day', '?')}")
    summary["key_ok"] = True

    # 1. Лига и сезоны
    data, err = await _get(client, f"{HOCKEY}/leagues", search="KHL")
    leagues = [lg for lg in data.get("response") or [] if lg.get("name", "").upper() == "KHL"] if not err else []
    if not leagues:
        out(f"❌ КХЛ не найдена{': ' + err if err else ''}")
        summary["league"] = None
        return summary
    lg = leagues[0]
    seasons = sorted(lg.get("seasons") or [], key=lambda s: s.get("season", 0))
    summary["league"] = lg["id"]
    out(f"✅ КХЛ найдена (id {lg['id']}, {lg.get('country', {}).get('name', '?')}). "
        f"Сезоны: {', '.join(str(s.get('season')) for s in seasons[-6:]) or '—'}")
    current = next((s for s in seasons if s.get("current")), seasons[-1] if seasons else None)
    if current:
        cov = current.get("coverage") or {}
        flags = []
        for key, label in (("standings", "таблица"), ("odds", "кэфы"), ("players", "игроки")):
            if key in cov:
                flags.append(f"{label} {'да' if cov[key] else 'нет'}")
        games_cov = cov.get("games") or {}
        if isinstance(games_cov, dict) and "statistics" in games_cov:
            flags.append(f"статистика матчей {'да' if games_cov['statistics'] else 'нет'}")
        out(f"   Текущий сезон {current.get('season')} ({current.get('start')} — {current.get('end')}): "
            + (", ".join(flags) or "данных о покрытии нет"))

    # 2. Матчи: текущий сезон, а если тариф его не даёт — самый свежий доступный
    season_ok = None
    tried = [s.get("season") for s in reversed(seasons)][:3]
    for season in tried:
        data, err = await _get(client, f"{HOCKEY}/games", league=lg["id"], season=season)
        if err:
            out(f"   ❌ Матчи сезона {season}: {err}")
            continue
        games = data.get("response") or []
        finished = [g for g in games if (g.get("status") or {}).get("short") in ("FT", "AOT", "AP")]
        upcoming = sorted(
            (g for g in games if (g.get("status") or {}).get("short") == "NS"),
            key=lambda g: g.get("date", ""),
        )
        out(f"   ✅ Матчи сезона {season}: всего {len(games)}, сыграно {len(finished)}, впереди {len(upcoming)}")
        for g in upcoming[:3]:
            t = g.get("teams") or {}
            out(f"      {g.get('date', '')[:16]}  {t.get('home', {}).get('name')} — {t.get('away', {}).get('name')}")
        season_ok = season
        summary["games"] = len(games)
        summary["upcoming"] = len(upcoming)
        break
    summary["season"] = season_ok
    if season_ok is None:
        out("❌ Ни один из последних сезонов на этом тарифе не открыт.")
        return summary
    if current and season_ok != current.get("season"):
        out(f"⚠️ Текущий сезон {current.get('season')} на бесплатном тарифе закрыт — доступен только {season_ok}.")

    # 3. Таблица
    data, err = await _get(client, f"{HOCKEY}/standings", league=lg["id"], season=season_ok)
    if err:
        out(f"   ❌ Таблица: {err}")
    else:
        rows = [r for grp in data.get("response") or [] for r in (grp if isinstance(grp, list) else [grp])]
        out(f"   ✅ Таблица: {len(rows)} строк"
            + (f", первая — {rows[0].get('team', {}).get('name')}" if rows else ""))

    # 4. Коэффициенты и конторы
    data, err = await _get(client, f"{HOCKEY}/odds", league=lg["id"], season=season_ok)
    if err:
        out(f"❌ Кэфы: {err}")
        summary["books"] = []
    else:
        items = data.get("response") or []
        books, bets = Counter(), Counter()
        for it in items:
            for b in it.get("bookmakers") or []:
                books[b.get("name", "?")] += 1
                for bet in b.get("bets") or []:
                    bets[bet.get("name", "?")] += 1
        total = (data.get("paging") or {}).get("total")
        out(f"{'✅' if books else '❌'} Кэфы: матчей с линией {len(items)}"
            + (f" (страниц {total})" if total and total > 1 else ""))
        if books:
            ru = [n for n in books if _is_ru_book(n)]
            out("   Конторы: " + ", ".join(f"{n}{' 🇷🇺' if _is_ru_book(n) else ''}" for n, _ in books.most_common(15)))
            out("   Рынки: " + ", ".join(n for n, _ in bets.most_common(8)))
            out(f"   Российские конторы: {', '.join(ru) if ru else 'не найдены'}")
            summary["ru_books"] = ru
        summary["books"] = list(books)

    # 5. Бонус: футбол тем же ключом
    data, err = await _get(client, f"{FOOTBALL}/standings", league=CHAMPIONSHIP, season=season_ok)
    out(f"{'❌' if err else '✅'} Футбол (API-Football), Чемпионшип сезона {season_ok}: "
        + (err or "доступен"))

    data, _ = await _get(client, f"{HOCKEY}/status")
    req = (data.get("response") or {}).get("requests") or {}
    out(f"\nИзрасходовано сегодня: {req.get('current', '?')} из {req.get('limit_day', '?')} запросов.")
    return summary


async def main() -> None:
    load_dotenv()
    key = os.environ.get("API_SPORTS_KEY", "").strip()
    if not key:
        print("❌ API_SPORTS_KEY не задан. Зарегистрируйтесь на api-sports.io, скопируйте ключ из кабинета и\n"
              "   впишите его:  bash scripts/set.sh API_SPORTS_KEY ваш_ключ")
        return
    async with httpx.AsyncClient(headers={"x-apisports-key": key}, timeout=30) as client:
        await run(client)


if __name__ == "__main__":
    asyncio.run(main())
