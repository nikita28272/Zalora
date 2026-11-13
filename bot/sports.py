"""Виды спорта, их исходы (рынки) и подписи для сообщений.

Коды исходов:
  футбол: home / draw / away (1X2), over / under (тотал, по умолчанию 2.5)
  хоккей: ml_home / ml_away (победа с учётом овертайма и буллитов),
          home / draw / away (исход основного времени, 1X2),
          over / under (тотал матча с овертаймом, линия букмекера, обычно 5.5–6.5),
          hcp_home / hcp_away (фора, обычно ±1.5, с овертаймом)
  баскетбол: ml_home / ml_away (победа с учётом овертайма), hcp_home / hcp_away (фора по очкам),
          over / under (тотал очков) — всё с овертаймом
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Sport:
    key: str
    emoji: str
    name_ru: str
    outcomes: tuple[str, ...]
    groups: tuple[tuple[str, ...], ...]  # взаимоисключающие исходы одного рынка
    odds_markets: str  # рынки для The Odds API
    odds_markets_fallback: str | None = None  # если API не знает какой-то рынок
    default_total: float = 2.5


FOOTBALL = Sport(
    key="football",
    emoji="⚽",
    name_ru="Футбол",
    outcomes=("home", "draw", "away", "over", "under"),
    groups=(("home", "draw", "away"), ("over", "under")),
    odds_markets="h2h,totals",
)

HOCKEY = Sport(
    key="hockey",
    emoji="🏒",
    name_ru="Хоккей",
    outcomes=("ml_home", "ml_away", "home", "draw", "away", "over", "under", "hcp_home", "hcp_away"),
    groups=(("ml_home", "ml_away"), ("home", "draw", "away"), ("over", "under"), ("hcp_home", "hcp_away")),
    odds_markets="h2h,spreads,totals,h2h_3_way",
    odds_markets_fallback="h2h,spreads,totals",
    default_total=5.5,
)

BASKETBALL = Sport(
    key="basketball",
    emoji="🏀",
    name_ru="Баскетбол",
    outcomes=("ml_home", "ml_away", "hcp_home", "hcp_away", "over", "under"),
    groups=(("ml_home", "ml_away"), ("hcp_home", "hcp_away"), ("over", "under")),
    odds_markets="h2h,spreads,totals",
    default_total=225.5,
)

SPORTS = {s.key: s for s in (FOOTBALL, HOCKEY, BASKETBALL)}


def group_of(sport: Sport, selection: str) -> tuple[str, ...]:
    for g in sport.groups:
        if selection in g:
            return g
    raise ValueError(f"Исход {selection} не относится к виду спорта {sport.key}")


def _fmt_line(x: float | None) -> str:
    if x is None:
        return ""
    return f"{x:+g}" if x else "0"


def line_for(selection: str, lines: dict) -> float | None:
    if selection in ("over", "under"):
        return lines.get("total")
    if selection == "hcp_home":
        return lines.get("hcp_home")
    if selection == "hcp_away":
        h = lines.get("hcp_home")
        return -h if h is not None else None
    return None


def sel_short(sport: Sport, selection: str, line: float | None = None) -> str:
    """Короткая подпись: П1, X, ТБ2.5, П1 ОТ, Ф1(-1.5)."""
    hockey = sport.key == "hockey"
    if sport.key == "basketball" and selection in ("ml_home", "ml_away"):
        return "П1" if selection == "ml_home" else "П2"  # в баскетболе ничьих нет, ОТ всегда включён
    if selection == "home":
        return "П1 осн" if hockey else "П1"
    if selection == "draw":
        return "X осн" if hockey else "X"
    if selection == "away":
        return "П2 осн" if hockey else "П2"
    if selection == "ml_home":
        return "П1 ОТ"
    if selection == "ml_away":
        return "П2 ОТ"
    if selection == "over":
        return f"ТБ{line if line is not None else sport.default_total:g}"
    if selection == "under":
        return f"ТМ{line if line is not None else sport.default_total:g}"
    if selection == "hcp_home":
        return f"Ф1({_fmt_line(line)})"
    if selection == "hcp_away":
        return f"Ф2({_fmt_line(line)})"
    return selection


def sel_ru(sport: Sport, selection: str, line: float | None = None) -> str:
    """Подробная подпись для сигнала."""
    hockey = sport.key == "hockey"
    total = line if line is not None else sport.default_total
    if sport.key == "basketball":
        return {
            "ml_home": "П1 — победа хозяев (с учётом овертайма)",
            "ml_away": "П2 — победа гостей (с учётом овертайма)",
            "over": f"ТБ {total:g} — больше {total:g} очков (с овертаймом)",
            "under": f"ТМ {total:g} — меньше {total:g} очков (с овертаймом)",
            "hcp_home": f"Ф1({_fmt_line(line)}) — фора хозяев {_fmt_line(line)} очков (с овертаймом)",
            "hcp_away": f"Ф2({_fmt_line(line)}) — фора гостей {_fmt_line(line)} очков (с овертаймом)",
        }.get(selection, selection)
    names = {
        "home": "П1 — победа хозяев в основное время" if hockey else "П1 (победа хозяев)",
        "draw": "X — ничья в основное время" if hockey else "X (ничья)",
        "away": "П2 — победа гостей в основное время" if hockey else "П2 (победа гостей)",
        "ml_home": "П1 с ОТ — победа хозяев с учётом овертайма и буллитов",
        "ml_away": "П2 с ОТ — победа гостей с учётом овертайма и буллитов",
        "over": f"ТБ {total:g} — больше {total:g} голов" + (" (с овертаймом)" if hockey else ""),
        "under": f"ТМ {total:g} — меньше {total:g} голов" + (" (с овертаймом)" if hockey else ""),
        "hcp_home": f"Ф1({_fmt_line(line)}) — фора хозяев {_fmt_line(line)} (с овертаймом)",
        "hcp_away": f"Ф2({_fmt_line(line)}) — фора гостей {_fmt_line(line)} (с овертаймом)",
    }
    return names.get(selection, selection)
