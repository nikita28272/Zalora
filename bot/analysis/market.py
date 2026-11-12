"""Что «думает» рынок: вероятности из коэффициентов без маржи букмекера."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median

from ..data.odds import BookOdds, Event

SHARP_BOOKS = ("pinnacle", "betfair_ex_eu", "matchbook")

SELECTIONS_1X2 = ("home", "draw", "away")
SELECTIONS_TOTAL = ("over", "under")
ALL_SELECTIONS = SELECTIONS_1X2 + SELECTIONS_TOTAL

_BOOK_FIELD = {"home": "home", "draw": "draw", "away": "away", "over": "over25", "under": "under25"}


def devig(odds: list[float]) -> list[float]:
    """Пропорциональное снятие маржи: 1/k, нормированные к сумме 1."""
    inv = [1.0 / o for o in odds]
    total = sum(inv)
    return [x / total for x in inv]


def margin(odds: list[float]) -> float:
    return sum(1.0 / o for o in odds) - 1.0


@dataclass
class MarketView:
    fair: dict[str, float]  # «честные» вероятности рынка
    avg_odds: dict[str, float]  # средний коэффициент по букмекерам
    best_odds: dict[str, float]  # лучший коэффициент
    n_books: int
    avg_margin_1x2: float | None
    source: str  # pinnacle / consensus


def _books_with(books: list[BookOdds], fields: tuple[str, ...]) -> list[BookOdds]:
    return [b for b in books if all(getattr(b, f) for f in fields)]


def market_view(event: Event) -> MarketView | None:
    fields_1x2 = ("home", "draw", "away")
    fields_tot = ("over25", "under25")
    books_1x2 = _books_with(event.books, fields_1x2)
    if not books_1x2:
        return None
    books_tot = _books_with(event.books, fields_tot)

    fair: dict[str, float] = {}
    source = "consensus"

    def fair_for(books: list[BookOdds], fields: tuple[str, ...], names: tuple[str, ...]) -> dict[str, float]:
        nonlocal source
        sharp = [b for b in books if b.bookmaker in SHARP_BOOKS]
        if sharp:
            source = sharp[0].bookmaker
            probs = devig([getattr(sharp[0], f) for f in fields])
            return dict(zip(names, probs))
        per_book = [devig([getattr(b, f) for f in fields]) for b in books]
        med = [median(p[i] for p in per_book) for i in range(len(fields))]
        total = sum(med)
        return {n: m / total for n, m in zip(names, med)}

    fair.update(fair_for(books_1x2, fields_1x2, SELECTIONS_1X2))
    if books_tot:
        fair.update(fair_for(books_tot, fields_tot, SELECTIONS_TOTAL))

    avg_odds, best_odds = {}, {}
    for sel in ALL_SELECTIONS:
        prices = [getattr(b, _BOOK_FIELD[sel]) for b in event.books if getattr(b, _BOOK_FIELD[sel])]
        if prices:
            avg_odds[sel] = round(mean(prices), 3)
            best_odds[sel] = max(prices)

    return MarketView(
        fair=fair,
        avg_odds=avg_odds,
        best_odds=best_odds,
        n_books=len(books_1x2),
        avg_margin_1x2=mean(margin([b.home, b.draw, b.away]) for b in books_1x2),  # type: ignore[list-item]
        source=source,
    )
