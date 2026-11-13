"""Что «думает» рынок: вероятности из коэффициентов без маржи букмекера.

Для каждого рынка вида спорта (1X2, тотал, фора, победа с ОТ) берём основную линию —
ту, что предлагает больше всего букмекеров, — снимаем маржу и получаем «честные»
вероятности. Если есть Pinnacle или биржа (самые точные линии), берём их, иначе медиану.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from statistics import mean, median

from ..data.odds import BookOdds, Event
from ..sports import SPORTS, Sport

SHARP_BOOKS = ("pinnacle", "betfair_ex_eu", "matchbook")

# Хоккей: доля ничьих в основное время в НХЛ — около 22–24%. Нужна, только если
# в линии нет 1X2 основного времени и его приходится оценивать по П1/П2 с ОТ.
NHL_REG_DRAW = 0.23

# Отсев испорченных котировок. Честная линия всегда с маржой: сумма 1/кэф ≥ 1.
# Меньше — значит букмекер отдал под этим рынком другой (например, основное время вместо «с ОТ»).
MAX_BOOK_MARGIN = 0.25  # шире — подозрительно (ошибка данных или экзотическая линия)
MAX_DEVIATION = 0.10  # котировка, которая расходится с медианой рынка больше чем на 10 п.п., — выброс
ESTIMATED_MARGIN = 0.07  # типичная маржа букмекера на 1X2 в хоккее — для оценочного кэфа


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
    avg_margin_1x2: float | None  # маржа основного рынка
    source: str  # pinnacle / consensus
    lines: dict[str, float] = field(default_factory=dict)  # {"total": 6.5, "hcp_home": -1.5}
    estimated: list[str] = field(default_factory=list)  # исходы с оценочным, а не реальным кэфом


def _main_line(counts: Counter, preferred: float | None, default: float) -> float | None:
    if not counts:
        return None
    if preferred is not None and preferred in counts:
        return preferred
    top = max(counts.values())
    return min((p for p, c in counts.items() if c == top), key=lambda p: abs(p - default))


def _pick_lines(sport: Sport, books: list[BookOdds]) -> dict[str, float]:
    lines: dict[str, float] = {}
    if "over" in sport.outcomes:
        totals = Counter(
            p for b in books for p in b.points("over") if b.price("under", p) is not None
        )
        preferred = 2.5 if sport.key == "football" else None
        line = _main_line(totals, preferred, sport.default_total)
        if line is not None:
            lines["total"] = line
    if "hcp_home" in sport.outcomes:
        hcps = Counter(
            p for b in books for p in b.points("hcp_home") if b.price("hcp_away", -p) is not None
        )
        line = _main_line(hcps, None, -1.5)
        if line is not None:
            lines["hcp_home"] = line
    return lines


def _point(selection: str, lines: dict[str, float]) -> float | None:
    if selection in ("over", "under"):
        return lines.get("total")
    if selection == "hcp_home":
        return lines.get("hcp_home")
    if selection == "hcp_away":
        return -lines["hcp_home"] if "hcp_home" in lines else None
    return None


def clean_books(books: list[BookOdds], group: tuple[str, ...], pts: dict) -> list[BookOdds]:
    """Выбрасываем невозможные и выбивающиеся котировки, чтобы они не портили средний и лучший кэф."""
    sane = []
    for b in books:
        m = margin([b.price(s, pts[s]) for s in group])
        if 0 <= m <= MAX_BOOK_MARGIN:
            sane.append(b)
    if len(sane) < 3:
        return sane
    per_book = {id(b): devig([b.price(s, pts[s]) for s in group]) for b in sane}
    med = [median(p[i] for p in per_book.values()) for i in range(len(group))]
    return [b for b in sane if max(abs(per_book[id(b)][i] - med[i]) for i in range(len(group))) <= MAX_DEVIATION]


def market_view(event: Event) -> MarketView | None:
    sport = SPORTS[event.league.sport]
    lines = _pick_lines(sport, event.books)

    fair: dict[str, float] = {}
    avg_odds: dict[str, float] = {}
    best_odds: dict[str, float] = {}
    source = "consensus"
    main_books: list[BookOdds] = []
    main_margin: float | None = None

    for gi, group in enumerate(sport.groups):
        if any(s in ("over", "under", "hcp_home", "hcp_away") for s in group) and not all(
            _point(s, lines) is not None for s in group
        ):
            continue  # нет основной линии для тотала/форы
        pts = {s: _point(s, lines) for s in group}
        books = clean_books(
            [b for b in event.books if all(b.price(s, pts[s]) for s in group)], group, pts
        )
        if not books:
            continue
        sharp = [b for b in books if b.bookmaker in SHARP_BOOKS]
        if sharp:
            probs = devig([sharp[0].price(s, pts[s]) for s in group])
            group_source = sharp[0].bookmaker
        else:
            per_book = [devig([b.price(s, pts[s]) for s in group]) for b in books]
            med = [median(p[i] for p in per_book) for i in range(len(group))]
            total = sum(med)
            probs = [m / total for m in med]
            group_source = "consensus"
        fair.update(zip(group, probs))
        for s in group:
            prices = [b.price(s, pts[s]) for b in books]
            avg_odds[s] = round(mean(prices), 3)
            best_odds[s] = max(prices)
        if gi == 0:
            main_books, source = books, group_source
            main_margin = mean(margin([b.price(s, pts[s]) for s in group]) for b in books)

    if not main_books:
        return None  # нет основного рынка (1X2 в футболе, П1/П2 с ОТ в хоккее)

    estimated: list[str] = []
    if sport.key == "hockey" and "home" not in fair and "ml_home" in fair:
        # 1X2 основного времени нет в линии — оцениваем по П1/П2 с ОТ.
        # Сильная команда чаще выигрывает и овертайм, поэтому делим ничью пропорционально.
        d = NHL_REG_DRAW
        fair["home"] = fair["ml_home"] * (1 - d)
        fair["away"] = fair["ml_away"] * (1 - d)
        fair["draw"] = d
        for s in ("home", "draw", "away"):
            avg_odds[s] = best_odds[s] = round(1 / (fair[s] * (1 + ESTIMATED_MARGIN)), 2)
            estimated.append(s)

    return MarketView(
        fair=fair,
        avg_odds=avg_odds,
        best_odds=best_odds,
        n_books=len(main_books),
        avg_margin_1x2=main_margin,
        source=source,
        lines=lines,
        estimated=estimated,
    )
