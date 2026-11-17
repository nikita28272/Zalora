"""Доп. рынки без расхода кредитов Odds API: индивидуальные тоталы команд и тотал 1-го периода (НХЛ) /
1-й половины (НБА).

Честная цена считается из основной линии рынка, которая уже есть в досье:
  * хоккей — по честным вероятностям «П1 с ОТ» и «ТБ основной линии» подбираем ожидаемые голы
    хозяев и гостей (Пуассон, ничья в основное время → +1 гол победителю ОТ/буллитов);
    1-й период — доля голов основного времени (в НХЛ ≈ 31%);
  * баскетбол — по основной форе и тоталу: ожидаемые очки каждой команды и 1-й половины
    (нормальное распределение).
Линии этих рынков у конторы мы не знаем (их загрузка стоила бы кредитов), поэтому в досье даём
таблицу «линия → честный шанс → минимальный кэф для ставки», а кэф владелец сверяет в своей конторе.
Тотал команды — с овертаймом (как и основной тотал в боте), 1-й период/половина — без овертайма.
"""

from __future__ import annotations

import math
from statistics import NormalDist

DERIVED = ("it1_over", "it1_under", "it2_over", "it2_under", "p1_over", "p1_under")
DERIVED_GROUPS = (("it1_over", "it1_under"), ("it2_over", "it2_under"), ("p1_over", "p1_under"))
MARGIN = 0.07  # маржа конторы на доп. рынки обычно выше, чем на основные
P1_SHARE_NHL = 0.31  # доля голов основного времени в 1-м периоде НХЛ
H1_SHARE_NBA = 0.505  # доля очков матча в 1-й половине НБА
SD_TEAM_NBA = 11.0  # разброс очков одной команды НБА
SD_H1_NBA = 11.0  # разброс суммы очков 1-й половины НБА
MAX_GOALS = 12
_N = NormalDist()


def est_odds(p: float) -> float:
    return round(1 / (p * (1 + MARGIN)), 2)


def _pmf(lam: float, k: int) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


def _pois_over(lam: float, line: float) -> float:
    """P(X > line) для Пуассона; line — половинчатая."""
    k = math.floor(line)
    return 1 - sum(_pmf(lam, i) for i in range(k + 1))


def hockey_final(lh: float, la: float) -> dict:
    """Распределения итогового счёта с ОТ/буллитами: победа хозяев, сумма голов и голы каждой команды."""
    ph = [_pmf(lh, i) for i in range(MAX_GOALS + 1)]
    pa = [_pmf(la, j) for j in range(MAX_GOALS + 1)]
    w = lh / (lh + la) if lh + la > 0 else 0.5  # кто сильнее — чаще выигрывает ОТ/буллиты
    ml_home = 0.0
    total: dict[int, float] = {}
    home: dict[int, float] = {}
    away: dict[int, float] = {}

    def add(d: dict, k: int, p: float) -> None:
        d[k] = d.get(k, 0.0) + p

    for i, p_i in enumerate(ph):
        for j, p_j in enumerate(pa):
            p = p_i * p_j
            if i != j:
                ml_home += p if i > j else 0.0
                add(total, i + j, p)
                add(home, i, p)
                add(away, j, p)
            else:  # ничья в основное время: +1 гол победителю ОТ/буллитов
                ml_home += p * w
                add(total, i + j + 1, p)
                add(home, i + 1, p * w)
                add(home, i, p * (1 - w))
                add(away, j + 1, p * (1 - w))
                add(away, j, p * w)
    s = sum(total.values())
    return {"ml_home": ml_home / s, "total": {k: v / s for k, v in total.items()},
            "home": {k: v / s for k, v in home.items()}, "away": {k: v / s for k, v in away.items()}}


def _over(dist: dict[int, float], line: float) -> float:
    return sum(p for k, p in dist.items() if k > line)


def fit_hockey(p_ml_home: float, p_over: float, total_line: float) -> tuple[float, float] | None:
    """Ожидаемые голы основного времени (хозяева, гости), при которых модель повторяет рынок."""
    if not (0.05 < p_ml_home < 0.95 and 0.05 < p_over < 0.95):
        return None
    mu, share = 5.6, 0.5  # сумма ожидаемых голов основного времени и доля хозяев
    for _ in range(4):  # поочерёдно подгоняем долю (по победе) и сумму (по тоталу)
        lo, hi = 0.2, 0.8
        for _ in range(30):
            share = (lo + hi) / 2
            got = hockey_final(mu * share, mu * (1 - share))["ml_home"]
            lo, hi = (lo, share) if got > p_ml_home else (share, hi)
        lo, hi = 3.0, 9.0
        for _ in range(30):
            mu = (lo + hi) / 2
            got = _over(hockey_final(mu * share, mu * (1 - share))["total"], total_line)
            lo, hi = (lo, mu) if got > p_over else (mu, hi)
    return mu * share, mu * (1 - share)


def _hockey_lams(ctx) -> tuple[float, float] | None:
    fair, line = ctx.market.fair, ctx.market.lines.get("total")
    if line is None or "ml_home" not in fair or "over" not in fair:
        return None
    return fit_hockey(fair["ml_home"], fair["over"], line)


def _nba_means(ctx) -> tuple[float, float, float] | None:
    from .altlines import _nba_margin_mean, _nba_total_mean

    m, t = _nba_margin_mean(ctx), _nba_total_mean(ctx)
    if m is None or t is None:
        return None
    return (t + m) / 2, (t - m) / 2, t * H1_SHARE_NBA


def derived_prob(ctx, selection: str, line: float | None) -> float | None:
    """Честная вероятность доп. рынка по основной линии; None — посчитать нельзя."""
    if selection not in DERIVED or line is None:
        return None
    over = selection.endswith("_over")
    if ctx.sport.key == "hockey":
        lams = _hockey_lams(ctx)
        if lams is None:
            return None
        lh, la = lams
        if selection.startswith("p1"):
            p = _pois_over((lh + la) * P1_SHARE_NHL, line)
        else:
            fin = hockey_final(lh, la)
            p = _over(fin["home" if selection.startswith("it1") else "away"], line)
    elif ctx.sport.key == "basketball":
        means = _nba_means(ctx)
        if means is None:
            return None
        home, away, h1 = means
        mu, sd = {"it1": (home, SD_TEAM_NBA), "it2": (away, SD_TEAM_NBA), "p1_": (h1, SD_H1_NBA)}[selection[:3]]
        p = 1 - _N.cdf((line - mu) / sd)
    else:
        return None
    p = p if over else 1 - p
    return min(max(p, 0.01), 0.99)


def derived_market(ctx, selection: str, line: float | None) -> tuple[float, float] | None:
    p = derived_prob(ctx, selection, line)
    return (p, est_odds(p)) if p is not None else None


def _half(x: float) -> float:
    return math.floor(x) + 0.5


def derived_text(ctx, min_ev: float = 0.03) -> str:
    """Блок досье «ДОП. РЫНКИ»: линии, честный шанс и минимальный кэф для ставки (шанс × кэф ≥ 1 + min_ev)."""
    sport = ctx.sport.key
    if sport == "hockey":
        lams = _hockey_lams(ctx)
        if lams is None:
            return ""
        lh, la = lams
        it_lines, p1_lines = (1.5, 2.5, 3.5), (0.5, 1.5, 2.5)
        means = (lh + 0.11, la + 0.11, (lh + la) * P1_SHARE_NHL)  # +ОТ ≈ 0.11 гола на команду
        unit, p1_name = "гола", "Тотал 1-го периода (без ОТ)"
    elif sport == "basketball":
        got = _nba_means(ctx)
        if got is None:
            return ""
        means = got
        c1, c2, ch = _half(got[0]), _half(got[1]), _half(got[2])
        it_lines = None
        p1_lines = (ch - 2, ch, ch + 2)
        unit, p1_name = "очка", "Тотал 1-й половины (без ОТ)"
    else:
        return ""

    def row(code: str, line: float) -> str:
        po = derived_prob(ctx, f"{code}_over", line)
        pu = derived_prob(ctx, f"{code}_under", line)
        if min(po, pu) < 0.2:  # кэф далеко за рабочим диапазоном — не загромождаем досье
            return ""
        need = lambda p: (1 + min_ev) / p  # noqa: E731
        return (f"    {line:g}: Б {po:.0%} (от {need(po):.2f}) / М {pu:.0%} (от {need(pu):.2f})"
                f"  — {code}_over@{line:g} / {code}_under@{line:g}")

    home, away = ctx.event.home_team, ctx.event.away_team
    out = ["ДОП. РЫНКИ — честная цена из основной линии (кредиты не тратятся; кэф и линию сверить в конторе, "
           "ставка — если кэф не ниже «от»):"]
    for code, team, mu, lines in (
        ("it1", home, means[0], it_lines or (c1 - 2, c1, c1 + 2)),
        ("it2", away, means[1], it_lines or (c2 - 2, c2, c2 + 2)),
    ):
        out.append(f"  ИТ {'хозяев' if code == 'it1' else 'гостей'} {team} (ожидание {mu:.1f} {unit}, с ОТ):")
        out += [r for ln in lines if (r := row(code, ln))]
    out.append(f"  {p1_name} (ожидание {means[2]:.1f}):")
    out += [r for ln in p1_lines if (r := row("p1", ln))]
    return "\n".join(out)
