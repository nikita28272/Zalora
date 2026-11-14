"""Самые вероятные исходы в рабочем диапазоне кэфов (MIN_ODDS–MAX_ODDS, по умолчанию 1.70–2.50).

Владелец берёт кэф 1.70–2.50 и внутри этого диапазона — исходы с самым высоким шансом захода.
Основные линии букмекеров ставятся так, чтобы обе стороны были около 50% (кэф ~1.9). Кэф 1.70 —
это шанс ~55–56%, поэтому дополнительно сдвигаем линию с небольшим запасом:
  * НБА — фора и тотал: по честной вероятности основной линии восстанавливаем ожидаемую разницу
    и сумму очков (нормальное распределение) и подбираем линию под нужный шанс;
  * футбол — двойной шанс 1X / X2 = сумма вероятностей исходов;
  * хоккей и остальное — исходы основной линии, у которых шанс уже 58–75% (фора +1.5 андердога,
    победа фаворита).
Кэф по запасным линиям оценочный (честный кэф с типичной маржой): владелец сверяет его в своей конторе.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

from ..sports import sel_short

from ..config import settings

MARGIN = 0.05  # типичная маржа конторы на такие линии
# Рабочий диапазон кэфов владельца; внутри него — самые вероятные исходы.
ODDS_RANGE = (settings.min_odds, settings.max_odds)
# Запасная линия подбирается под кэф чуть выше нижней границы: 1.72 ≈ шанс 55%.
TARGET = 1 / ((ODDS_RANGE[0] + 0.02) * (1 + MARGIN))
SD_MARGIN = 12.5  # разброс разницы очков НБА
SD_TOTAL = 18.0  # разброс суммы очков НБА
_N = NormalDist()


@dataclass
class SafeOption:
    selection: str
    line: float | None
    p: float  # честная вероятность по рынку
    odds: float  # оценочный кэф
    estimated: bool  # кэф оценочный (запасная линия), а не из линии букмекеров
    label: str

    @property
    def code(self) -> str:
        """Код для /verdict: «hcp_home@+12.5», «under@225.5», «dc_home»."""
        if self.line is None or not self.estimated:
            return self.selection
        return f"{self.selection}@{self.line:+g}" if self.selection.startswith("hcp") else \
            f"{self.selection}@{self.line:g}"


def est_odds(p: float) -> float:
    return round(1 / (p * (1 + MARGIN)), 2)


def _half(x: float) -> float:
    """Ближайшая «половинчатая» линия (…, 10.5, 11.5, …) — без возврата."""
    return math.floor(x) + 0.5


def _phi(x: float) -> float:
    return _N.cdf(x)


def _nba_margin_mean(ctx) -> float | None:
    """Ожидаемая разница очков (хозяева − гости) по честной вероятности основной форы."""
    line, p = ctx.market.lines.get("hcp_home"), ctx.market.fair.get("hcp_home")
    if line is None or p is None or not 0 < p < 1:
        return None
    # хозяева покрывают фору line, если разница + line > 0: P = Φ((m + line) / sd)
    return SD_MARGIN * _N.inv_cdf(p) - line


def _nba_total_mean(ctx) -> float | None:
    line, p = ctx.market.lines.get("total"), ctx.market.fair.get("over")
    if line is None or p is None or not 0 < p < 1:
        return None
    return line - SD_TOTAL * _N.inv_cdf(1 - p)


def alt_market(ctx, selection: str, line: float | None) -> tuple[float, float] | None:
    """Честная вероятность и оценочный кэф для запасной линии или двойного шанса; None — посчитать нельзя."""
    sport = ctx.sport.key
    fair = ctx.market.fair
    if sport == "football" and selection in ("dc_home", "dc_away"):
        side = "home" if selection == "dc_home" else "away"
        if side not in fair or "draw" not in fair:
            return None
        p = fair[side] + fair["draw"]
        return p, est_odds(p)
    if sport != "basketball" or line is None:
        return None
    if selection in ("hcp_home", "hcp_away"):
        m = _nba_margin_mean(ctx)
        if m is None:
            return None
        home_line = line if selection == "hcp_home" else -line
        p_home = _phi((m + home_line) / SD_MARGIN)
        p = p_home if selection == "hcp_home" else 1 - p_home
        return p, est_odds(p)
    if selection in ("over", "under"):
        t = _nba_total_mean(ctx)
        if t is None:
            return None
        p_under = _phi((line - t) / SD_TOTAL)
        p = p_under if selection == "under" else 1 - p_under
        return p, est_odds(p)
    return None


def _fit(ctx, sel: str, ideal: float) -> SafeOption | None:
    """Половинчатая линия рядом с идеальной: самый высокий шанс при оценочном кэфе в диапазоне."""
    best = None
    for line in (_half(ideal) - 1, _half(ideal), _half(ideal) + 1):
        got = alt_market(ctx, sel, line)
        if got and ODDS_RANGE[0] <= got[1] <= ODDS_RANGE[1] and (best is None or got[0] > best[1]):
            best = (line, got[0], got[1])
    if best is None:
        return None
    line, p, o = best
    return SafeOption(sel, line, p, o, True, sel_short(ctx.sport, sel, line))


def safe_options(ctx) -> list[SafeOption]:
    """Исходы с кэфом в рабочем диапазоне для досье — самые вероятные первыми."""
    out: list[SafeOption] = []
    sport, fair, odds = ctx.sport, ctx.market.fair, ctx.market.avg_odds
    # 1. исходы основной линии, у которых шанс уже высокий, а кэф не слишком низкий
    for sel, p in fair.items():
        o = odds.get(sel)
        if sel in ctx.market.estimated or o is None:
            continue
        if ODDS_RANGE[0] <= o <= ODDS_RANGE[1]:
            out.append(SafeOption(sel, ctx.line(sel), p, o, False, ctx.short(sel)))
    # 2. футбол — двойной шанс
    if sport.key == "football":
        for sel in ("dc_home", "dc_away"):
            got = alt_market(ctx, sel, None)
            if got and ODDS_RANGE[0] <= got[1] <= ODDS_RANGE[1]:
                out.append(SafeOption(sel, None, got[0], got[1], True, sel_short(sport, sel)))
    # 3. НБА — фора и тотал с запасом под шанс ~65%
    if sport.key == "basketball":
        m, t = _nba_margin_mean(ctx), _nba_total_mean(ctx)
        z = _N.inv_cdf(TARGET)
        if m is not None:
            for sel, ideal in (("hcp_home", SD_MARGIN * z - m), ("hcp_away", SD_MARGIN * z + m)):
                if (opt := _fit(ctx, sel, ideal)):
                    out.append(opt)
        if t is not None:
            for sel, ideal in (("under", t + SD_TOTAL * z), ("over", t - SD_TOTAL * z)):
                if (opt := _fit(ctx, sel, ideal)):
                    out.append(opt)
    out.sort(key=lambda x: -x.p)
    return out


def safe_text(ctx) -> str:
    opts = safe_options(ctx)
    if not opts:
        return ""
    lines = [f"САМЫЕ ВЕРОЯТНЫЕ ИСХОДЫ С КЭФОМ {ODDS_RANGE[0]:.2f}–{ODDS_RANGE[1]:.2f} (сверху — выше шанс; "
             "* — линия с запасом, кэф оценочный, сверить в конторе):"]
    for o in opts[:6]:
        lines.append(f"  {o.code} = {o.label}: вероятность {o.p:.1%}, кэф {'≈' if o.estimated else ''}{o.odds:.2f}"
                     f"{' *' if o.estimated else ''}")
    return "\n".join(lines)
