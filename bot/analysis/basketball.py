"""Модель НБА: рейтинги нападения и защиты в очках.

Для каждой команды оцениваем, на сколько очков она набирает больше среднего (нападение)
и на сколько больше среднего пропускает (защита), с поправкой на соперников, площадку
и время: свежие матчи весят больше, прошлый сезон — меньше, а несколько стартовых
матчей не дают крайних оценок (сглаживание к среднему).

Ожидаемые очки:  хозяева = среднее + ½ преимущества площадки + нападение хозяев + защита гостей,
                 гости   = среднее − ½ преимущества площадки + нападение гостей + защита хозяев.
Второй матч за два дня (back-to-back) отнимает у уставшей команды ~2 очка разницы.

Разница и сумма очков распределены примерно нормально; разброс берём из остатков модели
(в НБА обычно 12–13 очков для разницы и 17–20 для суммы). Отсюда победа, фора и тотал.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from ..data.stats import Result
from .poisson import ModelOutput

HALF_LIFE_DAYS = 90.0
PRIOR_GAMES = 8.0  # сглаживание: «виртуальные» средние матчи
PREV_SEASON_FACTOR = 0.5  # в НБА составы меняются сильнее, чем в НХЛ
MIN_GAMES = 5
ITERATIONS = 30
DEFAULT_HCA = 2.5  # преимущество своей площадки в очках разницы
B2B_PENALTY = 1.0  # уставшая команда: −1 очко в нападении и +1 пропущенное
B2B_HOURS = 30  # матч накануне (с запасом на часовые пояса)
MARGIN_SD = (11.5, 14.5)  # допустимый разброс разницы очков
TOTAL_SD = (16.0, 22.0)  # допустимый разброс суммы очков


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class Rating:
    offense: float  # +4 — набирает на 4 очка больше среднего
    defense: float  # −3 — пропускает на 3 очка меньше среднего (минус — хорошо)
    games: float


class NbaModel:
    def __init__(self, results: list[Result], now: datetime | None = None, season_start: datetime | None = None):
        self.now = now or datetime.now(timezone.utc)
        self.season_start = season_start
        self.results = sorted(results, key=lambda r: r.utc_date)
        self.counts: dict[str, int] = {}
        self._fit()

    def weight(self, r: Result) -> float:
        days = max((self.now - r.utc_date).total_seconds() / 86400.0, 0.0)
        w = 0.5 ** (days / HALF_LIFE_DAYS)
        if self.season_start is not None and r.utc_date < self.season_start:
            w *= PREV_SEASON_FACTOR
        return w

    def current_season(self, team: str | None = None) -> list[Result]:
        rows = [r for r in self.results if self.season_start is None or r.utc_date >= self.season_start]
        return [r for r in rows if team in (r.home, r.away)] if team else rows

    def _fit(self) -> None:
        games = [(r, self.weight(r), *r.final) for r in self.results]
        w_sum = sum(w for _, w, _, _ in games)
        self.avg = sum(w * (h + a) / 2 for _, w, h, a in games) / w_sum if w_sum else 112.0
        hca = sum(w * (h - a) for _, w, h, a in games) / w_sum if w_sum else DEFAULT_HCA
        self.hca = min(max(hca, 0.0), 5.0)

        played: dict[str, list[tuple[str, float, int, int, bool]]] = {}
        for r, w, h, a in games:
            played.setdefault(r.home, []).append((r.away, w, h, a, True))
            played.setdefault(r.away, []).append((r.home, w, a, h, False))
            self.counts[r.home] = self.counts.get(r.home, 0) + 1
            self.counts[r.away] = self.counts.get(r.away, 0) + 1

        off = {t: 0.0 for t in played}
        dfn = {t: 0.0 for t in played}
        k = PRIOR_GAMES
        for _ in range(ITERATIONS):
            for t, rows in played.items():
                num = sum(w * (pf - self.avg - (self.hca / 2 if home else -self.hca / 2) - dfn[opp])
                          for opp, w, pf, _, home in rows)
                off[t] = num / (sum(w for _, w, _, _, _ in rows) + k)
            for t, rows in played.items():
                num = sum(w * (pa - self.avg - (-self.hca / 2 if home else self.hca / 2) - off[opp])
                          for opp, w, _, pa, home in rows)
                dfn[t] = num / (sum(w for _, w, _, _, _ in rows) + k)
            mo, md = sum(off.values()) / len(off), sum(dfn.values()) / len(dfn)
            off = {t: v - mo for t, v in off.items()}
            dfn = {t: v - md for t, v in dfn.items()}

        self.teams: dict[str, Rating] = {
            t: Rating(off[t], dfn[t], sum(w for _, w, _, _, _ in rows)) for t, rows in played.items()
        }
        # разброс: по остаткам модели, с запасом (в выборке ошибки всегда меньше, чем на новых матчах)
        res_m, res_t, ws = 0.0, 0.0, 0.0
        for r, w, h, a in games:
            eh, ea = self._expected(r.home, r.away)
            res_m += w * ((h - a) - (eh - ea)) ** 2
            res_t += w * ((h + a) - (eh + ea)) ** 2
            ws += w
        sd_m = math.sqrt(res_m / ws) + 0.5 if ws else 12.5
        sd_t = math.sqrt(res_t / ws) + 0.5 if ws else 18.5
        self.margin_sd = min(max(sd_m, MARGIN_SD[0]), MARGIN_SD[1])
        self.total_sd = min(max(sd_t, TOTAL_SD[0]), TOTAL_SD[1])

    def _expected(self, home: str, away: str) -> tuple[float, float]:
        h, a = self.teams[home], self.teams[away]
        eh = self.avg + self.hca / 2 + h.offense + a.defense
        ea = self.avg - self.hca / 2 + a.offense + h.defense
        return eh, ea

    def played_before(self, team: str, kickoff: datetime) -> bool:
        """Играла ли команда накануне (второй матч за два дня)."""
        return any(
            team in (r.home, r.away) and timedelta(0) < kickoff - r.utc_date <= timedelta(hours=B2B_HOURS)
            for r in self.results[-200:]
        )

    def predict(self, home: str, away: str, kickoff: datetime | None = None) -> ModelOutput | None:
        if home not in self.teams or away not in self.teams:
            return None
        if self.counts.get(home, 0) < MIN_GAMES or self.counts.get(away, 0) < MIN_GAMES:
            return None
        eh, ea = self._expected(home, away)
        notes = []
        if kickoff is not None:
            for team, side in ((home, "хозяева"), (away, "гости")):
                if self.played_before(team, kickoff):
                    notes.append(f"{side} играли накануне (back-to-back)")
                    if team == home:
                        eh, ea = eh - B2B_PENALTY, ea + B2B_PENALTY
                    else:
                        eh, ea = eh + B2B_PENALTY, ea - B2B_PENALTY
        out = ModelOutput(eh, ea, basketball_probs(eh, ea, {}, (self.margin_sd, self.total_sd)),
                          self.counts[home], self.counts[away])
        out.sds = (self.margin_sd, self.total_sd)
        out.note = "; ".join(notes)
        return out

    def form(self, team: str, last: int = 5) -> str:
        games = [r for r in self.results if team in (r.home, r.away)][-last:]
        out = []
        for r in games:
            fh, fa = r.final
            gf, ga = (fh, fa) if r.home == team else (fa, fh)
            out.append("W" if gf > ga else "L")
        return "".join(out)

    def last_results(self, team: str, last: int = 5) -> list[str]:
        games = [r for r in self.results if team in (r.home, r.away)][-last:]
        out = []
        for r in games:
            fh, fa = r.final
            ot = " ОТ" if r.decided == "OT" else ""
            out.append(f"{r.utc_date:%d.%m} {r.home} {fh}:{fa}{ot} {r.away}")
        return out


def _cover(mu: float, sd: float, line: float) -> tuple[float, float]:
    """Целочисленная величина X ~ N(mu, sd): (P(X > line), P(X < line)) с поправкой на дискретность.
    При целой линии равенство — возврат, его вероятность из обеих частей убираем."""
    above = math.floor(line) + 1  # наименьшее целое строго больше линии
    below = math.ceil(line) - 1  # наибольшее целое строго меньше линии
    p_over = 1.0 - _phi((above - 0.5 - mu) / sd)
    p_under = _phi((below + 0.5 - mu) / sd)
    return p_over, p_under


def _two_way(a: float, b: float) -> tuple[float, float]:
    s = a + b
    return (a / s, b / s) if s else (0.5, 0.5)


def basketball_probs(exp_home: float, exp_away: float, lines: dict[str, float],
                     sds: tuple[float, float] | None = None) -> dict[str, float]:
    """Победа (с ОТ), фора хозяев по линии и тотал по линии."""
    sd_m, sd_t = sds or (12.5, 18.5)
    mu, total = exp_home - exp_away, exp_home + exp_away
    win, lose = _cover(mu, sd_m, 0.0)
    tie = max(1.0 - win - lose, 0.0)  # ничья после 48 минут — овертайм, почти поровну
    ml_home = win + tie * 0.5
    probs = {"ml_home": ml_home, "ml_away": 1.0 - ml_home}
    hcp = lines.get("hcp_home")
    if hcp is not None:
        # фора хозяев hcp (например −5.5): выигрыш, если разница + hcp > 0
        h_win, h_lose = _cover(mu, sd_m, -hcp)
        probs["hcp_home"], probs["hcp_away"] = _two_way(h_win, h_lose)
    tot = lines.get("total")
    if tot is not None:
        probs["over"], probs["under"] = _two_way(*_cover(total, sd_t, tot))
    return probs
