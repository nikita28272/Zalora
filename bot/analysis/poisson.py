"""Статистический аналитик: модель Пуассона по результатам текущего сезона.

Для каждой команды оцениваем две силы: атаку и оборону. Считаем их по ВСЕМ матчам
команды (дома и в гостях вместе), а разницу между домом и выездом учитываем общей
для лиги поправкой. Так данных на команду вдвое больше, чем при раздельном счёте.

Силы уточняются итеративно с учётом соперников: голы в ворота сильной обороны
стоят больше, чем в ворота слабой. Свежие матчи весят больше (затухание по времени),
а небольшое сглаживание к среднему не даёт 3–4 матчам дать экстремальную оценку.
По ожидаемым голам строим матрицу счетов: вероятности 1X2 и тотала 2.5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from ..data.stats import Result

HALF_LIFE_DAYS = 120.0
PRIOR_GAMES = 4.0  # «виртуальные» средние матчи; подобрано на симуляции сезонов (лучшая калибровка)
MIN_GAMES = 4  # меньше — модель не строим
MAX_GOALS = 12
ITERATIONS = 40
DRAW_PRIOR_GAMES = 60  # сглаживание фактической доли ничьих к прогнозу модели
# Матчи прошлого сезона дополнительно умножаются на этот вес (состав и тренер могли смениться).
# Подобрано на симуляции: при 11 турах ошибка 1X2 ниже на ~16–18%, чем без прошлого сезона.
PREV_SEASON_FACTOR = 0.7


@dataclass
class TeamStrength:
    attack: float  # 1.0 — средняя атака лиги, 1.3 — забивает на 30% больше среднего
    defense: float  # 1.0 — средняя оборона, 0.7 — пропускает на 30% меньше среднего
    games: float  # взвешенное число матчей


@dataclass
class ModelOutput:
    lambda_home: float
    lambda_away: float
    probs: dict[str, float]  # вероятности исходов (набор зависит от вида спорта)
    home_games: int
    away_games: int
    draw_boost: float = 1.0  # калибровка доли ничьих (см. PoissonModel._calibrate_draws)


def _weight(when: datetime, now: datetime) -> float:
    days = max((now - when).total_seconds() / 86400.0, 0.0)
    return 0.5 ** (days / HALF_LIFE_DAYS)


class PoissonModel:
    def __init__(self, results: list[Result], now: datetime | None = None, season_start: datetime | None = None):
        self.now = now or datetime.now(timezone.utc)
        self.season_start = season_start
        self.results = sorted(results, key=lambda r: r.utc_date)
        self.counts: dict[str, list[int]] = {}
        self._fit()

    def weight(self, r: Result) -> float:
        w = _weight(r.utc_date, self.now)
        if self.season_start is not None and r.utc_date < self.season_start:
            w *= PREV_SEASON_FACTOR
        return w

    def current_season(self, team: str | None = None) -> list[Result]:
        rows = [r for r in self.results if self.season_start is None or r.utc_date >= self.season_start]
        return [r for r in rows if team in (r.home, r.away)] if team else rows

    def _fit(self) -> None:
        games = [(r, self.weight(r)) for r in self.results]
        w_sum = sum(w for _, w in games)
        self.avg_home = sum(w * r.home_goals for r, w in games) / w_sum if w_sum else 1.5
        self.avg_away = sum(w * r.away_goals for r, w in games) / w_sum if w_sum else 1.2

        # (соперник, вес, забил, пропустил, дома ли) для каждой команды
        played: dict[str, list[tuple[str, float, int, int, bool]]] = {}
        for r, w in games:
            played.setdefault(r.home, []).append((r.away, w, r.home_goals, r.away_goals, True))
            played.setdefault(r.away, []).append((r.home, w, r.away_goals, r.home_goals, False))
            self.counts.setdefault(r.home, [0, 0])[0] += 1
            self.counts.setdefault(r.away, [0, 0])[1] += 1

        att = {t: 1.0 for t in played}
        dfn = {t: 1.0 for t in played}
        k = PRIOR_GAMES
        mean_base = (self.avg_home + self.avg_away) / 2
        for _ in range(ITERATIONS):
            for t, rows in played.items():
                # атака: забитые против ожидаемых с учётом обороны соперника и поля
                scored = sum(w * gf for _, w, gf, _, _ in rows)
                expected = sum(
                    w * (self.avg_home if home else self.avg_away) * dfn[opp] for opp, w, _, _, home in rows
                )
                att[t] = (scored + k * mean_base) / (expected + k * mean_base)
            for t, rows in played.items():
                conceded = sum(w * ga for _, w, _, ga, _ in rows)
                expected = sum(
                    w * (self.avg_away if home else self.avg_home) * att[opp] for opp, w, _, _, home in rows
                )
                dfn[t] = (conceded + k * mean_base) / (expected + k * mean_base)
            # нормировка: средняя команда лиги = 1.0
            ma = sum(att.values()) / len(att)
            md = sum(dfn.values()) / len(dfn)
            att = {t: v / ma for t, v in att.items()}
            dfn = {t: v / md for t, v in dfn.items()}

        self.teams: dict[str, TeamStrength] = {
            t: TeamStrength(att[t], dfn[t], sum(w for _, w, _, _, _ in rows)) for t, rows in played.items()
        }
        self.draw_boost = self._calibrate_draws()

    def _calibrate_draws(self) -> float:
        """Поправка на ничьи: независимый Пуассон занижает их долю (особенно в хоккее,
        где проигрывающая команда давит, а в конце снимает вратаря). Сравниваем долю ничьих
        в прогнозах модели с фактической долей в сезоне и масштабируем диагональ матрицы.
        """
        if not self.results:
            return 1.0
        pred = sum(
            sum(row[i] for i, row in enumerate(_matrix(
                self.avg_home * self.teams[r.home].attack * self.teams[r.away].defense,
                self.avg_away * self.teams[r.away].attack * self.teams[r.home].defense,
            )))
            for r in self.results
        ) / len(self.results)
        actual = sum(r.home_goals == r.away_goals for r in self.results) / len(self.results)
        n = len(self.results)
        actual = (n * actual + DRAW_PRIOR_GAMES * pred) / (n + DRAW_PRIOR_GAMES)  # сглаживание
        return min(max(actual / pred, 0.8), 1.6) if pred > 0 else 1.0

    def predict(self, home: str, away: str) -> ModelOutput | None:
        h, a = self.teams.get(home), self.teams.get(away)
        if h is None or a is None:
            return None
        n_home = sum(self.counts.get(home, [0, 0]))
        n_away = sum(self.counts.get(away, [0, 0]))
        if n_home < MIN_GAMES or n_away < MIN_GAMES:
            return None
        lam_h = self.avg_home * h.attack * a.defense
        lam_a = self.avg_away * a.attack * h.defense
        return ModelOutput(lam_h, lam_a, outcome_probs(lam_h, lam_a, draw_boost=self.draw_boost),
                           n_home, n_away, self.draw_boost)

    def form(self, team: str, last: int = 5) -> str:
        """Форма в виде «WDLWW» (последние матчи справа)."""
        games = sorted(
            (r for r in self.results if team in (r.home, r.away)), key=lambda r: r.utc_date
        )[-last:]
        out = []
        for r in games:
            gf, ga = (r.home_goals, r.away_goals) if r.home == team else (r.away_goals, r.home_goals)
            out.append("W" if gf > ga else "D" if gf == ga else "L")
        return "".join(out)

    def last_results(self, team: str, last: int = 5) -> list[str]:
        games = sorted(
            (r for r in self.results if team in (r.home, r.away)), key=lambda r: r.utc_date
        )[-last:]
        return [f"{r.utc_date:%d.%m} {r.home} {r.home_goals}:{r.away_goals} {r.away}" for r in games]


def _pmf(lam: float, k: int) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


def _matrix(lam_h: float, lam_a: float, draw_boost: float = 1.0) -> list[list[float]]:
    ph = [_pmf(lam_h, i) for i in range(MAX_GOALS + 1)]
    pa = [_pmf(lam_a, j) for j in range(MAX_GOALS + 1)]
    m = [[p_i * p_j * (draw_boost if i == j else 1.0) for j, p_j in enumerate(pa)] for i, p_i in enumerate(ph)]
    total = sum(map(sum, m))
    return [[p / total for p in row] for row in m]


def _two_way(win: float, lose: float) -> tuple[float, float]:
    """Вероятности при целой линии без учёта возврата (push): нормируем выигрыш и проигрыш."""
    s = win + lose
    return (win / s, lose / s) if s else (0.5, 0.5)


def outcome_probs(lam_h: float, lam_a: float, total_line: float = 2.5, draw_boost: float = 1.0) -> dict[str, float]:
    """Футбол: 1X2 и тотал."""
    m = _matrix(lam_h, lam_a, draw_boost)
    home = draw = away = over = under = 0.0
    for i, row in enumerate(m):
        for j, p in enumerate(row):
            if i > j:
                home += p
            elif i == j:
                draw += p
            else:
                away += p
            if i + j > total_line:
                over += p
            elif i + j < total_line:
                under += p
    over, under = _two_way(over, under)
    return {"home": home, "draw": draw, "away": away, "over": over, "under": under}


def hockey_probs(lam_h: float, lam_a: float, total_line: float = 5.5, hcp_home: float = -1.5,
                 draw_boost: float = 1.0) -> dict[str, float]:
    """Хоккей. lam_* — ожидаемые голы за основное время (60 минут).

    Если основное время закончилось вничью, исход решают овертайм или буллиты: победитель
    получает +1 гол к итоговому счёту. Шанс хозяев выиграть ОТ/буллиты берём пропорционально
    силе атак. Тотал и фора считаются по итоговому счёту, как у букмекеров.
    """
    m = _matrix(lam_h, lam_a, draw_boost)
    s_home = lam_h / (lam_h + lam_a) if lam_h + lam_a > 0 else 0.5
    reg_home = reg_draw = reg_away = 0.0
    over = under = 0.0
    h_win = h_lose = 0.0  # фора хозяев
    for i, row in enumerate(m):
        for j, p in enumerate(row):
            if i == j:
                reg_draw += p
                # ОТ/буллиты: итог (i+1):i или i:(i+1)
                for diff, q in ((1, s_home), (-1, 1 - s_home)):
                    tot = i + j + 1
                    if tot > total_line:
                        over += p * q
                    elif tot < total_line:
                        under += p * q
                    margin = diff + hcp_home
                    if margin > 0:
                        h_win += p * q
                    elif margin < 0:
                        h_lose += p * q
                continue
            if i > j:
                reg_home += p
            else:
                reg_away += p
            if i + j > total_line:
                over += p
            elif i + j < total_line:
                under += p
            margin = (i - j) + hcp_home
            if margin > 0:
                h_win += p
            elif margin < 0:
                h_lose += p
    over, under = _two_way(over, under)
    hcp_h, hcp_a = _two_way(h_win, h_lose)
    return {
        "ml_home": reg_home + reg_draw * s_home,
        "ml_away": reg_away + reg_draw * (1 - s_home),
        "home": reg_home,
        "draw": reg_draw,
        "away": reg_away,
        "over": over,
        "under": under,
        "hcp_home": hcp_h,
        "hcp_away": hcp_a,
    }


def sport_probs(sport_key: str, lam_h: float, lam_a: float, lines: dict[str, float],
                draw_boost: float = 1.0) -> dict[str, float]:
    if sport_key == "hockey":
        return hockey_probs(lam_h, lam_a, lines.get("total", 5.5), lines.get("hcp_home", -1.5), draw_boost)
    return outcome_probs(lam_h, lam_a, lines.get("total", 2.5), draw_boost)
