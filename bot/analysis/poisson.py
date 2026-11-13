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
MAX_GOALS = 10
ITERATIONS = 40


@dataclass
class TeamStrength:
    attack: float  # 1.0 — средняя атака лиги, 1.3 — забивает на 30% больше среднего
    defense: float  # 1.0 — средняя оборона, 0.7 — пропускает на 30% меньше среднего
    games: float  # взвешенное число матчей


@dataclass
class ModelOutput:
    lambda_home: float
    lambda_away: float
    probs: dict[str, float]  # home/draw/away/over/under
    home_games: int
    away_games: int


def _weight(when: datetime, now: datetime) -> float:
    days = max((now - when).total_seconds() / 86400.0, 0.0)
    return 0.5 ** (days / HALF_LIFE_DAYS)


class PoissonModel:
    def __init__(self, results: list[Result], now: datetime | None = None):
        self.now = now or datetime.now(timezone.utc)
        self.results = results
        self.counts: dict[str, list[int]] = {}
        self._fit()

    def _fit(self) -> None:
        games = [(r, _weight(r.utc_date, self.now)) for r in self.results]
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
        return ModelOutput(lam_h, lam_a, outcome_probs(lam_h, lam_a), n_home, n_away)

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


def outcome_probs(lam_h: float, lam_a: float) -> dict[str, float]:
    ph = [_pmf(lam_h, i) for i in range(MAX_GOALS + 1)]
    pa = [_pmf(lam_a, j) for j in range(MAX_GOALS + 1)]
    home = draw = away = under = 0.0
    for i, p_i in enumerate(ph):
        for j, p_j in enumerate(pa):
            p = p_i * p_j
            if i > j:
                home += p
            elif i == j:
                draw += p
            else:
                away += p
            if i + j <= 2:
                under += p
    total = home + draw + away
    under /= total
    return {
        "home": home / total,
        "draw": draw / total,
        "away": away / total,
        "over": 1.0 - under,
        "under": under,
    }
