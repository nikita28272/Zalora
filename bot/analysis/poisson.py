"""Статистический аналитик: модель Пуассона по результатам текущего сезона.

Для каждой команды считаем силу атаки и обороны отдельно дома и в гостях,
с затуханием по времени (свежие матчи важнее) и сглаживанием к среднему по
лиге (чтобы 3 матча не давали экстремальных оценок). По ожидаемым голам
строим матрицу счетов и получаем вероятности 1X2 и тотала 2.5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from ..data.stats import Result

HALF_LIFE_DAYS = 120.0
PRIOR_GAMES = 4.0  # «виртуальные» средние матчи для сглаживания
MIN_GAMES = 4  # меньше — модель не строим
MAX_GOALS = 10


@dataclass
class TeamStrength:
    attack_home: float
    defense_home: float
    attack_away: float
    defense_away: float
    games_home: float
    games_away: float


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
        w_sum = hg = ag = 0.0
        acc: dict[str, dict[str, float]] = {}

        def slot(team: str) -> dict[str, float]:
            return acc.setdefault(team, {k: 0.0 for k in ("hs", "hc", "hw", "as", "ac", "aw")})

        for r in self.results:
            w = _weight(r.utc_date, self.now)
            w_sum += w
            hg += w * r.home_goals
            ag += w * r.away_goals
            h, a = slot(r.home), slot(r.away)
            h["hs"] += w * r.home_goals
            h["hc"] += w * r.away_goals
            h["hw"] += w
            a["as"] += w * r.away_goals
            a["ac"] += w * r.home_goals
            a["aw"] += w
            self.counts.setdefault(r.home, [0, 0])[0] += 1
            self.counts.setdefault(r.away, [0, 0])[1] += 1

        self.avg_home = hg / w_sum if w_sum else 1.5
        self.avg_away = ag / w_sum if w_sum else 1.2

        self.teams: dict[str, TeamStrength] = {}
        k = PRIOR_GAMES
        for team, s in acc.items():
            # сглаживание: добавляем k средних матчей лиги
            self.teams[team] = TeamStrength(
                attack_home=((s["hs"] + k * self.avg_home) / (s["hw"] + k)) / self.avg_home,
                defense_home=((s["hc"] + k * self.avg_away) / (s["hw"] + k)) / self.avg_away,
                attack_away=((s["as"] + k * self.avg_away) / (s["aw"] + k)) / self.avg_away,
                defense_away=((s["ac"] + k * self.avg_home) / (s["aw"] + k)) / self.avg_home,
                games_home=s["hw"],
                games_away=s["aw"],
            )

    def predict(self, home: str, away: str) -> ModelOutput | None:
        h, a = self.teams.get(home), self.teams.get(away)
        if h is None or a is None:
            return None
        n_home = sum(self.counts.get(home, [0, 0]))
        n_away = sum(self.counts.get(away, [0, 0]))
        if n_home < MIN_GAMES or n_away < MIN_GAMES:
            return None
        lam_h = h.attack_home * a.defense_away * self.avg_home
        lam_a = a.attack_away * h.defense_home * self.avg_away
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
