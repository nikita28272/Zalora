"""НБА: факторы графика и усталости — бесплатно, по результатам ESPN, без линий букмекеров.

Для каждой команды перед матчем считаем:
  * отдых: дней с прошлой игры, второй матч за два дня (back-to-back), 3 игры за 4 дня, 4 за 6;
  * перелёт: км от города прошлой игры до города этой, сдвиг часовых поясов (на восток тяжелее);
  * выезд / домашняя серия: какая по счёту игра подряд в гостях или дома;
  * серия побед или поражений;
  * высота: гости в Денвере и Юте (1300–1600 м) устают быстрее, особенно после перелёта.

Из этого складываем «усталость» каждой команды; разница усталости — перекос графика.
Матчи с сильным перекосом (или с перекосом, совпадающим с разницей в силе команд) — кандидаты
на разбор: именно в них бывает информация, которую агенты могут проверить и уточнить.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..data.stats import Result
from ..data.teams import best_match

# (широта, долгота, часовой пояс: 0 — восток, 1 — центр, 2 — горы, 3 — запад, высота над морем)
ARENAS: dict[str, tuple[float, float, int, bool]] = {
    "Atlanta Hawks": (33.757, -84.396, 0, False),
    "Boston Celtics": (42.366, -71.062, 0, False),
    "Brooklyn Nets": (40.683, -73.975, 0, False),
    "Charlotte Hornets": (35.225, -80.839, 0, False),
    "Chicago Bulls": (41.881, -87.674, 1, False),
    "Cleveland Cavaliers": (41.496, -81.688, 0, False),
    "Dallas Mavericks": (32.790, -96.810, 1, False),
    "Denver Nuggets": (39.749, -105.008, 2, True),
    "Detroit Pistons": (42.341, -83.055, 0, False),
    "Golden State Warriors": (37.768, -122.388, 3, False),
    "Houston Rockets": (29.751, -95.362, 1, False),
    "Indiana Pacers": (39.764, -86.155, 0, False),
    "LA Clippers": (33.945, -118.341, 3, False),
    "Los Angeles Lakers": (34.043, -118.267, 3, False),
    "Memphis Grizzlies": (35.138, -90.051, 1, False),
    "Miami Heat": (25.781, -80.188, 0, False),
    "Milwaukee Bucks": (43.045, -87.917, 1, False),
    "Minnesota Timberwolves": (44.979, -93.276, 1, False),
    "New Orleans Pelicans": (29.949, -90.082, 1, False),
    "New York Knicks": (40.751, -73.993, 0, False),
    "Oklahoma City Thunder": (35.463, -97.515, 1, False),
    "Orlando Magic": (28.539, -81.384, 0, False),
    "Philadelphia 76ers": (39.901, -75.172, 0, False),
    "Phoenix Suns": (33.446, -112.071, 2, False),
    "Portland Trail Blazers": (45.532, -122.667, 3, False),
    "Sacramento Kings": (38.580, -121.500, 3, False),
    "San Antonio Spurs": (29.427, -98.438, 1, False),
    "Toronto Raptors": (43.643, -79.379, 0, False),
    "Utah Jazz": (40.768, -111.901, 2, True),
    "Washington Wizards": (38.898, -77.021, 0, False),
}
ZONE_RU = {0: "восточное", 1: "центральное", 2: "горное", 3: "тихоокеанское"}


def arena(team: str) -> tuple[float, float, int, bool] | None:
    name = team if team in ARENAS else best_match(team, ARENAS.keys())
    return ARENAS.get(name) if name else None


def km(a: tuple[float, float, int, bool], b: tuple[float, float, int, bool]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


@dataclass
class TeamSpot:
    team: str
    rest_days: int | None = None  # 0 — back-to-back
    games_4: int = 1  # игр за 4 дня, включая эту
    games_6: int = 1
    travel_km: float = 0.0
    tz_shift: int = 0  # >0 — летели на восток (часы «теряются»)
    road_game: int = 0  # какая по счёту игра выезда (для гостей)
    home_game: int = 0  # какая по счёту домашняя игра подряд (для хозяев)
    streak: str = ""  # «W3» / «L5»
    altitude: bool = False  # гости на высоте (Денвер, Юта)
    fatigue: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass
class Factors:
    home: TeamSpot
    away: TeamSpot
    edge: float  # усталость гостей − усталость хозяев: > 0 — график за хозяев
    score: float  # сила перекоса для отбора
    side: str  # «хозяева» / «гости» / «—»
    reasons: list[str]

    @property
    def side_gen(self) -> str:
        return {"хозяева": "хозяев", "гости": "гостей"}.get(self.side, self.side)

    def text(self) -> str:
        lines = ["ФАКТОРЫ ГРАФИКА (бот, по результатам сезона; агентам — проверить и дополнить):"]
        for label, t in (("хозяева", self.home), ("гости", self.away)):
            lines.append(f"  {t.team} ({label}): " + ("; ".join(t.notes) or "без особенностей")
                         + f" — усталость {t.fatigue:.1f}")
        if self.side != "—":
            lines.append(f"  Перекос графика в пользу {self.side_gen} ({self.edge:+.1f}). "
                         + "; ".join(self.reasons))
        else:
            lines.append("  Перекоса графика нет.")
        return "\n".join(lines)


def _team_games(results: list[Result], team: str, before: datetime) -> list[Result]:
    return [r for r in results if team in (r.home, r.away) and r.utc_date < before - timedelta(hours=3)]


def team_spot(results: list[Result], team: str, venue_team: str, kickoff: datetime, is_home: bool) -> TeamSpot:
    spot = TeamSpot(team)
    games = _team_games(results, team, kickoff)
    venue = arena(venue_team)
    if games:
        last = games[-1]
        hours = (kickoff - last.utc_date).total_seconds() / 3600
        spot.rest_days = max(int((hours + 6) // 24) - 1, 0)  # +6 ч: вечер → вечер следующего дня = 0 дней отдыха
        spot.games_4 = 1 + sum(1 for g in games if kickoff - g.utc_date <= timedelta(days=3, hours=6))
        spot.games_6 = 1 + sum(1 for g in games if kickoff - g.utc_date <= timedelta(days=5, hours=6))
        prev_venue = arena(last.home)
        if venue and prev_venue:
            spot.travel_km = km(prev_venue, venue)
            spot.tz_shift = prev_venue[2] - venue[2]  # из зоны 3 (запад) в 0 (восток) = +3
        # серия выездов / домашних игр подряд
        run = 0
        for g in reversed(games):
            if (g.home == team) == is_home:
                run += 1
            else:
                break
        if is_home:
            spot.home_game = run + 1
        else:
            spot.road_game = run + 1
        # серия побед / поражений
        n, kind = 0, None
        for g in reversed(games):
            fh, fa = g.final
            won = (fh > fa) == (g.home == team)
            if kind is None:
                kind = won
            if won != kind:
                break
            n += 1
        spot.streak = f"{'W' if kind else 'L'}{n}" if kind is not None else ""
    if not is_home and venue and venue[3]:
        spot.altitude = True

    f, notes = 0.0, []
    if spot.rest_days is not None:
        if spot.rest_days == 0:
            f += 3.0
            notes.append("второй матч за два дня (back-to-back)")
        elif spot.rest_days >= 2:
            f -= 1.0
            notes.append(f"отдых {spot.rest_days} дн.")
        else:
            notes.append("отдых 1 день")
    if spot.games_4 >= 3:
        f += 1.5
        notes.append(f"{spot.games_4}-я игра за 4 дня")
    elif spot.games_6 >= 4:
        f += 1.0
        notes.append(f"{spot.games_6}-я игра за 6 дней")
    if spot.travel_km >= 300:
        f += min(spot.travel_km / 1500, 2.0)
        notes.append(f"перелёт ~{spot.travel_km:,.0f} км".replace(",", " "))
    if spot.tz_shift >= 2:
        f += 1.0
        notes.append(f"сдвиг на {spot.tz_shift} ч на восток")
    elif spot.tz_shift <= -2:
        f += 0.5
        notes.append(f"сдвиг на {-spot.tz_shift} ч на запад")
    if spot.road_game >= 4:
        f += 1.0
        notes.append(f"{spot.road_game}-я игра выезда подряд")
    elif spot.road_game >= 2:
        notes.append(f"{spot.road_game}-я игра выезда")
    if spot.home_game >= 3:
        notes.append(f"{spot.home_game}-я домашняя игра подряд")
    if spot.altitude:
        f += 0.5 + (0.5 if spot.rest_days == 0 else 0.0)
        notes.append("игра на высоте (Денвер/Юта)")
    if spot.streak and int(spot.streak[1:]) >= 3:
        notes.append(f"серия {'побед' if spot.streak[0] == 'W' else 'поражений'} {spot.streak[1:]}")
    spot.fatigue, spot.notes = round(f, 2), notes
    return spot


def nba_factors(results: list[Result], home: str, away: str, kickoff: datetime,
                model_margin: float | None = None) -> Factors:
    """Перекос графика между командами; model_margin — ожидаемая разница очков по модели (хозяева − гости)."""
    h = team_spot(results, home, home, kickoff, is_home=True)
    a = team_spot(results, away, home, kickoff, is_home=False)
    edge = round(a.fatigue - h.fatigue, 2)
    reasons = []
    if (h.rest_days == 0) != (a.rest_days == 0):
        tired = "хозяева" if h.rest_days == 0 else "гости"
        reasons.append(f"{tired} на back-to-back, соперник отдохнул")
    if abs(h.travel_km - a.travel_km) >= 1500:
        reasons.append(f"{'гости' if a.travel_km > h.travel_km else 'хозяева'} летели заметно дальше")
    if a.road_game >= 4:
        reasons.append("длинный выезд у гостей")
    score = abs(edge)
    if model_margin is not None and edge * model_margin > 0 and abs(model_margin) >= 4:
        score += 1.0  # сильнее и свежее — перекос в одну сторону
        reasons.append("сильнее по модели та же сторона")
    side = "хозяева" if edge >= 1 else "гости" if edge <= -1 else "—"
    return Factors(h, a, edge, round(score, 2), side, reasons)
