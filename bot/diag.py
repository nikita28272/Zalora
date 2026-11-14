"""Диагностика модели: на чём она построила оценку команд.

Запуск на сервере:  bash scripts/diag.sh NHL "Vegas" "Nashville"
                    bash scripts/diag.sh NBA "Celtics" "Knicks"
                    bash scripts/diag.sh PL "Arsenal" "Chelsea"
Ключи не выводятся. Кредиты The Odds API не тратятся (линии не запрашиваются).
"""

from __future__ import annotations

import asyncio
import sys

from .analysis.poisson import PREV_SEASON_FACTOR, PoissonModel, sport_probs
from .config import ALL_LEAGUES
from datetime import datetime, timezone

from .analysis.basketball import NbaModel
from .data.nba import NbaClient
from .data.nba import season_start as nba_season_start
from .data.nhl import NhlClient
from .data.nhl import season_start as nhl_season_start
from .data.stats import StatsClient, football_season_start
from .data.teams import best_match


def _team_report(model: PoissonModel, team: str) -> list[str]:
    s = model.teams[team]
    all_games = [r for r in model.results if team in (r.home, r.away)]
    games = model.current_season(team)  # цифры голов — по текущему сезону
    gf = ga = fgf = fga = 0
    for r in games:
        home = r.home == team
        gf += r.home_goals if home else r.away_goals
        ga += r.away_goals if home else r.home_goals
        fh, fa = r.final
        fgf += fh if home else fa
        fga += fa if home else fh
    n_home, n_away = model.counts.get(team, [0, 0])
    out = [
        f"{team}: в текущем сезоне {len(games)} матчей, всего с прошлым сезоном {len(all_games)}"
        f" (дома {n_home}, в гостях {n_away})",
        f"  атака {s.attack:.2f}, оборона {s.defense:.2f} (1.00 — средняя команда; оборона < 1 — пропускает меньше)",
        f"  текущий сезон: голы основного времени {gf}:{ga} (разница {gf - ga:+d}),"
        f" итоговые с ОТ {fgf}:{fga} (разница {fgf - fga:+d})",
        "  последние 10 (осн. время / итог / развязка):",
    ]
    for r in games[-10:]:
        side = "дома " if r.home == team else "в гостях"
        fh, fa = r.final
        out.append(f"    {r.utc_date:%d.%m} {side} {r.home} {r.home_goals}:{r.away_goals} {r.away}"
                   f" | итог {fh}:{fa} {r.decided}")
    return out


async def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print('Использование: python -m bot.diag КОД_ЛИГИ "Хозяева" "Гости"  (коды: '
              + ", ".join(lg.fd_code for lg in ALL_LEAGUES) + ")")
        return 1
    code, home_q, away_q = argv[0].upper(), argv[1], argv[2]
    league = next((lg for lg in ALL_LEAGUES if lg.fd_code == code), None)
    if league is None:
        print(f"Нет лиги {code}")
        return 1
    today = datetime.now(timezone.utc).date()
    if league.sport == "basketball":
        return await _nba_diag(home_q, away_q)
    if league.sport == "hockey":
        results = await NhlClient().results_with_previous()
        start = nhl_season_start(today)
    else:
        results = await StatsClient().results_with_previous(code)
        start = football_season_start(today)
    if not results:
        print("Результатов сезона не получено — источник недоступен или формат изменился.")
        return 1
    model = PoissonModel(results, season_start=datetime(start.year, start.month, start.day, tzinfo=timezone.utc))
    home = best_match(home_q, model.teams.keys()) or next((t for t in model.teams if home_q.lower() in t.lower()), None)
    away = best_match(away_q, model.teams.keys()) or next((t for t in model.teams if away_q.lower() in t.lower()), None)
    print(f"Лига {league.name_ru}: матчей в выборке {len(results)}"
          f" (текущий сезон {len(model.current_season())}, прошлый {len(results) - len(model.current_season())},"
          f" вес прошлого сезона ×{PREV_SEASON_FACTOR})")
    print(f"Средние голы основного времени: хозяева {model.avg_home:.2f}, гости {model.avg_away:.2f}")
    print(f"Ничьих в основное время: {sum(r.home_goals == r.away_goals for r in results) / len(results):.1%},"
          f" поправка на ничьи ×{model.draw_boost:.2f}")
    for name, q in ((home, home_q), (away, away_q)):
        if name is None:
            print(f"\nКоманда «{q}» не найдена. Есть: " + ", ".join(sorted(model.teams)))
            return 1
        print()
        print("\n".join(_team_report(model, name)))
    pred = model.predict(home, away)
    if pred:
        lines = {"total": 5.5, "hcp_home": -1.5} if league.sport == "hockey" else {"total": 2.5}
        probs = sport_probs(league.sport, pred.lambda_home, pred.lambda_away, lines, pred.draw_boost)
        print(f"\nПрогноз модели {home} — {away}: ожидаемые голы {pred.lambda_home:.2f} : {pred.lambda_away:.2f}")
        print("  " + ", ".join(f"{k} {v:.1%}" for k, v in probs.items()))
    return 0


async def _nba_diag(home_q: str, away_q: str) -> int:
    results = await NbaClient().results_with_previous()
    if not results:
        print("Результатов НБА не получено — ESPN недоступен или формат изменился.")
        return 1
    start = nba_season_start(datetime.now(timezone.utc).date())
    model = NbaModel(results, season_start=datetime(start.year, start.month, start.day, tzinfo=timezone.utc))
    print(f"НБА: матчей в выборке {len(results)} (текущий сезон {len(model.current_season())})")
    print(f"Среднее очков команды {model.avg:.1f}, преимущество площадки {model.hca:+.1f},"
          f" разброс разницы {model.margin_sd:.1f}, суммы {model.total_sd:.1f}")
    names = []
    for q in (home_q, away_q):
        name = best_match(q, model.teams.keys()) or next((t for t in model.teams if q.lower() in t.lower()), None)
        if name is None:
            print(f"\nКоманда «{q}» не найдена. Есть: " + ", ".join(sorted(model.teams)))
            return 1
        r = model.teams[name]
        print(f"\n{name}: нападение {r.offense:+.1f}, защита {r.defense:+.1f} (минус — пропускает меньше),"
              f" рейтинг {r.offense - r.defense:+.1f}; форма {model.form(name)}")
        for line in model.last_results(name):
            print("   " + line)
        names.append(name)
    pred = model.predict(names[0], names[1], datetime.now(timezone.utc))
    if pred:
        print(f"\nПрогноз {names[0]} — {names[1]}: {pred.lambda_home:.1f} : {pred.lambda_away:.1f}"
              + (f" ({pred.note})" if pred.note else ""))
        print("  " + ", ".join(f"{k} {v:.1%}" for k, v in pred.probs.items()))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
