"""НХЛ: линии, модель, расчёт ставок с овертаймом/буллитами, цепочка досье → вердикт → итог."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from bot.analysis.market import NHL_REG_DRAW, market_view
from bot.analysis.poisson import PoissonModel, hockey_probs
from bot.config import LEAGUES_BY_ODDS_KEY
from bot.data.nhl import parse_game, season_start
from bot.data.odds import parse_event
from bot.data.stats import Result
from bot.db import DB
from bot.pipeline import Engine, MatchContext, settle_outcome
from bot.tg import formatting as fmt
from bot.tg.handlers import parse_verdict

NHL = LEAGUES_BY_ODDS_KEY["icehockey_nhl"]
NOW = datetime.now(timezone.utc)


def _book(key, ml_h, ml_a, total, over, under, hcp, hcp_h, hcp_a, three_way=None):
    markets = [
        {"key": "h2h", "outcomes": [{"name": "Toronto Maple Leafs", "price": ml_h},
                                    {"name": "Boston Bruins", "price": ml_a}]},
        {"key": "totals", "outcomes": [{"name": "Over", "price": over, "point": total},
                                       {"name": "Under", "price": under, "point": total}]},
        {"key": "spreads", "outcomes": [{"name": "Toronto Maple Leafs", "price": hcp_h, "point": hcp},
                                        {"name": "Boston Bruins", "price": hcp_a, "point": -hcp}]},
    ]
    if three_way:
        h, d, a = three_way
        markets.append({"key": "h2h_3_way", "outcomes": [
            {"name": "Toronto Maple Leafs", "price": h}, {"name": "Draw", "price": d},
            {"name": "Boston Bruins", "price": a}]})
    return {"key": key, "markets": markets}


def _raw(kickoff, three_way=True):
    tw = (2.40, 4.00, 2.85) if three_way else None  # с обычной маржой ~2%
    return {
        "id": "nhl-1",
        "commence_time": kickoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "home_team": "Toronto Maple Leafs",
        "away_team": "Boston Bruins",
        "bookmakers": [
            _book("pinnacle", 1.80, 2.08, 6.5, 2.05, 1.80, -1.5, 2.70, 1.50, tw),
            _book("unibet", 1.78, 2.05, 6.5, 2.00, 1.80, -1.5, 2.65, 1.48, tw),
            _book("betsson", 1.79, 2.06, 5.5, 1.60, 2.30, -1.5, 2.60, 1.49, tw),
        ],
    }


# ---------- линии ----------
def test_parse_hockey_markets_and_main_lines():
    ev = parse_event(_raw(NOW + timedelta(days=1)), NHL)
    b = ev.books[0]
    assert b.price("ml_home") == 1.80 and b.price("home") == 2.40 and b.price("draw") == 4.00
    assert b.price("over", 6.5) == 2.05 and b.price("hcp_home", -1.5) == 2.70 and b.price("hcp_away", 1.5) == 1.50

    mv = market_view(ev)
    assert mv.lines == {"total": 6.5, "hcp_home": -1.5}  # 6.5 — у большинства букмекеров
    assert mv.source == "pinnacle" and not mv.estimated
    for group in (("ml_home", "ml_away"), ("home", "draw", "away"), ("over", "under"), ("hcp_home", "hcp_away")):
        assert abs(sum(mv.fair[s] for s in group) - 1) < 1e-9


def test_regulation_1x2_estimated_when_missing():
    mv = market_view(parse_event(_raw(NOW + timedelta(days=1), three_way=False), NHL))
    assert set(mv.estimated) == {"home", "draw", "away"}
    assert mv.fair["draw"] == pytest.approx(NHL_REG_DRAW)
    assert mv.fair["home"] + mv.fair["draw"] + mv.fair["away"] == pytest.approx(1)
    assert mv.fair["home"] < mv.fair["ml_home"]  # без овертайма шансов меньше


# ---------- модель ----------
def test_hockey_probs_consistent():
    p = hockey_probs(3.2, 2.7, 6.5, -1.5)
    assert p["ml_home"] + p["ml_away"] == pytest.approx(1)
    assert p["home"] + p["draw"] + p["away"] == pytest.approx(1)
    assert p["ml_home"] > p["home"] and p["ml_home"] == pytest.approx(p["home"] + p["draw"] * 3.2 / 5.9)
    assert p["hcp_home"] < p["home"]  # выиграть с разницей 2+ труднее, чем просто выиграть
    boosted = hockey_probs(3.2, 2.7, 6.5, -1.5, draw_boost=1.3)
    assert boosted["draw"] > p["draw"]


def test_draw_calibration_learns_from_season():
    base = NOW - timedelta(days=40)
    teams = [f"T{i}" for i in range(8)]
    res, n = [], 0
    for rnd in range(6):
        for i in range(0, 8, 2):
            h, a = teams[(i + rnd) % 8], teams[(i + 1 + 2 * rnd) % 8]
            if h == a:
                continue
            tie = (n % 3 == 0)  # каждый третий матч — ничья в основное время (33%)
            hg, ag = (2, 2) if tie else (3, 1) if n % 2 else (1, 3)
            res.append(Result(n, "NHL", base + timedelta(days=rnd * 5 + i), h, a, hg, ag))
            n += 1
    m = PoissonModel(res, now=NOW)
    assert m.draw_boost > 1.0


# ---------- расчёт ставок ----------
def _r(rh, ra, fh=None, fa=None, decided="REG"):
    return Result(1, "NHL", NOW, "Toronto Maple Leafs", "Boston Bruins", rh, ra, fh, fa, decided)


def test_settle_rules_overtime_and_shootout():
    so = _r(2, 2, 3, 2, "SO")  # 2:2 в основное время, хозяева выиграли буллиты
    assert settle_outcome("hockey", "ml_home", None, so) == "won"
    assert settle_outcome("hockey", "draw", None, so) == "won"
    assert settle_outcome("hockey", "home", None, so) == "lost"
    assert settle_outcome("hockey", "over", 4.5, so) == "won"  # итог 3:2 = 5 голов
    assert settle_outcome("hockey", "hcp_home", -1.5, so) == "lost"
    assert settle_outcome("hockey", "hcp_away", 1.5, so) == "won"

    reg = _r(4, 1)
    assert settle_outcome("hockey", "hcp_home", -1.5, reg) == "won"
    assert settle_outcome("hockey", "under", 5.0, reg) == "void"  # ровно 5 при линии 5 — возврат
    assert settle_outcome("football", "over", 2.5, Result(1, "PL", NOW, "a", "b", 2, 1)) == "won"


# ---------- НХЛ API ----------
def test_parse_nhl_game_and_season_start():
    g = {"id": 7, "gameType": 2, "gameState": "OFF", "startTimeUTC": "2026-11-10T00:00:00Z",
         "homeTeam": {"placeName": {"default": "Toronto"}, "commonName": {"default": "Maple Leafs"}, "score": 3},
         "awayTeam": {"placeName": {"default": "Boston"}, "commonName": {"default": "Bruins"}, "score": 4},
         "gameOutcome": {"lastPeriodType": "OT"}}
    r = parse_game(g)
    assert (r.home, r.away) == ("Toronto Maple Leafs", "Boston Bruins")
    assert (r.home_goals, r.away_goals, r.final, r.decided) == (3, 3, (3, 4), "OT")
    assert parse_game({**g, "gameState": "FUT"}) is None
    assert parse_game({"broken": True}) is None
    assert season_start(datetime(2027, 2, 1).date()).isoformat() == "2026-10-01"


# ---------- цепочка досье → вердикт → итог ----------
def _ctx(kickoff):
    ev = parse_event(_raw(kickoff), NHL)
    return MatchContext(1, ev, market_view(ev))


def test_hockey_manual_flow_to_settlement():
    db = DB(":memory:")
    engine = Engine(db)
    ctx = _ctx(NOW + timedelta(days=1))
    num = engine.save_candidate(ctx)

    text = engine.dossier(ctx, num)
    assert "🏒" in text and "ml_home" in text and "hcp_home (Ф1(-1.5))" in text and "вратар" in text
    assert len(fmt.dossier_message(text)) < 4000

    n, sel, p, conf = parse_verdict(f"{num} ф1 0.42 medium")
    res = engine.manual_verdict(n, sel, p, conf)
    assert res.signal_id is not None, res.skip_reason
    msg = fmt.analysis(res)
    assert "Ф1(-1.5)" in msg and "СИГНАЛ" in msg
    assert abs(res.table["hcp_home"].p_final + res.table["hcp_away"].p_final - 1) < 1e-6
    sig = db.signal(res.signal_id)
    assert sig.sport == "hockey" and sig.line == -1.5

    # матч сыгран: 4:2 в основное время — фора -1.5 зашла
    db.conn.execute("UPDATE signals SET kickoff = ? WHERE id = ?",
                    ((NOW - timedelta(hours=5)).isoformat(), res.signal_id))

    async def fake_nhl(date_from, date_to):
        return [Result(9, "NHL", NOW - timedelta(hours=5), "Toronto Maple Leafs", "Boston Bruins", 4, 2, 4, 2, "REG")]

    engine.nhl.finished_between = fake_nhl
    settled = asyncio.run(engine.settle())
    assert len(settled) == 1 and settled[0].status == "won" and settled[0].score == "4:2"
    report = fmt.settle_report(settled, db.bankroll())
    assert "🏒" in report and "Ф1(-1.5)" in report


def test_wrong_outcome_for_sport_is_rejected():
    engine = Engine(DB(":memory:"))
    fb = _ctx(NOW + timedelta(days=1))
    num = engine.save_candidate(fb)
    with pytest.raises(ValueError):
        engine.manual_verdict(num, "home_or_what", 0.5, "low")


def test_diag_report_runs(monkeypatch, capsys):
    from bot import diag
    from bot.data import nhl

    base = NOW - timedelta(days=30)
    games = []
    for i in range(12):
        h, a = ("Vegas Golden Knights", "Nashville Predators") if i % 2 else ("Nashville Predators", "Vegas Golden Knights")
        games.append(Result(i, "NHL", base + timedelta(days=i), h, a, 3, 2 if i % 3 else 3, None, None, "REG"))

    async def fake_results(self):
        return games

    monkeypatch.setattr(nhl.NhlClient, "results_with_previous", fake_results)
    assert asyncio.run(diag.main(["NHL", "Vegas", "Nashville"])) == 0
    out = capsys.readouterr().out
    assert "атака" in out and "Прогноз модели" in out and "ml_home" in out



def test_previous_season_is_weighted_and_fetched(monkeypatch):
    from bot.analysis import poisson
    from bot.data import nhl

    start = NOW - timedelta(days=40)
    prev = Result(1, "NHL", start - timedelta(days=200), "A", "B", 5, 0)
    cur = Result(2, "NHL", start + timedelta(days=5), "A", "B", 1, 1)
    m = PoissonModel([cur, prev], now=NOW, season_start=start)
    assert m.results[0] is prev  # отсортированы по дате
    assert m.weight(prev) == pytest.approx(poisson._weight(prev.utc_date, NOW) * poisson.PREV_SEASON_FACTOR)
    assert m.weight(cur) == pytest.approx(poisson._weight(cur.utc_date, NOW))
    assert m.current_season() == [cur] and m.current_season("A") == [cur]

    calls = []

    async def fake_between(self, a, b):
        calls.append((a, b))
        return [Result(len(calls), "NHL", NOW, "A", "B", 1, 0)]

    monkeypatch.setattr(nhl.NhlClient, "games_between", fake_between)
    rows = asyncio.run(nhl.NhlClient().results_with_previous())
    (p_from, p_to), (c_from, _) = calls
    assert p_from.month == 10 and p_to.month == 6 and c_from.year == p_from.year + 1 and len(rows) == 2


def test_football_previous_season(monkeypatch):
    from bot.data import stats as stats_mod

    assert stats_mod.football_season_start(datetime(2026, 11, 13).date()).isoformat() == "2026-07-01"
    assert stats_mod.football_season_start(datetime(2027, 3, 1).date()).isoformat() == "2026-07-01"
    seasons = []

    async def fake(self, competition, season=None):
        seasons.append(season)
        if season is not None:
            raise RuntimeError("тариф не отдаёт прошлый сезон")
        return [Result(1, competition, NOW, "A", "B", 1, 0)]

    monkeypatch.setattr(stats_mod.StatsClient, "season_results", fake)
    rows = asyncio.run(stats_mod.StatsClient().results_with_previous("PL"))
    assert seasons[0] is None and seasons[1] == stats_mod.football_season_start(NOW.date()).year - 1
    assert len(rows) == 1  # без прошлого сезона бот продолжает работать
