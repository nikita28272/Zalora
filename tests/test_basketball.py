"""НБА: данные ESPN, модель рейтингов, линии, цепочка досье → вердикт → итог."""

import asyncio
import random
from datetime import datetime, timedelta, timezone

from bot.analysis.basketball import NbaModel, basketball_probs
from bot.analysis.market import market_view
from bot.config import LEAGUES_BY_ODDS_KEY
from bot.data.nba import parse_event as parse_espn
from bot.data.nba import standings_from_results
from bot.data.odds import parse_event
from bot.data.stats import Result
from bot.db import DB
from bot.pipeline import Engine, settle_outcome
from bot.sports import SPORTS, sel_short
from bot.tg import formatting as fmt

NBA = LEAGUES_BY_ODDS_KEY["basketball_nba"]
NOW = datetime.now(timezone.utc)
HOME, AWAY = "Boston Celtics", "Los Angeles Clippers"


def _espn(eid, when, home, away, hs, as_, period=4, completed=True, season_type=2, quarters=None):
    def comp(side, name, score, qs):
        c = {"homeAway": side, "score": str(score), "team": {"displayName": name}}
        if qs:
            c["linescores"] = [{"value": q} for q in qs]
        return c

    qh, qa = quarters or (None, None)
    return {
        "id": str(eid), "date": when.strftime("%Y-%m-%dT%H:%MZ"), "season": {"type": season_type},
        "competitions": [{
            "competitors": [comp("home", home, hs, qh), comp("away", away, as_, qa)],
            "status": {"period": period, "type": {"completed": completed}},
        }],
    }


# ---------- данные ESPN ----------
def test_parse_espn_regular_ot_and_skips():
    r = parse_espn(_espn(1, NOW, HOME, "LA Clippers", 112, 104))
    assert r.competition == "NBA" and r.final == (112, 104) and r.decided == "REG" and r.away == "LA Clippers"

    ot = parse_espn(_espn(2, NOW, HOME, "Miami Heat", 120, 118, period=5,
                          quarters=([30, 25, 27, 26, 12], [28, 27, 25, 28, 10])))
    assert ot.decided == "OT" and ot.final == (120, 118) and (ot.home_goals, ot.away_goals) == (108, 108)

    assert parse_espn(_espn(3, NOW, HOME, "Miami Heat", 0, 0, completed=False)) is None
    assert parse_espn(_espn(4, NOW, HOME, "Miami Heat", 99, 90, season_type=1)) is None  # предсезонка
    assert parse_espn({"id": "x"}) is None


def test_standings_from_results():
    rs = [Result(i, "NBA", NOW - timedelta(days=10 - i), HOME, "Miami Heat", 110, 100 + (i % 2) * 15,
                 110, 100 + (i % 2) * 15) for i in range(6)]
    rows = standings_from_results(rs)
    top = rows[0]
    assert top.team == HOME and top.played == 6 and top.points == 3 and "последние 10: 3-3" in top.form


# ---------- модель ----------
def _season(n_days=60, seed=1):
    random.seed(seed)
    strength = {HOME: 7, "Los Angeles Clippers": 0, "Miami Heat": 1, "Utah Jazz": -7, "Denver Nuggets": 4}
    names, out, i = list(strength), [], 0
    for d in range(n_days):
        for _ in range(2):
            h, a = random.sample(names, 2)
            m = round(random.gauss(strength[h] - strength[a] + 2.5, 12))
            t = round(random.gauss(226, 18))
            hp = (t + m) // 2
            ap = t - hp
            if hp == ap:
                hp += 1
            i += 1
            when = NOW - timedelta(days=n_days - d, hours=3)
            out.append(Result(i, "NBA", when, h, a, hp, ap, hp, ap))
    return out


def test_model_ratings_b2b_and_probabilities():
    model = NbaModel(_season(), now=NOW)
    assert model.teams[HOME].offense - model.teams[HOME].defense > model.teams["Utah Jazz"].offense - \
        model.teams["Utah Jazz"].defense
    rest = model.predict(HOME, "Utah Jazz", NOW + timedelta(days=3))
    assert rest.probs["ml_home"] > 0.75 and rest.note == ""
    # матч через 20 часов после последней игры хозяев — back-to-back, хозяева теряют ~2 очка разницы
    last = max(r.utc_date for r in model.results if HOME in (r.home, r.away))
    tired = model.predict(HOME, "Utah Jazz", last + timedelta(hours=20))
    fresh = model.predict(HOME, "Utah Jazz", last + timedelta(hours=60))
    assert "хозяева играли накануне" in tired.note
    gap = (fresh.lambda_home - fresh.lambda_away) - (tired.lambda_home - tired.lambda_away)
    assert abs(gap - 2.0) < 1e-6 or "гости" in tired.note

    p = basketball_probs(115, 110, {"hcp_home": -5.5, "total": 225.5}, (12.5, 18.5))
    assert abs(p["ml_home"] + p["ml_away"] - 1) < 1e-9 and 0.6 < p["ml_home"] < 0.7
    assert abs(p["hcp_home"] + p["hcp_away"] - 1) < 1e-9 and p["hcp_home"] < 0.5  # 5 очков против форы 5.5
    assert abs(p["over"] + p["under"] - 1) < 1e-9 and p["over"] < 0.5
    even = basketball_probs(115, 110, {"hcp_home": -5, "total": 225}, (12.5, 18.5))
    assert abs(even["hcp_home"] - 0.5) < 1e-9 and abs(even["over"] - 0.5) < 1e-9  # целая линия: возврат убран


def test_labels_and_settlement_rules():
    bb = SPORTS["basketball"]
    assert sel_short(bb, "ml_home") == "П1" and sel_short(bb, "hcp_away", 5.5) == "Ф2(+5.5)"
    assert sel_short(bb, "over", 225.5) == "ТБ225.5"
    r = Result(1, "NBA", NOW, HOME, AWAY, 108, 108, 120, 112, decided="OT")
    assert settle_outcome("basketball", "ml_home", None, r) == "won"
    assert settle_outcome("basketball", "hcp_away", 7.5, r) == "lost"  # проиграли 8, фора +7.5
    assert settle_outcome("basketball", "hcp_away", 8, r) == "void"
    assert settle_outcome("basketball", "over", 231.5, r) == "won"  # 232 с овертаймом


# ---------- линии и полный путь ----------
def _book(key, ml_h, ml_a, hcp, hcp_h, hcp_a, total, over, under):
    return {"key": key, "markets": [
        {"key": "h2h", "outcomes": [{"name": HOME, "price": ml_h}, {"name": AWAY, "price": ml_a}]},
        {"key": "spreads", "outcomes": [{"name": HOME, "price": hcp_h, "point": hcp},
                                        {"name": AWAY, "price": hcp_a, "point": -hcp}]},
        {"key": "totals", "outcomes": [{"name": "Over", "price": over, "point": total},
                                       {"name": "Under", "price": under, "point": total}]},
    ]}


def _raw(kickoff):
    return {
        "id": "nba-1", "commence_time": kickoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "home_team": HOME, "away_team": AWAY,
        "bookmakers": [
            _book("pinnacle", 1.55, 2.55, -5.5, 1.91, 1.91, 225.5, 1.90, 1.92),
            _book("unibet", 1.53, 2.50, -5.5, 1.88, 1.90, 225.5, 1.87, 1.90),
            _book("betsson", 1.54, 2.52, -6.5, 2.02, 1.80, 226.5, 1.92, 1.88),
        ],
    }


def test_market_view_basketball():
    mv = market_view(parse_event(_raw(NOW + timedelta(days=1)), NBA))
    assert mv.lines == {"total": 225.5, "hcp_home": -5.5}
    assert set(mv.fair) == {"ml_home", "ml_away", "hcp_home", "hcp_away", "over", "under"}
    assert mv.source == "pinnacle" and 0.6 < mv.fair["ml_home"] < 0.65 and not mv.estimated


def test_nba_dossier_verdict_and_settlement():
    db = DB(":memory:")
    engine = Engine(db)
    kickoff = NOW + timedelta(days=1)
    season = _season()
    # «LA Clippers» в данных ESPN — «Los Angeles Clippers» в линиях
    season = [Result(r.match_id, r.competition, r.utc_date, r.home.replace("Los Angeles", "LA"),
                     r.away.replace("Los Angeles", "LA"), *r.final, *r.final) for r in season]

    async def upcoming(max_age_minutes=None, force=False):
        return [parse_event(_raw(kickoff), NBA)]

    async def results():
        return season

    engine._upcoming = upcoming
    engine.nba.results_with_previous = results
    matches = asyncio.run(engine.load_matches())
    ctx = matches[0]
    assert ctx.fd_away == "LA Clippers" and ctx.model is not None
    assert {"hcp_home", "over"} <= set(ctx.model.probs) and ctx.home_row is not None

    num = engine.save_candidate(ctx)
    text = engine.dossier(ctx, num)
    assert "🏀 Баскетбол" in text and "ожидаемые очки" in text and "РЕЖИМ НБА: ПО АНАЛИЗУ" in text
    assert "ФАКТОРЫ ГРАФИКА" in text and "мораль" in text
    assert "hcp_home (Ф1(-5.5))" in text and "over (ТБ225.5)" in text
    assert len(fmt.dossier_message(text)) < 4000

    # по анализу: средняя уверенность — фиксированный 1% банка, даже без перевеса над линией
    top = engine.manual_top(num, [("hcp_home", 0.52), ("over", 0.5)], "medium")
    assert all(p.positive for p in top.picks) and all(p.stake > 0 for p in top.picks)
    by_p = sorted(top.picks, key=lambda p: p.p)
    assert by_p[0].stake <= by_p[-1].stake  # вероятнее — ставка не меньше
    msg = fmt.top_picks(top, db.bankroll())
    assert "📊 сильный разбор" in msg and "по анализу" in msg
    low = engine.manual_top(num, [("hcp_home", 0.52)], "low")
    assert not low.picks[0].positive and "слабый разбор" in fmt.top_picks(low, db.bankroll())
    top = engine.manual_top(num, [("hcp_home", 0.52), ("over", 0.5)], "medium")
    assert "отчёт о травмах" in engine.late_check(num)

    db.conn.execute("UPDATE signals SET kickoff = ?", ((NOW - timedelta(hours=4)).isoformat(),))

    async def finished(date_from, date_to):
        return [Result(99, "NBA", NOW - timedelta(hours=4), HOME, "LA Clippers", 108, 100, 115, 104)]

    engine.nba.finished_between = finished
    settled = asyncio.run(engine.settle())
    by_sel = {s.selection: s for s in settled}
    assert by_sel["hcp_home"].status == "won" and by_sel["over"].status == "lost"  # 115:104, сумма 219
    report = fmt.settle_report(settled, db.bankroll())
    assert "🏀" in report and "115:104" in report


# ---------- запасные пути загрузки ----------
def test_espn_range_rejected_falls_back_to_days_and_cdn():
    import httpx
    from datetime import date

    from bot.data.nba import NbaClient, parse_cdn_game

    client = NbaClient()
    calls = []
    day_events = {"20261110": [_espn(10, datetime(2026, 11, 10, 0, 30, tzinfo=timezone.utc), HOME, "Miami Heat",
                                     110, 100)]}

    async def fetch(url, params, key, ttl):
        calls.append(key)
        req = httpx.Request("GET", url)
        if key.startswith("r"):  # период одним запросом — ошибка
            raise httpx.HTTPStatusError("bad", request=req, response=httpx.Response(400, request=req))
        return {"events": day_events.get(key[1:], [])}

    client._fetch = fetch
    games = asyncio.run(client.games_between(date(2026, 11, 9), date(2026, 11, 11)))
    assert [g.match_id for g in games] == [10] and calls[0].startswith("r") and len(calls) == 4

    g = parse_cdn_game({"gameId": "0022600123", "gameStatus": 3, "gameStatusText": "Final/OT",
                        "gameDateTimeUTC": "2026-11-12T00:30:00Z",
                        "homeTeam": {"teamCity": "LA", "teamName": "Clippers", "score": 118},
                        "awayTeam": {"teamCity": "Utah", "teamName": "Jazz", "score": 115}})
    assert g.home == "LA Clippers" and g.decided == "OT" and g.final == (118, 115)
    assert parse_cdn_game({"gameId": "0012600001", "gameStatus": 3}) is None  # предсезонка


# ---------- факторы графика и отбор матчей НБА ----------
def test_schedule_factors_and_selection():
    from bot.analysis.nba_factors import km, nba_factors, ARENAS

    kickoff = datetime(2026, 11, 14, 0, 30, tzinfo=timezone.utc)
    rs = [
        # гости («Клипперс») вчера играли в Майами, до этого 3 выезда подряд
        Result(1, "NBA", kickoff - timedelta(days=5), "New York Knicks", "LA Clippers", 100, 110, 100, 110),
        Result(2, "NBA", kickoff - timedelta(days=3), "Atlanta Hawks", "LA Clippers", 100, 105, 100, 105),
        Result(3, "NBA", kickoff - timedelta(days=2), "Orlando Magic", "LA Clippers", 120, 100, 120, 100),
        Result(4, "NBA", kickoff - timedelta(days=1), "Miami Heat", "LA Clippers", 115, 99, 115, 99),
        # хозяева («Денвер») три дня отдыхали дома
        Result(5, "NBA", kickoff - timedelta(days=4), "Denver Nuggets", "Utah Jazz", 120, 101, 120, 101),
    ]
    f = nba_factors(rs, "Denver Nuggets", "LA Clippers", kickoff, model_margin=6.0)
    away, home = f.away, f.home
    assert away.rest_days == 0 and away.road_game == 5 and away.altitude and away.games_4 >= 3
    assert away.travel_km > 2500 and away.streak == "L2"
    assert home.rest_days == 3 and home.home_game == 2
    assert f.side == "хозяева" and f.score >= 5 and "back-to-back" in f.reasons[0]
    assert "ФАКТОРЫ ГРАФИКА" in f.text() and "игра на высоте" in f.text()
    assert 2500 < km(ARENAS["Miami Heat"], ARENAS["Denver Nuggets"]) < 3000

    # отбор: берём только матчи с сильным перекосом
    db = DB(":memory:")
    engine = Engine(db)
    ctx_a = _nba_ctx("a", NOW + timedelta(days=1), f)
    weak = nba_factors(rs, "Utah Jazz", "Denver Nuggets", kickoff)  # оба отдохнули
    ctx_b = _nba_ctx("b", NOW + timedelta(days=1), weak)
    assert weak.score < 2
    assert [m.event.event_id for m in engine.nba_top([ctx_a, ctx_b])] == ["a"]
    text = fmt.top_block([], engine.nba_top([ctx_a, ctx_b]))
    assert "НБА по анализу" in text and "график в пользу хозяев" in text


def _nba_ctx(eid, kickoff, factors):
    from bot.pipeline import MatchContext

    raw = _raw(kickoff)
    raw["id"] = eid
    ev = parse_event(raw, NBA)
    ctx = MatchContext(1, ev, market_view(ev))
    ctx.factors = factors
    return ctx



def test_daily_picks_two_hockey_two_nba():
    import sys
    sys.path.insert(0, "tests")
    from test_hockey import _ctx as hockey_ctx
    from bot.analysis.nba_factors import Factors, TeamSpot

    engine = Engine(DB(":memory:"))
    now = NOW
    hockey = []
    for i in range(3):
        c = hockey_ctx(now + timedelta(hours=10 + i))
        c.idx, c.event.event_id = i + 1, f"h{i}"
        hockey.append(c)

    def nba(i, score, hours):
        f = Factors(TeamSpot("A"), TeamSpot("B"), score, score, "хозяева" if score >= 1 else "—", ["гости на back-to-back"])
        c = _nba_ctx(f"n{i}", now + timedelta(hours=hours), f)
        c.idx = 10 + i
        return c

    nbas = [nba(0, 0.5, 12), nba(1, 4.0, 14), nba(2, 6.0, 40), nba(3, 2.0, 15)]
    picks = engine.daily_picks(hockey + nbas, now=now)
    sports = [m.sport.key for m, _ in picks]
    assert sports.count("hockey") == 2 and sports.count("basketball") == 2 and len(picks) == 4
    nba_ids = [m.event.event_id for m, _ in picks if m.sport.key == "basketball"]
    assert nba_ids == ["n1", "n3"]  # n2 сильнее, но не сегодня (через 40 ч) — ближайшие сутки в приоритете
    text = fmt.daily_block(picks, total=7)
    assert "Матчи дня к разбору" in text and "НБА" in text and "НХЛ" in text and "Все матчи" in text

    from bot.tg.handlers import daily_keyboard
    kb = daily_keyboard(picks, 7)
    flat = [b.callback_data for r in kb.inline_keyboard for b in r]
    assert sum(c.startswith("pick:") for c in flat) == 4 and "menu:all" in flat
