import asyncio
from datetime import timedelta

from bot import pipeline
from bot.analysis.derived import derived_prob, fit_hockey, hockey_final, _over
from bot.data.injuries import Injury
from bot.data.stats import Result
from bot.db import DB
from bot.pipeline import Engine, settle_outcome
from bot.tg import formatting as fmt
from bot.tg.handlers import _CONF_ALIASES, _SEL_ALIASES, _parse_prob, parse_verdict
from bot.watch import Condition, condition_met, diff_injuries, parse_condition
from tests.test_hockey import NOW, _ctx


def test_fit_hockey_reproduces_market():
    lh, la = fit_hockey(0.646, 0.557, 5.5)
    fin = hockey_final(lh, la)
    assert abs(fin["ml_home"] - 0.646) < 1e-3 and abs(_over(fin["total"], 5.5) - 0.557) < 1e-3
    assert lh > la


def test_derived_dossier_verdict_and_settlement(monkeypatch):
    db = DB(":memory:")
    engine = Engine(db)
    ctx = _ctx(NOW + timedelta(days=1))
    num = engine.save_candidate(ctx)
    text = engine.dossier(ctx, num)
    assert "ДОП. РЫНКИ" in text and "it2_under@2.5" in text and "1-го периода" in text
    p = derived_prob(ctx, "it2_under", 2.5)
    assert 0.3 < p < 0.8 and abs(p + derived_prob(ctx, "it2_over", 2.5) - 1) < 1e-9

    n, picks, conf = parse_verdict(f"{num} it2_under@2.5 0.6 p1_over@1.5 0.55 low")
    top = engine.manual_top(n, picks, conf)
    sigs = {db.signal(pk.signal_id).selection: db.signal(pk.signal_id) for pk in top.picks}
    assert sigs["it2_under"].line == 2.5 and fmt.sig_short(sigs["it2_under"]) == "ИТ2М2.5"
    assert fmt.sig_short(sigs["p1_over"]) == "1П ТБ1.5"

    db.conn.execute("UPDATE signals SET kickoff = ?", ((NOW - timedelta(hours=5)).isoformat(),))

    async def fake_nhl(date_from, date_to):
        return [Result(9, "NHL", NOW - timedelta(hours=5), "Toronto Maple Leafs", "Boston Bruins", 4, 2, 4, 2, "REG")]

    async def fake_periods(date_from, date_to):
        return [(NOW - timedelta(hours=5), "Toronto Maple Leafs", "Boston Bruins", 1, 1)]

    engine.nhl.finished_between = fake_nhl
    monkeypatch.setattr(pipeline, "espn_first_periods", fake_periods)
    settled = {s.selection: s for s in asyncio.run(engine.settle())}
    assert settled["it2_under"].status == "won"  # гости забили 2 < 2.5
    assert settled["p1_over"].status == "won"  # 1:1 в 1-м периоде — 2 > 1.5
    assert fmt.bet_type("hockey", "p1_over") == "Тотал 1-го периода больше"


def test_settle_derived_without_period_score_waits():
    r = Result(1, "NBA", NOW, "A", "B", 100, 98, 100, 98, "REG", p1=(55, 50))
    assert settle_outcome("basketball", "p1_over", 104.5, r) == "won"
    assert settle_outcome("basketball", "it1_under", 100.5, r) == "won"
    assert settle_outcome("basketball", "it2_over", 98.0, r) == "void"
    r.p1 = None
    try:
        settle_outcome("basketball", "p1_over", 104.5, r)
        assert False
    except LookupError:
        pass


def test_parse_condition_and_matching():
    c = parse_condition("38 Scottie Barnes out under@215.5 0.57 low", _SEL_ALIASES, _CONF_ALIASES, _parse_prob)
    assert isinstance(c, Condition)
    assert (c.num, c.player, c.status, c.selection, c.line, c.prob, c.confidence) == (
        38, "Scottie Barnes", "out", "under", 215.5, 0.57, "low")
    assert isinstance(parse_condition("38 Barnes maybe under 0.5", _SEL_ALIASES, _CONF_ALIASES, _parse_prob), str)
    teams = [[Injury("Scottie Barnes", "Out", "Illness")], []]
    assert condition_met(c, teams)
    assert not condition_met(c, [[Injury("Scottie Barnes", "Questionable")], []])
    c_in = Condition(38, "Barnes", "in", "over", None, 0.55)
    assert condition_met(c_in, [[], []]) and not condition_met(c_in, teams)
    ch = diff_injuries({"A": "Questionable", "B": "Out"}, {"A": "Out", "B": "Injured Reserve", "C": "Day-To-Day"})
    assert ch == [("A", "Questionable", "Out"), ("C", None, "Day-To-Day")]


def test_news_watch_alerts_and_condition(monkeypatch):
    db = DB(":memory:")
    engine = Engine(db)
    ctx = _ctx(NOW + timedelta(hours=6))
    num = engine.save_candidate(ctx)
    engine.manual_top(num, [("over", 0.5)], "low")
    cond = parse_condition(f"{num} Matthews out under 0.58 low", _SEL_ALIASES, _CONF_ALIASES, _parse_prob)
    engine.add_condition(cond)

    report = {"Toronto Maple Leafs": [Injury("Auston Matthews", "Out", "Knee")], "Boston Bruins": []}

    async def fetch(sport, max_age=None):
        return report

    monkeypatch.setattr(engine.injuries, "fetch", fetch)
    alerts = asyncio.run(engine.news_watch(now=NOW))
    assert len(alerts) == 1
    a = alerts[0]
    assert a.changes == [("Toronto Maple Leafs", "Auston Matthews", None, "Out")] and len(a.fired) == 1
    text = fmt.news_alert(a)
    assert "Новости перед матчем" in text and "не играет" in text and "Сработало условие босса" in text
    assert asyncio.run(engine.news_watch(now=NOW)) == []  # повторно не шлём
    top = engine.apply_condition(num, a.fired[0][0])
    assert [p.selection for p in top.picks] == ["under"]
