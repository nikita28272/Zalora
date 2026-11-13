import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from bot.agents import llm
from bot.agents.schemas import ChiefVerdict, DevilReport
from bot.analysis.bankroll import RiskState, decide_stake, edge, kelly, min_acceptable_odds
from bot.analysis.market import devig, market_view
from bot.analysis.poisson import PoissonModel, outcome_probs
from bot.config import LEAGUES, Settings
from bot.data.odds import BookOdds, Event
from bot.data.stats import Result
from bot.data.teams import best_match, normalize
from bot.db import DB

NOW = datetime(2026, 11, 12, 12, tzinfo=timezone.utc)


# ---------- рынок ----------
def test_devig_sums_to_one_and_removes_margin():
    probs = devig([2.0, 3.5, 4.0])
    assert abs(sum(probs) - 1) < 1e-9
    assert probs[0] < 0.5  # 1/2.0 = 0.5 до снятия маржи


def _event(books):
    return Event("e1", LEAGUES[0], NOW + timedelta(days=1), "Arsenal", "Chelsea", books)


def test_market_view_prefers_pinnacle():
    books = [
        BookOdds.football("pinnacle", 2.0, 3.6, 4.0, 1.9, 1.95),
        BookOdds.football("unibet", 1.8, 3.4, 4.2, 1.85, 1.9),
    ]
    mv = market_view(_event(books))
    assert mv.source == "pinnacle"
    assert mv.best_odds["home"] == 2.0
    assert abs(mv.fair["home"] + mv.fair["draw"] + mv.fair["away"] - 1) < 1e-9
    assert abs(mv.fair["over"] + mv.fair["under"] - 1) < 1e-9


def test_market_view_without_h2h_is_none():
    assert market_view(_event([BookOdds.football("x")])) is None


# ---------- модель ----------
def test_outcome_probs_symmetric():
    p = outcome_probs(1.3, 1.3)
    assert abs(p["home"] - p["away"]) < 1e-9
    assert abs(p["over"] + p["under"] - 1) < 1e-9


def _results():
    out, d = [], NOW - timedelta(days=60)
    teams = ["Strong FC", "Weak FC", "Mid FC", "Other FC"]
    # сильная команда всех обыгрывает, слабая всем проигрывает
    for rnd in range(6):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                hg = 3 if h == "Strong FC" else 0 if h == "Weak FC" else 1
                ag = 3 if a == "Strong FC" else 0 if a == "Weak FC" else 1
                out.append(Result(len(out), "PL", d + timedelta(days=rnd * 7), h, a, hg, ag))
    return out


def test_poisson_ranks_strong_team():
    m = PoissonModel(_results(), now=NOW)
    pred = m.predict("Strong FC", "Weak FC")
    assert pred is not None
    assert pred.probs["home"] > 0.7
    assert m.predict("Strong FC", "Unknown FC") is None
    assert len(m.form("Strong FC")) == 5


# ---------- финансист ----------
def test_kelly_and_edge():
    assert kelly(0.5, 2.0) == 0
    assert kelly(0.6, 2.0) == pytest.approx(0.2)
    assert edge(0.55, 2.0) == pytest.approx(0.10)
    assert min_acceptable_odds(0.5, 0.03) == pytest.approx(2.06)


def test_stake_capped_and_limits():
    cfg = Settings()
    ok = decide_stake(0.6, 2.2, RiskState(10000, 0, 0, 0), cfg)
    assert ok.allowed and ok.stake <= 10000 * cfg.max_stake_pct
    assert decide_stake(0.6, 2.2, RiskState(10000, 25, 0, 0), cfg).allowed  # лимита ставок в день нет
    assert not decide_stake(0.6, 2.2, RiskState(10000, 0, -10000, 0), cfg).allowed
    assert not decide_stake(0.6, 2.2, RiskState(10000, 0, 0, cfg.stop_after_losses), cfg).allowed
    assert not decide_stake(0.4, 2.2, RiskState(10000, 0, 0, 0), cfg).allowed


# ---------- названия команд ----------
@pytest.mark.parametrize(
    "odds_name, expected",
    [
        ("Bayern Munich", "FC Bayern München"),
        ("Inter Milan", "FC Internazionale Milano"),
        ("AC Milan", "AC Milan"),
        ("Manchester United", "Manchester United FC"),
        ("Manchester City", "Manchester City FC"),
        ("Brighton and Hove Albion", "Brighton & Hove Albion FC"),
        ("Wolverhampton Wanderers", "Wolverhampton Wanderers FC"),
        ("Paris Saint Germain", "Paris Saint-Germain FC"),
        ("Atletico Madrid", "Club Atlético de Madrid"),
        ("Real Madrid", "Real Madrid CF"),
        ("Athletic Bilbao", "Athletic Club"),
        ("Borussia Monchengladbach", "Borussia Mönchengladbach"),
        ("RB Leipzig", "RB Leipzig"),
        ("Tottenham Hotspur", "Tottenham Hotspur FC"),
    ],
)
def test_team_matching(odds_name, expected):
    candidates = [
        "FC Bayern München", "FC Internazionale Milano", "AC Milan", "Manchester United FC",
        "Manchester City FC", "Brighton & Hove Albion FC", "Wolverhampton Wanderers FC",
        "Paris Saint-Germain FC", "Paris FC", "Club Atlético de Madrid", "Real Madrid CF", "Athletic Club",
        "Borussia Mönchengladbach", "Borussia Dortmund", "RB Leipzig", "Tottenham Hotspur FC",
    ]
    assert best_match(odds_name, candidates) == expected


def test_no_false_match():
    assert best_match("Manchester United", ["Manchester City FC"]) is None
    assert normalize("FC Bayern München") == "bayern munchen"


# ---------- журнал ----------
def test_db_signals_and_streak_count_only_placed():
    db = DB(":memory:")
    base = dict(league_code="PL", league_name="АПЛ", home="A", away="B",
                kickoff=NOW.isoformat(), odds_ref=2.0, min_odds=1.9, p_final=0.55, p_market=0.5,
                p_model=0.56, edge=0.1, stake=100, confidence="medium", reasoning="r")
    s1 = db.add_signal(event_id="e1", selection="home", **base)
    assert db.add_signal(event_id="e1", selection="home", **base) is None  # дубль
    s2 = db.add_signal(event_id="e2", selection="away", **base)
    s3 = db.add_signal(event_id="e3", selection="away", **base)
    db.decide(s1, placed=True, odds=2.0, stake=100)
    db.decide(s2, placed=True, odds=2.1, stake=200)
    db.decide(s3, placed=False)
    assert db.in_play() == (2, 300)
    start = db.bankroll()
    db.settle(s1, "lost", "0:1", -100, -100)
    db.settle(s2, "lost", "2:0", -200, -100)
    db.settle(s3, "lost", "2:0", 0, -100)  # не ставили: банк не меняется
    assert db.bankroll() == start - 300
    assert db.risk_state().losing_streak == 2
    db.reset_streak()
    assert db.risk_state().losing_streak == 0


# ---------- агенты (без сети) ----------
class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        # копия на момент вызова: настоящий SDK сериализует запрос сразу
        self.calls.append(dict(kwargs, messages=list(kwargs["messages"])))
        return self.responses.pop(0)


def _resp(stop, content):
    usage = SimpleNamespace(input_tokens=1000, output_tokens=200, cache_read_input_tokens=0,
                            server_tool_use=SimpleNamespace(web_search_requests=1))
    return SimpleNamespace(stop_reason=stop, content=content, usage=usage)


def test_run_agent_handles_pause_and_nudge(monkeypatch):
    report = {"outcomes": [{"selection": o, "stance": "against", "comment": "c"}
                           for o in ("home", "draw", "away", "over", "under")],
              "hidden_risks": [], "severity": "low", "summary": "s"}
    msgs = FakeMessages([
        _resp("pause_turn", [SimpleNamespace(type="text", text="ищу")]),
        _resp("end_turn", [SimpleNamespace(type="text", text="готово")]),
        _resp("tool_use", [SimpleNamespace(type="tool_use", id="t1", name="submit_report", input=report)]),
    ])
    fake = SimpleNamespace(messages=msgs, beta=SimpleNamespace(messages=msgs))
    monkeypatch.setattr(llm, "_client", fake)
    run = asyncio.run(llm.run_agent(model="claude-opus-5-5", effort="low", system="s", prompt="p",
                                    schema=DevilReport, web_search=True))
    assert run.result.severity == "low"
    assert run.web_searches == 3 and run.input_tokens == 3000
    assert len(msgs.calls) == 3
    # после pause_turn не добавляем лишнее сообщение пользователя, после end_turn — добавляем
    last = msgs.calls[-1]["messages"]
    assert [m["role"] for m in last] == ["user", "assistant", "assistant", "user"]
    assert msgs.calls[0]["fallbacks"] == "default"
    assert msgs.calls[0]["tools"][0]["type"] == "web_search_20260209"


def test_run_agent_refusal(monkeypatch):
    msgs = FakeMessages([_resp("refusal", [])])
    monkeypatch.setattr(llm, "_client", SimpleNamespace(messages=msgs, beta=SimpleNamespace(messages=msgs)))
    with pytest.raises(llm.AgentError):
        asyncio.run(llm.run_agent(model="claude-opus-5-5", effort="low", system="s", prompt="p",
                                  schema=ChiefVerdict))


def test_run_agent_sends_incomplete_report_back(monkeypatch):
    from bot.agents.team import _outcomes_complete
    partial = {"outcomes": [{"selection": "home", "stance": "against", "comment": "c"}],
               "hidden_risks": [], "severity": "low", "summary": "s"}
    full = dict(partial, outcomes=[{"selection": o, "stance": "neutral", "comment": "c"}
                                   for o in ("home", "draw", "away", "over", "under")])
    msgs = FakeMessages([
        _resp("tool_use", [SimpleNamespace(type="tool_use", id="t1", name="submit_report", input=partial)]),
        _resp("tool_use", [SimpleNamespace(type="tool_use", id="t2", name="submit_report", input=full)]),
    ])
    monkeypatch.setattr(llm, "_client", SimpleNamespace(messages=msgs, beta=SimpleNamespace(messages=msgs)))
    run = asyncio.run(llm.run_agent(model="claude-opus-5-5", effort="low", system="s", prompt="p",
                                    schema=DevilReport, validate=_outcomes_complete))
    assert len(run.result.outcomes) == 5
    back = msgs.calls[1]["messages"][-1]["content"][0]
    assert back["type"] == "tool_result" and back["is_error"] and back["tool_use_id"] == "t1"


# ---------- экономичный режим и учёт расходов ----------
def test_economy_mode_defaults(monkeypatch):
    monkeypatch.delenv("LLM_MODEL_CHIEF", raising=False)
    monkeypatch.delenv("LLM_MODEL_AGENTS", raising=False)
    cfg = Settings()
    assert cfg.llm_model_chief == "claude-opus-5-5"
    assert cfg.llm_model_agents == "claude-sonnet-5-5"


def test_cost_depends_on_model():
    opus = llm.AgentRun(None, 100_000, 10_000, 2, "claude-opus-5-5")
    sonnet = llm.AgentRun(None, 100_000, 10_000, 2, "claude-sonnet-5-5")
    assert opus.cost_usd == pytest.approx(0.4 + 0.2 + 0.02)
    assert sonnet.cost_usd == pytest.approx(0.2 + 0.1 + 0.02)


def test_usage_accumulates_real_cost():
    db = DB(":memory:")
    db.add_llm_usage(analyses=1)
    db.add_llm_usage(input_tokens=1000, output_tokens=100, web_searches=1, cost_usd=0.12)
    db.add_llm_usage(input_tokens=2000, output_tokens=300, cost_usd=0.08)
    u = db.llm_usage_since(30)
    assert u["analyses"] == 1 and u["cost_usd"] == pytest.approx(0.20) and u["input_tokens"] == 3000


def test_sonnet_agent_uses_new_web_search_and_fallback(monkeypatch):
    report = {"outcomes": [{"selection": o, "stance": "neutral", "comment": "c"}
                           for o in ("home", "draw", "away", "over", "under")],
              "hidden_risks": [], "severity": "low", "summary": "s"}
    msgs = FakeMessages([_resp("tool_use", [SimpleNamespace(type="tool_use", id="t", name="submit_report",
                                                            input=report)])])
    monkeypatch.setattr(llm, "_client", SimpleNamespace(messages=msgs, beta=SimpleNamespace(messages=msgs)))
    asyncio.run(llm.run_agent(model="claude-sonnet-5-5", effort="low", system="s", prompt="p",
                              schema=DevilReport, web_search=True))
    call = msgs.calls[0]
    assert call["model"] == "claude-sonnet-5-5"
    assert call["tools"][0]["type"] == "web_search_20260209"
    assert call["fallbacks"] == "default"


def test_poisson_accounts_for_opponent_strength():
    """Одинаковые счета против сильных соперников ценятся выше, чем против слабых."""
    d = NOW - timedelta(days=30)
    res, n = [], 0

    def add(h, a, hg, ag, day):
        nonlocal n
        res.append(Result(n, "PL", d + timedelta(days=day), h, a, hg, ag))
        n += 1

    strong, weak = ["S1", "S2", "S3"], ["W1", "W2", "W3"]
    # сильные обыгрывают слабых, между собой ничьи
    day = 0
    for s in strong:
        for w in weak:
            add(s, w, 3, 0, day); add(w, s, 0, 2, day + 1); day += 2
    for i, s in enumerate(strong):
        for t in strong[i + 1:]:
            add(s, t, 1, 1, day); add(t, s, 1, 1, day + 1); day += 2
    # A и B: одинаковые результаты 1:1, но A играл с сильными, B — со слабыми
    for s in strong:
        add("A", s, 1, 1, day); add(s, "A", 1, 1, day + 1); day += 2
    for w in weak:
        add("B", w, 1, 1, day); add(w, "B", 1, 1, day + 1); day += 2

    m = PoissonModel(res, now=NOW)
    assert m.teams["A"].attack > m.teams["B"].attack
    assert m.teams["A"].defense < m.teams["B"].defense  # меньше — значит надёжнее
    assert m.predict("A", "B").probs["home"] > m.predict("A", "B").probs["away"]


# ---------- кредиты The Odds API ----------
def test_quota_tracking_persists_and_warns(monkeypatch):
    import httpx

    from bot.data import odds as odds_mod
    from bot.pipeline import Engine
    from bot.tg import formatting as fmt

    def handler(request):
        assert request.url.path.endswith("/v4/sports")  # бесплатный эндпоинт
        return httpx.Response(200, json=[], headers={"x-requests-remaining": "42", "x-requests-used": "458"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(odds_mod.httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))

    db = DB(":memory:")
    engine = Engine(db)
    assert "пока неизвестно" in fmt.quota_line(engine.quota())
    q = asyncio.run(engine.refresh_quota())
    assert (q.remaining, q.used, q.total) == (42, 458, 500)

    restarted = Engine(db)  # после перезапуска бот помнит остаток из базы
    line = fmt.quota_line(restarted.quota())
    assert "осталось <b>42</b> из 500" in line and "заканчиваются" in line


def test_old_database_is_migrated_with_sources(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
    CREATE TABLE signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, event_id TEXT NOT NULL,
        league_code TEXT NOT NULL, league_name TEXT NOT NULL, home TEXT NOT NULL, away TEXT NOT NULL,
        kickoff TEXT NOT NULL, selection TEXT NOT NULL, odds_ref REAL NOT NULL, min_odds REAL NOT NULL,
        p_final REAL NOT NULL, p_market REAL NOT NULL, p_model REAL, edge REAL NOT NULL, stake REAL NOT NULL,
        confidence TEXT NOT NULL, reasoning TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
        score TEXT, profit REAL, settled_at TEXT, UNIQUE(event_id, selection));
    INSERT INTO signals (created_at, event_id, league_code, league_name, home, away, kickoff, selection,
        odds_ref, min_odds, p_final, p_market, edge, stake, confidence, reasoning)
    VALUES ('2026-11-10T00:00:00+00:00', 'e1', 'PL', 'АПЛ', 'A', 'B', '2026-11-10T20:00:00+00:00', 'home',
        2.0, 1.9, 0.55, 0.5, 0.1, 100, 'medium', 'r');
    """)
    conn.commit()
    conn.close()

    db = DB(str(path))
    old = db.signal(1)
    assert old.source == "boss" and old.sport == "football" and old.user_decision == "pending"
    base = dict(league_code="PL", league_name="АПЛ", home="A", away="B", kickoff=old.kickoff.isoformat(),
                odds_ref=2.0, min_odds=1.9, p_final=0.55, p_market=0.5, edge=0.1, stake=100,
                confidence="user", reasoning="r")
    # тот же матч и исход, но ставка владельца — теперь допустимо
    assert db.add_signal(event_id="e1", selection="home", source="user", **base) is not None


def test_corrupted_quotes_are_dropped():
    from bot.analysis.market import clean_books

    books = [
        BookOdds.football("pinnacle", 1.62, 4.0, 5.2),
        BookOdds.football("b1", 1.60, 3.9, 5.0),
        BookOdds.football("b2", 1.58, 4.1, 5.3),
        BookOdds.football("arb", 1.93, 4.5, 3.28),  # невозможная линия: сумма вероятностей < 100%
        BookOdds.football("odd", 2.60, 3.4, 2.60),  # с маржой, но далеко от рынка — выброс
    ]
    pts = {"home": None, "draw": None, "away": None}
    kept = {b.bookmaker for b in clean_books(books, ("home", "draw", "away"), pts)}
    assert kept == {"pinnacle", "b1", "b2"}


def test_hockey_moneyline_ignores_regulation_prices_under_h2h():
    from bot.config import LEAGUES_BY_ODDS_KEY
    from bot.data.odds import parse_event

    def book(key, h, a):
        return {"key": key, "markets": [{"key": "h2h", "outcomes": [
            {"name": "Vegas Golden Knights", "price": h}, {"name": "Nashville Predators", "price": a}]}]}

    raw = {"id": "x", "commence_time": "2026-11-14T03:10:00Z", "home_team": "Vegas Golden Knights",
           "away_team": "Nashville Predators",
           "bookmakers": [book("pinnacle", 1.56, 2.58), book("b1", 1.53, 2.62), book("b2", 1.55, 2.55),
                          book("b3", 1.54, 2.60), book("bad", 1.93, 3.55)]}  # «bad» отдал кэфы основного времени
    mv = market_view(parse_event(raw, LEAGUES_BY_ODDS_KEY["icehockey_nhl"]))
    assert mv.best_odds["ml_away"] < 2.7 and mv.avg_odds["ml_away"] < 2.65


def test_model_far_from_market_is_not_a_candidate():
    from bot.analysis.poisson import ModelOutput
    from bot.config import LEAGUES
    from bot.pipeline import MatchContext

    ev = Event("e", LEAGUES[0], NOW + timedelta(days=1), "A", "B",
               [BookOdds.football("pinnacle", 1.62, 4.0, 5.2), BookOdds.football("b1", 1.60, 3.9, 5.0)])
    mv = market_view(ev)
    far = ModelOutput(1.2, 1.4, {"home": 0.40, "draw": 0.27, "away": 0.33}, 10, 10)  # П2: модель 33% vs рынок ~19%
    ctx = MatchContext(1, ev, mv, far)
    assert ctx.quant_candidates() == []
    assert ctx.model_gap("away") > 0.12


def test_tighter_filter_drops_outlier_prices_and_regulation_totals():
    from bot.analysis.market import clean_books, usable_prices

    # одиночный «щедрый» кэф 3.15 при обычных 2.80–2.90 не попадает ни в лучший, ни в средний
    assert usable_prices([2.85, 2.80, 2.90, 2.86, 3.15]) == [2.85, 2.80, 2.90, 2.86]
    assert usable_prices([2.0, 3.0]) == [2.0, 3.0]  # мало данных — не трогаем

    # тотал без учёта овертайма (больше ~2.25 / меньше ~1.62) выбивается из рынка тотала с ОТ
    def tot(name, over, under):
        b = BookOdds(name)
        b.put("over", over, 6.5)
        b.put("under", under, 6.5)
        return b

    books = [tot("pinnacle", 2.02, 1.85), tot("a", 2.00, 1.83), tot("b", 2.05, 1.80), tot("c", 2.03, 1.82),
             tot("reg_only", 2.25, 1.62)]
    kept = {b.bookmaker for b in clean_books(books, ("over", "under"), {"over": 6.5, "under": 6.5})}
    assert kept == {"pinnacle", "a", "b", "c"}



def test_drawdown_modes_and_exposure():
    from bot.analysis.bankroll import RiskState, decide_stake, plan_bet, risk_block, risk_mode
    from bot.config import settings as cfg

    normal = RiskState(10000, 0, 0, 0, peak=10000)
    assert risk_mode(normal, cfg).multiplier == 1
    caution = RiskState(8800, 0, 0, 0, peak=10000)  # −12%
    assert risk_mode(caution, cfg).multiplier == 0.5 and "осторожный" in risk_mode(caution, cfg).label
    defense = RiskState(7500, 0, 0, 2, peak=10000)  # −25% и 2 проигрыша подряд
    assert risk_mode(defense, cfg).multiplier == 0.25 * 0.75
    stop = RiskState(6900, 0, 0, 0, peak=10000)  # −31%
    assert risk_mode(stop, cfg).stop and "просел" in risk_block(stop, cfg)
    assert not decide_stake(0.6, 2.2, stop, cfg).allowed

    full = plan_bet(0.6, 2.2, normal, cfg)
    half = plan_bet(0.6, 2.2, caution, cfg)
    assert full.pct == cfg.max_stake_pct and "предохранитель" in full.reason  # сильный перевес — упёрлись
    assert half.pct == full.pct / 2 and half.stake == 220 and "×0.5" in half.reason

    week = RiskState(10000, 0, 0, 0, peak=10000, week_profit=-1000)
    assert "недельный" in risk_block(week, cfg)

    # открытые ставки уменьшают свободный банк и лимит «в игре»
    free = plan_bet(0.55, 2.0, RiskState(10000, 0, 0, 0, peak=10000, in_play=2000), cfg)
    base = plan_bet(0.55, 2.0, normal, cfg)
    assert free.stake < base.stake
    busy = RiskState(10000, 0, 0, 0, peak=10000, in_play=1100)  # в игре 11% из 12%
    capped = plan_bet(0.6, 2.2, busy, cfg)
    assert capped.stake <= 100 and "в игре" in capped.reason
    same_match = plan_bet(0.6, 2.2, normal, cfg, match_in_play=500)  # на матч уже 5% из 6%
    assert same_match.stake <= 100 and "один матч" in same_match.reason
    assert "лимит одновременных" in risk_block(RiskState(10000, 0, 0, 0, in_play=1200), cfg)


def test_peak_tracking_and_resume_baseline():
    from bot.db import DB

    db = DB(":memory:")
    db.set_bankroll(10000)
    assert db.bank_peak() == 10000
    db.set("bankroll", 11000)
    assert db.bank_peak() == 11000  # новый максимум
    db.set("bankroll", 7000)
    st = db.risk_state()
    assert st.peak == 11000 and round(st.drawdown, 2) == 0.36
    db.reset_peak()
    assert db.risk_state().drawdown == 0


# ---------- точность босса и типы ставок ----------
def test_accuracy_and_bet_types():
    from bot.tg import formatting as fmt

    db = DB(":memory:")
    base = dict(league_code="NBA", league_name="НБА", home="A", away="B", kickoff=NOW.isoformat(),
                odds_ref=1.75, min_odds=1.75, p_market=0.55, p_model=None, edge=0.0, stake=10,
                confidence="medium", reasoning="r", sport="basketball")
    # 10 прогнозов по 55% на тотал «меньше»: зашло 4 — босс завышает
    for i in range(10):
        sid = db.add_signal(event_id=f"u{i}", selection="under", p_final=0.55, line=225.5, **base)
        won = i < 4
        db.settle(sid, "won" if won else "lost", "1:0", 0.0, 7.5 if won else -10)
    # 2 прогноза на фору, оба зашли, один поставлен
    for i in range(2):
        sid = db.add_signal(event_id=f"h{i}", selection="hcp_home", p_final=0.62, line=-3.5, **base)
        if i == 0:
            db.decide(sid, True, 1.8, 50)
        db.settle(sid, "won", "1:0", 40.0 if i == 0 else 0.0, 7.5)

    acc = fmt.accuracy(db.settled())
    assert "ждали 55%, зашло 40% (4 из 10) — ⚠️ босс завышает" in acc
    assert "60–65%" in acc and "мало данных" in acc and "Босс против букмекеров" in acc

    types = fmt.bet_types(db.settled())
    assert "🏀 <b>Баскетбол</b>" in types
    assert "🧠 Тотал меньше: 4 из 10 (40%)" in types and "🧠 Фора: 2 из 2 (100%)" in types
    assert "💰 Фора: 1 из 1" in types and "💰 Тотал меньше" not in types
    assert "Лучше всего" in types or "от 10 прогнозов" in types
    assert fmt.bet_type("hockey", "ml_home") == "Победа с ОТ" and fmt.bet_type("hockey", "home") == "Победа в осн. время"
