import asyncio

import pytest
from datetime import datetime, timedelta, timezone

from bot.agents.llm import AgentError, AgentRun
from bot.agents.schemas import ALL_OUTCOMES, ChiefOutcome, ChiefVerdict, DevilReport, ExpertsReport, NewsReport, OutcomeView
from bot.analysis.market import market_view
from bot.analysis.poisson import ModelOutput, outcome_probs
from bot.config import LEAGUES
from bot.data.odds import BookOdds, Event
from bot.data.stats import Result
from bot.db import DB, month_start_utc
from bot.pipeline import AnalysisResult, Engine, MatchContext
from bot.tg import formatting as fmt
from bot.tg.handlers import chunks


def _ctx(kickoff):
    ev = Event("evt-1", LEAGUES[0], kickoff, "Arsenal", "Chelsea",
               [BookOdds.football("pinnacle", 2.10, 3.50, 3.70, 1.95, 1.90), BookOdds.football("unibet", 2.05, 3.40, 3.60, 1.90, 1.90)])
    mv = market_view(ev)
    # модель на ~11 п.п. выше рынка по П1: в пределах MAX_MODEL_GAP, перевес > 3%
    probs = {"home": 0.575, "draw": 0.245, "away": 0.18, "over": 0.52, "under": 0.48}
    model = ModelOutput(1.8, 0.9, probs, 10, 10)
    return MatchContext(1, ev, mv, model, "Arsenal FC", "Chelsea FC", home_form="WWDWW", away_form="LDLWL")


def _views(stance="neutral"):
    return [OutcomeView(selection=o, stance=stance, comment=f"мнение по {o}") for o in ALL_OUTCOMES]


def _news():
    return NewsReport(home_absences=["Saka (под вопросом)"], away_absences=[], key_factors=["дерби"],
                      outcomes=_views("for"), confidence="medium", summary="Новости.", sources=["bbc.co.uk"])


def _experts():
    return ExpertsReport(sources_checked=6, agreement="medium", popular_picks=["П1: 4 из 6"], outcomes=_views(),
                         notable_arguments=["форма"], summary="Эксперты.", sources=[])


def _devil():
    return DevilReport(outcomes=_views("against"), hidden_risks=[], severity="low", summary="Возражения слабые.")


def _verdict(decision="bet", p_home=0.56):
    rest = (1 - p_home) / 2
    probs = {"home": p_home, "draw": rest, "away": rest, "over": 0.55, "under": 0.45}
    return ChiefVerdict(
        outcomes=[ChiefOutcome(selection=o, probability=probs[o], comment="итог") for o in ALL_OUTCOMES],
        decision=decision, selection="home", confidence="medium",
        reasoning="Модель и новости за хозяев.", pros=["форма"], cons=["дерби"],
    )


def _result(ctx, decision="bet", p_home=0.56):
    res = AnalysisResult(ctx=ctx)
    res.candidate = ctx.quant_candidates()[0][0]
    res.news, res.experts, res.devil = _news(), _experts(), _devil()
    res.verdict = _verdict(decision, p_home)
    return res


def test_candidate_found_for_home():
    ctx = _ctx(datetime.now(timezone.utc) + timedelta(days=1))
    assert ctx.quant_candidates()[0][0] == "home"
    assert "ДОКЛАД СТАТИСТИКА" in ctx.numbers()


def test_finalize_creates_signal_and_clamps_probability():
    db = DB(":memory:")
    engine = Engine(db)
    ctx = _ctx(datetime.now(timezone.utc) + timedelta(days=1))
    res = _result(ctx, p_home=0.90)  # слишком самоуверенно
    engine._finalize(res)
    assert res.p_final <= ctx.market.fair["home"] + 0.08 + 1e-9
    assert set(res.table) == set(ALL_OUTCOMES)
    assert res.signal_id is not None, res.skip_reason
    text = fmt.analysis(res)
    assert "СИГНАЛ" in text and "не ниже" in text and "Совет по каждому исходу" in text
    res2 = _result(ctx, p_home=0.90)
    engine._finalize(res2)
    assert res2.signal_id is None and "уже" in res2.skip_reason


def test_no_signal_close_to_kickoff():
    engine = Engine(DB(":memory:"))
    res = _result(_ctx(datetime.now(timezone.utc) + timedelta(minutes=10)), p_home=0.9)
    engine._finalize(res)
    assert res.signal_id is None and "прематч" in res.skip_reason


def test_finalize_skip_when_no_edge():
    engine = Engine(DB(":memory:"))
    ctx = _ctx(datetime.now(timezone.utc) + timedelta(days=1))
    res = _result(ctx, p_home=ctx.market.fair["home"])
    engine._finalize(res)
    assert res.signal_id is None and "Перевес" in res.skip_reason
    assert "Без ставки" in fmt.analysis(res)


def test_chief_waits_for_all_subordinates(monkeypatch):
    """Если подчинённый не доложил даже после повтора, главный вердикт не выносит."""
    from bot.agents import team

    calls = {"chief": 0, "news": 0}

    def run(result):
        return AgentRun(result, 10, 10, 0, "claude-opus-5-5")

    async def news_fail(*a, **k):
        calls["news"] += 1
        raise AgentError("нет ответа")

    async def experts_ok(*a, **k):
        return run(_experts())

    async def devil_ok(*a, **k):
        return run(_devil())

    async def chief(*a, **k):
        calls["chief"] += 1
        return run(_verdict())

    async def no_headlines(self):
        return []

    monkeypatch.setattr(team, "news_agent", news_fail)
    monkeypatch.setattr(team, "experts_agent", experts_ok)
    monkeypatch.setattr(team, "devil_agent", devil_ok)
    monkeypatch.setattr(team, "chief_agent", chief)
    monkeypatch.setattr(Engine, "headlines", no_headlines)

    engine = Engine(DB(":memory:"))
    res = asyncio.run(engine.analyze(_ctx(datetime.now(timezone.utc) + timedelta(days=1))))
    assert calls["news"] == 2  # главный повторил задание
    assert calls["chief"] == 0
    assert res.verdict is None and "Совет не собран" in res.skip_reason


def test_full_flow_placed_bet_settles_into_bank():
    db = DB(":memory:")
    engine = Engine(db)
    kickoff = datetime.now(timezone.utc) + timedelta(days=1)
    res = _result(_ctx(kickoff), p_home=0.60)
    engine._finalize(res)
    sid = res.signal_id
    assert sid
    # владелец поставил 300 по 2.20
    db.decide(sid, placed=True, odds=2.20, stake=300)
    assert "Финансист записал" in fmt.placed_confirmation(db.signal(sid), db.bankroll(), db.in_play())
    # матч сыгран
    db.conn.execute("UPDATE signals SET kickoff = ? WHERE id = ?",
                    ((datetime.now(timezone.utc) - timedelta(hours=5)).isoformat(), sid))
    bank_before = db.bankroll()

    async def fake_finished(codes, date_from, date_to):
        return [Result(1, "PL", datetime.now(timezone.utc) - timedelta(hours=5), "Arsenal FC", "Chelsea FC", 2, 1)]

    engine.stats.finished_between = fake_finished
    settled = asyncio.run(engine.settle())
    assert len(settled) == 1 and settled[0].status == "won"
    assert db.bankroll() == bank_before + 300 * 1.20
    report = fmt.settle_report(settled, db.bankroll())
    assert "🧠 Прогноз: П1 — ✅ зашла" in report and "Вы поставили" in report and "+360" in report
    text = fmt.stats(db.settled(since=month_start_utc()), db.settled(), db.bankroll())
    assert "ROI +120.0%" in text


def test_undecided_signal_expires_and_does_not_touch_bank():
    db = DB(":memory:")
    engine = Engine(db)
    res = _result(_ctx(datetime.now(timezone.utc) + timedelta(days=1)), p_home=0.60)
    engine._finalize(res)
    db.conn.execute("UPDATE signals SET kickoff = ? WHERE id = ?",
                    ((datetime.now(timezone.utc) - timedelta(hours=5)).isoformat(), res.signal_id))
    bank_before = db.bankroll()

    async def fake_finished(codes, date_from, date_to):
        return [Result(1, "PL", datetime.now(timezone.utc) - timedelta(hours=5), "Arsenal FC", "Chelsea FC", 0, 1)]

    engine.stats.finished_between = fake_finished
    settled = asyncio.run(engine.settle())
    assert settled[0].user_decision == "expired" and settled[0].status == "lost"
    assert db.bankroll() == bank_before
    assert settled[0].paper_profit < 0
    assert "Вы не ставили" in fmt.settle_report(settled, db.bankroll())


def test_matches_list_and_chunks():
    ctx = _ctx(datetime.now(timezone.utc) + timedelta(days=1))
    text = fmt.matches_list([ctx])
    assert "Arsenal" in text and "🔎" in text
    parts = chunks("строка\n" * 2000, limit=500)
    assert all(len(p) <= 500 for p in parts)


# ---------- режим «босс в чате» ----------
def test_manual_flow_dossier_to_signal():
    from bot.tg.handlers import parse_verdict

    db = DB(":memory:")
    engine = Engine(db)
    ctx = _ctx(datetime.now(timezone.utc) + timedelta(days=1))
    num = engine.save_candidate(ctx)
    assert engine.save_candidate(ctx) == num  # тот же матч — тот же номер

    text = engine.dossier(ctx, num)
    assert f"ДОСЬЕ #{num} ДЛЯ БОССА" in text and f"/verdict {num}" in text
    assert "ДОКЛАД ФИНАНСИСТА" in text and "BOSS.md" in text
    msg = fmt.dossier_message(text)
    assert len(msg) < 4000  # влезает в одно сообщение Telegram
    assert msg.startswith('<pre><code class="language-dossier">') and msg.endswith("</code></pre>")

    # снимок переживает перезапуск бота
    engine2 = Engine(db)
    restored = engine2.load_candidate(num)
    assert restored.title == ctx.title and restored.market.fair == ctx.market.fair

    # второй исход без перевеса, но записан первым — бот ставит исход с перевесом первым
    n, picks, conf = parse_verdict(f"{num} ТМ {ctx.market.fair['under']:.3f} П1 54% medium")
    top = engine2.manual_top(n, picks, conf)
    assert [p.selection for p in top.picks] == ["home", "under"]
    assert [p.positive for p in top.picks] == [True, False]
    assert [p.rank for p in top.picks] == [1, 2]
    rows = db.boss_picks(ctx.event.event_id)
    assert [(s.selection, s.pick_rank) for s in rows] == [("home", 1), ("under", 2)]
    assert rows[0].positive and not rows[1].positive
    assert rows[1].min_odds > rows[1].odds_ref  # «в ноль» только по кэфу выше рыночного
    assert rows[1].paper_odds == rows[1].odds_ref  # бумажный учёт — по кэфу из досье
    msg = fmt.top_picks(top, db.bankroll())
    assert "1. П1</b> — ✅ есть перевес" in msg and "2. ТМ2.5</b> — ⚪ без перевеса" in msg

    # повторный вердикт заменяет прежний, пока ничего не отмечено
    top2 = engine2.manual_top(n, [("home", 0.55)], "high")
    assert top2.replaced and [s.selection for s in db.boss_picks(ctx.event.event_id)] == ["home"]
    # «Не ставил» замене не мешает
    db.decide(top2.picks[0].signal_id, placed=False)
    top3 = engine2.manual_top(n, [("away", 0.3), ("over", 0.5)], "high")
    assert top3.replaced and {s.selection for s in db.boss_picks(ctx.event.event_id)} == {"away", "over"}
    # а сделанная ставка — мешает
    db.decide(top3.picks[0].signal_id, placed=True, odds=3.5, stake=100)
    with pytest.raises(ValueError, match="уже есть ставка"):
        engine2.manual_top(n, [("home", 0.5)], "high")


def test_manual_verdict_rejections():
    db = DB(":memory:")
    engine = Engine(db)
    with pytest.raises(ValueError, match="не найдено"):
        engine.manual_top(99, [("home", 0.5)], "medium")

    ctx = _ctx(datetime.now(timezone.utc) + timedelta(days=1))
    num = engine.save_candidate(ctx)
    with pytest.raises(ValueError, match="повторяются"):
        engine.manual_top(num, [("home", 0.5), ("home", 0.5)])
    with pytest.raises(ValueError, match="больше 100%"):
        engine.manual_top(num, [("home", 0.7), ("away", 0.4)])
    # оба исхода одного рынка — можно, если сумма меньше 100%
    top = engine.manual_top(num, [("home", 0.5), ("away", 0.25)])
    assert len(top.picks) == 2

    late = _ctx(datetime.now(timezone.utc) + timedelta(minutes=5))
    late.event.event_id = "evt-late"
    num2 = engine.save_candidate(late)
    with pytest.raises(ValueError, match="прематч"):
        engine.manual_top(num2, [("home", 0.6)], "high")


def test_candidates_message_and_keyboard():
    from bot.tg.handlers import dossier_keyboard

    ctx = _ctx(datetime.now(timezone.utc) + timedelta(days=1))
    text = fmt.candidates_message([(5, ctx)], total=12)
    assert "#5" in text and "Arsenal" in text
    kb = dossier_keyboard([5, 6, 7])
    assert [b.callback_data for row in kb.inline_keyboard for b in row] == ["dossier:5", "dossier:6", "dossier:7"]
    assert "Кандидатов сегодня нет" in fmt.candidates_message([], total=12)


def test_parse_verdict_variants():
    from bot.tg.handlers import parse_verdict

    assert parse_verdict("7 home 0.54 medium") == (7, [("home", 0.54)], "medium")
    assert parse_verdict("#7 тб 0,57 высокая") == (7, [("over", 0.57)], "high")
    assert parse_verdict("7 X 30") == (7, [("draw", 0.30)], "medium")
    assert parse_verdict("8 ml_away 0.685 ф2 46.5% low") == (8, [("ml_away", 0.685), ("hcp_away", 0.465)], "low")
    assert parse_verdict("8 п2от 0.68 тм 0.52") == (8, [("ml_away", 0.68), ("under", 0.52)], "medium")
    assert parse_verdict("7 home 1.5") is None
    assert parse_verdict("7 home") is None
    assert parse_verdict("7 кто-то 0.5") is None
    assert parse_verdict("7 home 0.5 away 0.2 draw 0.3") is None  # больше двух исходов


# ---------- кнопки ----------
def test_keyboards():
    from bot.tg.handlers import (BTN_ANALYZE, bank_keyboard, main_keyboard, matches_keyboard,
                                 more_keyboard, signals_keyboard)

    kb = main_keyboard()
    texts = [b.text for row in kb.keyboard for b in row]
    assert BTN_ANALYZE in texts and len(texts) == 7 and kb.is_persistent

    ctxs = []
    for i in range(7):
        c = _ctx(datetime.now(timezone.utc) + timedelta(days=1))
        c.idx, c.event.event_id = i + 1, f"{i:032x}"
        ctxs.append(c)
    mk = matches_keyboard(ctxs)
    assert [len(r) for r in mk.inline_keyboard] == [5, 2]
    assert all(len(b.callback_data) <= 64 for r in mk.inline_keyboard for b in r)
    assert matches_keyboard([]) is None

    db = DB(":memory:")
    engine = Engine(db)
    res = _result(_ctx(datetime.now(timezone.utc) + timedelta(days=1)), p_home=0.6)
    engine._finalize(res)
    sk = signals_keyboard(db.pending())
    assert sk.inline_keyboard[0][0].callback_data == f"bet:{res.signal_id}:placed"
    db.decide(res.signal_id, placed=False)
    assert signals_keyboard(db.pending()) is None  # отмеченные сигналы без кнопок

    assert {b.callback_data for r in more_keyboard().inline_keyboard for b in r} >= {"menu:history", "menu:settle"}
    assert bank_keyboard().inline_keyboard[0][0].callback_data == "bank:edit"


def test_morning_report_groups_boss_prediction_and_user_bet():
    db = DB(":memory:")
    engine = Engine(db)
    ctx = _ctx(datetime.now(timezone.utc) + timedelta(days=1))
    num = engine.save_candidate(ctx)
    engine.manual_top(num, [("home", 0.56)], "medium")
    engine.record_user_bet(num, "over", 1.95, 200)
    db.conn.execute("UPDATE signals SET kickoff = ?",
                    ((datetime.now(timezone.utc) - timedelta(hours=5)).isoformat(),))
    bank_before = db.bankroll()

    async def fake_finished(codes, date_from, date_to):
        return [Result(1, "PL", datetime.now(timezone.utc) - timedelta(hours=5), "Arsenal FC", "Chelsea FC", 0, 1)]

    engine.stats.finished_between = fake_finished
    settled = asyncio.run(engine.settle())
    assert {s.source for s in settled} == {"boss", "user"}
    assert db.bankroll() == bank_before - 200  # банк меняет только ваша ставка
    report = fmt.settle_report(settled, db.bankroll())
    assert report.count("Arsenal — Chelsea") == 1  # один блок на матч
    assert "🧠 Прогноз 1: П1 — ❌ не зашла" in report and "записано в статистику" in report
    assert "💰 Ваша ставка: ТБ2.5" in report
    stats = fmt.stats(db.settled(), db.settled(), db.bankroll())
    assert "Прогнозы босса" in stats


def test_late_check_text_and_reminder_timing():
    db = DB(":memory:")
    engine = Engine(db)
    now = datetime.now(timezone.utc)
    ctx = _ctx(now + timedelta(minutes=40))  # футбол: составы за час, напоминание за 50 минут
    num = engine.save_candidate(ctx)
    with pytest.raises(ValueError, match="нет прогноза"):
        engine.late_check(num)
    # поздний вердикт за 40 минут до начала принимается (после составов)
    engine.manual_top(num, [("home", 0.56), ("under", 0.5)], "medium")
    text = engine.late_check(num)
    assert text.startswith(f"ПРОВЕРКА #{num} ДЛЯ БОССА") and "стартовые составы" in text
    assert "1. home = П1" in text and "не отмечен" in text and f"/verdict {num}" in text

    assert engine.due_late_checks(now) == [num]
    engine.mark_late_sent(num)
    assert engine.due_late_checks(now) == []  # напоминание — один раз
    # рано: за 3 часа до матча напоминания нет
    db.delete(f"late_sent:{ctx.event.event_id}")
    assert engine.due_late_checks(now - timedelta(hours=3)) == []
    # все исходы отмечены — проверка не нужна
    for s in db.boss_picks(ctx.event.event_id):
        db.decide(s.id, placed=False)
    assert engine.due_late_checks(now) == []
