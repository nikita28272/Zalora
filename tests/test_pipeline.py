import asyncio
from datetime import datetime, timedelta, timezone

from bot.agents.schemas import ChiefVerdict, DevilReport, ExpertsReport, NewsReport
from bot.analysis.market import market_view
from bot.analysis.poisson import ModelOutput, outcome_probs
from bot.config import LEAGUES
from bot.data.odds import BookOdds, Event
from bot.data.stats import Result
from bot.db import DB
from bot.pipeline import AnalysisResult, Engine, MatchContext
from bot.tg import formatting as fmt
from bot.tg.handlers import chunks


def _ctx(kickoff):
    ev = Event("evt-1", LEAGUES[0], kickoff, "Arsenal", "Chelsea",
               [BookOdds("pinnacle", 2.10, 3.50, 3.70, 1.95, 1.90), BookOdds("unibet", 2.05, 3.40, 3.60, 1.90, 1.90)])
    mv = market_view(ev)
    model = ModelOutput(1.8, 0.9, outcome_probs(1.8, 0.9), 10, 10)
    return MatchContext(1, ev, mv, model, "Arsenal FC", "Chelsea FC", home_form="WWDWW", away_form="LDLWL")


def _result(ctx, decision="bet", p=0.56):
    res = AnalysisResult(ctx=ctx)
    res.candidate = ctx.quant_candidates()[0][0]
    res.news = NewsReport(home_absences=["Saka (под вопросом)"], away_absences=[], key_factors=["дерби"],
                          lean_1x2="home", lean_total="none", confidence="medium", summary="Новости.",
                          sources=["bbc.co.uk"])
    res.experts = ExpertsReport(sources_checked=6, consensus_1x2="home", consensus_total="over", agreement="medium",
                                popular_picks=["П1: 4 из 6"], notable_arguments=["форма"], summary="Эксперты.",
                                sources=[])
    res.devil = DevilReport(counter_arguments=["ротация"], hidden_risks=[], severity="low",
                            recommend_skip=False, summary="Возражения слабые.")
    res.verdict = ChiefVerdict(decision=decision, selection="home", probability=p, confidence="medium",
                               reasoning="Модель и новости за хозяев.", pros=["форма"], cons=["дерби"])
    return res


def test_candidate_found_for_home():
    ctx = _ctx(datetime.now(timezone.utc) + timedelta(days=1))
    assert ctx.quant_candidates()[0][0] == "home"
    assert "СТАТИСТИЧЕСКАЯ МОДЕЛЬ" in ctx.numbers()


def test_finalize_creates_signal_and_clamps_probability():
    db = DB(":memory:")
    engine = Engine(db)
    ctx = _ctx(datetime.now(timezone.utc) + timedelta(days=1))
    res = _result(ctx, p=0.90)  # слишком самоуверенно
    engine._finalize(res)
    assert res.p_final <= ctx.market.fair["home"] + 0.08 + 1e-9
    assert res.signal_id is not None, res.skip_reason
    text = fmt.analysis(res)
    assert "СИГНАЛ" in text and "не ниже" in text
    # второй раз на тот же матч сигнал не выдаём
    res2 = _result(ctx, p=0.90)
    engine._finalize(res2)
    assert res2.signal_id is None and "уже" in res2.skip_reason


def test_finalize_skip_when_no_edge():
    engine = Engine(DB(":memory:"))
    ctx = _ctx(datetime.now(timezone.utc) + timedelta(days=1))
    res = _result(ctx, p=ctx.market.fair["home"])  # согласен с рынком → перевеса нет
    engine._finalize(res)
    assert res.signal_id is None and "Перевес" in res.skip_reason
    assert "Без ставки" in fmt.analysis(res)


def test_settle_marks_won_and_updates_bank():
    db = DB(":memory:")
    engine = Engine(db)
    kickoff = datetime.now(timezone.utc) - timedelta(hours=5)
    res = _result(_ctx(kickoff), p=0.60)
    engine._finalize(res)
    assert res.signal_id
    bank_before = db.bankroll()

    async def fake_finished(codes, date_from, date_to):
        return [Result(1, "PL", kickoff, "Arsenal FC", "Chelsea FC", 2, 1)]

    engine.stats.finished_between = fake_finished
    settled = asyncio.run(engine.settle())
    assert len(settled) == 1 and settled[0].status == "won"
    assert db.bankroll() > bank_before
    assert "✅" in fmt.signal_line(settled[0])
    assert "Сигналов: 1" in fmt.stats(db.settled(), db.bankroll())


def test_matches_list_and_chunks():
    ctx = _ctx(datetime.now(timezone.utc) + timedelta(days=1))
    text = fmt.matches_list([ctx])
    assert "Arsenal" in text and "🔎" in text
    parts = chunks("строка\n" * 2000, limit=500)
    assert all(len(p) <= 500 for p in parts)
