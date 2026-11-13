"""Маршрутизация кнопок на настоящем диспетчере aiogram (сеть Telegram подменена)."""

import asyncio
from datetime import datetime, timezone

import pytest

from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.methods import AnswerCallbackQuery, SendMessage
from aiogram.types import Chat, Message, Update, User

from bot.config import settings
from bot.db import DB
from bot.pipeline import Engine
from bot.tg import handlers

OWNER = 777


class FakeSession(BaseSession):
    def __init__(self):
        super().__init__()
        self.sent = []

    async def make_request(self, bot, method, timeout=None):
        self.sent.append(method)
        if isinstance(method, SendMessage):
            return Message(message_id=len(self.sent), date=datetime.now(timezone.utc),
                           chat=Chat(id=method.chat_id, type="private"), text=method.text)
        if isinstance(method, AnswerCallbackQuery):
            return True
        return True

    async def stream_content(self, *a, **k):  # pragma: no cover
        yield b""

    async def close(self):
        pass


@pytest.fixture(autouse=True)
def owner():
    old = settings.owner_id
    object.__setattr__(settings, "owner_id", OWNER)  # настройки заморожены — подменяем на время теста
    yield
    object.__setattr__(settings, "owner_id", old)


def _setup(monkeypatch):
    session = FakeSession()
    bot = Bot("123:ABC", session=session)
    dp = Dispatcher()
    db = DB(":memory:")
    for r in handlers.build_routers(Engine(db), db):
        dp.include_router(r)
    return bot, dp, session, db


def _msg(text, uid=OWNER):
    return Update(update_id=1, message={
        "message_id": 1, "date": 0, "chat": {"id": uid, "type": "private"},
        "from": {"id": uid, "is_bot": False, "first_name": "x"}, "text": text})


def _cb(data):
    return Update(update_id=2, callback_query={
        "id": "c1", "chat_instance": "ci", "data": data,
        "from": {"id": OWNER, "is_bot": False, "first_name": "x"},
        "message": {"message_id": 5, "date": 0, "chat": {"id": OWNER, "type": "private"}, "text": "menu"}})


def _texts(session):
    return [m.text for m in session.sent if isinstance(m, SendMessage)]


def test_menu_buttons_route_correctly(monkeypatch):
    bot, dp, session, db = _setup(monkeypatch)

    asyncio.run(dp.feed_update(bot, _msg(handlers.BTN_BANK)))
    assert "Банк" in _texts(session)[-1]

    asyncio.run(dp.feed_update(bot, _cb("bank:edit")))
    asyncio.run(dp.feed_update(bot, _msg("25 000")))
    assert db.bankroll() == 25000

    asyncio.run(dp.feed_update(bot, _msg("привет")))
    assert "кнопками" in _texts(session)[-1]

    asyncio.run(dp.feed_update(bot, _cb("menu:limits")))
    assert "Лимиты финансиста" in _texts(session)[-1]

    asyncio.run(dp.feed_update(bot, _msg(handlers.BTN_MORE)))
    assert _texts(session)[-1] == "Что открыть?"


def test_strangers_are_ignored(monkeypatch):
    bot, dp, session, db = _setup(monkeypatch)
    asyncio.run(dp.feed_update(bot, _msg(handlers.BTN_BANK, uid=999)))
    assert _texts(session) == []


# ---------- досье: «Ставил / Не ставил» под окном копирования ----------
def _engine_with_match(db):
    import sys
    sys.path.insert(0, "tests")
    from datetime import timedelta
    from test_pipeline import _ctx

    engine = Engine(db)
    ctx = _ctx(datetime.now(timezone.utc) + timedelta(days=1))
    engine.matches = [ctx]
    return engine, ctx


def _setup_with_match():
    session = FakeSession()
    bot = Bot("123:ABC", session=session)
    dp = Dispatcher()
    db = DB(":memory:")
    engine, ctx = _engine_with_match(db)
    for r in handlers.build_routers(engine, db):
        dp.include_router(r)
    return bot, dp, session, db, engine, ctx


def _markups(session):
    return [m.reply_markup for m in session.sent if isinstance(m, SendMessage)]


def test_dossier_then_top2_verdict_with_buttons_per_pick():
    from aiogram.methods import EditMessageReplyMarkup

    bot, dp, session, db, engine, ctx = _setup_with_match()

    asyncio.run(dp.feed_update(bot, _msg(handlers.BTN_ANALYZE)))
    pick = _markups(session)[-1].inline_keyboard[0][0].callback_data
    assert pick.startswith("pick:")

    asyncio.run(dp.feed_update(bot, _cb(pick)))
    dossier = [m for m in session.sent if isinstance(m, SendMessage) and "ДОСЬЕ #" in m.text][-1]
    assert dossier.reply_markup is None  # под досье кнопок нет: они появятся под вердиктом
    num = int(dossier.text.split("ДОСЬЕ #")[1].split()[0])

    # босс дал два исхода; второй без перевеса
    under = ctx.market.fair["under"]
    asyncio.run(dp.feed_update(bot, _msg(f"/verdict {num} under {under:.3f} home 0.56 medium")))
    reply = [m for m in session.sent if isinstance(m, SendMessage)][-1]
    assert "Топ-2 босса" in reply.text
    rows = reply.reply_markup.inline_keyboard
    assert [[b.text for b in r] for r in rows] == [["✅ Сделал: П1", "❌ Не ставил"],
                                                  ["✅ Сделал: ТМ2.5", "❌ Не ставил"],
                                                  ["🥅 Поздняя проверка"]]

    # поздняя проверка: блок для копирования
    asyncio.run(dp.feed_update(bot, _cb(rows[2][0].callback_data)))
    check = [m for m in session.sent if isinstance(m, SendMessage)][-2]
    assert f"ПРОВЕРКА #{num} ДЛЯ БОССА" in check.text and "language-dossier" in check.text

    # «✅ Сделал» по первому исходу: совет финансиста → «Другая сумма» → сумма → кэф
    asyncio.run(dp.feed_update(bot, _cb(rows[0][0].callback_data)))
    advice = [m for m in session.sent if isinstance(m, SendMessage)][-1]
    assert "Ставьте" in advice.text
    other = [b for r in advice.reply_markup.inline_keyboard for b in r if b.text == "✏️ Другая сумма"][0]
    asyncio.run(dp.feed_update(bot, _cb(other.callback_data)))
    asyncio.run(dp.feed_update(bot, _msg("300")))
    assert "По какому кэфу" in _texts(session)[-1]
    asyncio.run(dp.feed_update(bot, _msg("2.20")))
    home = db.find_signal(ctx.event.event_id, "home", "boss")
    assert home.placed and home.actual_odds == 2.20 and home.actual_stake == 300
    assert "Финансист записал" in _texts(session)[-1] and "Прогноз по банку" in _texts(session)[-1]

    # «❌ Не ставил» по второму: прогноз остаётся для статистики
    asyncio.run(dp.feed_update(bot, _cb(rows[1][1].callback_data)))
    assert db.find_signal(ctx.event.event_id, "under", "boss").user_decision == "skipped"
    assert "статистику" in _texts(session)[-1]

    # повторный вердикт после отметок не принимается
    asyncio.run(dp.feed_update(bot, _msg(f"/verdict {num} away 0.3")))
    assert "уже есть ставка" in _texts(session)[-1]
    assert any(isinstance(m, EditMessageReplyMarkup) for m in session.sent)  # кнопки совета убраны после ответа


def test_bad_verdict_shows_usage():
    bot, dp, session, db, engine, ctx = _setup_with_match()
    asyncio.run(dp.feed_update(bot, _msg("/verdict 5 home")))
    assert "Формат" in _texts(session)[-1]


def test_old_dossier_skip_button_still_works():
    bot, dp, session, db, engine, ctx = _setup_with_match()
    num = engine.save_candidate(ctx)
    top = engine.manual_top(num, [("home", 0.56)], "medium")
    asyncio.run(dp.feed_update(bot, _cb(f"dbet:{num}:skip")))
    assert db.signal(top.picks[0].signal_id).user_decision == "skipped"
    assert "статистику" in _texts(session)[-1]


def test_morning_message_has_analyze_button():
    bot, dp, session, db, engine, ctx = _setup_with_match()

    async def cached(max_age_minutes=None):
        return [ctx]

    engine.load_matches = cached
    asyncio.run(handlers.run_morning_matches(bot, engine, OWNER))
    msg = [m for m in session.sent if isinstance(m, SendMessage)][-1]
    assert "Матчи дня к разбору" in msg.text
    assert msg.reply_markup.inline_keyboard[0][0].callback_data.startswith("pick:")


def test_pressed_pick_row_is_removed():
    from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="a", callback_data="bet:1:placed"), InlineKeyboardButton(text="b", callback_data="bet:1:skip")],
        [InlineKeyboardButton(text="c", callback_data="bet:12:placed"), InlineKeyboardButton(text="d", callback_data="bet:12:skip")],
    ])
    left = handlers._without(kb, 1)
    assert [b.callback_data for r in left.inline_keyboard for b in r] == ["bet:12:placed", "bet:12:skip"]
    assert handlers._without(left, 12) is None
    late = InlineKeyboardMarkup(inline_keyboard=kb.inline_keyboard[:1] + [
        [InlineKeyboardButton(text="🥅", callback_data="late:7")]])
    assert handlers._without(late, 1) is None  # остальные исходы отмечены — убираем и проверку


def test_late_check_reminder_is_sent_once_outside_quiet_hours(monkeypatch):
    from datetime import timedelta

    bot, dp, session, db, engine, ctx = _setup_with_match()
    ctx.event.kickoff = datetime.now(timezone.utc) + timedelta(minutes=30)
    num = engine.save_candidate(ctx)
    engine.manual_top(num, [("home", 0.56)], "medium")
    monkeypatch.setattr(handlers, "_quiet_now", lambda now: False)
    asyncio.run(handlers.run_late_checks(bot, engine, OWNER))
    asyncio.run(handlers.run_late_checks(bot, engine, OWNER))
    texts = _texts(session)
    assert sum("пора поздней проверки" in t for t in texts) == 1
    assert any(f"ПРОВЕРКА #{num}" in t for t in texts)


def test_quiet_hours(monkeypatch):
    from bot.config import settings as st

    old = st.quiet_hours
    try:
        object.__setattr__(st, "quiet_hours", "0-7")
        assert handlers._quiet_now(datetime(2026, 1, 1, 3, 0)) and not handlers._quiet_now(datetime(2026, 1, 1, 8, 0))
        object.__setattr__(st, "quiet_hours", "23-7")
        assert handlers._quiet_now(datetime(2026, 1, 1, 23, 30)) and not handlers._quiet_now(datetime(2026, 1, 1, 12, 0))
        object.__setattr__(st, "quiet_hours", "")
        assert not handlers._quiet_now(datetime(2026, 1, 1, 3, 0))
    finally:
        object.__setattr__(st, "quiet_hours", old)



def test_financier_plan_from_owner_odds():
    from bot.analysis.bankroll import RiskState, plan_bet
    from bot.config import settings as st

    bot, dp, session, db, engine, ctx = _setup_with_match()
    num = engine.save_candidate(ctx)
    top = engine.manual_top(num, [("home", 0.56)], "medium")
    sid = top.picks[0].signal_id

    # шаг 1: «✅ Сделал» — финансист сразу говорит сумму по банку
    asyncio.run(dp.feed_update(bot, _cb(f"bet:{sid}:placed")))
    advice = [m for m in session.sent if isinstance(m, SendMessage)][-1]
    assert "Финансист" in advice.text and "54%" in advice.text and "Ставьте" in advice.text
    ok = [b for r in advice.reply_markup.inline_keyboard for b in r][0]
    assert ok.text.startswith("✅ Поставил") and db.signal(sid).user_decision == "pending"
    stake = float(ok.callback_data.split(":")[3])

    # шаг 2: «✅ Поставил N» — бот спрашивает кэф
    asyncio.run(dp.feed_update(bot, _cb(ok.callback_data)))
    assert "По какому кэфу" in _texts(session)[-1]

    # шаг 3: кэф — ставка записана, разбор и прогноз по банку
    asyncio.run(dp.feed_update(bot, _msg("2.10")))
    sig = db.signal(sid)
    assert sig.placed and sig.actual_odds == 2.10 and sig.actual_stake == stake
    review = _texts(session)[-1]
    assert "Финансист записал" in review and "Зайдёт" in review and "Не зайдёт" in review
    assert db.discipline()[1] == 0  # по совету — нарушений нет

    # свой кэф хуже: перевеса нет — не рекомендует, но можно поставить свою сумму
    state = RiskState(10000, 0, 0, 0)
    bad = plan_bet(0.56, 1.70, state, st)
    assert bad.stake == 0 and bad.verdict == "не ставить" and bad.max_own == 50
    good = plan_bet(0.56, 2.10, state, st)
    assert good.stake > 0 and good.pct <= st.max_stake_pct and good.win == round(good.stake * 1.1, 2)
    nba = plan_bet(0.6, 1.80, state, st, by_analysis=True, confidence="high")
    assert nba.stake == 300 and "высокая" in nba.reason  # 1.5% × 1.5 (high) × 1.33 (шанс 60%)
    nba_low = plan_bet(0.6, 1.80, state, st, by_analysis=True, confidence="low")
    assert nba_low.stake == 0 and nba_low.verdict == "на свой риск"
    blocked = plan_bet(0.6, 2.2, RiskState(10000, 0, 0, 9), st)
    assert blocked.stake == 0 and "подряд" in blocked.reason


def test_financier_own_amount_with_warning():
    bot, dp, session, db, engine, ctx = _setup_with_match()
    num = engine.save_candidate(ctx)
    sid = engine.manual_top(num, [("home", 0.56)], "medium").picks[0].signal_id
    asyncio.run(dp.feed_update(bot, _cb(f"fin:own:{sid}:2.1")))
    assert "Какую сумму" in _texts(session)[-1]
    asyncio.run(dp.feed_update(bot, _msg("1500")))
    sig = db.signal(sid)
    assert sig.placed and sig.actual_stake == 1500 and sig.actual_odds == 2.1
    assert "больше совета" in _texts(session)[-1] and "очень рискованно" in _texts(session)[-1]
    ok, bad, notes = db.discipline()
    assert bad == 1 and "П1" in notes[-1]
    asyncio.run(dp.feed_update(bot, _msg(handlers.BTN_BANK)))
    assert "Дисциплина: 0 из 1" in _texts(session)[-1] and "Режим финансиста" in _texts(session)[-1]



def test_fire_marks_from_verdict_and_command():
    from bot.tg.handlers import parse_fire

    assert parse_fire("7 home 0.54 under 0.5 medium fire") == ("7 home 0.54 under 0.5 medium", 1)
    assert parse_fire("7 home 0.54 under 0.5 🔥🔥") == ("7 home 0.54 under 0.5", 2)
    assert parse_fire("7 home 0.54") == ("7 home 0.54", 0)

    bot, dp, session, db, engine, ctx = _setup_with_match()
    num = engine.save_candidate(ctx)
    under = ctx.market.fair["under"]
    # второй исход без перевеса, но босс поставил 🔥 на оба
    asyncio.run(dp.feed_update(bot, _msg(f"/verdict {num} under {under:.3f} home 0.56 medium fire2")))
    picks = {s.selection: s for s in db.boss_picks(ctx.event.event_id)}
    assert picks["under"].hot and picks["home"].hot

    # /fire снимает и ставит отметку
    asyncio.run(dp.feed_update(bot, _msg(f"/fire {picks['under'].id}")))
    assert not db.signal(picks["under"].id).hot and "Снял" in _texts(session)[-1]
    asyncio.run(dp.feed_update(bot, _msg(f"/fire {picks['under'].id}")))
    assert db.signal(picks["under"].id).hot

    # в списке открытых — 🔥 наверху, в утреннем сообщении — блок «Обратить внимание»
    asyncio.run(dp.feed_update(bot, _msg(handlers.BTN_SIGNALS)))
    text = _texts(session)[-1]
    assert "🔥 — босс советует" in text and "⏳🔥" in text
    from bot.tg import formatting as fmt
    hot = fmt.hot_block(db.hot_open())
    assert "Обратить внимание" in hot and "Arsenal" in hot
