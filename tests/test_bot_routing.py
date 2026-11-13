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
