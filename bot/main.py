"""Точка входа: python -m bot.main"""

from __future__ import annotations

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import BotCommand
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from .config import settings
from .db import DB
from .pipeline import Engine
from .tg.handlers import build_routers, main_keyboard, run_daily, run_settle

COMMANDS = [
    BotCommand(command="matches", description="Ближайшие матчи"),
    BotCommand(command="analyze", description="Разбор или досье матча: /analyze N"),
    BotCommand(command="scan", description="Кандидаты или сигналы сейчас"),
    BotCommand(command="verdict", description="Вердикт босса: /verdict 7 home 0.54"),
    BotCommand(command="signals", description="Открытые сигналы"),
    BotCommand(command="stats", description="Статистика журнала"),
    BotCommand(command="history", description="Последние рассчитанные"),
    BotCommand(command="bank", description="Банк: /bank 10000"),
    BotCommand(command="limits", description="Лимиты риска"),
    BotCommand(command="credits", description="Остаток кредитов The Odds API"),
    BotCommand(command="usage", description="Расход на Claude API"),
    BotCommand(command="settle", description="Рассчитать итоги"),
    BotCommand(command="help", description="Помощь"),
]


def _hm(value: str) -> tuple[int, int]:
    h, m = value.split(":")
    return int(h), int(m)


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # httpx пишет в лог полные URL, а в них бывают ключи API
    logging.getLogger("httpx").setLevel(logging.WARNING)

    missing = settings.missing_keys()
    if missing:
        sys.exit("Не заданы переменные окружения: " + ", ".join(missing) + ". Заполните файл .env")

    db = DB()
    engine = Engine(db)
    bot = Bot(settings.telegram_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()
    for router in build_routers(engine, db):
        dp.include_router(router)
    await bot.set_my_commands(COMMANDS)

    scheduler = AsyncIOScheduler(timezone=settings.timezone)
    if settings.owner_id:
        sh, sm = _hm(settings.scan_time)
        th, tm = _hm(settings.settle_time)
        scheduler.add_job(run_daily, CronTrigger(hour=sh, minute=sm), args=[bot, engine, settings.owner_id])
        scheduler.add_job(run_settle, CronTrigger(hour=th, minute=tm), args=[bot, engine, db, settings.owner_id])
        scheduler.start()
    else:
        logging.warning("TELEGRAM_OWNER_ID не задан: напишите боту, он пришлёт ваш ID")

    logging.info("Бот запущен")
    if settings.owner_id:
        try:  # меню появляется сразу после запуска или обновления, без /start
            await bot.send_message(settings.owner_id, "✅ Бот запущен. Меню — внизу экрана 👇",
                                   reply_markup=main_keyboard())
        except Exception as exc:  # noqa: BLE001 — например, владелец ещё не писал боту
            logging.warning("Не удалось отправить меню владельцу: %s", type(exc).__name__)
    try:
        await dp.start_polling(bot)
    finally:
        if scheduler.running:
            scheduler.shutdown(wait=False)
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
