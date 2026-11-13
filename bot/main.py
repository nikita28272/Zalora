"""Точка входа: python -m bot.main"""

from __future__ import annotations

import asyncio
import logging
import sys
from zoneinfo import ZoneInfo

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


def schedule_jobs(scheduler: AsyncIOScheduler, bot, engine, db) -> None:
    """Ежедневные задачи по часовому поясу TIMEZONE.

    Часовой пояс передаётся каждому триггеру явно: CronTrigger без него считает время по
    поясу сервера (в контейнере это UTC), и «11:00» превращалось в 14:00 по Москве.
    misfire_grace_time: если в момент запуска бот был занят, задача выполнится в течение часа.
    """
    sh, sm = _hm(settings.scan_time)
    th, tm = _hm(settings.settle_time)
    common = dict(misfire_grace_time=3600, coalesce=True, max_instances=1)
    scheduler.add_job(run_daily, CronTrigger(hour=sh, minute=sm, timezone=settings.timezone),
                      args=[bot, engine, settings.owner_id], id="daily", **common)
    scheduler.add_job(run_settle, CronTrigger(hour=th, minute=tm, timezone=settings.timezone),
                      args=[bot, engine, db, settings.owner_id], id="settle", **common)


def _next_runs_text(scheduler: AsyncIOScheduler) -> str:
    names = {"daily": "матчи дня", "settle": "итоги"}
    parts = []
    for job in sorted(scheduler.get_jobs(), key=lambda j: j.next_run_time):
        when = job.next_run_time.astimezone(ZoneInfo(settings.timezone))
        parts.append(f"{names.get(job.id, job.id)} — {when:%d.%m %H:%M}")
    return "Ближайшие: " + "; ".join(parts) + f" ({settings.timezone})"


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
        schedule_jobs(scheduler, bot, engine, db)
        scheduler.start()
        logging.info(_next_runs_text(scheduler))
    else:
        logging.warning("TELEGRAM_OWNER_ID не задан: напишите боту, он пришлёт ваш ID")

    logging.info("Бот запущен")
    if settings.owner_id:
        try:  # меню появляется сразу после запуска или обновления, без /start
            await bot.send_message(settings.owner_id,
                                   "✅ Бот запущен. Меню — внизу экрана 👇\n" + _next_runs_text(scheduler),
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
