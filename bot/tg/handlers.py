"""Команды Telegram-бота."""

from __future__ import annotations

import logging

from aiogram import Bot, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import Message

from ..agents.llm import PRICES, WEB_SEARCH_PRICE
from ..config import settings
from ..db import DB
from ..pipeline import Engine
from . import formatting as fmt

log = logging.getLogger(__name__)

TG_LIMIT = 4000


def chunks(text: str, limit: int = TG_LIMIT) -> list[str]:
    parts, current = [], ""
    for line in text.split("\n"):
        while len(line) > limit:  # очень длинная строка
            parts.append(line[:limit])
            line = line[limit:]
        if len(current) + len(line) + 1 > limit:
            parts.append(current)
            current = ""
        current += line + "\n"
    if current.strip():
        parts.append(current)
    return parts


async def send_long(bot: Bot, chat_id: int, text: str) -> None:
    for part in chunks(text):
        await bot.send_message(chat_id, part, disable_web_page_preview=True)


def build_routers(engine: Engine, db: DB) -> list[Router]:
    owner = Router(name="owner")
    owner.message.filter(lambda m: m.from_user is not None and m.from_user.id == settings.owner_id)

    @owner.message(Command("start", "help"))
    async def cmd_help(message: Message) -> None:
        await message.answer(fmt.HELP)

    @owner.message(Command("matches"))
    async def cmd_matches(message: Message) -> None:
        await message.answer("Загружаю линии и статистику, это займёт около минуты…")
        try:
            matches = await engine.load_matches()
        except Exception as exc:  # noqa: BLE001
            log.exception("load_matches failed")
            await message.answer(f"Не удалось загрузить матчи: {type(exc).__name__}")
            return
        await send_long(message.bot, message.chat.id, fmt.matches_list(matches))

    @owner.message(Command("analyze"))
    async def cmd_analyze(message: Message, command: CommandObject) -> None:
        if not command.args or not command.args.strip().isdigit():
            await message.answer("Укажите номер матча из /matches, например: /analyze 3")
            return
        await message.answer("Агенты работают: новости, эксперты, адвокат дьявола, главный аналитик. Это 1–3 минуты…")
        try:
            res = await engine.analyze_by_index(int(command.args.strip()))
        except ValueError as exc:
            await message.answer(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            log.exception("analyze failed")
            await message.answer(f"Ошибка разбора: {type(exc).__name__}")
            return
        await send_long(message.bot, message.chat.id, fmt.analysis(res))

    @owner.message(Command("scan"))
    async def cmd_scan(message: Message) -> None:
        await message.answer("Ищу матчи с перевесом и запускаю агентов…")
        await run_scan(message.bot, engine, message.chat.id, verbose=True)

    @owner.message(Command("signals"))
    async def cmd_signals(message: Message) -> None:
        pending = db.pending()
        if not pending:
            await message.answer("Открытых сигналов нет.")
            return
        await send_long(message.bot, message.chat.id,
                        "<b>Открытые сигналы</b>\n" + "\n".join(fmt.signal_line(s) for s in pending))

    @owner.message(Command("history"))
    async def cmd_history(message: Message) -> None:
        recent = db.recent(15)
        if not recent:
            await message.answer("Рассчитанных сигналов пока нет.")
            return
        await send_long(message.bot, message.chat.id,
                        "<b>Последние сигналы</b>\n" + "\n".join(fmt.signal_line(s) for s in recent))

    @owner.message(Command("stats"))
    async def cmd_stats(message: Message) -> None:
        await message.answer(fmt.stats(db.settled(), db.bankroll()))

    @owner.message(Command("bank"))
    async def cmd_bank(message: Message, command: CommandObject) -> None:
        if command.args:
            raw = command.args.replace(" ", "").replace(",", ".")
            try:
                value = float(raw)
                if value < 0:
                    raise ValueError
            except ValueError:
                await message.answer("Укажите сумму числом, например: /bank 15000")
                return
            db.set_bankroll(value)
        await message.answer(
            f"Банк: <b>{db.bankroll():,.0f}</b>\n"
            f"Максимальная ставка: {settings.max_stake_pct:.0%} банка = {db.bankroll() * settings.max_stake_pct:,.0f}"
            .replace(",", " ")
        )

    @owner.message(Command("limits"))
    async def cmd_limits(message: Message) -> None:
        st = db.risk_state()
        await message.answer(
            "<b>Лимиты финансиста</b>\n"
            f"Ставка: {settings.kelly_fraction:g} Келли, не больше {settings.max_stake_pct:.0%} банка\n"
            f"Ставок в день: {st.bets_today} из {settings.max_bets_per_day}\n"
            f"Дневной лимит убытка: {settings.daily_loss_limit_pct:.0%} банка (сегодня {st.profit_today:+,.0f})\n"
            f"Проигрышей подряд: {st.losing_streak} (пауза после {settings.stop_after_losses})\n"
            f"Минимальный перевес: {settings.min_edge:.0%}, коэффициенты {settings.min_odds}–{settings.max_odds}\n"
            f"Разборов агентами сегодня: {db.analyses_today()} из {settings.max_llm_analyses_per_day}"
            .replace(",", " ")
        )

    @owner.message(Command("resume"))
    async def cmd_resume(message: Message) -> None:
        db.reset_streak()
        await message.answer("Пауза снята. Серия проигрышей обнулена.")

    @owner.message(Command("usage"))
    async def cmd_usage(message: Message) -> None:
        u = db.llm_usage_since(30)
        p_in, p_out = PRICES.get(settings.llm_model_agents, PRICES["claude-opus-5-5"])
        cost = u["input_tokens"] / 1e6 * p_in + u["output_tokens"] / 1e6 * p_out + u["web_searches"] * WEB_SEARCH_PRICE
        await message.answer(
            "<b>Claude API за 30 дней</b>\n"
            f"Разборов матчей: {u['analyses']}\n"
            f"Токенов: вход {u['input_tokens']:,}, выход {u['output_tokens']:,}\n"
            f"Веб-поисков: {u['web_searches']}\n"
            f"Оценка расходов: ≈ ${cost:.2f}\n"
            f"Модели: главный {settings.llm_model_chief}, помощники {settings.llm_model_agents}\n"
            f"Кредитов The Odds API осталось: {engine.odds.requests_remaining or 'неизвестно'}"
            .replace(",", " ")
        )

    @owner.message(Command("settle"))
    async def cmd_settle(message: Message) -> None:
        await run_settle(message.bot, engine, db, message.chat.id, verbose=True)

    # Всем остальным: подсказка, как стать владельцем.
    public = Router(name="public")

    @public.message()
    async def stranger(message: Message) -> None:
        uid = message.from_user.id if message.from_user else 0
        if not settings.owner_id:
            await message.answer(
                f"Ваш Telegram ID: <code>{uid}</code>\n"
                "Впишите его в TELEGRAM_OWNER_ID в файле .env и перезапустите бота."
            )
        # чужим пользователям ничего не отвечаем

    return [owner, public]


async def run_scan(bot: Bot, engine: Engine, chat_id: int, verbose: bool = False) -> None:
    try:
        matches, results = await engine.scan()
    except Exception as exc:  # noqa: BLE001
        log.exception("scan failed")
        await bot.send_message(chat_id, f"Скан не удался: {type(exc).__name__}")
        return
    signals = [r for r in results if r.signal_id]
    for r in results:
        if r.signal_id or verbose:
            await send_long(bot, chat_id, fmt.analysis(r))
    cost = sum(r.cost_usd for r in results)
    summary = (
        f"Скан завершён: матчей в линии {len(matches)}, разобрано агентами {len(results)}, "
        f"сигналов {len(signals)}. Стоимость ≈ ${cost:.2f}."
    )
    if not results:
        summary += "\nМодель не нашла матчей с перевесом (или исчерпан дневной лимит разборов)."
    if signals or verbose:
        await bot.send_message(chat_id, summary + "\n\n" + fmt.DISCLAIMER)
    elif results:
        await bot.send_message(chat_id, summary)


async def run_settle(bot: Bot, engine: Engine, db: DB, chat_id: int, verbose: bool = False) -> None:
    try:
        settled = await engine.settle()
    except Exception as exc:  # noqa: BLE001
        log.exception("settle failed")
        await bot.send_message(chat_id, f"Расчёт не удался: {type(exc).__name__}")
        return
    if not settled:
        if verbose:
            await bot.send_message(chat_id, "Нет завершённых матчей для расчёта.")
        return
    text = "<b>Итоги сигналов</b>\n" + "\n".join(fmt.signal_line(s) for s in settled)
    text += f"\n\nБанк: {db.bankroll():,.0f}".replace(",", " ")
    await send_long(bot, chat_id, text)
