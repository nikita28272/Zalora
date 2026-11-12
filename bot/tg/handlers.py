"""Команды Telegram-бота."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from ..config import settings
from ..db import DB, month_start_utc
from ..pipeline import AnalysisResult, Engine
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


def decision_keyboard(signal_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✅ Поставил", callback_data=f"bet:{signal_id}:placed"),
        InlineKeyboardButton(text="❌ Не ставил", callback_data=f"bet:{signal_id}:skip"),
    ]])


async def send_long(bot: Bot, chat_id: int, text: str, keyboard: InlineKeyboardMarkup | None = None) -> None:
    parts = chunks(text)
    for i, part in enumerate(parts):
        markup = keyboard if i == len(parts) - 1 else None  # кнопки под последней частью
        await bot.send_message(chat_id, part, disable_web_page_preview=True, reply_markup=markup)


async def send_analysis(bot: Bot, chat_id: int, res: AnalysisResult) -> None:
    kb = decision_keyboard(res.signal_id) if res.signal_id else None
    await send_long(bot, chat_id, fmt.analysis(res), kb)


def parse_bet_input(text: str) -> tuple[float, float | None] | None:
    """«2.15», «2,15 500» или «2.15 1 000» -> (коэффициент, сумма | None)."""
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(.*)$", text.strip().replace(",", "."))
    if not m:
        return None
    odds = float(m.group(1))
    if not 1.01 <= odds <= 50:
        return None
    rest = re.sub(r"[^\d.]", "", m.group(2))
    try:
        stake = float(rest) if rest else None
    except ValueError:
        stake = None
    return odds, stake if stake and stake > 0 else None


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
        await send_analysis(message.bot, message.chat.id, res)

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
        await message.answer(fmt.stats(db.settled(since=month_start_utc()), db.settled(), db.bankroll()))

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
        cost = u["cost_usd"]
        avg = cost / u["analyses"] if u["analyses"] else 0
        await message.answer(
            "<b>Claude API за 30 дней</b>\n"
            f"Разборов матчей: {u['analyses']}\n"
            f"Токенов: вход {u['input_tokens']:,}, выход {u['output_tokens']:,}\n"
            f"Веб-поисков: {u['web_searches']}\n"
            f"Оценка расходов: ≈ ${cost:.2f} (в среднем ${avg:.2f} за разбор)\n"
            f"Модели: главный {settings.llm_model_chief}, помощники {settings.llm_model_agents}\n"
            f"Кредитов The Odds API осталось: {engine.odds.requests_remaining or 'неизвестно'}"
            .replace(",", " ")
        )

    @owner.message(Command("settle"))
    async def cmd_settle(message: Message) -> None:
        await run_settle(message.bot, engine, db, message.chat.id, verbose=True)

    # ---------- отметка «поставил / не ставил» ----------
    owner.callback_query.filter(lambda c: c.from_user is not None and c.from_user.id == settings.owner_id)

    @owner.callback_query(F.data.startswith("bet:"))
    async def on_decision(cb: CallbackQuery) -> None:
        _, sid, action = cb.data.split(":")
        sig = db.signal(int(sid))
        if sig is None:
            await cb.answer("Сигнал не найден", show_alert=True)
            return
        if sig.kickoff <= datetime.now(timezone.utc):
            await cb.answer("Матч уже начался: работаем только в прематче, отметку не принимаю.", show_alert=True)
            return
        if action == "skip":
            db.decide(sig.id, placed=False)
            db.set("awaiting_bet", None)
            await cb.answer("Записал: не ставили")
            await cb.message.answer(
                f"🧠 Главный: принял, по сигналу #{sig.id} вы не ставите. "
                "Финансист не трогает банк, аналитик всё равно проверит исход для статистики."
            )
            return
        db.set("awaiting_bet", sig.id)
        await cb.answer()
        await cb.message.answer(
            f"По какому коэффициенту поставили на #{sig.id}? Напишите число, например <code>{sig.min_odds:.2f}</code>.\n"
            f"Если сумма отличается от рекомендованной ({sig.stake:,.0f}), напишите её вторым числом: "
            f"<code>{sig.min_odds:.2f} 500</code>".replace(",", " ")
        )

    @owner.message(F.text & ~F.text.startswith("/"))
    async def on_bet_input(message: Message) -> None:
        sid = db.get("awaiting_bet")
        if not sid:
            await message.answer("Команды: /help")
            return
        sig = db.signal(int(sid))
        if sig is None:
            db.set("awaiting_bet", None)
            return
        if sig.kickoff <= datetime.now(timezone.utc):
            db.set("awaiting_bet", None)
            await message.answer("Матч уже начался: работаем только в прематче, отметку не принимаю.")
            return
        parsed = parse_bet_input(message.text or "")
        if parsed is None:
            await message.answer("Не понял. Напишите коэффициент числом, например 2.15 (или «2.15 500» с суммой).")
            return
        odds, stake = parsed
        db.decide(sig.id, placed=True, odds=odds, stake=stake or sig.stake)
        db.set("awaiting_bet", None)
        await message.answer(fmt.placed_confirmation(db.signal(sig.id), db.bankroll(), db.in_play()))

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
            await send_analysis(bot, chat_id, r)
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
    await send_long(bot, chat_id, fmt.settle_report(settled, db.bankroll()))
