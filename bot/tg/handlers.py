"""Команды Telegram-бота."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)

from ..analysis.bankroll import plan_bet
from ..config import RISK_NAMES_RU, settings
from ..db import DB, month_start_utc, week_start_utc
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


_SEL_ALIASES = {
    "home": "home", "п1": "home", "1": "home", "w1": "home",
    "draw": "draw", "x": "draw", "х": "draw", "н": "draw", "ничья": "draw",
    "away": "away", "п2": "away", "2": "away", "w2": "away",
    "over": "over", "тб": "over", "тб2.5": "over", "тб2,5": "over", "o2.5": "over",
    "under": "under", "тм": "under", "тм2.5": "under", "тм2,5": "under", "u2.5": "under",
    # хоккей
    "ml_home": "ml_home", "п1от": "ml_home", "п1_от": "ml_home", "п1(от)": "ml_home",
    "ml_away": "ml_away", "п2от": "ml_away", "п2_от": "ml_away", "п2(от)": "ml_away",
    "hcp_home": "hcp_home", "ф1": "hcp_home",
    "hcp_away": "hcp_away", "ф2": "hcp_away",
    # двойной шанс (футбол)
    "dc_home": "dc_home", "1x": "dc_home", "1х": "dc_home",
    "dc_away": "dc_away", "x2": "dc_away", "х2": "dc_away",
}
_CONF_ALIASES = {
    "low": "low", "низкая": "low", "medium": "medium", "средняя": "medium", "high": "high", "высокая": "high",
}
VERDICT_USAGE = (
    "Формат: <code>/verdict НОМЕР_ДОСЬЕ ИСХОД1 ВЕРОЯТНОСТЬ1 ИСХОД2 ВЕРОЯТНОСТЬ2 [УВЕРЕННОСТЬ]</code>\n"
    "Например: <code>/verdict 7 home 0.54 under 0.56 medium</code>\n"
    "Исход — код из досье: home/draw/away/over/under, в хоккее ещё ml_home/ml_away/hcp_home/hcp_away.\n"
    "Вероятность: 0.54 или 54%. Можно и один исход.\n"
    "Линия с запасом: <code>hcp_home@+12.5</code>, <code>under@225.5</code>; двойной шанс: <code>1x</code>, <code>x2</code>.\n"
    "🔥 в конце: <code>fire</code> — на первый исход, <code>fire2</code> — на оба."
)


def _parse_prob(raw: str) -> float | None:
    raw = raw.replace(",", ".")
    percent = raw.endswith("%")
    try:
        p = float(raw.rstrip("%"))
    except ValueError:
        return None
    if percent or p >= 2:
        p /= 100  # «54» или «54%»
    elif p > 1:
        return None  # «1.5» — непонятно, что имелось в виду
    return p if 0 < p < 1 else None


_FIRE = {"fire": 1, "🔥": 1, "огонь": 1, "fire2": 2, "🔥🔥": 2}


_LINK = {"link", "связка", "🔗"}


def parse_link(args: str) -> tuple[str, bool]:
    """Снимает из вердикта отметку 🔗 «один сценарий — вместе не брать»."""
    words = args.split()
    kept = [w for w in words if w.lower() not in _LINK]
    return " ".join(kept), len(kept) != len(words)


def parse_fire(args: str) -> tuple[str, int]:
    """Снимает из вердикта отметку 🔥: «fire» — на первый исход, «fire2» — на оба."""
    words, n = [], 0
    for w in args.split():
        if w.lower() in _FIRE:
            n = max(n, _FIRE[w.lower()])
        else:
            words.append(w)
    return " ".join(words), n


def parse_verdict(args: str) -> tuple[int, list[tuple[str, float]], str] | None:
    """«7 home 0.54 under 0.56 medium» -> (7, [("home", 0.54), ("under", 0.56)], "medium")."""
    parts = args.replace("#", " ").split()
    if len(parts) < 3 or not parts[0].isdigit():
        return None
    num, rest, conf = int(parts[0]), parts[1:], "medium"
    if len(rest) % 2 == 1:  # последним словом — уверенность
        conf = _CONF_ALIASES.get(rest[-1].lower())
        rest = rest[:-1]
        if conf is None:
            return None
    if not 2 <= len(rest) <= 4:
        return None
    picks = []
    for i in range(0, len(rest), 2):
        code, _, raw_line = rest[i].lower().partition("@")  # запасная линия: hcp_home@+12.5, under@225.5
        sel, p = _SEL_ALIASES.get(code), _parse_prob(rest[i + 1])
        if sel is None or p is None:
            return None
        if raw_line:
            try:
                picks.append((sel, p, float(raw_line.replace(",", "."))))
            except ValueError:
                return None
        else:
            picks.append((sel, p))
    return num, picks, conf


def picks_keyboard(signals, num: int | None = None) -> InlineKeyboardMarkup | None:
    """Под двумя исходами босса: по строке кнопок на каждый и кнопка поздней проверки."""
    rows = [
        [InlineKeyboardButton(text=f"✅ Сделал: {fmt.sig_short(s)}", callback_data=f"bet:{s.id}:placed"),
         InlineKeyboardButton(text="❌ Не ставил", callback_data=f"bet:{s.id}:skip")]
        for s in signals if s.user_decision == "pending"
    ]
    if rows and num is not None:
        rows.append([InlineKeyboardButton(text="🥅 Поздняя проверка", callback_data=f"late:{num}")])
    return InlineKeyboardMarkup(inline_keyboard=rows) if rows else None


def odds_question(sig, stake: float) -> str:
    example = sig.odds_ref or sig.min_odds
    return (f"Ставка {stake:,.0f} на <b>{fmt.escape(fmt.sig_short(sig))}</b> ({fmt.escape(sig.home)} — "
            f"{fmt.escape(sig.away)}). По какому кэфу поставили? Например <code>{example:.2f}</code>"
            ).replace(",", " ")


def clear_states(db: DB) -> None:
    for key in ("awaiting_bank", "awaiting_dossier_bet", "awaiting_stake", "awaiting_bet",
                "awaiting_amount", "awaiting_odds"):
        db.set(key, None)


def advice_keyboard(signal_id: int, stake: float) -> InlineKeyboardMarkup:
    """Под советом финансиста: поставил рекомендованную сумму / другую / не ставлю."""
    rows = []
    if stake > 0:
        rows.append([
            InlineKeyboardButton(text=f"✅ Поставил {stake:,.0f}".replace(",", " "),
                                 callback_data=f"fin:amt:{signal_id}:{stake:g}"),
            InlineKeyboardButton(text="✏️ Другая сумма", callback_data=f"fin:other:{signal_id}:0"),
        ])
    else:
        rows.append([InlineKeyboardButton(text="✏️ Всё равно поставил", callback_data=f"fin:other:{signal_id}:0")])
    rows.append([InlineKeyboardButton(text="❌ Не ставлю", callback_data=f"bet:{signal_id}:skip")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def plan_keyboard(signal_id: int, odds: float, stake: float) -> InlineKeyboardMarkup:
    """Под разбором финансиста."""
    rows = []
    if stake > 0:
        rows.append([
            InlineKeyboardButton(text=f"✅ Ставлю {stake:,.0f}".replace(",", " "),
                                 callback_data=f"fin:ok:{signal_id}:{odds:g}:{stake:g}"),
            InlineKeyboardButton(text="✏️ Своя сумма", callback_data=f"fin:own:{signal_id}:{odds:g}"),
        ])
    else:
        rows.append([InlineKeyboardButton(text="✏️ Всё равно ставлю", callback_data=f"fin:own:{signal_id}:{odds:g}")])
    rows.append([InlineKeyboardButton(text="❌ Не ставлю", callback_data=f"bet:{signal_id}:skip")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def signal_plan(db: DB, sig, odds: float):
    """Разбор финансиста для сигнала по кэфу владельца (от текущего банка и лимитов)."""
    by_analysis = sig.sport == "basketball" and settings.nba_mode == "analysis" and sig.source == "boss"
    plan = plan_bet(sig.p_final, odds, db.risk_state(), settings, by_analysis, sig.confidence,
                    match_in_play=db.match_in_play(sig.event_id, exclude_id=sig.id))
    partner = db.signal(sig.linked_to) if sig.linked_to else None
    if partner is not None and partner.placed:  # 🔗 второй исход того же сценария
        from dataclasses import replace
        plan = replace(plan, stake=0, pct=0, verdict="не ставить",
                       reason=f"🔗 вы уже стоите на этом сценарии: #{partner.id} {fmt.sig_short(partner)} — "
                              "вторая ставка удвоит риск на один исход матча", max_own=0)
    return plan


def check_discipline(db: DB, sig, plan, amount: float, advised: float | None = None) -> str:
    """Записывает, следует ли владелец совету, и возвращает предупреждение финансиста (или пусто).
    advised — сумма, которую финансист советовал до ставки (если была)."""
    st = db.risk_state()
    warn = []
    rec = advised if advised is not None else plan.stake
    if rec <= 0:
        warn.append("финансист советовал не ставить")
    elif amount > 1.5 * rec:
        warn.append(f"сумма {amount:,.0f} больше совета {rec:,.0f}".replace(",", " "))
    if amount > settings.max_stake_pct * st.bankroll:
        warn.append(f"больше {settings.max_stake_pct:.0%} банка на одну ставку — это уже очень рискованно")
    db.log_discipline(not warn, f"#{sig.id} {fmt.sig_short(sig)}: " + ", ".join(warn) if warn else "")
    if not warn:
        return ""
    text = "\n⚠️ Финансист: " + "; ".join(warn) + "."
    if st.losing_streak >= 2:
        text += " После проигрышей не отыгрывайтесь — это главный путь к потере банка."
    return text


def late_check_keyboard(num: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="🥅 Поздняя проверка", callback_data=f"late:{num}"),
    ]])


def dossier_keyboard(items: list[int]) -> InlineKeyboardMarkup:
    rows, row = [], []
    for num in items:
        row.append(InlineKeyboardButton(text=f"📋 Досье #{num}", callback_data=f"dossier:{num}"))
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return InlineKeyboardMarkup(inline_keyboard=rows)


# ---------- кнопки ----------
BTN_ANALYZE = "🔍 Анализ"
BTN_CANDIDATES = "📋 Кандидаты"  # кнопка старой версии меню — продолжает работать
BTN_MATCHES = "📅 Матчи"
BTN_SIGNALS = "📌 Прогнозы и ставки"
BTN_SIGNALS_OLD = "📌 Мои сигналы"
BTN_STATS = "📊 Статистика"
BTN_BANK = "💰 Банк"
BTN_CREDITS = "💳 Кредиты"
BTN_ACCURACY = "🎯 Точность босса"
BTN_TYPES = "📂 Типы ставок"
BTN_MORE = "⚙️ Ещё"


def main_keyboard() -> ReplyKeyboardMarkup:
    """Постоянное меню внизу экрана вместо обычной клавиатуры."""
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=BTN_MATCHES), KeyboardButton(text=BTN_ANALYZE)],
            [KeyboardButton(text=BTN_SIGNALS), KeyboardButton(text=BTN_STATS)],
            [KeyboardButton(text=BTN_ACCURACY), KeyboardButton(text=BTN_TYPES)],
            [KeyboardButton(text=BTN_BANK), KeyboardButton(text=BTN_CREDITS)],
            [KeyboardButton(text=BTN_MORE)],
        ],
        resize_keyboard=True,
        is_persistent=False,  # меню можно свернуть значком в поле ввода
        input_field_placeholder="Выберите действие или вставьте /verdict …",
    )


def _quick_row(top=None, nba=None) -> list[InlineKeyboardButton]:
    """Кнопки быстрых досье: ⭐ — матчи с возможным перевесом, 🏀 — НБА по анализу."""
    row = [InlineKeyboardButton(text=f"⭐ {m.idx}", callback_data=f"pick:{m.event.event_id}"[:64])
           for m, _, _ in (top or [])[:5]]
    row += [InlineKeyboardButton(text=f"🏀 {m.idx}", callback_data=f"pick:{m.event.event_id}"[:64])
            for m in (nba or [])[:3]]
    return row[:8]


def daily_keyboard(picks, total: int, report=None) -> InlineKeyboardMarkup | None:
    """Кнопки досье только для матчей дня и «📃 Все матчи»."""
    rows, row = [], []
    for m, _ in picks:
        row.append(InlineKeyboardButton(text=f"📋 {m.idx}", callback_data=f"pick:{m.event.event_id}"[:64]))
        if len(row) == 4:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    if total > len(picks):
        rows.append([InlineKeyboardButton(text=f"📃 Все матчи ({total})", callback_data="menu:all")])
    force = force_row(report)
    if force:
        rows.append(force)
    return InlineKeyboardMarkup(inline_keyboard=rows) if rows else None


async def send_daily(bot: Bot, chat_id: int, engine: Engine, matches, title: str = "") -> None:
    """Матчи дня (2 хоккея, 2 НБА, 2 футбола) с кнопками; полный список — по кнопке."""
    picks = engine.daily_picks(matches)
    text = (title + "\n\n" if title else "") + fmt.daily_block(picks, len(matches))
    hot = fmt.hot_block(engine.db.hot_open())
    if hot:
        text = hot + "\n\n" + text
    extra = fmt.schedule_only_block(engine.last_report)
    if extra:
        text += "\n\n" + extra
    text += "\n\n" + fmt.finance_status(engine.db.risk_state(), settings, short=True)
    text += "\n" + fmt.odds_footer(engine.last_report, engine.quota())
    await send_long(bot, chat_id, text, daily_keyboard(picks, len(matches), engine.last_report))


def nba_in_play(matches, report) -> bool:
    """Есть ли сегодня НБА: с линиями или только в расписании (линии не грузили — нечего разбирать)."""
    return any(m.sport.key == "basketball" for m in matches) or bool(
        report and any(name == "НБА" for name, *_ in report.schedule_only))


def force_row(report) -> list[InlineKeyboardButton]:
    """Кнопка «загрузить отложенные лиги», если бот отложил их ради дневного бюджета."""
    if report is None or report.cached or not report.budget_skipped:
        return []
    return [InlineKeyboardButton(text="⏩ Загрузить отложенные: " + ", ".join(report.budget_skipped)[:40],
                                 callback_data="odds:force")]


def with_force(markup: InlineKeyboardMarkup | None, report) -> InlineKeyboardMarkup | None:
    row = force_row(report)
    if not row:
        return markup
    rows = list(markup.inline_keyboard) if markup else []
    return InlineKeyboardMarkup(inline_keyboard=rows + [row])


def analyze_button(top=None, nba=None) -> InlineKeyboardMarkup:
    """Под утренним списком: «🔍 Анализ» и сразу досье отобранных матчей одним нажатием."""
    rows = [[InlineKeyboardButton(text="🔍 Анализ", callback_data="menu:analyze")]]
    quick = _quick_row(top, nba)
    if quick:
        rows.append(quick)
    return InlineKeyboardMarkup(inline_keyboard=rows)


def outcome_keyboard(ctx, num: int, highlight: set[str] | None = None) -> InlineKeyboardMarkup:
    """На какой исход поставили: кнопки по рынкам вида спорта, по рынку в строке."""
    highlight = highlight or set()
    rows = []
    for group in ctx.sport.groups:
        row = [
            InlineKeyboardButton(text=("⭐ " if sel in highlight else "") + ctx.short(sel),
                                 callback_data=f"dsel:{num}:{sel}")
            for sel in group if sel in ctx.market.fair
        ]
        if row:
            rows.append(row)
    return InlineKeyboardMarkup(inline_keyboard=rows)


def more_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🗂 История", callback_data="menu:history"),
         InlineKeyboardButton(text="🛡 Лимиты", callback_data="menu:limits")],
        [InlineKeyboardButton(text="🔄 Проверить итоги", callback_data="menu:settle"),
         InlineKeyboardButton(text="▶️ Снять паузу", callback_data="menu:resume")],
        [InlineKeyboardButton(text="🤖 Расход Claude", callback_data="menu:usage"),
         InlineKeyboardButton(text="❓ Помощь", callback_data="menu:help")],
    ])


def matches_keyboard(matches, top=None, nba=None) -> InlineKeyboardMarkup | None:
    """Кнопки с номерами матчей: нажал — получил досье (или разбор в режиме api).
    Первой строкой — ⭐ матчи, которые стоит разобрать."""
    if not matches:
        return None
    rows, row = [], []
    quick = _quick_row(top, nba)
    if quick:
        rows.append(quick)
    for m in matches[:30]:
        row.append(InlineKeyboardButton(text=f"📋 {m.idx}", callback_data=f"pick:{m.event.event_id}"[:64]))
        if len(row) == 5:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return InlineKeyboardMarkup(inline_keyboard=rows)


def signals_keyboard(pending) -> InlineKeyboardMarkup | None:
    """У неотмеченных сигналов (до начала матча) — «Сделал / Не ставил»; внизу — «🔄 Обновить»."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    rows = [
        [InlineKeyboardButton(text=f"✅ Сделал #{s.id} {fmt.sig_short(s)}", callback_data=f"bet:{s.id}:placed"),
         InlineKeyboardButton(text=f"❌ Не ставил #{s.id}", callback_data=f"bet:{s.id}:skip")]
        for s in sorted(pending, key=lambda x: x.kickoff) if s.user_decision == "pending" and s.kickoff > now
    ]
    if pending:
        rows.append([InlineKeyboardButton(text="🔄 Обновить", callback_data="board:refresh")])
    return InlineKeyboardMarkup(inline_keyboard=rows) if rows else None


def bank_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✏️ Изменить банк", callback_data="bank:edit"),
    ]])


def _without(markup: InlineKeyboardMarkup | None, signal_id: int) -> InlineKeyboardMarkup | None:
    if markup is None:
        return None
    rows = [r for r in markup.inline_keyboard
            if not any((b.callback_data or "").startswith(f"bet:{signal_id}:") for b in r)]
    if not any((b.callback_data or "").startswith("bet:") for r in rows for b in r):
        return None  # все исходы отмечены — поздняя проверка уже не нужна
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def _drop_row(cb: CallbackQuery, signal_id: int) -> None:
    """Убрать из сообщения кнопки исхода, по которому уже отметились."""
    if not cb.message or not cb.message.reply_markup:
        return
    try:
        await cb.message.edit_reply_markup(reply_markup=_without(cb.message.reply_markup, signal_id))
    except Exception as exc:  # noqa: BLE001 — сообщение старое или уже изменено
        log.info("edit markup: %s", type(exc).__name__)


def build_routers(engine: Engine, db: DB) -> list[Router]:
    owner = Router(name="owner")
    owner.message.filter(lambda m: m.from_user is not None and m.from_user.id == settings.owner_id)

    @owner.message(Command("start", "help"))
    async def cmd_help(message: Message) -> None:
        await message.answer(fmt.help_text(), reply_markup=main_keyboard())

    @owner.message(Command("matches"))
    async def cmd_matches(message: Message) -> None:
        await message.answer("Загружаю линии и статистику (первый раз за день — до 2–3 минут)…")
        try:
            matches = await engine.load_matches()
        except Exception as exc:  # noqa: BLE001
            log.exception("load_matches failed")
            await message.answer(f"Не удалось загрузить матчи: {type(exc).__name__}")
            return
        await send_daily(message.bot, message.chat.id, engine, matches)

    async def show_all(message: Message) -> None:
        """«📃 Все матчи»: полный список с кнопками всех матчей."""
        matches = engine.matches
        if not matches:
            await message.answer("Матчей нет. Нажмите «📅 Матчи».")
            return
        await send_long(message.bot, message.chat.id,
                        fmt.matches_list(matches, report=engine.last_report) + "\n\nНажмите номер матча, чтобы получить досье.",
                        matches_keyboard(matches))

    async def pick_match(message: Message, idx: int) -> None:
        """Досье матча (режим «босс в чате») или полный разбор агентами (режим api)."""
        if not settings.api_mode:
            try:
                num, ctx = await engine.dossier_by_index(idx)
            except ValueError as exc:
                await message.answer(str(exc))
                return
            await message.answer(fmt.dossier_message(engine.dossier(ctx, num)))
            await message.answer(fmt.DOSSIER_HINT)
            return
        await message.answer("Агенты работают: новости, эксперты, адвокат дьявола, главный аналитик. Это 1–3 минуты…")
        try:
            res = await engine.analyze_by_index(idx)
        except ValueError as exc:
            await message.answer(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            log.exception("analyze failed")
            await message.answer(f"Ошибка разбора: {type(exc).__name__}")
            return
        await send_analysis(message.bot, message.chat.id, res)

    @owner.message(Command("analyze"))
    async def cmd_analyze(message: Message, command: CommandObject) -> None:
        if not command.args or not command.args.strip().isdigit():
            await message.answer("Нажмите «📅 Матчи» и выберите номер матча кнопкой (или: /analyze 3).")
            return
        await pick_match(message, int(command.args.strip()))

    @owner.message(Command("scan"))
    async def cmd_scan(message: Message) -> None:
        if not settings.api_mode:
            await message.answer("Ищу кандидатов дня…")
            await run_candidates(message.bot, engine, message.chat.id)
            return
        await message.answer("Ищу матчи с перевесом и запускаю агентов…")
        await run_scan(message.bot, engine, message.chat.id, verbose=True)

    @owner.message(Command("verdict"))
    async def cmd_verdict(message: Message, command: CommandObject) -> None:
        args, link = parse_link(command.args or "")
        args, fire = parse_fire(args)
        parsed = parse_verdict(args)
        if parsed is None:
            await message.answer(VERDICT_USAGE)
            return
        num, picks, conf = parsed
        try:
            top = engine.manual_top(num, picks, conf, hot={pk[0] for pk in picks[:fire]}, link=link)
        except ValueError as exc:
            await message.answer(str(exc))
            return
        await message.answer(fmt.top_picks(top, db.bankroll()),
                             reply_markup=picks_keyboard([db.signal(p.signal_id) for p in top.picks], num))

    async def send_late_check(message: Message, num: int) -> None:
        try:
            text = engine.late_check(num)
        except ValueError as exc:
            await message.answer(str(exc))
            return
        await message.answer(fmt.dossier_message(text))
        await message.answer(fmt.LATE_HINT)

    @owner.message(Command("check"))
    async def cmd_check(message: Message, command: CommandObject) -> None:
        if not command.args or not command.args.strip().lstrip("#").isdigit():
            await message.answer("Формат: <code>/check НОМЕР_ДОСЬЕ</code>, например <code>/check 7</code>.")
            return
        await send_late_check(message, int(command.args.strip().lstrip("#")))

    @owner.message(Command("signals"))
    async def cmd_signals(message: Message) -> None:
        pending = db.pending()
        if not pending:
            await message.answer("📌 Открытых прогнозов и ставок нет. Сыгранные — в «📊 Статистика».")
            return
        await send_long(message.bot, message.chat.id, fmt.signals_board(pending), signals_keyboard(pending))

    @owner.callback_query(F.data == "board:refresh")
    async def on_board_refresh(cb: CallbackQuery) -> None:
        pending = db.pending()
        text = fmt.signals_board(pending) if pending else "📌 Открытых прогнозов и ставок нет."
        await cb.answer("Обновлено")
        try:
            if len(text) <= 4000:
                await cb.message.edit_text(text, reply_markup=signals_keyboard(pending),
                                           disable_web_page_preview=True)
                return
        except Exception:  # noqa: BLE001 — текст не изменился или сообщение слишком старое
            return
        await send_long(cb.bot, cb.message.chat.id, text, signals_keyboard(pending))

    @owner.message(Command("fire"))
    async def cmd_fire(message: Message, command: CommandObject) -> None:
        raw = (command.args or "").strip().lstrip("#")
        if not raw.isdigit():
            await message.answer("Формат: <code>/fire 9</code> — поставить или снять 🔥 у прогноза #9 "
                                 "(номер из «📌 Прогнозы и ставки»).")
            return
        sig = db.signal(int(raw))
        if sig is None or sig.status != "pending":
            await message.answer("Открытого прогноза с таким номером нет.")
            return
        db.set_hot(sig.id, not sig.hot)
        await message.answer(("🔥 Отметил" if not sig.hot else "Снял 🔥 с")
                             + f" #{sig.id} {fmt.escape(fmt.sig_short(sig))} "
                               f"({fmt.escape(sig.home)} — {fmt.escape(sig.away)}).")

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
        await message.answer(fmt.stats(db.settled(since=month_start_utc()), db.settled(), db.bankroll(),
                                       week=db.settled(since=week_start_utc())))

    @owner.message(Command("accuracy"))
    async def cmd_accuracy(message: Message) -> None:
        await send_long(message.bot, message.chat.id, fmt.accuracy(db.settled()))

    @owner.message(Command("types"))
    async def cmd_types(message: Message) -> None:
        await send_long(message.bot, message.chat.id, fmt.bet_types(db.settled()))

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
        await show_bank(message)

    async def show_bank(message: Message) -> None:
        await message.answer(fmt.finance_status(db.risk_state(), settings, db.discipline()),
                             reply_markup=bank_keyboard())

    @owner.message(Command("limits"))
    async def cmd_limits(message: Message) -> None:
        st = db.risk_state()
        text = (
            "<b>Лимиты финансиста</b>\n"
            f"Профиль риска: {RISK_NAMES_RU.get(settings.risk_profile, settings.risk_profile)} — "
            f"{settings.kelly_fraction:g} Келли от свободного банка, предохранитель {settings.max_stake_pct:.0%} на ставку\n"
            f"Ставок сегодня: {st.bets_today}"
            + (f" из {settings.max_bets_per_day}\n" if settings.max_bets_per_day else " (без лимита)\n")
            + f"Дневной лимит убытка: {settings.daily_loss_limit_pct:.0%} банка (сегодня {st.profit_today:+,.0f})\n"
            f"Проигрышей подряд: {st.losing_streak} (пауза после {settings.stop_after_losses})\n"
            f"Минимальный перевес: {settings.min_edge:.0%}, коэффициенты {settings.min_odds}–{settings.max_odds}\n"
        )
        if settings.api_mode:
            text += f"Разборов агентами сегодня: {db.analyses_today()} из {settings.max_llm_analyses_per_day}"
        else:
            text += "Режим: босс в чате"
        await message.answer(text.replace(",", " "))

    @owner.message(Command("resume"))
    async def cmd_resume(message: Message) -> None:
        from ..analysis.bankroll import risk_mode

        st = db.risk_state()
        db.reset_streak()
        db.reset_week()
        text = "Пауза снята: серия проигрышей и недельный счёт обнулены."
        if risk_mode(st, settings).stop:
            db.reset_peak()
            text += (f"\n💰 Финансист: банк {st.bankroll:,.0f} — новая точка отсчёта просадки. "
                     "Начинаем осторожно: если снова просядем, режим защиты включится раньше.").replace(",", " ")
        await message.answer(text)

    @owner.message(Command("usage"))
    async def cmd_usage(message: Message) -> None:
        if not settings.api_mode:
            await message.answer(
                "Режим «босс в чате»: Claude API не используется, бот ничего не тратит на нейросеть.\n"
                + fmt.quota_line(engine.quota())
            )
            return
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
            .replace(",", " ")
            + fmt.quota_line(engine.quota())
        )

    @owner.message(Command("credits"))
    async def cmd_credits(message: Message) -> None:
        try:
            q = await engine.refresh_quota()
        except Exception as exc:  # noqa: BLE001
            log.warning("quota check failed: %s", type(exc).__name__)
            await message.answer("Не удалось связаться с The Odds API. Последние известные данные:\n"
                                 + fmt.quota_line(engine.quota()))
            return
        await message.answer(fmt.quota_line(q) + "\n<i>Проверка остатка кредиты не тратит.</i>")

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
            await _drop_row(cb, sig.id)
            await cb.answer("Записал: не ставили")
            await cb.message.answer(
                f"🧠 Главный: принял, на {fmt.escape(fmt.sig_short(sig))} ({fmt.escape(sig.home)} — "
                f"{fmt.escape(sig.away)}) вы не ставите. Банк не меняется, итог прогноза попадёт в статистику 📊."
            )
            return
        clear_states(db)
        await _drop_row(cb, sig.id)  # если передумаете — кнопки есть в «📌 Прогнозы и ставки»
        await cb.answer()
        # шаг 1: финансист сразу говорит, сколько ставить (по кэфу из досье и текущему банку)
        plan = signal_plan(db, sig, sig.odds_ref or sig.min_odds)
        await cb.message.answer(fmt.bet_advice(sig, plan, db.risk_state()),
                                reply_markup=advice_keyboard(sig.id, plan.stake))

    @owner.callback_query(F.data.startswith("dossier:"))
    async def on_dossier(cb: CallbackQuery) -> None:
        num = int(cb.data.split(":")[1])
        ctx = engine.load_candidate(num)
        if ctx is None:
            await cb.answer("Досье устарело. Запросите новое: /scan", show_alert=True)
            return
        await cb.answer()
        await cb.message.answer(fmt.dossier_message(engine.dossier(ctx, num)))
        await cb.message.answer(fmt.DOSSIER_HINT)

    # ---------- кнопки постоянного меню ----------
    async def show_analyze(message: Message) -> None:
        """«🔍 Анализ»: какой матч разобрать — кнопки с номерами."""
        if not engine.matches:
            await message.answer("Загружаю матчи…")
            try:
                await engine.load_matches()
            except Exception as exc:  # noqa: BLE001
                log.exception("load_matches failed")
                await message.answer(f"Не удалось загрузить матчи: {type(exc).__name__}")
                return
        if not engine.matches:
            await message.answer("Ближайших матчей нет.")
            return
        await send_daily(message.bot, message.chat.id, engine, engine.matches, "🔍 <b>Какой матч разобрать?</b>")

    @owner.message(F.text == BTN_ANALYZE)
    async def b_analyze(message: Message) -> None:
        await show_analyze(message)

    @owner.message(F.text == BTN_CANDIDATES)
    async def b_candidates(message: Message) -> None:
        await show_analyze(message)

    @owner.message(F.text == BTN_MATCHES)
    async def b_matches(message: Message) -> None:
        await cmd_matches(message)

    @owner.message(F.text.in_({BTN_SIGNALS, BTN_SIGNALS_OLD}))
    async def b_signals(message: Message) -> None:
        await cmd_signals(message)

    @owner.message(F.text == BTN_STATS)
    async def b_stats(message: Message) -> None:
        await cmd_stats(message)

    @owner.message(F.text == BTN_ACCURACY)
    async def b_accuracy(message: Message) -> None:
        await cmd_accuracy(message)

    @owner.message(F.text == BTN_TYPES)
    async def b_types(message: Message) -> None:
        await cmd_types(message)

    @owner.message(F.text == BTN_BANK)
    async def b_bank(message: Message) -> None:
        await show_bank(message)

    @owner.message(F.text == BTN_CREDITS)
    async def b_credits(message: Message) -> None:
        await cmd_credits(message)

    @owner.message(F.text == BTN_MORE)
    async def b_more(message: Message) -> None:
        await message.answer("Что открыть?", reply_markup=more_keyboard())

    @owner.callback_query(F.data.startswith("menu:"))
    async def on_menu(cb: CallbackQuery) -> None:
        action = cb.data.split(":", 1)[1]
        await cb.answer()
        handlers = {
            "history": cmd_history, "limits": cmd_limits, "settle": cmd_settle,
            "resume": cmd_resume, "usage": cmd_usage, "help": cmd_help, "analyze": show_analyze,
            "all": show_all,
        }
        handler = handlers.get(action)
        if handler:
            await handler(cb.message)

    @owner.callback_query(F.data.startswith("pick:"))
    async def on_pick(cb: CallbackQuery) -> None:
        event_id = cb.data.split(":", 1)[1]
        ctx = next((m for m in engine.matches if m.event.event_id.startswith(event_id)), None)
        if ctx is None:
            await cb.answer("Список матчей обновился. Нажмите «📅 Матчи» ещё раз.", show_alert=True)
            return
        await cb.answer("Готовлю…")
        await pick_match(cb.message, ctx.idx)

    @owner.callback_query(F.data.startswith("dbet:"))
    async def on_dossier_bet(cb: CallbackQuery) -> None:
        _, num_s, action = cb.data.split(":")
        num = int(num_s)
        if action == "skip":
            try:
                ctx = engine.skip_dossier(num)
            except ValueError as exc:
                await cb.answer(str(exc), show_alert=True)
                return
            db.set("awaiting_dossier_bet", None)
            await cb.answer("Записал: не ставили")
            await cb.message.answer(
                f"🧠 Принял: на матч {fmt.escape(ctx.title)} вы не ставите. Если по нему есть мой прогноз "
                "(/verdict), он попадёт в статистику на бумаге, банк не меняется."
            )
            return
        ctx = engine.load_candidate(num)
        if ctx is None or ctx.event.kickoff <= fmt.now_utc():
            await cb.answer("Матч уже начался или досье устарело: отметку не принимаю.", show_alert=True)
            return
        boss = {s.selection for s in db.pending() if s.event_id == ctx.event.event_id and s.source == "boss"}
        await cb.answer()
        await cb.message.answer(
            f"На какой исход поставили? {fmt.escape(ctx.title)}"
            + ("\n⭐ — прогноз босса" if boss else ""),
            reply_markup=outcome_keyboard(ctx, num, boss),
        )

    @owner.callback_query(F.data.startswith("dsel:"))
    async def on_dossier_selection(cb: CallbackQuery) -> None:
        _, num_s, sel = cb.data.split(":")
        num = int(num_s)
        ctx = engine.load_candidate(num)
        if ctx is None or ctx.event.kickoff <= fmt.now_utc():
            await cb.answer("Матч уже начался или досье устарело: отметку не принимаю.", show_alert=True)
            return
        db.set("awaiting_bet", None)
        db.set("awaiting_bank", None)
        db.set("awaiting_dossier_bet", {"num": num, "sel": sel})
        boss = db.find_signal(ctx.event.event_id, sel, "boss")
        default = boss.stake if boss else db.bankroll() * 0.01
        example = f"{boss.min_odds:.2f}" if boss else f"{ctx.market.avg_odds.get(sel, 2.0):.2f}"
        await cb.answer()
        await cb.message.answer(
            f"<b>{fmt.escape(ctx.short(sel))}</b>. По какому коэффициенту и на какую сумму поставили?\n"
            f"Например: <code>{example} 500</code>. Если сумму не написать, запишу {default:,.0f}."
            .replace(",", " ")
        )

    @owner.callback_query(F.data.startswith("late:"))
    async def on_late(cb: CallbackQuery) -> None:
        await cb.answer()
        await send_late_check(cb.message, int(cb.data.split(":")[1]))

    @owner.callback_query(F.data.startswith("fin:"))
    async def on_fin(cb: CallbackQuery) -> None:
        parts = cb.data.split(":")
        action, sid, odds = parts[1], int(parts[2]), float(parts[3])
        sig = db.signal(sid)
        if sig is None or sig.kickoff <= datetime.now(timezone.utc):
            await cb.answer("Матч уже начался: работаем только в прематче.", show_alert=True)
            return
        if action in ("amt", "other"):
            advised = signal_plan(db, sig, sig.odds_ref or sig.min_odds).stake
            clear_states(db)
            await cb.answer()
            try:
                await cb.message.edit_reply_markup(reply_markup=None)
            except Exception as exc:  # noqa: BLE001
                log.info("edit markup: %s", type(exc).__name__)
            if action == "amt":  # шаг 2: сумма известна — спрашиваем кэф
                db.set("awaiting_odds", {"sid": sid, "stake": odds, "advised": advised})
                await cb.message.answer(odds_question(sig, odds))
            else:
                db.set("awaiting_amount", {"sid": sid, "advised": advised})
                await cb.message.answer(f"Сколько поставили на {fmt.escape(fmt.sig_short(sig))}? "
                                        "Напишите сумму числом, например 300.")
            return
        if action == "own":
            db.set("awaiting_bet", None)
            db.set("awaiting_bank", None)
            db.set("awaiting_stake", {"sid": sid, "odds": odds})
            await cb.answer()
            await cb.message.answer(f"Какую сумму ставите на {fmt.escape(fmt.sig_short(sig))} по {odds:.2f}? "
                                    "Напишите число, например 300.")
            return
        stake = float(parts[4])
        db.log_discipline(True)
        db.decide(sid, placed=True, odds=odds, stake=stake)
        try:
            await cb.message.edit_reply_markup(reply_markup=None)
        except Exception as exc:  # noqa: BLE001
            log.info("edit markup: %s", type(exc).__name__)
        await cb.answer("Записал")
        await cb.message.answer(fmt.placed_confirmation(db.signal(sid), db.bankroll(), db.in_play()))

    @owner.callback_query(F.data == "odds:force")
    async def on_force(cb: CallbackQuery) -> None:
        await cb.answer("Загружаю отложенные лиги…")
        try:
            matches = await engine.load_matches(force=True)
        except Exception as exc:  # noqa: BLE001
            log.exception("forced load failed")
            await cb.message.answer(f"Не удалось загрузить: {type(exc).__name__}")
            return
        await send_daily(cb.message.bot, cb.message.chat.id, engine, matches)

    @owner.callback_query(F.data == "bank:edit")
    async def on_bank_edit(cb: CallbackQuery) -> None:
        db.set("awaiting_bet", None)
        db.set("awaiting_stake", None)
        db.set("awaiting_dossier_bet", None)
        db.set("awaiting_bank", True)
        await cb.answer()
        await cb.message.answer("Напишите новый банк числом, например <code>15000</code>.")

    @owner.message(F.text & ~F.text.startswith("/"))
    async def on_bet_input(message: Message) -> None:
        amount_state = db.get("awaiting_amount")
        if amount_state:
            raw = (message.text or "").replace(" ", "").replace(",", ".")
            try:
                amount = float(raw)
                if amount <= 0:
                    raise ValueError
            except ValueError:
                await message.answer("Не понял. Напишите сумму числом, например 300.")
                return
            sig = db.signal(int(amount_state["sid"]))
            db.set("awaiting_amount", None)
            if sig is None or sig.kickoff <= datetime.now(timezone.utc):
                await message.answer("Матч уже начался: работаем только в прематче, отметку не принимаю.")
                return
            db.set("awaiting_odds", {"sid": sig.id, "stake": amount, "advised": amount_state["advised"]})
            await message.answer(odds_question(sig, amount))
            return
        odds_state = db.get("awaiting_odds")
        if odds_state:
            parsed = parse_bet_input(message.text or "")
            if parsed is None:
                await message.answer("Не понял. Напишите кэф числом, например 2.15.")
                return
            odds, override = parsed
            sig = db.signal(int(odds_state["sid"]))
            db.set("awaiting_odds", None)
            if sig is None or sig.kickoff <= datetime.now(timezone.utc):
                await message.answer("Матч уже начался: работаем только в прематче, отметку не принимаю.")
                return
            stake = override or float(odds_state["stake"])
            before = db.risk_state()
            plan = signal_plan(db, sig, odds)  # разбор с вашим реальным кэфом
            warn = check_discipline(db, sig, plan, stake, advised=float(odds_state.get("advised") or 0))
            db.decide(sig.id, placed=True, odds=odds, stake=stake)
            await message.answer(fmt.bet_review(db.signal(sig.id), plan, before, settings) + warn)
            return
        pending_bet = db.get("awaiting_dossier_bet")
        if pending_bet:
            parsed = parse_bet_input(message.text or "")
            if parsed is None:
                await message.answer("Не понял. Напишите коэффициент и сумму, например 2.15 500.")
                return
            odds, stake = parsed
            try:
                sig = engine.record_user_bet(pending_bet["num"], pending_bet["sel"], odds, stake)
            except ValueError as exc:
                db.set("awaiting_dossier_bet", None)
                await message.answer(str(exc))
                return
            db.set("awaiting_dossier_bet", None)
            await message.answer(fmt.placed_confirmation(sig, db.bankroll(), db.in_play()))
            return
        if db.get("awaiting_bank"):
            raw = (message.text or "").replace(" ", "").replace(",", ".")
            try:
                value = float(raw)
                if value < 0:
                    raise ValueError
            except ValueError:
                await message.answer("Не понял. Напишите банк числом, например 15000.")
                return
            db.set("awaiting_bank", None)
            db.set_bankroll(value)
            await show_bank(message)
            return
        own = db.get("awaiting_stake")
        if own:
            raw = (message.text or "").replace(" ", "").replace(",", ".")
            try:
                amount = float(raw)
                if amount <= 0:
                    raise ValueError
            except ValueError:
                await message.answer("Не понял. Напишите сумму числом, например 300.")
                return
            db.set("awaiting_stake", None)
            sig = db.signal(int(own["sid"]))
            if sig is None or sig.kickoff <= datetime.now(timezone.utc):
                await message.answer("Матч уже начался: работаем только в прематче, отметку не принимаю.")
                return
            plan = signal_plan(db, sig, float(own["odds"]))
            warn = check_discipline(db, sig, plan, amount)
            db.decide(sig.id, placed=True, odds=float(own["odds"]), stake=amount)
            await message.answer(fmt.placed_confirmation(db.signal(sig.id), db.bankroll(), db.in_play()) + warn)
            return
        sid = db.get("awaiting_bet")
        if not sid:
            await message.answer("Пользуйтесь кнопками меню внизу 👇", reply_markup=main_keyboard())
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
        db.set("awaiting_bet", None)
        if stake is None:  # только кэф — сначала разбор финансиста
            plan = signal_plan(db, sig, odds)
            await message.answer(fmt.bet_plan(sig, plan, db.bankroll()),
                                 reply_markup=plan_keyboard(sig.id, odds, plan.stake))
            return
        warn = check_discipline(db, sig, signal_plan(db, sig, odds), stake)
        db.decide(sig.id, placed=True, odds=odds, stake=stake)
        await message.answer(fmt.placed_confirmation(db.signal(sig.id), db.bankroll(), db.in_play()) + warn)

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
    from ..analysis.bankroll import risk_mode

    before = risk_mode(db.risk_state(), settings)
    bank_before = db.bankroll()
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
    st = db.risk_state()
    after = risk_mode(st, settings)
    text = fmt.settle_report(settled, db.bankroll()) + "\n\n" + fmt.bank_change(bank_before, st)
    if after.multiplier != before.multiplier:
        worse = after.multiplier < before.multiplier
        text += ("\n" + ("⚠️" if worse else "✅") + f" Финансист: режим меняется на «{fmt.escape(after.label)}» — "
                 + ("ставки остановлены." if after.multiplier == 0 else f"ставки ×{after.multiplier:g}.")
                 + (" Не отыгрывайтесь: размер ставок сам вернётся, когда банк восстановится." if worse else ""))
    await send_long(bot, chat_id, text)
    pending = db.pending()
    if pending:  # после итогов — свежий список: сыгранные ушли, остались открытые
        await send_long(bot, chat_id, fmt.signals_board(pending), signals_keyboard(pending))


async def run_candidates(bot: Bot, engine: Engine, chat_id: int) -> None:
    """Режим «босс в чате»: кандидаты дня с кнопками досье."""
    try:
        matches, items = await engine.candidates()
    except Exception as exc:  # noqa: BLE001
        log.exception("candidates failed")
        await bot.send_message(chat_id, f"Не удалось подобрать кандидатов: {type(exc).__name__}")
        return
    kb = dossier_keyboard([num for num, _ in items]) if items else None
    text = fmt.candidates_message(items, len(matches)) + "\n\n" + fmt.odds_footer(engine.last_report, engine.quota())
    await bot.send_message(chat_id, text, reply_markup=kb)


async def run_morning_matches(bot: Bot, engine: Engine, chat_id: int) -> None:
    """SCAN_TIME: матчи дня к разбору (2 хоккея, 2 НБА, 2 футбола, если есть) и кнопка «📃 Все матчи»."""
    try:
        # линии, загруженные сегодня днём, переиспользуем — кредиты не тратятся
        matches = await engine.load_matches(max_age_minutes=engine.daily_max_age())
    except Exception as exc:  # noqa: BLE001
        log.exception("morning matches failed")
        await bot.send_message(chat_id, f"Не удалось загрузить матчи: {type(exc).__name__}")
        return
    await send_daily(bot, chat_id, engine, matches, "☀️ <b>Доброго дня! Матчи на сегодня</b>")


async def run_daily(bot: Bot, engine: Engine, chat_id: int) -> None:
    if settings.api_mode:
        await run_scan(bot, engine, chat_id)
    else:
        await run_morning_matches(bot, engine, chat_id)


def _quiet_now(now_local: datetime) -> bool:
    """QUIET_HOURS «0-7»: с 00:00 до 07:00 напоминаний нет (бывает и «23-7»)."""
    raw = settings.quiet_hours.strip()
    if not raw:
        return False
    try:
        start, end = (int(x) for x in raw.split("-"))
    except ValueError:
        return False
    h = now_local.hour
    return start <= h < end if start <= end else (h >= start or h < end)


async def run_late_checks(bot: Bot, engine: Engine, chat_id: int) -> None:
    """Каждые 10 минут: перед матчами с неотмеченными прогнозами — напоминание о поздней проверке."""
    try:
        due = engine.due_late_checks()
    except Exception:  # noqa: BLE001
        log.exception("late checks failed")
        return
    if not due or _quiet_now(fmt.local(fmt.now_utc())):
        return
    for num in due:
        try:
            text = engine.late_check(num)
        except ValueError:
            continue
        engine.mark_late_sent(num)
        ctx = engine.load_candidate(num)
        picks = [s for s in engine.db.pending() if ctx and s.event_id == ctx.event.event_id]
        placed = [s for s in picks if s.placed]
        when = f"{fmt.local(ctx.event.kickoff):%H:%M}" if ctx else ""
        head = (f"⏰ <b>Скоро матч: {fmt.escape(ctx.title) if ctx else f'#{num}'}</b>"
                + (f" — начало в {when}" if when else ""))
        if placed:
            head += ("\n💰 Ваша ставка: " + ", ".join(
                f"{fmt.escape(fmt.sig_short(s))} {fmt._money(s.actual_stake or 0)} × {(s.actual_odds or s.min_odds):.2f}"
                for s in placed)
                + "\nПоздняя проверка покажет, не случилось ли что-то перед игрой (травмы, составы, вратари).")
        if any(s.user_decision == "pending" for s in picks):
            head += "\n❔ Есть прогноз без отметки: проверьте и решите — «Сделал» или «Не ставил»."
        await bot.send_message(chat_id, head + "\nСкопируйте блок ниже в чат с Claude.")
        await bot.send_message(chat_id, fmt.dossier_message(text))
