"""Тексты сообщений в Telegram (HTML-разметка)."""

from __future__ import annotations

from html import escape

from ..agents.schemas import ALL_OUTCOMES, OutcomeView
from ..config import settings
from ..db import Signal, Summary, summarize
from ..pipeline import SEL_RU, SEL_SHORT, AnalysisResult, MatchContext, local

LEVEL_RU = {"low": "низкая", "medium": "средняя", "high": "высокая"}
STANCE = {"for": "за", "against": "против", "neutral": "нейтр."}

DISCLAIMER = (
    "⚠️ Это аналитика, а не гарантия. Букмекер всегда имеет преимущество в марже, "
    "и даже хорошие сигналы часто проигрывают. Ставьте только деньги, которые готовы потерять."
)

HELP = f"""<b>Бот сигналов на футбол</b> (топ-5 лиг + Лига чемпионов, только прематч)

<b>Иерархия</b>
👤 Вы: окончательное решение
🧠 Главный аналитик (босс): собирает доклады и выносит вердикт
Подчинённые:
  💹 рынок: вероятности из линий без маржи
  📊 статистик: модель по сезону, проверка исходов матчей
  📰 новости: травмы, составы, мотивация
  🌍 зарубежные аналитики: самые частые прогнозы
  😈 адвокат дьявола: аргументы против
  💰 финансист: банк, размер ставки, лимиты

Каждый подчинённый докладывает по каждому исходу (П1, X, П2, ТБ2.5, ТМ2.5).
Главный не выдаёт вердикт, пока не выслушает всех.

{{flow}}

<b>Команды</b>
/matches: ближайшие матчи с вероятностями
/analyze N: {{analyze_help}}
/scan: {{scan_help}}
/verdict: принять вердикт босса из чата (режим «босс в чате»)
/signals: открытые сигналы и ваши отметки
/stats: ROI за месяц и за всё время
/history: последние рассчитанные сигналы
/bank [сумма]: показать или задать банк
/limits: лимиты финансиста
/resume: снять паузу после серии проигрышей
/usage: расход на Claude API
/settle: проверить исходы вручную

Автоскан в {settings.scan_time}, проверка исходов в {settings.settle_time} ({settings.timezone}).

{DISCLAIMER}"""

FLOW_API = """<b>Ваш порядок действий</b>
1. Приходит сигнал с полным разбором.
2. Жмёте «✅ Поставил» и пишете коэффициент (и сумму, если она другая) или «❌ Не ставил».
3. После матча главный докладывает итог и новый баланс."""

FLOW_MANUAL = """<b>Режим «босс в чате»</b>
1. Бот присылает кандидатов дня. Жмёте «📋 Досье» и копируете текст.
2. Вставляете досье в чат с Claude: босс запускает субагентов и выносит вердикт.
3. Строку <code>/verdict …</code> из ответа босса отправляете сюда.
4. Финансист считает сумму, приходит сигнал: «✅ Поставил» или «❌ Не ставил».
5. После матча бот докладывает итог и новый баланс."""


def help_text() -> str:
    if settings.api_mode:
        return HELP.format(flow=FLOW_API, analyze_help="полный разбор матча N агентами",
                           scan_help="найти сигналы сейчас (то же делает ежедневный автоскан)")
    return HELP.format(flow=FLOW_MANUAL, analyze_help="досье матча N для босса",
                       scan_help="кандидаты дня с досье (то же делает ежедневный автоскан)")


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.0%}"


def _money(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def matches_list(matches: list[MatchContext], limit: int = 30) -> str:
    if not matches:
        return "Ближайших матчей с линией не найдено."
    lines = ["<b>Ближайшие матчи</b> (рынок / модель, %)", ""]
    league = None
    for m in matches[:limit]:
        if m.event.league.name_ru != league:
            league = m.event.league.name_ru
            lines.append(f"<b>{escape(league)}</b>")
        f = m.market.fair
        mp = m.model.probs if m.model else None
        probs = (
            f"П1 {_pct(f['home'])}/{_pct(mp['home']) if mp else '—'} · "
            f"X {_pct(f['draw'])}/{_pct(mp['draw']) if mp else '—'} · "
            f"П2 {_pct(f['away'])}/{_pct(mp['away']) if mp else '—'}"
        )
        flag = " 🔎" if m.quant_candidates() else ""
        lines.append(
            f"<b>{m.idx}.</b> {local(m.event.kickoff):%d.%m %H:%M} {escape(m.title)}{flag}\n    {probs}"
        )
    if len(matches) > limit:
        lines.append(f"…и ещё {len(matches) - limit}")
    lines += ["", "🔎 — модель видит возможный перевес. Разбор: /analyze N"]
    return "\n".join(lines)


def _list(items: list[str], limit: int = 4) -> str:
    return "\n".join(f"  • {escape(x)}" for x in items[:limit]) or "  —"


def _views(items: list[OutcomeView] | None) -> dict[str, OutcomeView]:
    return {v.selection: v for v in (items or [])}


def outcome_table(res: AnalysisResult) -> str:
    """Моноширинная таблица: рынок, модель, главный, кэф и перевес по каждому исходу."""
    rows = ["Исход  Рынок Модель Босс  Кэф  Перевес"]
    for sel in ALL_OUTCOMES:
        r = res.table.get(sel)
        if r is None or r.p_market is None:
            continue
        mark = "▶" if res.verdict and res.verdict.selection == sel else " "
        edge_txt = f"{r.edge:+.1%}" if r.edge is not None else "—"
        rows.append(
            f"{mark}{SEL_SHORT[sel]:<6}{_pct(r.p_market):>5} {_pct(r.p_model):>6} {_pct(r.p_final):>4} "
            f"{(r.odds or 0):>5.2f} {edge_txt:>7}"
        )
    return "<pre>" + escape("\n".join(rows)) + "</pre>"


def council(res: AnalysisResult) -> str:
    """Кто что сказал по каждому исходу."""
    news_v = _views(res.news.outcomes if res.news else None)
    exp_v = _views(res.experts.outcomes if res.experts else None)
    devil_v = _views(res.devil.outcomes if res.devil else None)
    chief_v = {o.selection: o for o in (res.verdict.outcomes if res.verdict else [])}
    lines = ["<b>Совет по каждому исходу</b>"]
    for sel in ALL_OUTCOMES:
        if sel not in res.ctx.market.fair:
            continue
        lines.append(f"<b>{SEL_SHORT[sel]}</b>")
        if sel in news_v:
            lines.append(f"  📰 {STANCE[news_v[sel].stance]}: {escape(news_v[sel].comment)}")
        if sel in exp_v:
            lines.append(f"  🌍 {STANCE[exp_v[sel].stance]}: {escape(exp_v[sel].comment)}")
        if sel in devil_v:
            lines.append(f"  😈 {escape(devil_v[sel].comment)}")
        if sel in chief_v:
            lines.append(f"  🧠 {escape(chief_v[sel].comment)}")
    return "\n".join(lines)


def analysis(res: AnalysisResult) -> str:
    ctx, v = res.ctx, res.verdict
    head = (
        f"<b>{escape(ctx.title)}</b>\n{escape(ctx.event.league.name_ru)}, "
        f"{local(ctx.event.kickoff):%d.%m %H:%M} · прематч"
    )
    out = []
    if res.signal_id and v and res.stake:
        out.append(f"🟢 <b>СИГНАЛ #{res.signal_id}</b> · доклад главного аналитика")
        out.append(head)
        out.append("")
        out.append(f"Ставка: <b>{SEL_RU[v.selection]}</b>")
        out.append(f"Брать при коэффициенте <b>не ниже {res.min_odds:.2f}</b> (средний по рынку {res.odds_ref:.2f})")
        out.append(
            f"💰 Финансист: <b>{_money(res.stake.stake)}</b> ({res.stake.stake_pct:.1%} банка; "
            f"{escape(res.stake.reason)})"
        )
        out.append(
            f"Наша вероятность {res.p_final:.1%} против рыночной {ctx.market.fair[v.selection]:.1%}, "
            f"перевес {res.final_edge:+.1%}. Уверенность: {LEVEL_RU[v.confidence]}."
        )
    else:
        out.append("⚪️ <b>Без ставки</b> · доклад главного аналитика")
        out.append(head)
        if res.skip_reason:
            out.append(f"Причина: {escape(res.skip_reason)}")

    if res.table:
        out += ["", outcome_table(res)]
    if v:
        out += [f"<b>🧠 Главный аналитик:</b> {escape(v.reasoning)}"]
        out += ["<b>За:</b>", _list(v.pros), "<b>Против:</b>", _list(v.cons)]

    out.append("")
    if res.news or res.experts or res.devil:
        out.append(council(res))
        out.append("")
    out.append("<b>Коротко от подчинённых</b>")
    if ctx.model:
        out.append(
            f"📊 Статистик: ожидаемые голы {ctx.model.lambda_home:.2f}:{ctx.model.lambda_away:.2f}, "
            f"форма {ctx.home_form or '—'} / {ctx.away_form or '—'}"
        )
    else:
        out.append("📊 Статистик: мало данных для модели")
    if res.news:
        n = res.news
        out.append(f"📰 Новости ({LEVEL_RU[n.confidence]}): {escape(n.summary)}")
        absences = n.home_absences[:3] + n.away_absences[:3]
        if absences:
            out.append("   Потери: " + escape("; ".join(absences)))
    if res.experts:
        e = res.experts
        out.append(
            f"🌍 Эксперты ({e.sources_checked} источн., согласие {LEVEL_RU[e.agreement]}): {escape(e.summary)}"
        )
    if res.devil:
        out.append(f"😈 Адвокат дьявола ({LEVEL_RU[res.devil.severity]}): {escape(res.devil.summary)}")
    for err in res.errors:
        out.append(f"⚠️ {escape(err)}")
    if res.cost_usd:
        out.append(f"\n<i>Стоимость разбора ≈ ${res.cost_usd:.2f}</i>")
    if res.signal_id:
        out.append("\nОтметьте, поставили ли вы. Без отметки до начала матча ставка считается несделанной.")
    return "\n".join(out)


DECISION_RU = {"pending": "ждёт отметки", "placed": "поставлено", "skipped": "не ставили", "expired": "без отметки"}


def signal_line(s: Signal) -> str:
    status = {"pending": "⏳", "won": "✅", "lost": "❌", "void": "↩️"}[s.status]
    if s.placed:
        bet = f"{_money(s.actual_stake or 0)} по {s.actual_odds or s.min_odds:.2f}"
    else:
        bet = f"от {s.min_odds:.2f}, {DECISION_RU[s.user_decision]}"
    tail = ""
    if s.status in ("won", "lost"):
        tail = f" → {s.score}"
        tail += f", {s.profit:+,.0f}".replace(",", " ") if s.placed else " (вы не ставили)"
    elif s.status == "void":
        tail = " → возврат"
    return (
        f"{status} #{s.id} {local(s.kickoff):%d.%m %H:%M} {escape(s.home)} — {escape(s.away)}: "
        f"<b>{SEL_SHORT[s.selection]}</b> {bet}{tail}"
    )


def placed_confirmation(s: Signal, bank: float, in_play: tuple[int, float]) -> str:
    warn = ""
    if s.actual_odds and s.actual_odds < s.min_odds:
        warn = (
            f"\n⚠️ Коэффициент {s.actual_odds:.2f} ниже минимального {s.min_odds:.2f}: "
            "по нашей оценке у этой ставки нет перевеса."
        )
    return (
        f"🧠 Главный: принял, передал финансисту.\n"
        f"💰 Финансист записал ставку #{s.id}: {SEL_SHORT[s.selection]} на матч {escape(s.home)} — {escape(s.away)}, "
        f"{_money(s.actual_stake or 0)} по {s.actual_odds:.2f}.\n"
        f"Банк: {_money(bank)}. В игре: {in_play[0]} ставок на {_money(in_play[1])}.{warn}"
    )


def settle_report(settled: list[Signal], bank: float) -> str:
    lines = ["🧠 <b>Доклад главного аналитика: итоги матчей</b>", ""]
    real_total = 0.0
    for s in settled:
        if s.status == "void":
            lines.append(f"↩️ #{s.id} {escape(s.home)} — {escape(s.away)}: результат не найден, сигнал закрыт без расчёта.")
            continue
        outcome = "зашла ✅" if s.status == "won" else "не зашла ❌"
        lines.append(
            f"📊 Статистик проверил исход: {escape(s.home)} — {escape(s.away)} {s.score}. "
            f"{SEL_SHORT[s.selection]} {outcome}"
        )
        if s.placed:
            real_total += s.profit or 0
            lines.append(f"   💰 Финансист: {s.profit:+,.0f}".replace(",", " "))
        else:
            lines.append(f"   Вы не ставили. На бумаге: {s.paper_profit:+,.0f}".replace(",", " "))
    lines.append("")
    lines.append(f"💰 Итог по вашим ставкам: {real_total:+,.0f}. Банк: <b>{_money(bank)}</b>".replace(",", " "))
    return "\n".join(lines)


def _summary_block(title: str, s: Summary | None) -> str:
    if s is None:
        return f"<b>{title}</b>\n  пока нет рассчитанных ставок"
    return (
        f"<b>{title}</b>\n"
        f"  ставок {s.count} (✅ {s.won} / ❌ {s.count - s.won}), проход {s.won / s.count:.0%} "
        f"(ожидали {s.expected_wr:.0%})\n"
        f"  средний кэф {s.avg_odds:.2f}, оборот {_money(s.staked)}\n"
        f"  прибыль {s.profit:+,.0f}, <b>ROI {s.roi:+.1%}</b>, макс. просадка {_money(s.max_drawdown)}"
        .replace(",", " ")
    )


def stats(month: list[Signal], all_time: list[Signal], bankroll: float) -> str:
    real_m = summarize(month, real=True)
    parts = [
        _summary_block("Ваши ставки за месяц", real_m),
        _summary_block("Ваши ставки за всё время", summarize(all_time, real=True)),
        _summary_block("Все сигналы системы (на бумаге, по мин. кэфу)", summarize(all_time, real=False)),
        f"<b>Банк:</b> {_money(bankroll)}",
    ]
    if real_m and bankroll - real_m.profit > 0:
        growth = real_m.profit / (bankroll - real_m.profit)
        parts[-1] += f" (за месяц {growth:+.1%})"
    total = summarize(all_time, real=False)
    if total is None or total.count < 50:
        parts.append("<i>Меньше 50 рассчитанных сигналов: результат пока во многом случаен.</i>")
    parts.append(
        "<i>Ориентир: устойчивый ROI +3…8% на линии топ-лиг — уже сильный результат.</i>"
    )
    return "\n\n".join(parts)


def candidates_message(items: list[tuple[int, MatchContext]], total: int) -> str:
    if not items:
        return (f"📋 Кандидатов сегодня нет: среди {total} матчей модель не видит перевеса. "
                "Можно запросить досье на любой матч: /matches, затем /analyze N.")
    lines = ["📋 <b>Кандидаты дня</b> (модель видит перевес)", ""]
    for num, ctx in items:
        sel, ev = ctx.quant_candidates()[0]
        lines.append(
            f"<b>#{num}</b> {local(ctx.event.kickoff):%d.%m %H:%M} {escape(ctx.title)}\n"
            f"    {escape(ctx.event.league.name_ru)} · модель: {SEL_SHORT[sel]}, перевес {ev:+.1%}"
        )
    lines += ["", "Жмите «📋 Досье», копируйте текст и отправляйте в чат с Claude (боссу)."]
    return "\n".join(lines)


def dossier_message(text: str) -> str:
    """Досье в блоке кода: в Telegram у него есть кнопка «Копировать»."""
    return "<pre>" + escape(text) + "</pre>"
