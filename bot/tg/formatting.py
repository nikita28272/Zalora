"""Тексты сообщений в Telegram (HTML-разметка)."""

from __future__ import annotations

from html import escape

from ..config import settings
from ..db import Signal
from ..pipeline import SEL_RU, SEL_SHORT, AnalysisResult, MatchContext, local

LEVEL_RU = {"low": "низкая", "medium": "средняя", "high": "высокая"}
LEAN_RU = {"home": "хозяева", "draw": "ничья", "away": "гости", "over": "ТБ 2.5", "under": "ТМ 2.5", "none": "нет"}

DISCLAIMER = (
    "⚠️ Это аналитика, а не гарантия. Букмекер всегда имеет преимущество в марже, "
    "и даже хорошие сигналы часто проигрывают. Ставьте только деньги, которые готовы потерять."
)

HELP = f"""<b>Бот сигналов на футбол</b> (топ-5 лиг + Лига чемпионов)

Как он работает:
1. Берёт линии букмекеров и снимает с них маржу. Получается «мнение рынка».
2. Статистическая модель (Пуассон по сезону) даёт свою оценку.
3. Если модель видит перевес, подключаются агенты: новости, зарубежные аналитики и адвокат дьявола.
4. Главный аналитик взвешивает всё и выносит вердикт.
5. Финансист считает размер ставки и следит за лимитами.

<b>Команды</b>
/matches — ближайшие матчи с вероятностями
/analyze N — полный разбор матча N из списка
/scan — найти сигналы сейчас (так же работает ежедневный автоскан)
/signals — открытые сигналы
/stats — статистика журнала
/history — последние рассчитанные сигналы
/bank [сумма] — показать или задать банк
/limits — лимиты риска
/resume — снять паузу после серии проигрышей
/usage — расход на Claude API за 30 дней
/settle — рассчитать завершённые матчи

Автоскан каждый день в {settings.scan_time}, расчёт итогов в {settings.settle_time} ({settings.timezone}).
Все сигналы ведутся в журнале как «бумажные» ставки, чтобы проверить точность до игры на реальные деньги.

{DISCLAIMER}"""


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.0%}"


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


def analysis(res: AnalysisResult) -> str:
    ctx, v = res.ctx, res.verdict
    head = (
        f"<b>{escape(ctx.title)}</b>\n{escape(ctx.event.league.name_ru)}, "
        f"{local(ctx.event.kickoff):%d.%m %H:%M}"
    )
    out = []
    if res.signal_id and v and res.stake:
        out.append(f"🟢 <b>СИГНАЛ #{res.signal_id}</b>")
        out.append(head)
        out.append("")
        out.append(f"Ставка: <b>{SEL_RU[v.selection]}</b>")
        out.append(f"Брать при коэффициенте <b>не ниже {res.min_odds:.2f}</b> (средний по рынку {res.odds_ref:.2f})")
        out.append(
            f"Сумма: <b>{res.stake.stake:,.0f}</b> ({res.stake.stake_pct:.1%} банка; {escape(res.stake.reason)})"
            .replace(",", " ")
        )
        out.append(
            f"Наша вероятность {res.p_final:.1%} против рыночной {ctx.market.fair[v.selection]:.1%}, "
            f"перевес {res.final_edge:+.1%}. Уверенность: {LEVEL_RU[v.confidence]}."
        )
    else:
        out.append("⚪️ <b>Без ставки</b>")
        out.append(head)
        if res.skip_reason:
            out.append(f"Причина: {escape(res.skip_reason)}")
        if v and res.p_final is not None:
            out.append(
                f"Ближайший вариант: {SEL_RU[v.selection]}, наша вероятность {res.p_final:.1%}, "
                f"рынок {ctx.market.fair.get(v.selection, 0):.1%}, перевес {res.final_edge:+.1%}"
            )

    if v:
        out += ["", f"<b>Главный аналитик:</b> {escape(v.reasoning)}"]
        out += ["<b>За:</b>", _list(v.pros), "<b>Против:</b>", _list(v.cons)]

    out.append("")
    out.append("<b>Отчёты помощников</b>")
    if ctx.model:
        mp = ctx.model.probs
        out.append(
            f"📊 Модель: голы {ctx.model.lambda_home:.2f}:{ctx.model.lambda_away:.2f}; "
            f"П1 {_pct(mp['home'])}, X {_pct(mp['draw'])}, П2 {_pct(mp['away'])}, ТБ2.5 {_pct(mp['over'])}"
        )
    else:
        out.append("📊 Модель: мало данных")
    f = ctx.market.fair
    out.append(
        f"💹 Рынок: П1 {_pct(f.get('home'))}, X {_pct(f.get('draw'))}, П2 {_pct(f.get('away'))}"
        + (f", ТБ2.5 {_pct(f.get('over'))}" if "over" in f else "")
    )
    if res.news:
        n = res.news
        absences = n.home_absences[:3] + n.away_absences[:3]
        out.append(f"📰 Новости ({LEVEL_RU[n.confidence]}): {escape(n.summary)}")
        if absences:
            out.append("   Потери: " + escape("; ".join(absences)))
    if res.experts:
        e = res.experts
        out.append(
            f"🌍 Эксперты ({e.sources_checked} источн., согласие {LEVEL_RU[e.agreement]}): "
            f"1X2 → {LEAN_RU[e.consensus_1x2]}, тотал → {LEAN_RU[e.consensus_total]}"
        )
        if e.popular_picks:
            out.append("   " + escape("; ".join(e.popular_picks[:3])))
    if res.devil:
        d = res.devil
        out.append(f"😈 Адвокат дьявола ({LEVEL_RU[d.severity]}): {escape(d.summary)}")
    for err in res.errors:
        out.append(f"⚠️ {escape(err)}")
    out.append(f"\n<i>Стоимость разбора ≈ ${res.cost_usd:.2f}</i>")
    return "\n".join(out)


def signal_line(s: Signal) -> str:
    status = {"pending": "⏳", "won": "✅", "lost": "❌", "void": "↩️"}[s.status]
    tail = ""
    if s.status != "pending":
        tail = f" → {s.score}, {s.profit:+,.0f}".replace(",", " ")
    return (
        f"{status} #{s.id} {local(s.kickoff):%d.%m %H:%M} {escape(s.home)} — {escape(s.away)}: "
        f"<b>{SEL_SHORT[s.selection]}</b> от {s.min_odds:.2f}, ставка {s.stake:,.0f}{tail}".replace(",", " ")
    )


def stats(signals: list[Signal], bankroll: float) -> str:
    done = [s for s in signals if s.status in ("won", "lost")]
    if not done:
        return f"Рассчитанных сигналов пока нет.\nБанк: {bankroll:,.0f}".replace(",", " ")
    won = sum(s.status == "won" for s in done)
    staked = sum(s.stake for s in done)
    profit = sum(s.profit or 0 for s in done)
    avg_odds = sum(s.odds_ref for s in done) / len(done)
    expected_wr = sum(s.p_final for s in done) / len(done)
    # максимальная просадка по накопленной прибыли
    peak = cum = dd = 0.0
    for s in done:
        cum += s.profit or 0
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    text = f"""<b>Статистика журнала</b>
Сигналов: {len(done)} (✅ {won} / ❌ {len(done) - won})
Проход: {won / len(done):.0%} (ожидали {expected_wr:.0%})
Средний коэффициент: {avg_odds:.2f}
Оборот: {staked:,.0f}
Прибыль: {profit:+,.0f}
ROI: {profit / staked:+.1%}
Макс. просадка: {dd:,.0f}
Банк: {bankroll:,.0f}""".replace(",", " ")
    if len(done) < 50:
        text += "\n\n<i>Меньше 50 ставок: результат пока во многом случаен, выводы делать рано.</i>"
    return text
