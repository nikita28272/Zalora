"""Тексты сообщений в Telegram (HTML-разметка)."""

from __future__ import annotations

from html import escape

from ..agents.schemas import OutcomeView
from ..config import settings
from ..db import Signal, Summary, summarize
from ..data.odds import Quota, ScanReport
from ..pipeline import AnalysisResult, MatchContext, TopPicks, local
from ..sports import SPORTS, sel_short

LEVEL_RU = {"low": "низкая", "medium": "средняя", "high": "высокая"}
STANCE = {"for": "за", "against": "против", "neutral": "нейтр."}

DISCLAIMER = (
    "⚠️ Это аналитика, а не гарантия. Букмекер всегда имеет преимущество в марже, "
    "и даже хорошие сигналы часто проигрывают. Ставьте только деньги, которые готовы потерять."
)

HELP = f"""<b>Бот сигналов</b>: ⚽ топ-5 лиг и Лига чемпионов, 🏒 НХЛ, 🏀 НБА. Только прематч.

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

Каждый подчинённый докладывает по каждому исходу: в футболе П1, X, П2, ТБ, ТМ; в хоккее ещё победа с ОТ и фора;
в баскетболе — победа, фора и тотал очков (с овертаймом).
Главный не выдаёт вердикт, пока не выслушает всех.

{{flow}}

<b>Кнопки меню внизу экрана</b>
📋 Кандидаты · 📅 Матчи (номер матча — кнопкой) · 📌 Мои сигналы · 📊 Статистика · 💰 Банк · 💳 Кредиты · ⚙️ Ещё

<b>Те же действия командами</b>
/matches: ближайшие матчи с вероятностями
/analyze N: {{analyze_help}}
/scan: {{scan_help}}
/verdict: принять вердикт босса из чата (режим «босс в чате»)
/check N: поздняя проверка перед матчем (вратари, составы)
/fire N: поставить или снять 🔥 у прогноза N
/signals: открытые сигналы и ваши отметки
/stats: ROI за месяц и за всё время
/history: последние рассчитанные сигналы
/bank [сумма]: показать или задать банк
/limits: лимиты финансиста
/resume: снять паузу после серии проигрышей
/credits: остаток кредитов The Odds API (проверка бесплатная)
/usage: расход на Claude API
/settle: проверить исходы вручную

Автоскан в {settings.scan_time}, проверка исходов в {settings.settle_time} ({settings.timezone}).

{DISCLAIMER}"""

FLOW_API = """<b>Ваш порядок действий</b>
1. Приходит сигнал с полным разбором.
2. Жмёте «✅ Поставил» и пишете коэффициент (и сумму, если она другая) или «❌ Не ставил».
3. После матча главный докладывает итог и новый баланс."""

FLOW_MANUAL = f"""<b>Как это работает</b>
1. В {settings.scan_time} бот присылает матчи дня к разбору: 2 хоккея, 2 НБА, 2 футбола (если есть).
2. «🔍 Анализ» → номер матча → досье. Копируете его и отправляете в чат с Claude (боссу).
3. Босс разбирает все исходы и даёт 2 лучших (с перевесом — первыми). Строку <code>/verdict …</code>
   из его ответа отправляете сюда.
4. Бот показывает эти 2 исхода. Под каждым — «✅ Сделал» или «❌ Не ставил».
   После «✅ Сделал» напишите кэф вашей конторы: 💰 финансист посчитает шанс, перевес
   и сумму от текущего банка («💰 Банк» → «✏️ Изменить»), затем «✅ Ставлю» или «✏️ Своя сумма».
   Перед матчем бот напомнит о поздней проверке (вратари, составы) — блок «ПРОВЕРКА» тоже в чат с Claude.
5. В {settings.settle_time} — итоги по этим исходам: ваши ставки меняют банк,
   остальные прогнозы идут в статистику 📊 на бумаге."""


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


def sig_short(s: Signal) -> str:
    return sel_short(SPORTS.get(s.sport, SPORTS["football"]), s.selection, s.line)


_sig_short = sig_short


def _sig_emoji(s: Signal) -> str:
    return SPORTS.get(s.sport, SPORTS["football"]).emoji


def flag_of(m: MatchContext) -> str:
    """Метка в списке: 🔎 — модель видит перевес; 🏀 — матч НБА, подходящий для разбора по анализу."""
    if m.sport.key == "basketball" and settings.nba_mode == "analysis":
        return " 🏀" if m.factors and m.factors.score >= settings.nba_min_factor else ""
    return " 🔎" if m.quant_candidates() else ""


def schedule_only_block(report: ScanReport | None) -> str:
    """Матчи лиг, по которым линии не загружались (нечего разбирать): только расписание."""
    if report is None or not report.schedule_only:
        return ""
    from datetime import datetime
    lines = []
    league = None
    for name, home, away, when in sorted(report.schedule_only, key=lambda x: (x[0], x[3])):
        if name != league:
            league = name
            lines.append(f"<b>{escape(name)}</b> — без линий: перекоса графика нет, кредиты не тратились")
        try:
            t = f"{local(datetime.fromisoformat(when.replace('Z', '+00:00'))):%d.%m %H:%M} "
        except ValueError:
            t = ""
        lines.append(f"   {t}{escape(home)} — {escape(away)}")
    return "\n".join(lines)


def matches_list(matches: list[MatchContext], limit: int = 30, report: ScanReport | None = None) -> str:
    extra = schedule_only_block(report)
    if not matches:
        return ("Ближайших матчей с линией не найдено." + ("\n\n" + extra if extra else ""))
    lines = ["<b>Ближайшие матчи</b> (рынок / модель, %)", ""]
    league = None
    for m in matches[:limit]:
        if m.event.league.name_ru != league:
            league = m.event.league.name_ru
            lines.append(f"<b>{m.sport.emoji} {escape(league)}</b>")
        f = m.market.fair
        mp = m.model.probs if m.model else {}
        shown = ("home", "draw", "away") if m.sport.key == "football" else ("ml_home", "ml_away", "over")
        probs = " · ".join(
            f"{m.short(o)} {_pct(f.get(o))}/{_pct(mp.get(o)) if mp else '—'}" for o in shown if o in f
        )
        lines.append(
            f"<b>{m.idx}.</b> {local(m.event.kickoff):%d.%m %H:%M} {escape(m.title)}{flag_of(m)}\n    {probs}"
        )
    if len(matches) > limit:
        lines.append(f"…и ещё {len(matches) - limit}")
    if extra:
        lines += ["", extra]
    lines += ["", "🔎 — модель видит возможный перевес; 🏀 — НБА с перекосом графика. Разбор: «🔍 Анализ»."]
    return "\n".join(lines)


def hot_block(hot: list[Signal]) -> str:
    """🔥 Обратить внимание: открытые прогнозы, отмеченные боссом, самые вероятные первыми."""
    if not hot:
        return ""
    lines = ["🔥 <b>Обратить внимание</b> (отметил босс):"]
    for s in hot:
        lines.append(f"#{s.id} {_sig_emoji(s)} {local(s.kickoff):%d.%m %H:%M} {escape(s.home)} — {escape(s.away)}: "
                     f"<b>{escape(sig_short(s))}</b>, шанс {s.p_final:.0%}, кэф {s.odds_ref:.2f}")
    return "\n".join(lines)


def daily_block(picks: list[tuple[MatchContext, str]], total: int) -> str:
    """Матчи дня к разбору: по 2 хоккея и НБА (и футбола, если есть) — с причиной выбора."""
    if not picks:
        return "🎯 <b>Матчи дня:</b> подходящих матчей нет."
    lines = ["🎯 <b>Матчи дня к разбору</b>"]
    sport = None
    for m, why in picks:
        if m.sport.key != sport:
            sport = m.sport.key
            lines += ["", f"{m.sport.emoji} <b>{escape(m.event.league.name_ru)}</b>"]
        lines.append(f"<b>{m.idx}.</b> {local(m.event.kickoff):%d.%m %H:%M} {escape(m.title)}\n    {escape(why)}")
    lines += ["", "Нажмите 📋 с номером — придёт досье, его — в чат с Claude."]
    if total > len(picks):
        lines.append(f"<i>Всего матчей в линии: {total} — остальные по кнопке «📃 Все матчи».</i>")
    return "\n".join(lines)


def top_block(top: list[tuple[MatchContext, str, float]], nba: list[MatchContext] | None = None) -> str:
    """Наверху утреннего списка: какие матчи разобрать, чтобы не тратить лимит Claude впустую."""
    if not top:
        lines = ["⭐ <b>Стоит разобрать:</b> сегодня нет матчей, где перевес вероятен."]
    else:
        lines = ["⭐ <b>Стоит разобрать</b> (самый перспективный исход, оценка до агентов):"]
        for m, sel, ev in top:
            lines.append(f"<b>{m.idx}.</b> {m.sport.emoji} {local(m.event.kickoff):%d.%m %H:%M} {escape(m.title)}"
                         f" — {escape(m.short(sel))} {m.market.avg_odds[sel]:.2f}, {ev:+.1%}")
    if nba is not None and settings.nba_mode == "analysis":
        lines.append("")
        if nba:
            lines.append("🏀 <b>НБА по анализу</b> (сильный перекос графика и усталости):")
            for m in nba:
                f = m.factors
                why = "; ".join(f.reasons) or "перекос по сумме факторов"
                lines.append(f"<b>{m.idx}.</b> {local(m.event.kickoff):%d.%m %H:%M} {escape(m.title)} — график в пользу "
                             f"{f.side_gen} ({f.score:.1f}): {escape(why)}")
        else:
            lines.append("🏀 <b>НБА:</b> сегодня нет матчей с заметным перекосом графика — разбирать нечего.")
    lines.append("<i>Остальные матчи без зацепки для анализа; разбирать их — тратить лимит.</i>")
    return "\n".join(lines)


def _list(items: list[str], limit: int = 4) -> str:
    return "\n".join(f"  • {escape(x)}" for x in items[:limit]) or "  —"


def _views(items: list[OutcomeView] | None) -> dict[str, OutcomeView]:
    return {v.selection: v for v in (items or [])}


def outcome_table(res: AnalysisResult) -> str:
    """Моноширинная таблица: рынок, модель, главный, кэф и перевес по каждому исходу."""
    ctx = res.ctx
    w = 6 if ctx.sport.key == "football" else 9
    rows = [f"{'Исход':<{w + 1}}Рынок Модель Босс  Кэф   Перевес"]
    for sel in ctx.sport.outcomes:
        r = res.table.get(sel)
        if r is None or r.p_market is None:
            continue
        mark = "▶" if res.verdict and res.verdict.selection == sel else " "
        edge_txt = f"{r.edge:+.1%}" if r.edge is not None else "—"
        star = "*" if sel in ctx.market.estimated else " "
        rows.append(
            f"{mark}{ctx.short(sel):<{w}}{_pct(r.p_market):>5} {_pct(r.p_model):>6} {_pct(r.p_final):>4} "
            f"{(r.odds or 0):>5.2f}{star}{edge_txt:>7}"
        )
    if ctx.market.estimated:
        rows.append("* кэф оценочный: 1X2 основного времени нет в линии, проверьте в своей БК")
    return "<pre>" + escape("\n".join(rows)) + "</pre>"


def council(res: AnalysisResult) -> str:
    """Кто что сказал по каждому исходу."""
    news_v = _views(res.news.outcomes if res.news else None)
    exp_v = _views(res.experts.outcomes if res.experts else None)
    devil_v = _views(res.devil.outcomes if res.devil else None)
    chief_v = {o.selection: o for o in (res.verdict.outcomes if res.verdict else [])}
    lines = ["<b>Совет по каждому исходу</b>"]
    for sel in res.ctx.sport.outcomes:
        if sel not in res.ctx.market.fair:
            continue
        lines.append(f"<b>{escape(res.ctx.short(sel))}</b>")
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
        f"{ctx.sport.emoji} <b>{escape(ctx.title)}</b>\n{escape(ctx.event.league.name_ru)}, "
        f"{local(ctx.event.kickoff):%d.%m %H:%M} · прематч"
    )
    out = []
    if res.signal_id and v and res.stake:
        out.append(f"🟢 <b>СИГНАЛ #{res.signal_id}</b> · доклад главного аналитика")
        out.append(head)
        out.append("")
        out.append(f"Ставка: <b>{escape(ctx.ru(v.selection))}</b>")
        if v.selection in ctx.market.estimated:
            out.append(f"Брать при коэффициенте <b>не ниже {res.min_odds:.2f}</b> "
                       f"(в линии нет этого рынка: проверьте кэф в своей БК)")
        else:
            out.append(f"Брать при коэффициенте <b>не ниже {res.min_odds:.2f}</b> "
                       f"(средний по рынку {res.odds_ref:.2f})")
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
        if v.pros:
            out += ["<b>За:</b>", _list(v.pros)]
        if v.cons:
            out += ["<b>Против:</b>", _list(v.cons)]

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
    elif s.source == "boss" and not s.positive:
        bet = f"кэф {s.odds_ref:.2f}, без перевеса, {DECISION_RU[s.user_decision]}"
    else:
        bet = f"от {s.min_odds:.2f}, {DECISION_RU[s.user_decision]}"
    tail = ""
    if s.status in ("won", "lost"):
        tail = f" → {s.score}"
        tail += f", {s.profit:+,.0f}".replace(",", " ") if s.placed else " (вы не ставили)"
    elif s.status == "void":
        tail = " → возврат"
    fire = "🔥" if s.hot else ""
    return (
        f"{status}{fire} #{s.id} {_sig_emoji(s)} {local(s.kickoff):%d.%m %H:%M} {escape(s.home)} — {escape(s.away)}: "
        f"<b>{escape(_sig_short(s))}</b> {bet}{tail}"
    )


def placed_confirmation(s: Signal, bank: float, in_play: tuple[int, float]) -> str:
    warn = ""
    if s.source == "boss" and not s.positive:
        warn = "\n⚪ Это прогноз без перевеса: ставка на ваш риск."
    elif s.actual_odds and s.actual_odds < s.min_odds:
        warn = (
            f"\n⚠️ Коэффициент {s.actual_odds:.2f} ниже минимального {s.min_odds:.2f}: "
            "по нашей оценке у этой ставки нет перевеса."
        )
    return (
        f"🧠 Главный: принял, передал финансисту.\n"
        f"💰 Финансист записал ставку #{s.id}: {escape(_sig_short(s))} на матч {escape(s.home)} — {escape(s.away)}, "
        f"{_money(s.actual_stake or 0)} по {s.actual_odds:.2f}.\n"
        f"Банк: {_money(bank)}. В игре: {in_play[0]} ставок на {_money(in_play[1])}.{warn}"
    )


def bet_plan(s: Signal, plan, bank: float) -> str:
    """Разбор финансиста по кэфу владельца: шанс, перевес, сумма, что будет с банком."""
    lines = [
        f"💰 <b>Финансист: {escape(sig_short(s))}</b> ({escape(s.home)} — {escape(s.away)}), ваш кэф {plan.odds:.2f}",
        "",
        f"Шанс захода (оценка босса): <b>{plan.p:.0%}</b>. Ставка окупается, если шанс выше "
        f"{1 / plan.odds:.0%}: перевес при вашем кэфе <b>{plan.ev:+.1%}</b>.",
    ]
    if plan.stake > 0:
        lines += [
            f"✅ Рекомендую: <b>{_money(plan.stake)}</b> ({plan.pct:.1%} банка) — {escape(plan.reason)}.",
            f"Зайдёт: +{_money(plan.win)} → банк {_money(bank + plan.win)}. "
            f"Не зайдёт: −{_money(plan.stake)} → банк {_money(bank - plan.stake)}.",
            f"В среднем на дистанции такая ставка даёт {plan.stake * plan.ev:+,.0f}.".replace(",", " "),
            f"<i>Даже при шансе {plan.p:.0%} из 10 таких ставок в среднем около {round(10 * (1 - plan.p))} "
            "проигрывают — это нормально, решает дистанция.</i>",
        ]
    else:
        lines.append(f"❌ Не рекомендую ставить: {escape(plan.reason)}.")
        if plan.max_own > 0:
            lines.append(f"Если всё же ставите — не больше {_money(plan.max_own)}.")
    return "\n".join(lines)


def settle_report(settled: list[Signal], bank: float) -> str:
    """09:00: итоги только по матчам, на которые давался прогноз. Сгруппировано по матчу."""
    lines = ["🧠 <b>Доклад главного аналитика: итоги прогнозов</b>"]
    by_event: dict[str, list[Signal]] = {}
    for s in settled:
        by_event.setdefault(s.event_id, []).append(s)
    real_total = 0.0
    boss_total = [0, 0]  # рассчитано, зашло
    for group in by_event.values():
        first = group[0]
        score = first.score if first.score and first.score != "нет данных" else "результат не найден"
        lines += ["", f"{_sig_emoji(first)} <b>{escape(first.home)} — {escape(first.away)}</b>: {escape(score)}"]
        for s in sorted(group, key=lambda x: (x.source != "boss", x.pick_rank or 99)):
            mark = {"won": "✅ зашла", "lost": "❌ не зашла", "void": "↩️ возврат"}.get(s.status, s.status)
            label = escape(_sig_short(s))
            if s.source == "boss":
                rank = f" {s.pick_rank}" if s.pick_rank else ""
                line = f"   🧠 Прогноз{rank}: {'🔥 ' if s.hot else ''}{label} — {mark}"
                lines.append(line)
                if s.status in ("won", "lost"):
                    boss_total[0] += 1
                    boss_total[1] += s.status == "won"
                    lines.append(
                        f"      босс {s.p_final:.0%}, кэф {s.odds_ref:.2f}, "
                        f"{'с перевесом' if s.positive else 'без перевеса'}; на бумаге {s.paper_profit:+,.0f}"
                        .replace(",", " "))
                if s.placed:
                    real_total += s.profit or 0
                    lines.append(f"   💰 Вы поставили {_money(s.actual_stake or 0)} по {s.actual_odds or s.min_odds:.2f}: "
                                 f"{(s.profit or 0):+,.0f}".replace(",", " "))
                else:
                    lines.append("   Вы не ставили — записано в статистику 📊")
            else:
                real_total += s.profit or 0
                lines.append(f"   💰 Ваша ставка: {label} {_money(s.actual_stake or 0)} по "
                             f"{s.actual_odds or s.min_odds:.2f} — {mark}, {(s.profit or 0):+,.0f}".replace(",", " "))
    if boss_total[0]:
        lines += ["", f"🧠 Прогнозы босса: зашло {boss_total[1]} из {boss_total[0]}"]
        hot = [s for s in settled if s.source == "boss" and s.hot and s.status in ("won", "lost")]
        if hot:
            lines.append(f"🔥 Отмеченные: зашло {sum(s.status == 'won' for s in hot)} из {len(hot)}")
    lines += [f"💰 Итог по вашим ставкам: {real_total:+,.0f}. Банк: <b>{_money(bank)}</b>".replace(",", " ")]
    if not boss_total[0]:
        lines.insert(-1, "")
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
        _summary_block("Прогнозы босса (на бумаге)", summarize(all_time, real=False)),
        f"<b>Банк:</b> {_money(bankroll)}",
    ]
    if real_m and bankroll - real_m.profit > 0:
        growth = real_m.profit / (bankroll - real_m.profit)
        parts[-1] += f" (за месяц {growth:+.1%})"
    sports = sorted({s.sport for s in all_time if s.status in ("won", "lost")})
    if len(sports) > 1:  # раздельно по видам спорта: видно, где система работает
        per = []
        for key in sports:
            sm = summarize([s for s in all_time if s.sport == key], real=False)
            if sm:
                per.append(f"  {SPORTS[key].emoji} {SPORTS[key].name_ru}: {sm.count} сигн., ROI {sm.roi:+.1%}")
        parts.insert(3, "<b>Сигналы системы по видам спорта</b>\n" + "\n".join(per))
    boss = [s for s in all_time if s.source == "boss"]
    pos, neg = summarize([s for s in boss if s.positive], False), summarize([s for s in boss if not s.positive], False)
    hot = summarize([s for s in boss if s.hot], False)
    if pos or neg:
        split = []
        for title, sm in (("🔥 отмеченные боссом", hot), ("✅ с перевесом", pos), ("⚪ без перевеса", neg)):
            split.append(f"  {title}: " + (f"{sm.won} из {sm.count}, ROI {sm.roi:+.1%}" if sm else "пока нет"))
        parts.insert(3, "<b>Прогнозы босса по типу</b>\n" + "\n".join(split))
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
            f"<b>#{num}</b> {ctx.sport.emoji} {local(ctx.event.kickoff):%d.%m %H:%M} {escape(ctx.title)}\n"
            f"    {escape(ctx.event.league.name_ru)} · модель: {escape(ctx.short(sel))}, перевес {ev:+.1%}"
        )
    lines += ["", "Жмите «📋 Досье», копируйте текст и отправляйте в чат с Claude (боссу)."]
    return "\n".join(lines)


def dossier_message(text: str) -> str:
    """Досье в блоке кода с подписью: у такого блока Telegram показывает кнопку «Копировать»."""
    return '<pre><code class="language-dossier">' + escape(text) + "</code></pre>"


DOSSIER_HINT = (
    "☝️ Нажмите «Копировать» в правом верхнем углу блока выше, затем вставьте текст в чат с Claude.\n"
    "Босс назовёт 2 лучших исхода. Строку <code>/verdict …</code> из его ответа отправьте сюда — "
    "появятся кнопки «✅ Сделал / ❌ Не ставил» по каждому исходу."
)


LATE_HINT = (
    "☝️ Скопируйте блок и отправьте в чат с Claude. Если босс даст новый <code>/verdict</code>, "
    "он заменит прежний — если вы ещё не нажали «✅ Сделал»."
)


def top_picks(top: TopPicks, bank: float) -> str:
    """Ответ на /verdict: два лучших исхода босса с советом финансиста."""
    ctx = top.ctx
    lines = [
        f"🧠 <b>Топ-2 босса по досье #{top.num}</b>"
        + (" (заменил прежний вердикт)" if top.replaced else ""),
        f"{ctx.sport.emoji} {escape(ctx.title)} · {local(ctx.event.kickoff):%d.%m %H:%M}",
    ]
    by_analysis = ctx.sport.key == "basketball" and settings.nba_mode == "analysis"
    for pk in top.picks:
        if by_analysis:
            tag = "📊 сильный разбор" if pk.positive else "⚪ слабый разбор"
        else:
            tag = "✅ есть перевес" if pk.positive else "⚪ без перевеса"
        lines += [
            "",
            f"<b>{pk.rank}. {'🔥 ' if pk.hot else ''}{escape(ctx.short(pk.selection))}</b> — {tag}",
            f"   вероятность {pk.p:.0%} (рынок {pk.p_market:.0%}), кэф {pk.odds:.2f}, перевес {pk.edge:+.1%}",
        ]
        if by_analysis:
            lines.append(f"   💰 {_money(pk.stake)} — {escape(pk.note)}")
            if pk.p * pk.odds < 1:
                lines.append(f"   ⚠️ по нашей же оценке кэф мал: в плюс только от {pk.min_odds:.2f}")
            continue
        if pk.positive:
            lines.append(f"   💰 {_money(pk.stake)} — {escape(pk.note)}; брать по кэфу не ниже {pk.min_odds:.2f}")
        else:
            lines.append(f"   💰 {escape(pk.note)} (≈{_money(pk.stake)}); в плюс только по кэфу от {pk.min_odds:.2f}")
    if not any(pk.positive for pk in top.picks):
        lines += ["", "⚠️ Разбор слабый: это лучшие из исходов, а не рекомендация ставить." if by_analysis
                  else "⚠️ Перевеса нет ни в одном исходе: это лучшие из имеющихся, а не рекомендация ставить."]
    lines += ["", f"Уверенность: {LEVEL_RU.get(top.confidence, top.confidence)}. Банк: {_money(bank)}.",
              "Отметьте по каждому исходу 👇 Итог пришлю на следующий день.",
              "🥅 Перед матчем можно сделать позднюю проверку ("
              + {"hockey": "вратари", "basketball": "травмы и составы"}.get(ctx.sport.key, "составы")
              + ") — бот сам напомнит."]
    return "\n".join(lines)


LOW_QUOTA = 50


def quota_line(q: Quota | None) -> str:
    if q is None:
        return "💳 Кредиты The Odds API: пока неизвестно (обновить: /credits)"
    from datetime import datetime
    when = local(datetime.fromisoformat(q.checked_at)).strftime("%d.%m %H:%M")
    text = f"💳 Кредиты The Odds API: осталось <b>{q.remaining}</b> из {q.total} (на {when})"
    if q.remaining < LOW_QUOTA:
        text += ("\n⚠️ Кредиты заканчиваются. Новые линии могут перестать приходить до начала следующего месяца. "
                 "Можно отключить часть лиг: <code>bash scripts/set.sh ENABLED_LEAGUES PL,BL1,NHL</code>")
    return text


def scan_report_line(r: ScanReport | None) -> str:
    """Сколько стоило обновление линий."""
    if r is None:
        return ""
    from datetime import datetime
    when = local(datetime.fromisoformat(r.fetched_at)).strftime("%H:%M") if r.fetched_at else "?"
    if r.cached:
        return f"♻️ Линии из кеша (обновлены в {when}), кредиты не тратились."
    parts = [f"🧾 Линии обновлены: потрачено <b>{r.spent}</b> кредит(ов) на {len(r.paid)} лиг(и)"]
    if r.reused:
        parts.append("из сохранённых бесплатно: " + escape(", ".join(r.reused)))
    if r.empty:
        parts.append(f"без матчей, пропущено бесплатно: {len(r.empty)}")
    text = "; ".join(parts) + "."
    if r.budget_skipped:
        text += ("\n⏸ Отложено, чтобы кредитов хватило до конца месяца: "
                 + escape(", ".join(r.budget_skipped)))
    if r.gated:
        text += ("\n🏀 Без матчей для разбора (нет перекоса графика), линии не загружались, кредиты целы: "
                 + escape(", ".join(r.gated)))
    if r.failed:
        text += "\n⚠️ Не удалось загрузить: " + escape(", ".join(r.failed))
    return text


def odds_footer(report: ScanReport | None, quota: Quota | None) -> str:
    line = scan_report_line(report)
    return (line + "\n" if line else "") + quota_line(quota)


def now_utc():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)
