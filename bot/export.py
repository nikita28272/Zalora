"""Выгрузка журнала в Excel (.xlsx): открывается на телефоне в Excel, Google Таблицах, WPS и т.п."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .config import settings
from .db import DB, Signal, summarize
from .sports import SPORTS, sel_short

STATUS_RU = {"pending": "ждёт", "won": "зашло", "lost": "не зашло", "void": "возврат"}
DECISION_RU = {"placed": "поставил", "skipped": "не ставил", "expired": "без отметки", "pending": "ждёт отметки"}
CONF_RU = {"high": "высокая", "medium": "средняя", "low": "низкая"}
GREEN, RED, GREY = "E2F4E2", "FBE3E3", "EEEEEE"


def _local(dt: datetime) -> datetime:
    return dt.astimezone(ZoneInfo(settings.timezone)).replace(tzinfo=None)


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _style_header(ws, widths: list[int]) -> None:
    for i, w in enumerate(widths, 1):
        cell = ws.cell(row=1, column=i)
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="DDE6F5")
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"


def _picks_sheet(ws, rows: list[Signal]) -> None:
    head = ["№", "Дата", "Спорт", "Матч", "Исход", "Шанс босса", "Уверенность", "🔥", "Кэф из досье",
            "Решение", "Ваш кэф", "Ваша сумма", "Результат", "Счёт", "Прибыль (ваша)", "На бумаге (по 100)"]
    ws.append(head)
    for s in rows:
        sport = SPORTS.get(s.sport, SPORTS["football"])
        ws.append([
            s.id, _local(s.kickoff), sport.name_ru, f"{s.home} — {s.away}",
            sel_short(sport, s.selection, s.line), round(s.p_final, 3), CONF_RU.get(s.confidence, s.confidence),
            "🔥" if s.hot else "", s.odds_ref, DECISION_RU.get(s.user_decision, s.user_decision),
            s.actual_odds if s.placed else None, s.actual_stake if s.placed else None,
            STATUS_RU.get(s.status, s.status), s.score or "",
            s.profit if s.placed and s.status != "pending" else None,
            s.paper_flat if s.status != "pending" else None,
        ])
        fill = {"won": GREEN, "lost": RED, "void": GREY}.get(s.status)
        if fill:
            for c in ws[ws.max_row]:
                c.fill = PatternFill("solid", fgColor=fill)
    for r in range(2, ws.max_row + 1):
        ws.cell(row=r, column=2).number_format = "DD.MM.YYYY HH:MM"
        ws.cell(row=r, column=6).number_format = "0%"
    _style_header(ws, [6, 16, 11, 34, 14, 10, 12, 4, 10, 13, 9, 11, 11, 9, 13, 13])


def _weeks_sheet(ws, rows: list[Signal]) -> None:
    ws.append(["Неделя (с понедельника)", "Прогнозов босса", "Зашло", "Проход", "ROI на бумаге",
               "Ваших ставок", "Ваша прибыль", "Ваш ROI"])
    done = [s for s in rows if s.status in ("won", "lost") and _parse(s.settled_at)]
    weeks: dict[datetime, list[Signal]] = {}
    for s in done:
        d = _local(_parse(s.settled_at))
        monday = (d - timedelta(days=d.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        weeks.setdefault(monday, []).append(s)
    for monday in sorted(weeks, reverse=True):
        part = weeks[monday]
        paper, real = summarize(part, real=False), summarize(part, real=True)
        ws.append([
            monday.strftime("%d.%m.%Y"),
            paper.count if paper else 0, paper.won if paper else 0,
            round(paper.won / paper.count, 3) if paper else None,
            round(paper.roi, 3) if paper else None,
            real.count if real else 0, round(real.profit, 2) if real else 0,
            round(real.roi, 3) if real else None,
        ])
    for r in range(2, ws.max_row + 1):
        for col in (4, 5, 8):
            ws.cell(row=r, column=col).number_format = "0.0%"
    _style_header(ws, [22, 15, 8, 9, 14, 13, 13, 10])


def build_xlsx(db: DB, path: str) -> int:
    """Журнал в Excel: лист «Прогнозы» (все, новые сверху) и «По неделям». Возвращает число строк."""
    rows = [db._row(r) for r in db.conn.execute("SELECT * FROM signals ORDER BY kickoff DESC, id DESC")]
    wb = Workbook()
    ws = wb.active
    ws.title = "Прогнозы"
    _picks_sheet(ws, rows)
    _weeks_sheet(wb.create_sheet("По неделям"), rows)
    wb.save(path)
    return len(rows)
