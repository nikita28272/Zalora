"""Сторож новостей перед матчем и условия босса.

Каждые NEWS_WATCH_MINUTES бот сверяет отчёт о травмах ESPN по матчам с прогнозом босса с тем, что было
в досье (или при прошлой проверке). Изменения («под вопросом → не играет», новая травма, игрок вернулся)
приходят владельцу, кроме тихих ночных часов NEWS_QUIET_HOURS: тогда бот молчит и ничего не запоминает,
поэтому утром пришлёт всё, что изменилось за ночь, по ещё не начавшимся матчам.

Условие босса — заранее решённая ставка на случай новости:
  /if 38 Scottie Barnes out under@215.5 0.57 low
  = «если Скотти Барнс не играет — прогноз ТМ215.5 с шансом 57%». Когда сторож видит, что условие
выполнено, он присылает его с кнопкой «Применить» (это тот же /verdict). Статусы: out — не играет,
in — играет (снят из отчёта о травмах или «скорее сыграет»).
Вратари НХЛ (DailyFaceoff): start — подтверждён в воротах, bench — подтверждены оба вратаря матча, и это
не он. Например: /if 47 Ersson start it2_over@2.5 0.56 low.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .data.injuries import Injury

STATUS_WORDS = {
    "out": "out", "не_играет": "out", "неиграет": "out", "нет": "out", "выбыл": "out",
    "in": "in", "играет": "in", "сыграет": "in",
    "start": "start", "starter": "start", "в_воротах": "start", "вворотах": "start", "вратарь": "start",
    "bench": "bench", "запасной": "bench", "не_в_воротах": "bench",
}
GOALIE_STATUSES = ("start", "bench")


@dataclass
class Condition:
    num: int
    player: str
    status: str  # out / in / start / bench (вратарь)
    selection: str
    line: float | None
    prob: float
    confidence: str = "low"
    fired: bool = False

    @property
    def code(self) -> str:
        if self.line is None:
            return self.selection
        return f"{self.selection}@{self.line:+g}" if self.selection.startswith("hcp") else \
            f"{self.selection}@{self.line:g}"

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Condition":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def parse_condition(args: str, sel_aliases: dict, conf_aliases: dict, parse_prob) -> Condition | str:
    """«38 Scottie Barnes out under@215.5 0.57 low» -> Condition; при ошибке — текст ошибки."""
    parts = args.replace("#", " ").split()
    if len(parts) < 5 or not parts[0].isdigit():
        return "мало данных"
    num, rest = int(parts[0]), parts[1:]
    conf = "low"
    if rest and rest[-1].lower() in conf_aliases:
        conf = conf_aliases[rest[-1].lower()]
        rest = rest[:-1]
    if len(rest) < 4:
        return "мало данных"
    prob = parse_prob(rest[-1])
    if prob is None:
        return f"не понял вероятность «{rest[-1]}»"
    code, _, raw_line = rest[-2].lower().partition("@")
    sel = sel_aliases.get(code)
    if sel is None:
        return f"не понял исход «{rest[-2]}»"
    line = None
    if raw_line:
        try:
            line = float(raw_line.replace(",", "."))
        except ValueError:
            return f"не понял линию «{raw_line}»"
    status = STATUS_WORDS.get(rest[-3].lower())
    if status is None:
        return (f"статус «{rest[-3]}»: нужно out (не играет), in (играет), "
                "start (вратарь в воротах) или bench (вратарь не в воротах)")
    player = " ".join(rest[:-3]).strip()
    if len(player) < 3:
        return "не указан игрок"
    return Condition(num, player, status, sel, line, prob, conf)


def _same_player(a: str, b: str) -> bool:
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return True
    # «Barnes» в условии ↔ «Scottie Barnes» в отчёте: совпадение по фамилии
    return a.split()[-1] == b.split()[-1] and (len(a.split()) == 1 or len(b.split()) == 1)


def player_status(name: str, teams: list[list[Injury]]) -> Injury | None:
    for inj in teams:
        for i in inj:
            if _same_player(name, i.name):
                return i
    return None


def condition_met(c: Condition, teams: list[list[Injury]], goalies: list | None = None) -> bool:
    """goalies — стартовые вратари обеих команд (GoalieStart) или None, если данных нет."""
    if c.status in GOALIE_STATUSES:
        if not goalies:
            return False
        if c.status == "start":
            return any(g.confirmed and _same_player(c.player, g.name) for g in goalies)
        return len(goalies) == 2 and all(g.confirmed and not _same_player(c.player, g.name) for g in goalies)
    found = player_status(c.player, teams)
    if c.status == "out":
        return bool(found and found.out)
    # «играет»: в отчёте его нет или «скорее сыграет»
    return found is None or found.status.lower().startswith(("probable", "available"))


def _key(status: str | None) -> str:
    return (status or "").lower()


def _is_out(status: str | None) -> bool:
    return bool(status) and Injury("", status).out


def diff_injuries(before: dict[str, str], after: dict[str, str]) -> list[tuple[str, str | None, str | None]]:
    """Изменения статусов: (игрок, было, стало); None — игрока нет в отчёте (здоров).
    «Не играет» → «список травмированных» и подобные смены внутри «не играет» не считаем новостью."""
    out = []
    for name in sorted(set(before) | set(after)):
        old, new = before.get(name), after.get(name)
        if _key(old) == _key(new):
            continue
        if _is_out(old) and _is_out(new):
            continue
        out.append((name, old, new))
    return out


def status_ru(status: str | None) -> str:
    return Injury("", status).status_ru if status else "здоров (нет в отчёте)"
