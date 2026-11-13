"""Настройки бота. Всё читается из переменных окружения (файл .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class League:
    odds_key: str  # ключ в The Odds API
    fd_code: str  # код лиги (для футбола — код football-data.org)
    name_ru: str
    sport: str = "football"  # football / hockey / basketball


ALL_LEAGUES: tuple[League, ...] = (
    League("soccer_epl", "PL", "Англия. Премьер-лига"),
    League("soccer_spain_la_liga", "PD", "Испания. Ла Лига"),
    League("soccer_italy_serie_a", "SA", "Италия. Серия А"),
    League("soccer_germany_bundesliga", "BL1", "Германия. Бундеслига"),
    League("soccer_france_ligue_one", "FL1", "Франция. Лига 1"),
    League("soccer_uefa_champs_league", "CL", "Лига чемпионов"),
    League("icehockey_nhl", "NHL", "НХЛ", sport="hockey"),
    League("basketball_nba", "NBA", "НБА", sport="basketball"),
)


def _enabled_leagues() -> tuple[League, ...]:
    # ENABLED_LEAGUES=PL,NHL,BL1 — какие лиги сканировать и в каком порядке приоритета
    # (при нехватке кредитов первыми пропускаются последние). Пусто — все лиги.
    raw = os.getenv("ENABLED_LEAGUES", "").strip()
    if not raw:
        return ALL_LEAGUES
    by_code = {lg.fd_code: lg for lg in ALL_LEAGUES}
    codes = [c.strip().upper() for c in raw.split(",") if c.strip()]
    return tuple(dict.fromkeys(by_code[c] for c in codes if c in by_code))


LEAGUES: tuple[League, ...] = _enabled_leagues()
LEAGUES_BY_ODDS_KEY = {lg.odds_key: lg for lg in ALL_LEAGUES}


# Профили риска финансиста: доля Келли, предохранитель на ставку, всё в игре, один матч, база ставки НБА «по анализу»
RISK_PROFILES = {
    "careful": {"kelly": 0.25, "cap": 0.02, "exposure": 0.06, "match": 0.03, "nba": 0.01},
    "moderate": {"kelly": 0.4, "cap": 0.05, "exposure": 0.12, "match": 0.06, "nba": 0.015},
    "aggressive": {"kelly": 0.6, "cap": 0.08, "exposure": 0.20, "match": 0.10, "nba": 0.025},
}
RISK_NAMES_RU = {"careful": "осторожный", "moderate": "умеренный", "aggressive": "агрессивный"}


def _risk_profile() -> str:
    name = os.getenv("RISK_PROFILE", "moderate").strip().lower()
    aliases = {"осторожный": "careful", "умеренный": "moderate", "агрессивный": "aggressive"}
    name = aliases.get(name, name)
    return name if name in RISK_PROFILES else "moderate"


def _profiled(env: str, key: str) -> float:
    """Значение из .env, если задано явно, иначе — из профиля риска."""
    raw = os.getenv(env, "").strip()
    if raw:
        try:
            return float(raw.replace(",", "."))
        except ValueError:
            pass
    return RISK_PROFILES[_risk_profile()][key]


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _float(name: str, default: float) -> float:
    raw = _env(name)
    return float(raw) if raw else default


def _int(name: str, default: int) -> int:
    raw = _env(name)
    return int(raw) if raw else default


def _bool(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on", "да"}


@dataclass(frozen=True)
class Settings:
    # --- ключи ---
    telegram_token: str = field(default_factory=lambda: _env("TELEGRAM_BOT_TOKEN"))
    owner_id: int = field(default_factory=lambda: _int("TELEGRAM_OWNER_ID", 0))
    odds_api_key: str = field(default_factory=lambda: _env("ODDS_API_KEY"))
    football_data_key: str = field(default_factory=lambda: _env("FOOTBALL_DATA_API_KEY"))

    # --- режим агентов ---
    # manual — «босс в чате»: бот готовит досье, разбор делает Claude в чате, вердикт приходит командой /verdict;
    # api    — агенты работают через Claude API прямо в боте (нужен ANTHROPIC_API_KEY).
    agents_mode: str = field(default_factory=lambda: _env("AGENTS_MODE", "manual").lower())
    manual_candidates_per_scan: int = field(default_factory=lambda: _int("MANUAL_CANDIDATES_PER_SCAN", 3))

    # --- LLM (только для AGENTS_MODE=api) ---
    llm_model_chief: str = field(default_factory=lambda: _env("LLM_MODEL_CHIEF", "claude-opus-5-5"))
    # Экономичный режим: босс на Opus, помощники на Sonnet (примерно вдвое дешевле Opus)
    llm_model_agents: str = field(default_factory=lambda: _env("LLM_MODEL_AGENTS", "claude-sonnet-5-5"))
    llm_effort_chief: str = field(default_factory=lambda: _env("LLM_EFFORT_CHIEF", "medium"))
    llm_effort_agents: str = field(default_factory=lambda: _env("LLM_EFFORT_AGENTS", "low"))
    web_search_enabled: bool = field(default_factory=lambda: _bool("WEB_SEARCH_ENABLED", True))
    web_search_max_uses: int = field(default_factory=lambda: _int("WEB_SEARCH_MAX_USES", 3))
    refusal_fallback: bool = field(default_factory=lambda: _bool("LLM_REFUSAL_FALLBACK", True))
    max_llm_matches_per_scan: int = field(default_factory=lambda: _int("MAX_LLM_MATCHES_PER_SCAN", 2))
    max_llm_analyses_per_day: int = field(default_factory=lambda: _int("MAX_LLM_ANALYSES_PER_DAY", 3))

    # --- модель и сигналы ---
    min_edge: float = field(default_factory=lambda: _float("MIN_EDGE", 0.03))
    # насколько модель может сдвинуть оценку рынка: 0.35 — на треть разницы между ними
    model_weight: float = field(default_factory=lambda: _float("MODEL_WEIGHT", 0.35))
    # расхождение модели с рынком больше этого — почти всегда ошибка модели: такой исход не кандидат
    max_model_gap: float = field(default_factory=lambda: _float("MAX_MODEL_GAP", 0.12))
    # Отбор матчей для разбора: сколько показывать наверху и с какой оценки перевеса (−1% = почти в ноль:
    # агенты и кэф вашей конторы могут довести такой исход до перевеса)
    # НБА: «analysis» — ставки по анализу (агенты, новости, график), а не по перевесу над линией;
    # «value» — как в остальных лигах
    nba_mode: str = field(default_factory=lambda: _env("NBA_MODE", "analysis").lower())
    # НБА «по анализу»: сначала бесплатно смотрим расписание и график; линии (3 кредита) грузим,
    # только если есть матч с перекосом графика
    nba_odds_gate: bool = field(default_factory=lambda: _env("NBA_ODDS_GATE", "0") not in ("0", "false", "no"))
    # Матчи дня к разбору: сколько матчей каждого вида спорта предлагать ежедневно
    # 🔥 автоматически: перевес по оценке босса не меньше этого (НБА «по анализу» — высокая уверенность)
    hot_min_edge: float = field(default_factory=lambda: _float("HOT_MIN_EDGE", 0.03))
    daily_hockey: int = field(default_factory=lambda: _int("DAILY_HOCKEY", 2))
    daily_nba: int = field(default_factory=lambda: _int("DAILY_NBA", 2))
    daily_football: int = field(default_factory=lambda: _int("DAILY_FOOTBALL", 2))
    daily_window_hours: int = field(default_factory=lambda: _int("DAILY_WINDOW_HOURS", 30))  # «ближайшие сутки»
    nba_top: int = field(default_factory=lambda: _int("NBA_TOP", 3))  # сколько матчей НБА в день предлагать
    nba_min_factor: float = field(default_factory=lambda: _float("NBA_MIN_FACTOR", 2.0))  # порог перекоса графика
    nba_stake_pct: float = field(default_factory=lambda: _profiled("NBA_STAKE_PCT", "nba"))  # база ставки НБА
    nba_max_prob_deviation: float = field(default_factory=lambda: _float("NBA_MAX_PROB_DEVIATION", 0.12))
    # Если линии уже загружались сегодня (например, днём нажали «🔍 Анализ») не раньше чем за столько часов,
    # ежедневный список в SCAN_TIME берёт их из кеша и не тратит кредиты; 0 — всегда загружать заново
    daily_reuse_hours: int = field(default_factory=lambda: _int("DAILY_REUSE_HOURS", 8))
    top_matches: int = field(default_factory=lambda: _int("TOP_MATCHES", 5))
    top_min_edge: float = field(default_factory=lambda: _float("TOP_MIN_EDGE", -0.01))
    max_prob_deviation: float = field(default_factory=lambda: _float("MAX_PROB_DEVIATION", 0.08))
    min_odds: float = field(default_factory=lambda: _float("MIN_ODDS", 1.40))
    max_odds: float = field(default_factory=lambda: _float("MAX_ODDS", 4.50))
    lookahead_hours: int = field(default_factory=lambda: _int("LOOKAHEAD_HOURS", 48))
    # --- экономия кредитов The Odds API ---
    # /matches и досье берут линии из кеша, если им не больше N минут
    odds_cache_minutes: int = field(default_factory=lambda: _int("ODDS_CACHE_MINUTES", 120))
    # ежедневный скан обновляет линии, если им больше N минут
    odds_scan_max_age_minutes: int = field(default_factory=lambda: _int("ODDS_SCAN_MAX_AGE_MINUTES", 60))
    # день месяца, когда The Odds API обновляет 500 кредитов (смотрите в личном кабинете)
    odds_reset_day: int = field(default_factory=lambda: _int("ODDS_RESET_DAY", 1))
    # неприкосновенный запас кредитов на ручные запросы
    odds_reserve: int = field(default_factory=lambda: _int("ODDS_RESERVE", 20))
    # только прематч: матчи, до начала которых меньше этого времени, не разбираем
    min_minutes_before_kickoff: int = field(default_factory=lambda: _int("MIN_MINUTES_BEFORE_KICKOFF", 60))
    # Режим «босс в чате»: вердикт принимается до этого срока (поздняя проверка идёт после составов)
    verdict_cutoff_minutes: int = field(default_factory=lambda: _int("VERDICT_CUTOFF_MINUTES", 10))
    # Напоминание о поздней проверке: в хоккее вратарей подтверждают за 1–3 часа,
    # в футболе составы объявляют за час до начала
    late_check_hockey_minutes: int = field(default_factory=lambda: _int("LATE_CHECK_HOCKEY_MINUTES", 120))
    # НБА: официальный отчёт о травмах уточняется до последнего часа, составы — за 30 минут
    late_check_basketball_minutes: int = field(default_factory=lambda: _int("LATE_CHECK_BASKETBALL_MINUTES", 60))
    late_check_football_minutes: int = field(default_factory=lambda: _int("LATE_CHECK_FOOTBALL_MINUTES", 50))
    # Тихие часы для напоминаний: «0-7» — не присылать с 00:00 до 07:00 по TIMEZONE; пусто (по умолчанию) — присылать всегда
    quiet_hours: str = field(default_factory=lambda: _env("QUIET_HOURS", ""))

    # --- банкролл и лимиты (финансист) ---
    start_bankroll: float = field(default_factory=lambda: _float("START_BANKROLL", 10000))
    # Профиль риска: careful / moderate / aggressive (осторожный / умеренный / агрессивный)
    risk_profile: str = field(default_factory=_risk_profile)
    kelly_fraction: float = field(default_factory=lambda: _profiled("KELLY_FRACTION", "kelly"))
    max_stake_pct: float = field(default_factory=lambda: _profiled("MAX_STAKE_PCT", "cap"))  # предохранитель
    # Управление банком: просадка от пика -> режим ставок
    dd_caution: float = field(default_factory=lambda: _float("DD_CAUTION", 0.10))  # ×0.5
    dd_defense: float = field(default_factory=lambda: _float("DD_DEFENSE", 0.20))  # ×0.25
    dd_stop: float = field(default_factory=lambda: _float("DD_STOP", 0.30))  # стоп до /resume
    weekly_loss_limit_pct: float = field(default_factory=lambda: _float("WEEKLY_LOSS_LIMIT_PCT", 0.10))
    max_exposure_pct: float = field(default_factory=lambda: _profiled("MAX_EXPOSURE_PCT", "exposure"))  # всё в игре
    max_match_pct: float = field(default_factory=lambda: _profiled("MAX_MATCH_PCT", "match"))  # на один матч
    # сумма бумажного учёта (и потолок совета) для прогнозов босса без перевеса
    paper_stake_pct: float = field(default_factory=lambda: _float("PAPER_STAKE_PCT", 0.005))
    max_bets_per_day: int = field(default_factory=lambda: _int("MAX_BETS_PER_DAY", 3))
    daily_loss_limit_pct: float = field(default_factory=lambda: _float("DAILY_LOSS_LIMIT_PCT", 0.05))
    stop_after_losses: int = field(default_factory=lambda: _int("STOP_AFTER_LOSSES", 4))

    # --- расписание ---
    timezone: str = field(default_factory=lambda: _env("TIMEZONE", "Europe/Moscow"))
    scan_time: str = field(default_factory=lambda: _env("SCAN_TIME", "11:00"))
    settle_time: str = field(default_factory=lambda: _env("SETTLE_TIME", "09:00"))

    db_path: str = field(default_factory=lambda: _env("DB_PATH", "data/bot.db"))

    def missing_keys(self) -> list[str]:
        missing = []
        if not self.telegram_token:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not self.odds_api_key:
            missing.append("ODDS_API_KEY")
        if not self.football_data_key:
            missing.append("FOOTBALL_DATA_API_KEY")
        if self.api_mode and not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
            missing.append("ANTHROPIC_API_KEY")
        return missing

    @property
    def api_mode(self) -> bool:
        return self.agents_mode == "api"


settings = Settings()
