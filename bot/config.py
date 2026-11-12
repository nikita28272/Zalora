"""Настройки бота. Всё читается из переменных окружения (файл .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class League:
    odds_key: str  # ключ в The Odds API
    fd_code: str  # код в football-data.org
    name_ru: str


LEAGUES: tuple[League, ...] = (
    League("soccer_epl", "PL", "Англия. Премьер-лига"),
    League("soccer_spain_la_liga", "PD", "Испания. Ла Лига"),
    League("soccer_italy_serie_a", "SA", "Италия. Серия А"),
    League("soccer_germany_bundesliga", "BL1", "Германия. Бундеслига"),
    League("soccer_france_ligue_one", "FL1", "Франция. Лига 1"),
    League("soccer_uefa_champs_league", "CL", "Лига чемпионов"),
)
LEAGUES_BY_ODDS_KEY = {lg.odds_key: lg for lg in LEAGUES}


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

    # --- LLM ---
    llm_model_chief: str = field(default_factory=lambda: _env("LLM_MODEL_CHIEF", "claude-opus-5-5"))
    llm_model_agents: str = field(default_factory=lambda: _env("LLM_MODEL_AGENTS", "claude-opus-5-5"))
    llm_effort_chief: str = field(default_factory=lambda: _env("LLM_EFFORT_CHIEF", "medium"))
    llm_effort_agents: str = field(default_factory=lambda: _env("LLM_EFFORT_AGENTS", "low"))
    web_search_enabled: bool = field(default_factory=lambda: _bool("WEB_SEARCH_ENABLED", True))
    web_search_max_uses: int = field(default_factory=lambda: _int("WEB_SEARCH_MAX_USES", 3))
    refusal_fallback: bool = field(default_factory=lambda: _bool("LLM_REFUSAL_FALLBACK", True))
    max_llm_matches_per_scan: int = field(default_factory=lambda: _int("MAX_LLM_MATCHES_PER_SCAN", 2))
    max_llm_analyses_per_day: int = field(default_factory=lambda: _int("MAX_LLM_ANALYSES_PER_DAY", 3))

    # --- модель и сигналы ---
    min_edge: float = field(default_factory=lambda: _float("MIN_EDGE", 0.03))
    max_prob_deviation: float = field(default_factory=lambda: _float("MAX_PROB_DEVIATION", 0.08))
    min_odds: float = field(default_factory=lambda: _float("MIN_ODDS", 1.40))
    max_odds: float = field(default_factory=lambda: _float("MAX_ODDS", 4.50))
    lookahead_hours: int = field(default_factory=lambda: _int("LOOKAHEAD_HOURS", 48))

    # --- банкролл и лимиты (финансист) ---
    start_bankroll: float = field(default_factory=lambda: _float("START_BANKROLL", 10000))
    kelly_fraction: float = field(default_factory=lambda: _float("KELLY_FRACTION", 0.25))
    max_stake_pct: float = field(default_factory=lambda: _float("MAX_STAKE_PCT", 0.02))
    max_bets_per_day: int = field(default_factory=lambda: _int("MAX_BETS_PER_DAY", 3))
    daily_loss_limit_pct: float = field(default_factory=lambda: _float("DAILY_LOSS_LIMIT_PCT", 0.05))
    stop_after_losses: int = field(default_factory=lambda: _int("STOP_AFTER_LOSSES", 4))

    # --- расписание ---
    timezone: str = field(default_factory=lambda: _env("TIMEZONE", "Europe/Moscow"))
    scan_time: str = field(default_factory=lambda: _env("SCAN_TIME", "11:00"))
    settle_time: str = field(default_factory=lambda: _env("SETTLE_TIME", "09:30"))

    db_path: str = field(default_factory=lambda: _env("DB_PATH", "data/bot.db"))

    def missing_keys(self) -> list[str]:
        missing = []
        if not self.telegram_token:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not self.odds_api_key:
            missing.append("ODDS_API_KEY")
        if not self.football_data_key:
            missing.append("FOOTBALL_DATA_API_KEY")
        if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
            missing.append("ANTHROPIC_API_KEY")
        return missing


settings = Settings()
