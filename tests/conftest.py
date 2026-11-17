import pytest

from bot.config import settings


def set_setting(name: str, value):
    object.__setattr__(settings, name, value)  # Settings — frozen dataclass


@pytest.fixture(autouse=True)
def _relaxed_stake_rules(monkeypatch):
    """Старые тесты писались до правил «шанс ≥ 50% и шанс × кэф ≥ 1.03»: по умолчанию их отключаем,
    а проверяем отдельными тестами (test_min_stake_*). Отчёты о травмах в тестах не скачиваем."""
    old = settings.min_stake_prob, settings.min_stake_ev
    set_setting("min_stake_prob", 0.0)
    set_setting("min_stake_ev", -1.0)

    async def no_injuries(self, sport, max_age=None):
        return {}

    monkeypatch.setattr("bot.data.injuries.InjuryClient.fetch", no_injuries)
    yield
    set_setting("min_stake_prob", old[0])
    set_setting("min_stake_ev", old[1])
