import pytest

from bot.config import settings


def set_setting(name: str, value):
    object.__setattr__(settings, name, value)  # Settings — frozen dataclass


@pytest.fixture(autouse=True)
def _no_min_stake_prob():
    """Старые тесты писались до правила «не ставить при шансе ниже 55%»: по умолчанию его отключаем,
    а проверяем отдельным тестом (test_min_stake_prob_*)."""
    old = settings.min_stake_prob
    set_setting("min_stake_prob", 0.0)
    yield
    set_setting("min_stake_prob", old)
