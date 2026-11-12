"""Форматы отчётов, которые подчинённые сдают главному, и вердикт главного.

Иерархия требует, чтобы каждый подчинённый высказался по КАЖДОМУ исходу:
П1, X, П2, ТБ 2.5, ТМ 2.5.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Level = Literal["low", "medium", "high"]
Selection = Literal["home", "draw", "away", "over", "under"]
Stance = Literal["for", "against", "neutral"]

ALL_OUTCOMES: tuple[str, ...] = ("home", "draw", "away", "over", "under")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OutcomeView(_Strict):
    selection: Selection
    stance: Stance = Field(description="for — данные говорят за этот исход, against — против, neutral — нет влияния")
    comment: str = Field(description="Одно короткое предложение по-русски: почему")


class NewsReport(_Strict):
    home_absences: list[str] = Field(description="Травмы, дисквалификации, вопросы по составу хозяев")
    away_absences: list[str] = Field(description="Травмы, дисквалификации, вопросы по составу гостей")
    key_factors: list[str] = Field(
        description="Мотивация, календарь и усталость, ротация, смена тренера, погода, конфликты"
    )
    outcomes: list[OutcomeView] = Field(description="Мнение по каждому из пяти исходов: home, draw, away, over, under")
    confidence: Level
    summary: str = Field(description="2-4 предложения по-русски: что важного и как это влияет на матч")
    sources: list[str] = Field(description="URL или названия источников")


class ExpertsReport(_Strict):
    sources_checked: int = Field(description="Сколько разных прогнозов/аналитиков удалось найти")
    agreement: Level = Field(description="Насколько эксперты согласны между собой")
    popular_picks: list[str] = Field(
        description="Самые частые прогнозы с долей, например «Победа хозяев: 5 из 8»"
    )
    outcomes: list[OutcomeView] = Field(
        description="По каждому из пяти исходов: сколько экспертов за/против, в comment укажи долю"
    )
    notable_arguments: list[str] = Field(description="Главные аргументы экспертов")
    summary: str = Field(description="2-4 предложения по-русски")
    sources: list[str]


class DevilReport(_Strict):
    outcomes: list[OutcomeView] = Field(
        description="По каждому из пяти исходов: самый сильный аргумент против него (stance обычно against)"
    )
    hidden_risks: list[str] = Field(description="Что может пойти не так, чего не видно в цифрах")
    severity: Level = Field(description="Насколько серьёзны возражения против кандидата на ставку")
    summary: str = Field(description="2-3 предложения по-русски")


class ChiefOutcome(_Strict):
    selection: Selection
    probability: float = Field(description="Ваша оценка истинной вероятности исхода, от 0 до 1")
    comment: str = Field(description="Одно предложение по-русски: итог по этому исходу с учётом всех отчётов")


class ChiefVerdict(_Strict):
    outcomes: list[ChiefOutcome] = Field(description="Оценка КАЖДОГО из пяти исходов: home, draw, away, over, under")
    decision: Literal["bet", "skip"]
    selection: Selection = Field(description="Рекомендуемый исход. При skip — самый близкий к ставке вариант")
    confidence: Level
    reasoning: str = Field(description="4-7 предложений по-русски: почему такое решение")
    pros: list[str] = Field(description="Аргументы за выбранный исход")
    cons: list[str] = Field(description="Аргументы против выбранного исхода")


def missing_outcomes(items: list) -> list[str]:
    """Каких исходов не хватает в отчёте (по полю selection)."""
    present = {getattr(x, "selection", None) for x in items}
    return [o for o in ALL_OUTCOMES if o not in present]
