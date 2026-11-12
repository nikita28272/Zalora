"""Форматы отчётов, которые агенты сдают главному аналитику."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Outcome1X2 = Literal["home", "draw", "away", "none"]
OutcomeTotal = Literal["over", "under", "none"]
Level = Literal["low", "medium", "high"]
Selection = Literal["home", "draw", "away", "over", "under"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NewsReport(_Strict):
    home_absences: list[str] = Field(description="Травмы, дисквалификации, вопросы по составу хозяев")
    away_absences: list[str] = Field(description="Травмы, дисквалификации, вопросы по составу гостей")
    key_factors: list[str] = Field(
        description="Мотивация, календарь и усталость, ротация, смена тренера, погода, конфликты"
    )
    lean_1x2: Outcome1X2 = Field(description="Куда новости склоняют исход 1X2, или none")
    lean_total: OutcomeTotal = Field(description="Куда новости склоняют тотал 2.5, или none")
    confidence: Level
    summary: str = Field(description="2-4 предложения по-русски: что важного и как это влияет на матч")
    sources: list[str] = Field(description="URL или названия источников")


class ExpertsReport(_Strict):
    sources_checked: int = Field(description="Сколько разных прогнозов/аналитиков удалось найти")
    consensus_1x2: Outcome1X2
    consensus_total: OutcomeTotal
    agreement: Level = Field(description="Насколько эксперты согласны между собой")
    popular_picks: list[str] = Field(
        description="Самые частые прогнозы с долей, например «Победа хозяев: 5 из 8»"
    )
    notable_arguments: list[str] = Field(description="Главные аргументы экспертов")
    summary: str = Field(description="2-4 предложения по-русски")
    sources: list[str]


class DevilReport(_Strict):
    counter_arguments: list[str] = Field(description="Сильнейшие аргументы ПРОТИВ кандидата на ставку")
    hidden_risks: list[str] = Field(description="Что может пойти не так, чего не видно в цифрах")
    severity: Level = Field(description="Насколько серьёзны возражения")
    recommend_skip: bool
    summary: str = Field(description="2-3 предложения по-русски")


class ChiefVerdict(_Strict):
    decision: Literal["bet", "skip"]
    selection: Selection = Field(
        description="Исход, о котором вердикт. При skip — самый близкий к ставке вариант"
    )
    probability: float = Field(description="Ваша оценка истинной вероятности этого исхода, от 0 до 1")
    confidence: Level
    reasoning: str = Field(description="4-7 предложений по-русски: почему такое решение")
    pros: list[str] = Field(description="Аргументы за")
    cons: list[str] = Field(description="Аргументы против")
