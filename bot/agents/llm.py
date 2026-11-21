"""Обёртка над Claude API: один вызов агента с гарантированно структурированным ответом.

Агент может искать в интернете (серверный инструмент web_search) и обязан
вернуть итог через инструмент submit_report со строгой JSON-схемой.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from ..config import settings

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

SUBMIT_TOOL = "submit_report"
MAX_ROUNDS = 6

# Цены за 1M токенов (вход, выход), USD. Для оценки расходов в /usage.
PRICES = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}
WEB_SEARCH_PRICE = 10.0 / 1000  # USD за один поиск

_client: anthropic.AsyncAnthropic | None = None


def client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(max_retries=3, timeout=300)
    return _client


class AgentError(RuntimeError):
    pass


@dataclass
class AgentRun(Generic[T]):
    result: T
    input_tokens: int
    output_tokens: int
    web_searches: int
    model: str

    @property
    def cost_usd(self) -> float:
        p_in, p_out = PRICES.get(self.model, PRICES["claude-opus-5-5"])
        return self.input_tokens / 1e6 * p_in + self.output_tokens / 1e6 * p_out + self.web_searches * WEB_SEARCH_PRICE


def _supports_server_fallback(model: str) -> bool:
    return model.startswith(("claude-opus-5", "claude-sonnet-5-5", "claude-fable-5"))


def _web_search_tool(model: str, max_uses: int) -> dict:
    # Версия с динамической фильтрацией есть у Opus/Sonnet/Fable; у Haiku — базовая.
    tool_type = "web_search_20250305" if model.startswith("claude-haiku") else "web_search_20260209"
    return {"type": tool_type, "name": "web_search", "max_uses": max_uses}


async def run_agent(
    *,
    model: str,
    effort: str,
    system: str,
    prompt: str,
    schema: type[T],
    web_search: bool = False,
    max_searches: int = 3,
    validate: Callable[[T], str | None] | None = None,
) -> AgentRun[T]:
    """validate(result) возвращает текст замечания, если отчёт нужно доработать, иначе None."""
    submit_tool = {
        "name": SUBMIT_TOOL,
        "description": "Отправить итоговый отчёт главному аналитику. Вызывается ровно один раз, в конце работы.",
        "strict": True,
        "input_schema": schema.model_json_schema(),
    }
    tools: list[dict] = []
    if web_search:
        tools.append(_web_search_tool(model, max_searches))
    tools.append(submit_tool)

    messages: list[dict] = [{"role": "user", "content": prompt}]
    request = dict(
        model=model,
        max_tokens=16000,
        system=system,
        tools=tools,
        tool_choice={"type": "auto"},
        output_config={"effort": effort},
    )
    use_fallback = settings.refusal_fallback and _supports_server_fallback(model)

    in_tok = out_tok = searches = 0
    for _ in range(MAX_ROUNDS):
        try:
            if use_fallback:
                resp = await client().beta.messages.create(
                    **request,
                    messages=messages,
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                )
            else:
                resp = await client().messages.create(**request, messages=messages)
        except anthropic.AuthenticationError as exc:
            raise AgentError("Неверный ANTHROPIC_API_KEY") from exc
        except anthropic.RateLimitError as exc:
            raise AgentError("Превышен лимит запросов Claude API, попробуйте позже") from exc
        except anthropic.APIStatusError as exc:
            raise AgentError(f"Ошибка Claude API ({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise AgentError("Нет связи с Claude API") from exc

        usage = resp.usage
        in_tok += (usage.input_tokens or 0) + (getattr(usage, "cache_read_input_tokens", 0) or 0)
        out_tok += usage.output_tokens or 0
        stu = getattr(usage, "server_tool_use", None)
        searches += (getattr(stu, "web_search_requests", 0) or 0) if stu else 0

        if resp.stop_reason == "refusal":
            raise AgentError("Модель отказалась отвечать на этот запрос")

        submit = next(
            (b for b in resp.content if b.type == "tool_use" and b.name == SUBMIT_TOOL), None
        )
        messages.append({"role": "assistant", "content": resp.content})

        if submit is not None:
            try:
                result = schema.model_validate(submit.input)
                problem = validate(result) if validate else None
            except ValidationError as exc:
                problem = f"Отчёт не соответствует схеме: {exc}"
            if problem is None:
                return AgentRun(result, in_tok, out_tok, searches, model)
            # Возвращаем отчёт на доработку
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": submit.id,
                    "is_error": True,
                    "content": f"Отчёт не принят: {problem} Исправь и отправь submit_report заново.",
                }],
            })
            continue

        if resp.stop_reason == "pause_turn":
            continue  # сервер продолжит поиск с того же места
        messages.append(
            {"role": "user", "content": f"Заверши работу: отправь итог через инструмент {SUBMIT_TOOL}."}
        )

    raise AgentError("Агент не вернул отчёт за отведённое число шагов")
