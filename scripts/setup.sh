#!/usr/bin/env bash
# Первичная настройка и запуск бота. Удобно для телефона: ключи спрашиваются по очереди.
# Запуск из папки проекта:  bash scripts/setup.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
    cp .env.example .env
fi
chmod 600 .env

set_var() {  # set_var KEY VALUE — записать значение в .env
    local key="$1" value="$2"
    if grep -q "^${key}=" .env; then
        # разделитель | : в токенах его не бывает
        sed -i "s|^${key}=.*|${key}=${value}|" .env
    else
        echo "${key}=${value}" >> .env
    fi
}

current() {
    grep "^$1=" .env | head -n1 | cut -d= -f2- || true
}

ask() {  # ask KEY "Вопрос" [значение по умолчанию]
    local key="$1" prompt="$2" def="${3:-}" cur value
    cur="$(current "$key")"
    [ -n "$cur" ] && def="$cur"
    if [ -n "$def" ]; then
        read -r -p "$prompt [Enter — оставить ${def:0:6}…]: " value
    else
        read -r -p "$prompt: " value
    fi
    value="$(echo "${value:-$def}" | tr -d '[:space:]')"
    set_var "$key" "$value"
}

echo "=== Настройка бота сигналов ==="
echo "Вставляйте ключи по одному и жмите Enter."
echo
ask TELEGRAM_BOT_TOKEN    "1/4 Токен бота от @BotFather"
ask ODDS_API_KEY          "2/4 Ключ The Odds API"
ask FOOTBALL_DATA_API_KEY "3/4 Ключ football-data.org"
ask START_BANKROLL        "4/4 Ваш банк в рублях" "10000"

# Режим агентов: по умолчанию «босс в чате», ключ Claude API не нужен
if [ "$(current AGENTS_MODE)" = "api" ]; then
    ask ANTHROPIC_API_KEY "Ключ Claude API (console.anthropic.com)"
else
    set_var AGENTS_MODE manual
fi
echo

missing=""
required="TELEGRAM_BOT_TOKEN ODDS_API_KEY FOOTBALL_DATA_API_KEY"
[ "$(current AGENTS_MODE)" = "api" ] && required="$required ANTHROPIC_API_KEY"
for k in $required; do
    [ -z "$(current "$k")" ] && missing="$missing $k"
done
if [ -n "$missing" ]; then
    echo "Не заполнено:$missing. Запустите скрипт ещё раз: bash scripts/setup.sh"
    exit 1
fi

if [ "${SKIP_DOCKER:-0}" = "1" ]; then
    echo "Файл .env готов (запуск Docker пропущен)."
    exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "Устанавливаю Docker (1–3 минуты)…"
    curl -fsSL https://get.docker.com | sh
fi

echo "Собираю и запускаю бота (первый раз 2–5 минут)…"
docker compose up -d --build

echo
echo "Проверяю ключи и источники данных…"
docker compose run --rm bot python -m bot.check || true

echo
echo "✅ Бот запущен."
echo "Дальше: напишите своему боту в Telegram любое сообщение — он пришлёт ваш ID."
echo "Затем выполните:  bash scripts/set.sh TELEGRAM_OWNER_ID ВАШ_ID"
