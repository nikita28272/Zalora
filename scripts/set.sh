#!/usr/bin/env bash
# Поменять одну настройку и перезапустить бота.
# Пример:  bash scripts/set.sh TELEGRAM_OWNER_ID 123456789
#          bash scripts/set.sh MAX_LLM_ANALYSES_PER_DAY 2
set -euo pipefail

cd "$(dirname "$0")/.."

if [ $# -ne 2 ]; then
    echo "Использование: bash scripts/set.sh НАСТРОЙКА ЗНАЧЕНИЕ"
    echo "Например:      bash scripts/set.sh TELEGRAM_OWNER_ID 123456789"
    exit 1
fi

key="$1"
value="$(echo "$2" | tr -d '[:space:]')"

if ! [[ "$key" =~ ^[A-Z_]+$ ]]; then
    echo "Название настройки пишется заглавными латинскими буквами, например MIN_EDGE"
    exit 1
fi
if [[ "$value" == *"|"* ]]; then
    echo "Значение не должно содержать символ |"
    exit 1
fi

[ -f .env ] || cp .env.example .env
chmod 600 .env

if grep -q "^${key}=" .env; then
    sed -i "s|^${key}=.*|${key}=${value}|" .env
else
    echo "${key}=${value}" >> .env
fi
echo "Записал ${key}."

if [ "${SKIP_DOCKER:-0}" = "1" ]; then
    exit 0
fi
# up -d пересоздаёт контейнер с новыми переменными (restart их не перечитывает)
docker compose up -d
echo "✅ Бот перезапущен с новой настройкой."
