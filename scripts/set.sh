#!/usr/bin/env bash
# Поменять одну или несколько настроек и перезапустить бота один раз.
# Примеры:  bash scripts/set.sh TELEGRAM_OWNER_ID 123456789
#           bash scripts/set.sh TIMEZONE Asia/Novosibirsk SCAN_TIME 16:00 SETTLE_TIME 15:00
set -euo pipefail

cd "$(dirname "$0")/.."

if [ $# -lt 2 ] || [ $(( $# % 2 )) -ne 0 ]; then
    echo "Использование: bash scripts/set.sh НАСТРОЙКА ЗНАЧЕНИЕ [НАСТРОЙКА ЗНАЧЕНИЕ ...]"
    echo "Например:      bash scripts/set.sh SCAN_TIME 16:00 SETTLE_TIME 15:00"
    exit 1
fi

[ -f .env ] || cp .env.example .env
chmod 600 .env

while [ $# -gt 0 ]; do
    key="$1"
    value="$(echo "$2" | tr -d '[:space:]')"
    shift 2

    if ! [[ "$key" =~ ^[A-Z_]+$ ]]; then
        echo "Название настройки пишется заглавными латинскими буквами, например MIN_EDGE (получено: $key)"
        exit 1
    fi
    if [[ "$value" == *"|"* ]]; then
        echo "Значение не должно содержать символ |"
        exit 1
    fi

    if grep -q "^${key}=" .env; then
        sed -i "s|^${key}=.*|${key}=${value}|" .env
    else
        echo "${key}=${value}" >> .env
    fi
    echo "Записал ${key}."
done

if [ "${SKIP_DOCKER:-0}" = "1" ]; then
    exit 0
fi
# up -d пересоздаёт контейнер с новыми переменными (restart их не перечитывает)
docker compose up -d
echo "✅ Бот перезапущен с новыми настройками."
