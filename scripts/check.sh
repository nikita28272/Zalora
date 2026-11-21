#!/usr/bin/env bash
# Проверить все ключи и источники данных (ничего не тратит).
# Запуск из папки проекта:  bash scripts/check.sh
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose build -q bot && docker compose run --rm bot python -m bot.check
