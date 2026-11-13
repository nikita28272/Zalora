#!/usr/bin/env bash
# Диагностика модели: на чём она построила оценку команд (кредиты не тратит).
# Пример:  bash scripts/diag.sh NHL "Vegas" "Nashville"
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose run --rm bot python -m bot.diag "$@"
