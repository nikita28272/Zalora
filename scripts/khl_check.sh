#!/usr/bin/env bash
# Проверка КХЛ в API-Sports (около 8 запросов из 100 бесплатных в сутки).
# Сначала впишите ключ:  bash scripts/set.sh API_SPORTS_KEY ваш_ключ
# Запуск из папки проекта:  bash scripts/khl_check.sh
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose build -q bot && docker compose run --rm bot python -m bot.khl_check
