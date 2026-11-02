#!/usr/bin/env bash
# Та же сетка конфигураций, что run_baseline.py для малых моделей.
# Каждая конфигурация — отдельный процесс.
# Использование: bash run_large.sh llama3-8b
set -e
M=${1:?укажите модель: phi3-mini | llama3-8b}

for cfg in "16 0 0" "8 128 0" "4 -1 0" "4 128 0" "4 128 1" "4 64 0" "3 128 0" "2 128 0"; do
  set -- $cfg
  HF_HUB_OFFLINE=1 python run_large.py --model "$M" --bits "$1" --group-size "$2" --sym "$3"
done
