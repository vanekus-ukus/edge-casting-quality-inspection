# Reference snapshot: edge casting NIR 2026

Репозиторий содержит воспроизводимый снимок НИР по теме:
«Методика адаптации моделей компьютерного зрения для контроля качества литых изделий на edge-устройствах с различными ресурсными ограничениями».

## Датасет

Используется Kaggle dataset `ravirajsinh45/real-life-industrial-dataset-of-casting-product`.
Датасет не включен в snapshot из-за лицензии и размера. Его нужно скачать отдельно через Kaggle API или вручную.

## Структура

- `src/`, `scripts/`, `configs/`, `tests/` - код пайплайна.
- `experiments/research_20260601/dedup_training/tables/` - финальные таблицы clean split.
- `experiments/research_20260601/dedup_training/metrics/` - метрики и prediction-файлы.
- `experiments/research_20260601/dedup_training/exported/` - exact TFLite FP32/INT8.
- `experiments/research_20260601/scientific_strengthening/dedup_split_protocol/` - протокол deduplicated split.
- `reports/final_nir/` - LaTeX, PDF и рисунки НИР.

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
```

## PDF

```bash
cd reports/final_nir
./build.sh
```

Готовый файл: `reports/final_nir/final_nir.pdf`.

## Ограничения

Latency является proxy-измерением на текущей машине. Raspberry Pi/OpenMV аппаратно не тестировались. Совместимость TFLite Micro/OpenMV firmware требует отдельной проверки.
