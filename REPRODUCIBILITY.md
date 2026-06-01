# Reproducibility

## Проверка окружения

```bash
python -m pytest
```

## Данные

```bash
make download-data
make prepare-data
```

Kaggle token должен лежать в `~/.kaggle/kaggle.json` или в каталоге, указанном через `KAGGLE_CONFIG_DIR`.

## Deduplicated split

```bash
python experiments/research_20260601/scientific_strengthening/dedup_split_protocol/create_dedup_split.py \
  --materialize-dir experiments/research_20260601/dedup_training/data/processed_clean
```

Финальные таблицы уже сохранены в `experiments/research_20260601/dedup_training/tables/`.

## Dedup training

Обучение в snapshot не запускается автоматически. Для полного воспроизведения используйте сохраненные конфиги проекта и отдельную output-папку, чтобы не перезаписать reference-результаты.

## Exact TFLite

Финальные exact TFLite артефакты:

- `experiments/research_20260601/dedup_training/exported/micro_edge_dedup_exact_fp32.tflite`
- `experiments/research_20260601/dedup_training/exported/micro_edge_dedup_exact_int8.tflite`

## Сборка PDF

```bash
cd reports/final_nir
./build.sh
```
