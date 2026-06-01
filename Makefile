.PHONY: setup test download-data prepare-data train-baseline train-edge export-tflite benchmark collect-results

PYTHON ?= python

setup:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e . --no-deps

test:
	$(PYTHON) -m pytest

download-data:
	$(PYTHON) scripts/00_download_data.py

prepare-data:
	$(PYTHON) scripts/01_prepare_dataset.py

train-baseline:
	$(PYTHON) scripts/02_train_baseline.py

train-edge:
	$(PYTHON) scripts/03_train_edge.py

export-tflite:
	$(PYTHON) scripts/04_export_tflite.py

benchmark:
	$(PYTHON) scripts/05_run_benchmarks.py

collect-results:
	$(PYTHON) scripts/06_collect_tables.py
