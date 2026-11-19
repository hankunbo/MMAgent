PYTHON ?= python3

.PHONY: install-dev format format-check lint test check

install-dev:
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install pytest ruff httpx

format:
	$(PYTHON) -m ruff format Algorithm_Tool agents apps packages tests

format-check:
	$(PYTHON) -m ruff format --check Algorithm_Tool agents apps packages tests

lint:
	$(PYTHON) -m ruff check Algorithm_Tool agents apps packages tests

test:
	$(PYTHON) -m pytest

check: format-check lint test
