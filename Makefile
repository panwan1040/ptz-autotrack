PYTHON ?= python3
APP ?= ptz-autotrack

.PHONY: install dev test lint typecheck run detect-only print-config docker-build

install:
	$(PYTHON) -m pip install -e .

dev:
	$(PYTHON) -m pip install -e .[dev]

test:
	pytest -q

lint:
	ruff check app tests

typecheck:
	mypy app

run:
	$(APP) run

detect-only:
	$(APP) detect-only

print-config:
	$(APP) print-config

docker-build:
	docker build -t ptz-autotrack:latest .
