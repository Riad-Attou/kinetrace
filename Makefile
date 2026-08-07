.PHONY: bootstrap models gemx dev api web test lint build

bootstrap:
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -e './backend[dev]'
	npm --prefix frontend install
	.venv/bin/python scripts/download_models.py

models:
	.venv/bin/python scripts/download_models.py

gemx:
	./scripts/setup_gemx.sh

dev:
	./scripts/dev.sh

api:
	.venv/bin/uvicorn kinetrace.api:app --reload --host 127.0.0.1 --port 8000

web:
	npm --prefix frontend run dev

test:
	.venv/bin/pytest backend/tests
	npm --prefix frontend run test

lint:
	.venv/bin/ruff check backend/src backend/tests scripts
	npm --prefix frontend run lint

build:
	npm --prefix frontend run build
