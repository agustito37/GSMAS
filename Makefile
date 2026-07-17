.PHONY: up down logs ps restart install run lint format typecheck check

up:
	docker compose up -d
	@echo "neo4j → http://localhost:7474 (bolt: 7687)"

down:
	docker compose down

logs:
	docker compose logs -f

ps:
	docker compose ps

restart:
	docker compose restart

install:
	uv sync

run:
	uv run python -m app.main

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run pyright

check: lint typecheck
