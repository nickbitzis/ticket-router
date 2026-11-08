PORT ?= 8836
TICKETS ?= 20000

.PHONY: install test lint format run eval gate drift feedback-sim screenshot docker-build clean

install:
	uv sync --python 3.12

test:
	uv run pytest -q

lint:
	uvx ruff check .
	uvx ruff format --check .

format:
	uvx ruff format .
	uvx ruff check . --fix

run:
	PORT=$(PORT) uv run ticketrouter serve --tickets $(TICKETS)

eval:
	uv run ticketrouter eval --tickets $(TICKETS) --out reports/eval.json

gate:  ## same check CI runs: fail if the operating point loses accuracy or goes over budget
	uv run ticketrouter eval --tickets $(TICKETS) --gate

drift:
	uv run ticketrouter drift --tickets $(TICKETS)

feedback-sim:
	uv run ticketrouter feedback-sim --tickets $(TICKETS) --out reports/feedback.json

screenshot:  ## start the server first with make run
	uv run python scripts/screenshot.py --port $(PORT)

docker-build:
	docker build -t ticket-router .

clean:
	rm -rf .pytest_cache .ruff_cache data reports
