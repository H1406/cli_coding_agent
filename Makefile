.PHONY: install dev lint test run docker-build

install:
	pip install -e .

dev:
	pip install -e .[dev]

lint:
	ruff check src tests

test:
	pytest

run:
	python -m cli_coding_agent.cli

docker-build:
	docker build -t cli-coding-agent .
