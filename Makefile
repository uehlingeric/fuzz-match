.PHONY: setup test lint format clean run

setup:
	uv sync

test:
	PYTHONPATH=src uv run pytest tests/ -v --cov=src/fuzz_match --cov-report=term-missing

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage build dist *.egg-info

run:
	uv run fuzz-match --help
