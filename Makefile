.PHONY: run format lint test check clean

# Run the main application
run:
	python -m src.main

# Auto-fix & format
format:
	ruff format src tests

# Linting (analysis with auto fix)
lint:
	ruff check src tests

# Run tests (with coverage by default via pytest.ini)
test:
	pytest --cov=src --cov=tests --cov-report=term-missing

# Run full code quality check
check: format lint test

# Clean up cache and artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml