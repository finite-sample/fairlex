.PHONY: help clean dev test test-cov lint format type-check docs build ci

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .ruff_cache/ .coverage htmlcov/ _site/ _doctest/ docs/_build/
	find . -type d -name __pycache__ -exec rm -rf {} +

dev: ## Install the package with every dependency group
	uv sync --all-groups
	uv run pre-commit install

test: ## Run tests
	uv run pytest

test-cov: ## Run tests with coverage
	uv run pytest --cov --cov-report=term-missing

lint: ## Run linter and formatter checks
	uv run ruff check .
	uv run ruff format --check .

format: ## Format code and apply safe lint fixes
	uv run ruff format .
	uv run ruff check --fix .

type-check: ## Run type checker
	uv run pyright

docs: ## Build the documentation the way CI does
	uv run sphinx-build -W -b html docs _site

build: ## Build package
	uv build

ci: ## Run the conformance checks CI runs
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright
	uvx --from pydoclint==0.9.1 pydoclint src/
	uvx preen check --strict
	uv run pytest --cov --cov-report=term-missing --cov-fail-under=94
