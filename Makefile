.PHONY: help install test lint format docs build serve train agent clean

help:
	@echo "FALL - Fully Autonomous Learning Language model"
	@echo ""
	@echo "make install    Install dependencies"
	@echo "make test       Run tests"
	@echo "make lint       Run linters"
	@echo "make format     Format code"
	@echo "make docs       Generate documentation"
	@echo "make build      Build Docker image"
	@echo "make serve      Start inference server"
	@echo "make train      Start training"
	@echo "make agent      Start autonomous agent"
	@echo "make clean      Clean build artifacts"

install:
	pip install -e ".[dev,gpu]"

test:
	pytest fall/tests/ -x -v

lint:
	ruff check fall/
	black --check fall/
	mypy fall/ --ignore-missing-imports

format:
	black fall/
	ruff check --fix fall/

docs:
	python -m fall.docs.generator

build:
	docker build -t fall:latest .

serve:
	python -m fall.inference.api

train:
	python -m fall.training.launch

agent:
	python -m fall.agent.runtime

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/