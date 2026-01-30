# Makefile
.PHONY: help
help:
	@echo "Ictonyx Development Commands"
	@echo "============================"
	@echo "  make install      Install in development mode"
	@echo "  make test         Run tests"
	@echo "  make coverage     Run tests with coverage"
	@echo "  make validate     Validate installation"
	@echo "  make benchmark    Run benchmarks"
	@echo "  make clean        Clean build artifacts"
	@echo "  make format       Format code with black"

.PHONY: install
install:
	pip install -e ".[dev]"

.PHONY: test
test:
	pytest tests/ -v

.PHONY: coverage
coverage:
	pytest tests/ --cov=ictonyx --cov-report=html --cov-report=term
	@echo "Coverage report: htmlcov/index.html"

.PHONY: validate
validate:
	python scripts/validate_installation.py

.PHONY: benchmark
benchmark:
	python scripts/benchmark.py

.PHONY: clean
clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov/ .coverage

.PHONY: format
format:
	black ictonyx/
	isort ictonyx/
