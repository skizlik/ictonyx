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
    @echo "  make install-ml   Install with TensorFlow, PyTorch, HuggingFace, MLflow"
    @echo "  make docs         Build Sphinx docs"

.PHONY: install
install:
	pip install -e ".[sklearn,isolation]"
	pip install pytest pytest-cov pytest-timeout black isort flake8 mypy pre-commit build twine
	pre-commit install

.PHONY: install-ml
install-ml: install
	pip install tensorflow
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -e ".[huggingface,mlflow,explain,tuning]"

.PHONY: docs
docs:
	$(MAKE) -C docs html
	@echo "Docs: docs/_build/html/index.html"

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
    rm -rf docs/_build/

.PHONY: format
format:
	black ictonyx/
	isort ictonyx/
