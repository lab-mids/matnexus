# Makefile for automating

# Autoformatting python code
pretty:
	black . 

# Check for code formatting issues
lint:
	pwd
	flake8 .

# Run python test suite based on `unittest`
test:
	python -m unittest discover tests/

# Run test with code coverage report
test-cov:
	pytest --cov-report=html --cov=MatNexus/
	firefox htmlcov/index.html

# Install library in developer mode
install-dev:
	pip install -e .

# Install library in normal mode
install:
	pip install .

# Run all pre-commit necessities to avoid pipeline failure
make all:
	make pretty
	make lint
	make test
