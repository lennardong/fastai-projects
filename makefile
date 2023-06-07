
################################################################################
# PYTHON
################################################################################

# Install dependencies
install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

# Lint code
lint:
	python -m pylint --disable=R,C **/*.py

# Format code
format:
	python -m black **/*.py

# Run tests
test:
	python -m pytest -vv --cov=vision --pyargs -k test_vision
