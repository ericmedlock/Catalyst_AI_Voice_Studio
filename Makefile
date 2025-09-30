.PHONY: setup test lint format run clean install

# Setup development environment
setup:
	pip install -e ".[dev]"
	pre-commit install

# Install package
install:
	pip install -e .

# Run tests
test:
	pytest tests/ -v --cov=catalyst_ai_voice_studio

# Run linting
lint:
	flake8 catalyst_ai_voice_studio tests
	black --check catalyst_ai_voice_studio tests
	isort --check-only catalyst_ai_voice_studio tests

# Format code
format:
	black catalyst_ai_voice_studio tests
	isort catalyst_ai_voice_studio tests

# Run development server
run:
	uvicorn catalyst_ai_voice_studio.web_streamer.api:app --reload --host 0.0.0.0 --port 8000

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Build package
build:
	python -m build

# Run pre-commit on all files
pre-commit:
	pre-commit run --all-files