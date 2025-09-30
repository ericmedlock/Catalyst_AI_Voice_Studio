# Contributing to Catalyst AI Voice Studio

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install development dependencies: `pip install -e ".[dev]"`
5. Install pre-commit hooks: `pre-commit install`

## Code Standards

- **Formatting**: Use `black` for code formatting
- **Import sorting**: Use `isort` for import organization
- **Linting**: Use `flake8` for code quality checks
- **Type hints**: Add type annotations where possible
- **Documentation**: Include docstrings for public APIs

## Testing

- Write unit tests for new features
- Ensure all tests pass: `pytest`
- Maintain test coverage above 80%
- Add integration tests for TTS models

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with tests
3. Run the full test suite
4. Update documentation if needed
5. Submit a pull request with clear description

## Code Review Guidelines

- Focus on code quality, performance, and maintainability
- Ensure backward compatibility
- Verify test coverage for new code
- Check for potential security issues