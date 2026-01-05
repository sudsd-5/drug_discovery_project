# Contributing to Drug Discovery Platform

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository and clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/drug_discovery_project.git
cd drug_discovery_project
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies including development tools:
```bash
pip install -r requirements.txt
```

## Development Workflow

1. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes following the coding standards below

3. Run tests to ensure everything works:
```bash
pytest tests/
```

4. Format your code:
```bash
black .
isort .
```

5. Check code quality:
```bash
flake8 .
```

6. Commit your changes:
```bash
git add .
git commit -m "Add: description of your changes"
```

7. Push to your fork and create a Pull Request

## Coding Standards

### Python Style
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Maximum line length: 88 characters (Black default)
- Use type hints where appropriate

### Documentation
- Write docstrings for all public functions and classes
- Use Google-style docstrings format
- Update README.md if you add new features

### Testing
- Write unit tests for new functionality
- Maintain test coverage above 80%
- Test edge cases and error conditions

### Commit Messages
Use conventional commit format:
- `Add:` for new features
- `Fix:` for bug fixes
- `Update:` for updates to existing features
- `Refactor:` for code refactoring
- `Docs:` for documentation changes
- `Test:` for test additions or modifications

Example:
```
Add: support for ChEMBL dataset preprocessing

- Implemented ChEMBL SQLite database reader
- Added data sampling functionality
- Updated documentation
```

## Testing

Run all tests:
```bash
pytest tests/
```

Run specific test file:
```bash
pytest tests/test_data_process.py
```

Run with coverage:
```bash
pytest --cov=. --cov-report=html tests/
```

## Pull Request Process

1. Ensure all tests pass
2. Update documentation as needed
3. Add a clear description of your changes in the PR
4. Reference any related issues
5. Wait for review from maintainers

## Code Review

All submissions require review. We use GitHub pull requests for this purpose. Reviewers will check:

- Code quality and style
- Test coverage
- Documentation
- Functionality
- Performance considerations

## Questions?

Feel free to open an issue for:
- Bug reports
- Feature requests
- Questions about the codebase
- Clarification on contribution guidelines

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
