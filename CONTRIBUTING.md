# Contributing to llama-stack-provider-lmeval

Thank you for your interest in contributing to llama-stack-provider-lmeval! This document provides guidelines and information for contributors.

## Development Setup

### Prerequisites

- Python 3.12 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd llama-stack-provider-lmeval
   ```

2. Install the project in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality and consistency. The hooks run automatically on every commit and can also be run manually.

### What Pre-commit Does

The pre-commit configuration includes:

- **Ruff linting**: Python code quality checks and formatting
- **Ruff formatting**: Automatic code formatting (similar to Black)
- **YAML validation**: Checks YAML files for syntax errors (supports multi-document YAML)
- **General file checks**: Trailing whitespace, end-of-file, merge conflicts, etc.
- **Unit tests**: Runs pytest to ensure tests pass

### Setup Pre-commit

**Manual setup**:
   ```bash
   # Install pre-commit if not already installed
   pip install pre-commit
   
   # Install the hooks
   pre-commit install
   ```

### Using Pre-commit

- **Automatic**: Hooks run automatically on every commit
- **Manual run on all files**:
  ```bash
  pre-commit run --all-files
  ```
- **Manual run on specific files**:
  ```bash
  pre-commit run --files path/to/file.py
  ```
- **Skip hooks** (use sparingly):
  ```bash
  git commit --no-verify -m "Emergency fix"
  ```

### Pre-commit.ci Integration

This project uses pre-commit.ci, which automatically runs pre-commit hooks on all pull requests and commits. This means:

- **No local setup required** for contributors - hooks run automatically in CI
- **Consistent checks** across all contributions
- **Automatic fixes** - if possible, pre-commit.ci will create a commit with fixes
- **Status checks** - PRs will show pre-commit status and block merging if checks fail

**Note**: Even with pre-commit.ci, it's still recommended to set up pre-commit locally for faster feedback during development.

### Pre-commit Configuration

The configuration is in `.pre-commit-config.yaml` and includes:

- **Ruff hooks**: Code linting and formatting
- **YAML validation**: With multi-document support for Kubernetes manifests
- **File hygiene**: Various file quality checks
- **Local pytest hook**: Ensures tests pass before commit

## Code Quality Standards

### Python Code Style

- **Line length**: 88 characters (Black's default)
- **Formatting**: Automatic with ruff
- **Linting**: Comprehensive checks with ruff
- **Python version**: 3.12+

### YAML Files

- **Multi-document YAML**: Supported for Kubernetes manifests
- **Syntax validation**: Automatic checking on commit
- **Format**: Consistent indentation and structure

## Running Tests

### Test Configuration

Tests are configured in `pytest.ini` and `pyproject.toml`:

- **Test paths**: `tests/` directory
- **Python files**: `test_*.py` pattern
- **Source paths**: `src/` directory
- **Options**: Verbose output enabled

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_lmeval.py

# Run with verbose output
pytest -v
```

## Development Workflow

1. **Create a feature branch** from main
2. **Make your changes** following the code style guidelines
3. **Run tests** to ensure everything works
4. **Commit your changes** (pre-commit hooks will run automatically)
5. **Push and create a pull request**

### Before Committing

- Ensure all tests pass: `pytest`
- Check code quality: `pre-commit run --all-files`
- Verify YAML syntax (if applicable)
- Update documentation if needed

## Additional Resources

- [Pre-commit documentation](https://pre-commit.com/)
- [Ruff documentation](https://docs.astral.sh/ruff/)
- [Pytest documentation](https://docs.pytest.org/)
- [Python development guide](https://docs.python.org/3/developing/)

---

Thank you for contributing to llama-stack-provider-lmeval!
