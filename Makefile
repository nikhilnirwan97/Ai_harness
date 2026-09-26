.PHONY: setup run test clean

VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip
PYTEST ?= $(VENV)/bin/pytest

# Optional arguments for non-interactive execution:
# make run TASK="Fix bugs in core" REPO="https://github.com/user/project"
TASK ?=
REPO ?=

setup:
	@echo "Setting up environment..."
	@python3 -m venv $(VENV)
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@echo "Environment setup completed successfully."

run:
	@echo "Starting AI Harness..."
	@if [ -n "$(TASK)" ]; then \
		$(PYTHON) main.py "$(TASK)" $(if $(REPO),"$(REPO)",); \
	else \
		$(PYTHON) main.py; \
	fi

test:
	@echo "Running tests..."
	@$(PYTEST) tests/

clean:
	@echo "Cleaning generated artefacts..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@rm -rf .pytest_cache
	@rm -f harness_run.jsonl patch.diff report.md
	@echo "Clean completed."
