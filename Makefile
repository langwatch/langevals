# export POETRY_VENV_PATH=$(PWD)/.venv

.PHONY: test lock lock-core lock-evaluators install install-core install-evaluators

test:
	@for dir in evaluators/*; do \
		if [ -d $$dir ]; then \
			echo "Running tests in $$dir"; \
			cd $$dir && poetry run pytest -s -vv && cd ../..; \
		fi \
	done

ensure-poetry:
	@if ! command -v poetry &> /dev/null; then \
		curl -sSL https://install.python-poetry.org | python3 -; \
	fi

setup:
	poetry run python scripts/generate_workspace.py

lock-core:
	@echo "Locking dependencies for langevals_core..."
	@cd langevals_core && poetry lock

lock-evaluators: lock-core install-core
	@for dir in evaluators/*; do \
		if [ -d $$dir ]; then \
			echo "Locking in $$dir"; \
			cd $$dir && poetry lock && cd ../..; \
		fi \
	done

install-core:
	@echo "Installing dependencies for langevals_core..."
	@cd langevals_core && poetry install

install-evaluators: install-core
	@for dir in evaluators/*; do \
		if [ -d $$dir ]; then \
			echo "Installing in $$dir"; \
			cd $$dir && poetry install && cd ../..; \
		fi \
	done

lock: ensure-poetry lock-core install-core lock-evaluators
	@echo "All packages locked."
	poetry lock

install: ensure-poetry install-core install-evaluators
	@echo "All evaluator packages installed."
	poetry install
	make setup

%:
	@: