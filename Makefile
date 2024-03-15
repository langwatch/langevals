.PHONY: test lock lock-core lock-evaluators install install-core install-evaluators run-docker

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
	poetry run python scripts/generate_evaluator_dependencies.py
	poetry run python scripts/generate_workspace.py
	poetry run python scripts/generate_evaluators_ts.py

lock-core:
	@echo "Locking dependencies for langevals_core..."
	@cd langevals_core && poetry lock --no-update

lock-evaluators: lock-core install-core
	@for dir in evaluators/*; do \
		if [ -d $$dir ]; then \
			echo "Locking in $$dir"; \
			cd $$dir && poetry lock --no-update && cd ../..; \
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
	poetry lock --no-update

install: ensure-poetry install-core install-evaluators
	@echo "All evaluator packages installed."
	poetry install --all-extras
	make setup && poetry lock --no-update && poetry install --all-extras

start:
	@echo "Starting the server..."
	poetry run python langevals/server.py

run-docker:
	@echo "Building and running the Docker container for the evaluator..."
	@docker build --build-arg EVALUATOR=$(EVALUATOR) -t langevals-$(EVALUATOR) .
	@docker run -p 80:80 langevals-$(EVALUATOR)

check-evaluator-versions:
	@echo "Checking all evaluator versions for changes..."
	@(cd langevals_core && ../scripts/check_version_bump.sh); \
	for dir in evaluators/*; do \
		if [ -d "$$dir" ]; then \
			echo "Checking $$dir"; \
			(cd "$$dir" && git add pyproject.toml && python ../../scripts/replace_develop_dependencies.py pyproject.toml && ../../scripts/check_version_bump.sh; exit_status=$$?; git checkout pyproject.toml; exit $$exit_status); \
		fi \
	done

%:
	@: