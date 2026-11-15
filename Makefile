.PHONY: dev format lint test

format:
	uv run ruff format .

lint:
	uv run ruff check .

test:
	uv run python -m unittest discover -s agentex/tests -t . -v

# make cannot activate the parent shell, so enter a shell in the uv environment.
dev:
	uv sync --all-groups --all-extras
	@echo "Entering the uv environment. Type 'exit' to leave."
	@bash --rcfile .venv/bin/activate -i
