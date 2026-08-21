CACHE = */__pycache__ */.mypy_cache */*.egg-info __pycache__ .mypy_cache *.egg-info
SRC= src

install:
	uv sync

run:
	uv run python -m $(SRC)

debug:
	uv run python3 -m pdb -m $(SRC)

clean:
	rm -rf $(CACHE)
	find . -type f -name "*.pyc" -delete
	find . -type d -name __pycache__ -exec rm -rf {} +
#	rm -rf .venv

lint:
	uv run python3 -m flake8 $(SRC)
	uv run python3 -m mypy $(SRC) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run python3 -m flake8 $(SRC)
	uv run python3 -m mypy $(SRC) --strict

.PHONY: install run debug clean lint lint-strict
