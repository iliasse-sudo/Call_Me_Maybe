CACHE = */__pycache__ */.mypy_cache */*.egg-info __pycache__ .mypy_cache *.egg-info
SRC= src
VENV_PATH := /goinfre/ibaya/uv_venv
UV_CACHE_DIR :=/goinfre/ibaya/uv_cache
HF_HOME :=/goinfre/ibaya/HF_cache

export UV_CACHE_DIR
export HF_HOME

install:
	rm -f .venv
	mkdir -p $(VENV_PATH)
	ln -s $(VENV_PATH) .venv
	uv sync

run: install
	uv run python -m $(SRC)

debug: install
	uv run python3 -m pdb -m $(SRC)

clean:
	rm -rf $(CACHE)
	find . -type f -name "*.pyc" -delete
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .venv $(VENV_PATH)

lint: install
	uv run python3 -m flake8 $(SRC)
	uv run python3 -m mypy $(SRC) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict: install
	uv run python3 -m flake8 $(SRC)
	uv run python3 -m mypy $(SRC) --strict

.PHONY: install run debug clean lint lint-strict
