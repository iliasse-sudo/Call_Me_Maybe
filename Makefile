CACHE = */__pycache__ */.mypy_cache */*.egg-info __pycache__ .mypy_cache *.egg-info
SRC= src
VENV_PATH= /goinfre/ibaya/uv_venv

install:
	mkdir $(VENV_PATH)
	ln -s $(VENV_PATH) .venv
	uv sync

run:
	export UV_CACHE_DIR=/goinfre/ibaya/uv_cache
	export HF_DIR=/goinfre/ibaya/HF_DIR
	uv run python -m $(SRC)

debug:
	uv run python3 -m pdb -m $(SRC)

clean:
	rm -rf $(CACHE)
	find . -type f -name "*.pyc" -delete
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .venv

lint:
	uv run python3 -m flake8 $(SRC)
	uv run python3 -m mypy $(SRC) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run python3 -m flake8 $(SRC)
	uv run python3 -m mypy $(SRC) --strict

.PHONY: install run debug clean lint lint-strict
