.PHONY: help install lint typecheck test data index eval report clean

PYTHON ?= python
CORPUS ?= contractnli
TOPK ?= 10

help:
	@echo "make install     - install package + dev deps via uv"
	@echo "make lint        - ruff check + format check"
	@echo "make typecheck   - mypy strict"
	@echo "make test        - pytest (skips slow + needs_api)"
	@echo "make test-all    - pytest with slow tests"
	@echo "make data        - download + prepare LegalBench-RAG corpora"
	@echo "make index CORPUS=contractnli - build BM25 + dense indices for a corpus"
	@echo "make eval CORPUS=contractnli  - run all retrievers, write results/"
	@echo "make report      - regenerate the markdown summary in results/"

install:
	uv sync --all-extras

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

typecheck:
	uv run mypy src

test:
	uv run pytest -m "not slow and not needs_api"

test-all:
	uv run pytest

data:
	uv run lrb data prepare

index:
	uv run lrb index build --corpus $(CORPUS)

eval:
	uv run lrb eval run --corpus $(CORPUS) --topk $(TOPK)

report:
	uv run lrb report build

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +


.PHONY: pdf test-artifacts
pdf:
	cd docs/_report && pandoc research_report.md -o ../research_report.pdf --pdf-engine=xelatex --toc --toc-depth=2 --number-sections -V geometry:margin=1in -V fontsize=11pt -V mainfont="Helvetica" -V monofont="Menlo" -V linkcolor=blue -V urlcolor=blue -V linestretch=1.15 || echo "pandoc + xelatex required; see https://pandoc.org/installing.html"

test-artifacts:
	uv run python ../../_meta/retrofit.py "$(notdir $(CURDIR))" "$(notdir $(CURDIR))"
