.PHONY: install test demo run report clean

install:
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev]"

test:
	.venv/bin/python -m pytest

demo:
	.venv/bin/python -m sentpred demo

run:
	.venv/bin/python -m sentpred run

report:
	.venv/bin/python -m sentpred report

clean:
	rm -rf data reports/index.html reports/charts
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
