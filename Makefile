PYTHON ?= python
.PHONY: test smoke lint
test:
	$(PYTHON) -m pytest -q
smoke:
	PYTHONPATH=src $(PYTHON) -m replication_smoke
lint:
	$(PYTHON) -m ruff check src tests
