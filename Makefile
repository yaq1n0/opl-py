.PHONY: install lint format typecheck test check init update train demo-core demo-analytics demo clean

install:
	pip install -e ".[dev,analytics]"

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	pyright src/

test:
	pytest tests/ -v --cov=src/opl --cov-report=term-missing

# Run all quality checks (CI equivalent)
check: lint typecheck test

# Database lifecycle
init:
	opl init $(if $(DB_PATH),--db-path $(DB_PATH),)

update:
	opl update $(if $(DB_PATH),--db-path $(DB_PATH),)

# Train trajectory model on the full dataset; outputs to pretrained/{csv_date}/model.joblib
# Pass LIMIT=5000 to cap the number of lifters (useful for testing)
train:
	python -m opl.analytics.scripts.train \
		$(if $(DB_PATH),--db-path $(DB_PATH),) \
		$(if $(LIMIT),--limit $(LIMIT),)

# E2E demos (require an initialised database; pass DB_PATH to override)
demo-core:
	python -m demo.demo_core $(if $(DB_PATH),--db-path $(DB_PATH),)

demo-analytics:
	python -m demo.demo_analytics $(if $(DB_PATH),--db-path $(DB_PATH),)

demo:
	python -m demo.demo_all $(if $(DB_PATH),--db-path $(DB_PATH),)

# Full demo including downloading the dataset (~160 MB) into a temp dir
demo-init:
	python -m demo.demo_all --include-init $(if $(DB_PATH),--db-path $(DB_PATH),)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name dist -exec rm -rf {} +
