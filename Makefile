.PHONY: setup data train test lint image

setup:     ## install the package and dev tools
	pip install -e ".[dev]"

data:      ## download the dataset (~45 MB) and build data/daily_net_cash.csv
	python -m cashflow.data

train:     ## ADF tests, backtest, save the best model to artifacts/model/ (~3 minutes)
	python -m cashflow.train

test:
	pytest -q

lint:
	ruff check .

image:     ## build the Lambda container image
	docker build -f deploy/Dockerfile -t cashflow-forecast .
