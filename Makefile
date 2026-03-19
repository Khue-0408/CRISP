install:
	pip install -e .

format:
	black src tests
	ruff check src tests --fix

lint:
	ruff check src tests
	mypy src

test:
	pytest tests -q

train-pranet-baseline:
	python -m crisp.scripts.train --config-path configs --config-name experiment/taskA_pranet_baseline

train-pranet-crisp:
	python -m crisp.scripts.train --config-path configs --config-name experiment/taskA_pranet_crisp

eval-pranet:
	python -m crisp.scripts.evaluate --config-path configs --config-name experiment/taskA_pranet_crisp checkpoint=$(CHECKPOINT)
