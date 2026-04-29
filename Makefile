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
	python -m crisp.scripts.evaluate --config-path configs --config-name experiment/taskA_pranet_crisp +checkpoint=$(CHECKPOINT)

train-unet-baseline:
	python -m crisp.scripts.train --config-path configs --config-name experiment/taskA_unet_baseline

train-unet-crisp:
	python -m crisp.scripts.train --config-path configs --config-name experiment/taskA_unet_crisp

eval-unet:
	python -m crisp.scripts.evaluate --config-path configs --config-name $(or $(CONFIG),experiment/taskA_unet_crisp) +checkpoint=$(CHECKPOINT)

bootstrap-local:
	python -m crisp.scripts.bootstrap_local_workspace

train-unet-baseline-local:
	python -m crisp.scripts.train --config-path configs --config-name experiment/task_local_unet_baseline

train-unet-crisp-local:
	python -m crisp.scripts.train --config-path configs --config-name experiment/task_local_unet_crisp

eval-unet-local:
	python -m crisp.scripts.evaluate --config-path configs --config-name $(or $(CONFIG),experiment/task_local_unet_crisp) +checkpoint=$(CHECKPOINT)
