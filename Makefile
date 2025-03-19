# Variables
SOURCE = pimmi

# Functions
define clean
	rm -rf *.egg-info .pytest_cache build dist
	find . -name "*.pyc" | xargs rm -f
	find . -name __pycache__ | xargs rm -rf
	rm -f *.spec
endef

define functional
    pimmi fill demo_dataset/small_dataset/  test/ressources/tmp/small -e -f --config-path test/ressources/config.yml
    pimmi query demo_dataset/small_dataset/  test/ressources/tmp/small --config-path test/ressources/config.yml \
	--output test/ressources/tmp/small_queries.csv
	pimmi clusters test/ressources/tmp/small test/ressources/tmp/small_queries.csv --config-path test/ressources/config.yml \
	-o test/ressources/tmp/small_clusters.csv
	pimmi viz test/ressources/tmp/small_clusters.csv -o test/ressources/tmp/small_viz.json

	pimmi query demo_dataset/small_dataset/  test/ressources/tmp/small --config-path test/ressources/config.yml \
	--nb-per-file 30 --output-template test/ressources/tmp/small_queries
	pimmi clusters test/ressources/tmp/small test/ressources/tmp/small_queries --config-path test/ressources/config.yml \
	-o test/ressources/tmp/small_clusters.csv
	pimmi viz test/ressources/tmp/small_clusters.csv -o test/ressources/tmp/small_viz.json

    echo "TESTS WITH STDIN"
	pimmi query demo_dataset/small_dataset/ test/ressources/tmp/small --config-path test/ressources/config.yml| \
	pimmi clusters test/ressources/tmp/small --config-path test/ressources/config.yml | \
	pimmi viz -o test/ressources/tmp/small_viz.json
endef

# Commands
all: lint test

publish: clean upload
	$(call clean)

clean:
	$(call clean)

test: functional unit

functional:
	$(call functional)

deps:
	pip3 install -U pip setuptools
	pip3 install -r requirements.txt
	pip3 install .

upload:
	python setup.py sdist bdist_wheel
	twine upload dist/*

unit:
	@echo Running unit tests...
	pytest -svvv
	@echo

lint:
	@echo Searching for unused imports...
	importchecker $(SOURCE) | grep -v __init__ || true
	importchecker test | grep -v __init__ || true
	@echo