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
    pimmi fill demo_dataset/small_dataset/ small --index-path test/ressources/tmp/ --erase --force
    pimmi query demo_dataset/small_dataset/ small --index-path test/ressources/tmp/
endef

# Commands
publish: clean upload
	$(call clean)

clean:
	$(call clean)

test: functional unit

functional:
	$(call functional)

deps:
	pip3 install -U pip
	pip3 install -r requirements.txt

upload:
	python setup.py sdist bdist_wheel
	twine upload dist/*

unit:
	@echo Running unit tests...
	pytest -svvv
	@echo