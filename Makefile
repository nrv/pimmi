# Variables
SOURCE = pimmi

# Functions
define clean
	rm -rf *.egg-info .pytest_cache build dist
	find . -name "*.pyc" | xargs rm -f
	find . -name __pycache__ | xargs rm -rf
	rm -f *.spec
endef

# Commands
test: unit

publish: clean upload
	$(call clean)

clean:
	$(call clean)

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