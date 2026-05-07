SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: audit build clean coverage deps help install lint publish push test version

clean: ## Remove build artifacts and caches
	rm -rf build dist src/*.egg-info .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -f src/fimdlp/cfimdlp.cpp src/fimdlp/*.so

deps:  ## Install development dependencies
	pip install -e ".[dev]"

test:  ## Run unit tests with coverage
	@ls src/fimdlp/cppfimdlp*.so >/dev/null 2>&1 || pip install -e . --quiet --no-deps
	coverage run -m unittest discover -v -s src

coverage:  ## Run tests and print coverage report
	make test
	coverage report -m

lint:  ## Format and lint sources
	black src
	flake8 --per-file-ignores="__init__.py:F401" src

push:  ## Push code with tags
	git push && git push --tags

build:  ## Build wheel and sdist (does not touch the editable extension)
	rm -rf dist build src/*.egg-info
	python -m build

install:  ## Install in editable mode
	make clean
	pip install -e .

publish:  ## Build and upload to PyPI
	make build
	twine check dist/*
	twine upload dist/*

sample_cpp: ## Build and execute c++ sample
	cd samples && rm -rf build 2>/dev/null && cmake -B build -S . && cmake --build build && cd build && ./sample -f iris

sample_py: ## Execute python sample
	cd samples && python sample.py iris

audit: ## Audit installed packages for known vulnerabilities
	pip-audit

version:  ## Show current versions
	@echo "Current Python version .: $(shell python --version)"
	@echo "Current FImdlp version .: $(shell python -c "from fimdlp import _version; print(_version.__version__)")"
	@echo "Current mdlp version ...: $(shell python -c "from fimdlp.cppfimdlp import CFImdlp; print(CFImdlp().get_version().decode())")"
	@echo "Installed FImdlp version: $(shell pip show fimdlp | grep Version | cut -d' ' -f2)"

help: ## Show help message
	@IFS=$$'\n' ; \
	help_lines=(`fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##/:/'`); \
	printf "%s\n\n" "Usage: make [task]"; \
	printf "%-20s %s\n" "task" "help" ; \
	printf "%-20s %s\n" "------" "----" ; \
	for help_line in $${help_lines[@]}; do \
		IFS=$$':' ; \
		help_split=($$help_line) ; \
		help_command=`echo $${help_split[0]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		help_info=`echo $${help_split[2]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		printf '\033[36m'; \
		printf "%-20s %s" $$help_command ; \
		printf '\033[0m'; \
		printf "%s\n" $$help_info; \
	done
