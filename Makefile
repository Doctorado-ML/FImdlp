SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: audit build build-clean clean coverage deps help install lint publish publish-test push sdist test version wheels

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

build-clean:  ## Remove dist/ and build/ before producing release artifacts
	rm -rf dist build src/*.egg-info

sdist:  ## Build the source distribution into dist/
	python -m build --sdist

wheels:  ## Build manylinux/macOS wheels for the current platform via cibuildwheel
	@command -v cibuildwheel >/dev/null 2>&1 || pip install --upgrade cibuildwheel
	@if [ "$$(uname)" = "Linux" ]; then \
		if command -v docker >/dev/null 2>&1; then \
			engine=docker; \
		elif command -v podman >/dev/null 2>&1; then \
			engine=podman; \
		else \
			echo "ERROR: cibuildwheel needs Docker or Podman on Linux to produce manylinux wheels." >&2; exit 1; \
		fi; \
		echo "Using container engine: $$engine"; \
		CIBW_CONTAINER_ENGINE=$$engine python -m cibuildwheel --output-dir dist; \
	else \
		python -m cibuildwheel --output-dir dist; \
	fi

build:  ## Clean and build sdist + wheels for the current platform
	make build-clean
	make sdist
	make wheels

install:  ## Install in editable mode
	make clean
	pip install -e .

publish:  ## Upload everything in dist/ to PyPI (build first, or drop in CI artifacts)
	@ls dist/*.whl >/dev/null 2>&1 || { echo "ERROR: no wheels in dist/. Run 'make build' or add CI artifacts first." >&2; exit 1; }
	twine check dist/*
	twine upload dist/*

publish-test:  ## Upload everything in dist/ to TestPyPI (manual)
	@ls dist/*.whl >/dev/null 2>&1 || { echo "ERROR: no wheels in dist/. Run 'make build' or add CI artifacts first." >&2; exit 1; }
	twine check dist/*
	twine upload --repository testpypi dist/*

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
