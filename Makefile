SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: coverage deps help lint push test build install audit

clean: ## Clean up
	rm -rf build dist src/*.egg-info
	if [ -f src/fimdlp/cfimdlp.cpp ]; then rm src/fimdlp/cfimdlp.cpp; fi;
	for file in src/fimdlp/*.so; do \
		if [ -f $${file} ]; then rm $${file}; fi; \
	done

test:
	coverage run -m unittest discover -v -s src
coverage:
	make test
	coverage report -m

submodule:
	git submodule update --remote src/cppmdlp
	git submodule update --merge

lint:  ## Lint and static-check
	black src
	flake8 --per-file-ignores="__init__.py:F401" src

push:  ## Push code with tags
	git push && git push --tags

build:  ## Build package
	make clean
	python -m build --wheel

install:  ## Build extension
	make clean
	pip install -e .

audit: ## Audit pip
	pip-audit

version:
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
