SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: coverage deps help lint push test doc build

clean: ## Clean up
	rm -rf build dist *.egg-info
	if [ -f fimdlp/cfimdlp.cpp ]; then rm fimdlp/cfimdlp.cpp; fi;
	if [ -f fimdlp/cppfimdlp.cpython-310-darwin.so ]; then rm fimdlp/cppfimdlp.cpython-310-darwin.so; fi;

lint:  ## Lint and static-check
	black fimdlp
	flake8 fimdlp

push:  ## Push code with tags
	git push && git push --tags

build:  ## Build package
	rm -fr dist/*
	rm -fr build/*
	python -m build

buildext:  ## Build extension
	rm -fr dist/*
	rm -fr build/*
	make clean
	python setup.py build_ext
	echo "Build extension success"; mv build/lib.macosx-12-x86_64-cpython-310/cppfimdlp.cpython-310-darwin.so fimdlp;

audit: ## Audit pip
	pip-audit

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
