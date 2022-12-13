SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: coverage deps help lint push test doc build

clean: ## Clean up
	rm -rf build dist *.egg-info
	if [ -f src/fimdlp/cfimdlp.cpp ]; then rm src/fimdlp/cfimdlp.cpp; fi;
	for file in src/fimdlp/*.so; do \
		if [ -f $${file} ]; then rm $${file}; fi; \
	done

test:
	coverage run -m unittest discover -v -s src
coverage:
	make test
	coverage report -m

lint:  ## Lint and static-check
	black src
	flake8 src

push:  ## Push code with tags
	git push && git push --tags

build:  ## Build package
	make clean
	python -m build --wheel

buildext:  ## Build extension
	make clean
	python setup.py build_ext
	echo "Build extension success"

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
