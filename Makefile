SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: coverage deps help lint push test doc build

clean: ## Clean up
	rm -rf build dist *.egg-info
	for name in fimdlp/cfimdlp.cpp fimdlp/fimdlp.cpython-310-darwin.so;do if [ -f $name ]; then rm $name; fi; done

lint:  ## Lint and static-check
	black fimdlp
	flake8 fimdlp

push:  ## Push code with tags
	git push && git push --tags

build:  ## Build package
	rm -fr dist/*
	rm -fr build/*
	#python setup.py build_ext
	python -m build

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
