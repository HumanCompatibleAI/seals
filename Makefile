SRC_FILES=src/ tests/ docs/conf.py setup.py

format:
	isort ${SRC_FILES}
	black ${SRC_FILES}

lintcheck:
	flake8 ${SRC_FILES}
	black --check ${SRC_FILES}
	codespell -I .codespell.skip --skip='*.pyc' ${SRC_FILES}
	if [ -x "`which circleci`" ]; then \
		circleci config validate; \
	fi

typecheck:
	pytype ${SRC_FILES}
	mypy ${SRC_FILES}

docscheck:
	pushd docs/
	make clean
	make html
	popd

ci: lintcheck typecheck docscheck