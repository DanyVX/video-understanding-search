.PHONY: setup lint format typecheck test test-fast bench run docker-build clean
setup lint format typecheck test test-fast bench run docker-build clean:
	python scripts/tasks.py $@
