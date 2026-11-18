# CPU-only checks of the published tree.
.PHONY: test check
test:
	python3 -m pytest -q tests --ignore=tests/models -p no:cacheprovider
check:
	bash tools/check_data_boundary.sh --tree HEAD
