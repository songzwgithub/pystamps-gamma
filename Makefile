PYTHON ?= python
DIST_DIR ?= dist

.PHONY: help check-python check-rust check-runtime install install-editable \
        build smoke native-release-bin web clean

help:
	@echo "pySTAMPS-GAMMA production targets"
	@echo "  make check-python       Check Python >= 3.12"
	@echo "  make check-rust         Check Cargo/Rust"
	@echo "  make check-runtime      Check SNAPHU and optional Triangle"
	@echo "  make install            Install from source"
	@echo "  make install-editable   Editable source install"
	@echo "  make build              Build sdist + wheel"
	@echo "  make smoke              Import Python and native extension"
	@echo "  make native-release-bin Build standalone Rust CLI"
	@echo "  make web                Run optional Rust web frontend"
	@echo "  make clean              Remove generated build artifacts"

check-python:
	@$(PYTHON) -c 'import sys; assert sys.version_info >= (3,12), "Python >= 3.12 is required"; print("Python:", sys.version.split()[0])'

check-rust:
	@command -v cargo >/dev/null 2>&1 || { echo "ERROR: cargo not found. Install Rust first."; exit 2; }
	@cargo --version
	@rustc --version

check-runtime:
	@command -v snaphu >/dev/null 2>&1 || { echo "ERROR: snaphu is required for Stage 6 but was not found in PATH"; exit 2; }
	@echo "SNAPHU : $$(command -v snaphu)"
	@if command -v triangle >/dev/null 2>&1; then \
		echo "Triangle: $$(command -v triangle)"; \
	else \
		echo "Triangle: not installed (OK; SciPy Delaunay fallback will be used)"; \
	fi

install: check-python check-rust
	$(PYTHON) -m pip install .

install-editable: check-python check-rust
	$(PYTHON) -m pip install -e .

build: check-python check-rust
	$(PYTHON) -m pip install --upgrade build setuptools-rust
	rm -rf build $(DIST_DIR)
	$(PYTHON) -m build --sdist --wheel

smoke:
	$(PYTHON) -c 'import pystamps; import pystamps.kernels._stage2_native as n; print("pySTAMPS:", pystamps.__version__); print("native:", n.__file__)'
	@command -v pystamps >/dev/null
	pystamps --help >/dev/null
	@echo "PASS: pySTAMPS installation smoke test"

native-release-bin: check-rust
	cargo build --release -p pystamps-core --bin pystamps-native

web: check-rust
	cargo run -p pystamps-web

clean:
	rm -rf build dist
	rm -rf ./*.egg-info
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
