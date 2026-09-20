# Convenience targets. `uv sync` alone is enough to build and install the engine.
# ARCH: 120 (5090), 86 (3090, default), 89 (Ada), 80 (A100), 90 (Hopper)
ARCH ?= 86
MODEL ?= models/Qwen3.5-4B

.PHONY: all build test test-gpu bench serve download clean

all: build

build:            ## build libcujev.so + cujev-bench with CMake (no Python)
	cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$(ARCH)
	cmake --build build -j

sync:             ## create .venv, build the engine into it, install the Python package
	CUJEV_CUDA_ARCH=$(ARCH) uv sync --extra dev

download:         ## download a checkpoint into ./models (MODEL=Qwen3.5-0.8B|2B|4B|9B)
	uv run scripts/download_model.py $(notdir $(MODEL))

test:             ## CPU-only tests
	uv run pytest -q tests/python/test_prompt.py

test-gpu:         ## numerical oracle + edge cases against Hugging Face (needs the dev extra)
	CUJEV_MODEL=$(MODEL) uv run pytest -q tests/python/test_oracle.py tests/python/test_engine.py

bench: build      ## raw engine timing on synthetic tokens
	./build/cujev-bench $(MODEL) 1024 16 40 8 5

serve:            ## start the Jev-compatible server on :8080
	uv run cujev serve --model $(MODEL) --port 8080

clean:
	rm -rf build dist *.egg-info
