# Everything this project downloads or builds lives inside the repo folder.
SHELL := /bin/bash
export HF_HOME := $(CURDIR)/models/.hf
export UV_CACHE_DIR := $(CURDIR)/.uv-cache

FEATURES ?= metal
MODEL_DIR := models/llama-3.2-1b-instruct
HF := https://huggingface.co
DENSE_REPO := unsloth/Llama-3.2-1B-Instruct
DENSE_FILES := config.json generation_config.json tokenizer.json tokenizer_config.json model.safetensors

.PHONY: models build test parity-ref parity lint teardown

models:
	@mkdir -p $(MODEL_DIR)
	@for f in $(DENSE_FILES); do \
		if [ ! -s $(MODEL_DIR)/$$f ]; then \
			echo "downloading $$f"; \
			curl -L --fail --progress-bar -o $(MODEL_DIR)/$$f.part $(HF)/$(DENSE_REPO)/resolve/main/$$f \
				&& mv $(MODEL_DIR)/$$f.part $(MODEL_DIR)/$$f || exit 1; \
		fi; \
	done

build:
	cargo build --release --features $(FEATURES)

test:
	cargo test --workspace

# Runs the HF transformers reference once in a throwaway venv, then deletes the venv.
parity-ref: models
	uv venv .venv --python 3.12
	VIRTUAL_ENV=.venv uv pip install torch transformers safetensors numpy
	.venv/bin/python -I scripts/parity/dump_reference.py --model $(MODEL_DIR) \
		--prompts tests/fixtures/parity/prompts.json --out tests/fixtures/parity/reference.safetensors
	rm -rf .venv .uv-cache

# f32 is the strict gate (token-for-token); f16 also allows flips at reference near-ties.
parity: models
	PARITY_DTYPE=f32 cargo test --release -p model --features $(FEATURES) --test parity -- --ignored --nocapture --test-threads=1
	PARITY_DTYPE=f16 cargo test --release -p model --features $(FEATURES) --test parity -- --ignored --nocapture --test-threads=1

lint:
	cargo fmt --all --check
	cargo clippy --workspace --all-targets -- -D warnings

teardown:
	-pkill -f 'target/.*/rlis' || true
	-pkill -f 'llama-server' || true
	rm -rf models target third_party .venv .uv-cache
	@du -sh .
	@echo "To remove the Rust toolchain too: rustup self uninstall"
