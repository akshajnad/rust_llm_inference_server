# Everything this project downloads or builds lives inside the repo folder.
SHELL := /bin/bash
export HF_HOME := $(CURDIR)/models/.hf
export UV_CACHE_DIR := $(CURDIR)/.uv-cache

FEATURES ?= metal
MODEL_DIR := models/llama-3.2-1b-instruct
HF := https://huggingface.co
DENSE_REPO := unsloth/Llama-3.2-1B-Instruct
DENSE_FILES := config.json generation_config.json tokenizer.json tokenizer_config.json model.safetensors

.PHONY: models build test test-metal parity-ref chat-ref parity lint fuzz teardown

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

# GPU-only unit tests (paged-attention kernel vs reference); no model download needed.
test-metal:
	cargo test --release -p model --lib --features metal

# Runs the HF transformers reference once in a throwaway venv, then deletes the venv.
parity-ref: models
	uv venv .venv --python 3.12
	VIRTUAL_ENV=.venv uv pip install torch transformers safetensors numpy
	.venv/bin/python -I scripts/parity/dump_reference.py --model $(MODEL_DIR) \
		--prompts tests/fixtures/parity/prompts.json --out tests/fixtures/parity/reference.safetensors
	rm -rf .venv .uv-cache

# f32 is the strict gate (token-for-token); f16 also allows flips at reference near-ties.
# Renders reference chat prompts with transformers (tokenizer only, no torch).
chat-ref: models
	uv venv .venv --python 3.12
	VIRTUAL_ENV=.venv uv pip install transformers jinja2
	.venv/bin/python -I scripts/parity/dump_chat_reference.py --model $(MODEL_DIR) \
		--out tests/fixtures/chat/llama32_reference.json
	rm -rf .venv .uv-cache

parity: models
	PARITY_DTYPE=f32 cargo test --release -p model --features $(FEATURES) --test parity -- --ignored --nocapture --test-threads=1
	PARITY_DTYPE=f16 cargo test --release -p model --features $(FEATURES) --test parity -- --ignored --nocapture --test-threads=1
	cargo test --release -p engine --test chat_template -- --ignored

lint:
	cargo fmt --all --check
	cargo clippy --workspace --all-targets -- -D warnings
	cargo clippy --workspace --all-targets --features model/metal -- -D warnings

# Needs `rustup toolchain install nightly` and `cargo install cargo-fuzz`.
FUZZ_SECS ?= 60
fuzz:
	cd fuzz && cargo +nightly fuzz run request_parser -- -max_total_time=$(FUZZ_SECS)
	cd fuzz && cargo +nightly fuzz run stop_matcher -- -max_total_time=$(FUZZ_SECS)

teardown:
	-pkill -f 'target/.*/rlis' || true
	-pkill -f 'llama-server' || true
	rm -rf models target third_party .venv .uv-cache fuzz/target fuzz/corpus fuzz/artifacts
	@du -sh .
	@echo "To remove the Rust toolchain too: rustup self uninstall"
