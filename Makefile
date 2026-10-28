# Everything this project downloads or builds lives inside the repo folder.
SHELL := /bin/bash
export HF_HOME := $(CURDIR)/models/.hf
export UV_CACHE_DIR := $(CURDIR)/.uv-cache

FEATURES ?= metal
MODEL_DIR := models/llama-3.2-1b-instruct
HF := https://huggingface.co
DENSE_REPO := unsloth/Llama-3.2-1B-Instruct
# Serving needs the tokenizer files and the default Q8_0 GGUF; parity, perplexity and the
# quantization comparison also need the bf16 safetensors and Q4_K_M.
GGUF_REPO := bartowski/Llama-3.2-1B-Instruct-GGUF
DEFAULT_GGUF := Llama-3.2-1B-Instruct-Q8_0.gguf

LLAMA_CPP_TAG := b11936
LLAMA_CPP_DIR := third_party/llama.cpp
LLAMA_CPP_CMAKE_FLAGS ?= -DGGML_METAL=ON
WIKITEXT_URL := $(HF)/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip

.PHONY: serve models models-all wikitext llama-cpp bench-mac bench-cuda bench-ablations bench-charts soak build test test-metal parity-ref chat-ref parity lint fuzz teardown

# One command from a fresh clone to a running server on http://127.0.0.1:8000.
serve: models
	cargo build --release --features $(FEATURES) -p server
	target/release/rlis serve

# $(call fetch,repo,file,destination)
define fetch
	@if [ ! -s $(3) ]; then \
		echo "downloading $(2)"; \
		mkdir -p $(dir $(3)); \
		curl -L --fail --progress-bar -o $(3).part $(HF)/$(1)/resolve/main/$(2) && mv $(3).part $(3); \
	fi
endef

models:
	$(call fetch,$(DENSE_REPO),config.json,$(MODEL_DIR)/config.json)
	$(call fetch,$(DENSE_REPO),generation_config.json,$(MODEL_DIR)/generation_config.json)
	$(call fetch,$(DENSE_REPO),tokenizer.json,$(MODEL_DIR)/tokenizer.json)
	$(call fetch,$(DENSE_REPO),tokenizer_config.json,$(MODEL_DIR)/tokenizer_config.json)
	$(call fetch,$(GGUF_REPO),$(DEFAULT_GGUF),models/$(DEFAULT_GGUF))

models-all: models
	$(call fetch,$(DENSE_REPO),model.safetensors,$(MODEL_DIR)/model.safetensors)
	$(call fetch,$(GGUF_REPO),Llama-3.2-1B-Instruct-Q4_K_M.gguf,models/Llama-3.2-1B-Instruct-Q4_K_M.gguf)

# WikiText-2 raw (the copy llama.cpp's perplexity tool uses), for quantization quality.
wikitext:
	@if [ ! -s models/wikitext-2-raw/wiki.test.raw ]; then \
		curl -L --fail -s -o models/wikitext-2-raw-v1.zip $(WIKITEXT_URL) \
			&& unzip -q -o models/wikitext-2-raw-v1.zip -d models \
			&& rm models/wikitext-2-raw-v1.zip; \
	fi
	@ls -l models/wikitext-2-raw/wiki.test.raw

# llama.cpp's server for the baseline, built inside the repo with cmake from a throwaway venv.
llama-cpp:
	@if [ ! -d $(LLAMA_CPP_DIR) ]; then \
		git clone --quiet --depth 1 --branch $(LLAMA_CPP_TAG) https://github.com/ggml-org/llama.cpp.git $(LLAMA_CPP_DIR); \
	fi
	uv venv .venv --python 3.12
	VIRTUAL_ENV=.venv uv pip install cmake
	.venv/bin/cmake -S $(LLAMA_CPP_DIR) -B $(LLAMA_CPP_DIR)/build -DCMAKE_BUILD_TYPE=Release \
		$(LLAMA_CPP_CMAKE_FLAGS) -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
	.venv/bin/cmake --build $(LLAMA_CPP_DIR)/build --target llama-server -j
	rm -rf .venv .uv-cache
	@ls -l $(LLAMA_CPP_DIR)/build/bin/llama-server

build:
	cargo build --release --features $(FEATURES)

test:
	cargo test --workspace

# GPU-only unit tests (paged-attention kernel vs reference); no model download needed.
test-metal:
	cargo test --release -p model --lib --features metal

# Runs the HF transformers reference once in a throwaway venv, then deletes the venv.
parity-ref: models-all
	uv venv .venv --python 3.12
	VIRTUAL_ENV=.venv uv pip install torch transformers safetensors numpy
	.venv/bin/python -I scripts/parity/dump_reference.py --model $(MODEL_DIR) \
		--prompts tests/fixtures/parity/prompts.json --out tests/fixtures/parity/reference.safetensors
	rm -rf .venv .uv-cache

# f32 is the strict gate (token-for-token); f16 also allows flips at reference near-ties.
# Renders reference chat prompts with transformers (tokenizer only, no torch).
chat-ref: models-all
	uv venv .venv --python 3.12
	VIRTUAL_ENV=.venv uv pip install transformers jinja2
	.venv/bin/python -I scripts/parity/dump_chat_reference.py --model $(MODEL_DIR) \
		--out tests/fixtures/chat/llama32_reference.json
	rm -rf .venv .uv-cache

parity: models-all
	PARITY_DTYPE=f32 cargo test --release -p model --features $(FEATURES) --test parity -- --ignored --nocapture --test-threads=1
	PARITY_DTYPE=f16 cargo test --release -p model --features $(FEATURES) --test parity -- --ignored --nocapture --test-threads=1 --skip gguf_tracks_reference
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

# rlis vs llama.cpp concurrency sweep (results/mac/*.json); needs `make llama-cpp`.
bench-mac: models
	cargo build --release --features $(FEATURES) -p server
	cargo build --release -p bench
	scripts/bench/sweep.sh

# The free CUDA run (M6): one Kaggle T4 session pinned to HEAD, which must be pushed to GitHub.
# Parity first, then the same sweep against llama.cpp (CUDA); results land in results/cuda/.
# Needs ~/.kaggle/kaggle.json from a phone-verified account (GPU and internet access).
bench-cuda:
	uv venv --clear .venv --python 3.12
	VIRTUAL_ENV=.venv uv pip install kaggle
	KAGGLE=.venv/bin/kaggle scripts/cuda/kaggle.sh; rc=$$?; rm -rf .venv .uv-cache; exit $$rc

bench-ablations: models
	cargo build --release --features $(FEATURES) -p server
	cargo build --release -p bench
	scripts/bench/ablations.sh

bench-charts:
	cargo build --release -p bench
	scripts/bench/charts.sh

# One hour at concurrency 32 with random client hang-ups (results/mac/soak.json).
SOAK_SECS ?= 3600
soak: models
	cargo build --release --features $(FEATURES) -p server
	cargo build --release -p bench
	scripts/bench/soak.sh $(SOAK_SECS)

teardown:
	-pkill -f 'target/.*/rlis' || true
	-pkill -f 'llama-server' || true
	rm -rf models target third_party .venv .uv-cache fuzz/target fuzz/corpus fuzz/artifacts
	@du -sh .
	@echo "To remove the Rust toolchain too: rustup self uninstall"
