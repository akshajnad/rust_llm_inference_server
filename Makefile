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
# Parallel compile jobs; empty means unlimited (fine for Metal, but CUDA's template-heavy
# kernels run a 31 GB machine out of memory that way).
LLAMA_CPP_JOBS ?=
WIKITEXT_URL := $(HF)/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip

.PHONY: serve chat install uninstall models models-qwen models-qwen-3b models-k2 models-qwen-14b models-all wikitext llama-cpp bench-mac bench-cuda bench-ablations bench-fork bench-fix bench-fix-3b bench-charts soak build test test-metal parity-ref parity-ref-qwen chat-ref chat-ref-qwen parity parity-qwen lint fuzz teardown

# One command from a fresh clone to a running server on http://127.0.0.1:8000.
serve: models
	cargo build --release --features $(FEATURES) -p server
	target/release/rlis serve

# Terminal chat; starts the server in the background (and stops it on exit) if none is running.
chat: models
	cargo build --release -p server -p chat --features server/$(FEATURES)
	target/release/rlis-chat

# Puts the chat on the PATH as `rlis` (rustup already adds ~/.cargo/bin), linked to this build
# so it runs from any directory and picks up rebuilds; `rlis serve` etc. reach the server.
INSTALL_DIR ?= $(HOME)/.cargo/bin
install: models
	cargo build --release -p server -p chat --features server/$(FEATURES)
	mkdir -p $(INSTALL_DIR)
	rm -f $(INSTALL_DIR)/rlis-chat
	ln -sf $(CURDIR)/target/release/rlis-chat $(INSTALL_DIR)/rlis
	@echo "installed: $(INSTALL_DIR)/rlis -> $(CURDIR)/target/release/rlis-chat"

uninstall:
	rm -f $(INSTALL_DIR)/rlis $(INSTALL_DIR)/rlis-chat

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

# Qwen2.5-Coder 1.5B for `rlis fix`: tokenizer files from the HF repo, Q8_0 GGUF (tied embeddings).
QWEN_DIR := models/qwen2.5-coder-1.5b-instruct
QWEN_REPO := Qwen/Qwen2.5-Coder-1.5B-Instruct
QWEN_GGUF_REPO := bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF
QWEN_GGUF := Qwen2.5-Coder-1.5B-Instruct-Q8_0.gguf

models-qwen:
	$(call fetch,$(QWEN_REPO),config.json,$(QWEN_DIR)/config.json)
	$(call fetch,$(QWEN_REPO),generation_config.json,$(QWEN_DIR)/generation_config.json)
	$(call fetch,$(QWEN_REPO),tokenizer.json,$(QWEN_DIR)/tokenizer.json)
	$(call fetch,$(QWEN_REPO),tokenizer_config.json,$(QWEN_DIR)/tokenizer_config.json)
	$(call fetch,$(QWEN_GGUF_REPO),$(QWEN_GGUF),models/$(QWEN_GGUF))

# The 3B (36 layers, 8 query heads per KV head), Q4_K_M to stay inside the memory budget.
QWEN3B_DIR := models/qwen2.5-coder-3b-instruct
QWEN3B_REPO := Qwen/Qwen2.5-Coder-3B-Instruct
QWEN3B_GGUF := Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf

models-qwen-3b:
	$(call fetch,$(QWEN3B_REPO),config.json,$(QWEN3B_DIR)/config.json)
	$(call fetch,$(QWEN3B_REPO),generation_config.json,$(QWEN3B_DIR)/generation_config.json)
	$(call fetch,$(QWEN3B_REPO),tokenizer.json,$(QWEN3B_DIR)/tokenizer.json)
	$(call fetch,$(QWEN3B_REPO),tokenizer_config.json,$(QWEN3B_DIR)/tokenizer_config.json)
	$(call fetch,bartowski/Qwen2.5-Coder-3B-Instruct-GGUF,$(QWEN3B_GGUF),models/$(QWEN3B_GGUF))

# `rlis code`'s default: K2 Horizon 3.7B (36 layers, 4 query heads per KV head), which thinks first
# and calls tools natively. Its chat template lives in its own file.
K2_DIR := models/k2-horizon-3.7b
K2_REPO := IFM/K2-Horizon-3.7B
K2_GGUF := K2-Horizon-3.7B-Q4_K_M.gguf

models-k2:
	$(call fetch,$(K2_REPO),config.json,$(K2_DIR)/config.json)
	$(call fetch,$(K2_REPO),generation_config.json,$(K2_DIR)/generation_config.json)
	$(call fetch,$(K2_REPO),tokenizer.json,$(K2_DIR)/tokenizer.json)
	$(call fetch,$(K2_REPO),tokenizer_config.json,$(K2_DIR)/tokenizer_config.json)
	$(call fetch,$(K2_REPO),chat_template.jinja,$(K2_DIR)/chat_template.jinja)
	$(call fetch,bartowski/K2-Horizon-3.7B-GGUF,$(K2_GGUF),models/$(K2_GGUF))

# The largest coder: the 14B (48 layers, 7 query heads per KV head).
QWEN14B_DIR := models/qwen2.5-coder-14b-instruct
QWEN14B_REPO := Qwen/Qwen2.5-Coder-14B-Instruct
QWEN14B_GGUF := Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf

models-qwen-14b:
	$(call fetch,$(QWEN14B_REPO),config.json,$(QWEN14B_DIR)/config.json)
	$(call fetch,$(QWEN14B_REPO),generation_config.json,$(QWEN14B_DIR)/generation_config.json)
	$(call fetch,$(QWEN14B_REPO),tokenizer.json,$(QWEN14B_DIR)/tokenizer.json)
	$(call fetch,$(QWEN14B_REPO),tokenizer_config.json,$(QWEN14B_DIR)/tokenizer_config.json)
	$(call fetch,bartowski/Qwen2.5-Coder-14B-Instruct-GGUF,$(QWEN14B_GGUF),models/$(QWEN14B_GGUF))

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
	.venv/bin/cmake --build $(LLAMA_CPP_DIR)/build --target llama-server -j $(LLAMA_CPP_JOBS)
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

# Qwen2.5-Coder reference from the GGUF's own weights (f16 storage, f32 compute: f32 weights
# would be ~6 GB). Prompts are the Llama fixture's.
parity-ref-qwen: models-qwen
	uv venv .venv --python 3.12
	VIRTUAL_ENV=.venv uv pip install torch transformers safetensors numpy gguf
	.venv/bin/python -I scripts/parity/dump_reference.py --model $(QWEN_DIR) --gguf models/$(QWEN_GGUF) \
		--prompts tests/fixtures/parity/prompts.json --out tests/fixtures/parity/qwen25_coder_reference.safetensors
	rm -rf .venv .uv-cache

chat-ref-qwen: models-qwen
	uv venv .venv --python 3.12
	VIRTUAL_ENV=.venv uv pip install transformers jinja2
	.venv/bin/python -I scripts/parity/dump_chat_reference.py --model $(QWEN_DIR) \
		--out tests/fixtures/chat/qwen25_coder_reference.json
	rm -rf .venv .uv-cache

parity: models-all
	PARITY_DTYPE=f32 cargo test --release -p model --features $(FEATURES) --test parity -- --ignored --nocapture --test-threads=1
	PARITY_DTYPE=f16 cargo test --release -p model --features $(FEATURES) --test parity -- --ignored --nocapture --test-threads=1 --skip gguf_tracks_reference
	cargo test --release -p engine --test chat_template -- --ignored llama32

# The GGUF is the model under test (f32 activations, f16 KV); its reference has f16-rounded
# weights, so it is held to the f16 gate.
parity-qwen: models-qwen
	PARITY_MODEL=qwen cargo test --release -p model --features $(FEATURES) --test parity -- --ignored --nocapture --test-threads=1 --skip gguf_tracks_reference
	cargo test --release -p engine --test chat_template -- --ignored qwen25

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

# Cost of N completions of one prompt vs one (results/mac/fork-*.json, docs/fix-bench.md);
# needs `make llama-cpp` and `make wikitext`.
bench-fork: models
	cargo build --release --features $(FEATURES) -p server
	cargo build --release -p bench
	scripts/bench/fork.sh

# `rlis fix` on the 10 buggy Python tasks in bench/fix-tasks (results/mac/fix/, docs/fix-bench.md).
bench-fix: models-qwen
	cargo build --release --features $(FEATURES) -p server
	cargo build --release -p chat
	scripts/bench/fix-tasks.sh

bench-fix-3b: models-qwen-3b
	cargo build --release --features $(FEATURES) -p server
	cargo build --release -p chat
	MODEL=models/$(QWEN3B_GGUF) TOKENIZER=$(QWEN3B_DIR) NAME=qwen2.5-coder-3b-instruct \
		OUT=results/mac/fix-3b scripts/bench/fix-tasks.sh

bench-charts:
	cargo build --release -p bench
	scripts/bench/charts.sh

# One hour at concurrency 32 with random client hang-ups (results/mac/soak.json).
SOAK_SECS ?= 3600
soak: models
	cargo build --release --features $(FEATURES) -p server
	cargo build --release -p bench
	scripts/bench/soak.sh $(SOAK_SECS)

teardown: uninstall
	-pkill -f 'target/.*/rlis' || true
	-pkill -f 'llama-server' || true
	rm -rf models target third_party .venv .uv-cache fuzz/target fuzz/corpus fuzz/artifacts
	@du -sh .
	@echo "To remove the Rust toolchain too: rustup self uninstall"
