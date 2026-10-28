"""Dump Hugging Face transformers reference outputs for the Rust parity test.

Runs the model in float32 on CPU and does greedy decoding for --steps tokens per prompt
without stopping at EOS. At each step it records the top-k logits and the full-vocab logsumexp
of the distribution the next token was chosen from. Run via `make parity-ref` (or
`make parity-ref-qwen`), which creates and then deletes a throwaway venv.

With --gguf, the weights come from that GGUF file instead of safetensors, into transformers'
own model class for the --model config. A 1.5B model in f32 is ~6 GB, over the memory budget,
so matrices are stored in f16 (Q8_0 dequantized, then rounded) and every matmul runs in f32;
norms and biases stay f32.
"""

import argparse
import json
import re

import numpy as np
import torch
import transformers
from safetensors.numpy import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# GGUF tensor names (llama.cpp) -> transformers parameter names, for unpermuted architectures
# (qwen2); llama's converter permutes Q/K rows, which this doesn't undo.
GGUF_NAMES = {
    "token_embd": "model.embed_tokens",
    "output_norm": "model.norm",
    "attn_norm": "input_layernorm",
    "ffn_norm": "post_attention_layernorm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}


def hf_name(gguf_name: str) -> str:
    m = re.fullmatch(r"blk\.(\d+)\.(\w+)\.(weight|bias)", gguf_name)
    if m:
        return f"model.layers.{m[1]}.{GGUF_NAMES[m[2]]}.{m[3]}"
    base, kind = gguf_name.rsplit(".", 1)
    return f"{GGUF_NAMES[base]}.{kind}"


def f32_linear(mod: torch.nn.Module) -> None:
    """Computes `mod` in f32 from its f16 weight, upcasting big matrices in row chunks."""
    w, b = mod.weight, mod.bias

    def forward(x):
        x = x.float()
        y = torch.cat([x @ w[i : i + 16384].float().T for i in range(0, w.shape[0], 16384)], -1)
        return y if b is None else y + b.float()

    mod.forward = forward


def load_gguf(model_dir: str, path: str):
    from gguf import GGUFReader
    from gguf.quants import dequantize

    config = AutoConfig.from_pretrained(model_dir)
    # Built without memory (meta) so the loaded tensors are the only copy of the weights.
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config, dtype=torch.float16, attn_implementation="eager"
        ).eval()
    state = {}
    for t in GGUFReader(path).tensors:
        shape = tuple(int(x) for x in reversed(t.shape))
        if len(shape) == 1:
            # Norms and biases stay exact.
            w = torch.from_numpy(dequantize(t.data, t.tensor_type).astype(np.float32))
        else:
            # Matrices in f16, filled a slice of rows at a time: a whole f32 copy of the
            # embedding alone is 0.9 GB, which pushed loading past the memory budget.
            w = torch.empty(shape, dtype=torch.float16)
            for i in range(0, shape[0], 16384):
                rows = dequantize(t.data[i : i + 16384], t.tensor_type)
                w[i : i + len(rows)] = torch.from_numpy(rows.reshape(len(rows), shape[1]))
        state[hf_name(t.name)] = w
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    # The LM head is tied to the embedding.
    assert set(missing) <= {"lm_head.weight"} and not unexpected, (missing, unexpected)
    model.tie_weights()
    # Buffers computed at construction (RoPE inverse frequencies) are still on meta.
    rotary = model.model.rotary_emb
    model.model.rotary_emb = type(rotary)(config=config)
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            f32_linear(m)
    model.model.embed_tokens.register_forward_hook(lambda m, i, o: o.float())
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--gguf", help="take the weights from this GGUF file (see module docstring)")
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.gguf:
        model = load_gguf(args.model, args.gguf)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32, attn_implementation="eager"
        ).eval()
    prompts = json.load(open(args.prompts))

    inv_freq = model.model.rotary_emb.inv_freq.float().tolist()
    print("inv_freq:", ", ".join(f"{x:.9e}" for x in inv_freq))

    tensors = {"inv_freq": np.asarray(inv_freq, dtype=np.float32)}
    with torch.inference_mode():
        for i, prompt in enumerate(prompts):
            ids = tok(prompt, return_tensors="pt").input_ids
            out = model(ids, use_cache=True)
            past, logits = out.past_key_values, out.logits[0, -1]
            greedy, top_v, top_i, lse = [], [], [], []
            for _ in range(args.steps):
                v, ix = torch.topk(logits, args.topk)
                top_v.append(v.numpy())
                top_i.append(ix.numpy().astype(np.uint32))
                lse.append(torch.logsumexp(logits, dim=-1).item())
                nxt = int(torch.argmax(logits))
                greedy.append(nxt)
                out = model(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
                past, logits = out.past_key_values, out.logits[0, -1]
            tensors[f"p{i}.prompt_ids"] = ids[0].numpy().astype(np.uint32)
            tensors[f"p{i}.greedy_ids"] = np.asarray(greedy, dtype=np.uint32)
            tensors[f"p{i}.topk_values"] = np.stack(top_v).astype(np.float32)
            tensors[f"p{i}.topk_indices"] = np.stack(top_i)
            tensors[f"p{i}.logsumexp"] = np.asarray(lse, dtype=np.float32)
            print(f"[{i + 1}/{len(prompts)}] {ids.shape[1]} prompt tokens: {tok.decode(greedy[:16])!r}")

    meta = {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "dtype": "float32",
        "weights": f"f16 from {args.gguf}" if args.gguf else "safetensors",
        "steps": str(args.steps),
        "topk": str(args.topk),
    }
    save_file(tensors, args.out, metadata=meta)
    print("wrote", args.out, meta)


if __name__ == "__main__":
    main()
