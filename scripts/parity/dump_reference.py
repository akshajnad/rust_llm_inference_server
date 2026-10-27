"""Dump Hugging Face transformers reference outputs for the Rust parity test.

Runs Llama in float32 on CPU and does greedy decoding for --steps tokens per prompt without
stopping at EOS. At each step it records the top-k logits and the full-vocab logsumexp of the
distribution the next token was chosen from. Run via `make parity-ref`, which creates and
then deletes a throwaway venv.
"""

import argparse
import json

import numpy as np
import torch
import transformers
from safetensors.numpy import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--topk", type=int, default=64)
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
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
        "steps": str(args.steps),
        "topk": str(args.topk),
    }
    save_file(tensors, args.out, metadata=meta)
    print("wrote", args.out, meta)


if __name__ == "__main__":
    main()
