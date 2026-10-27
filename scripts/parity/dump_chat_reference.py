"""Render fixed conversations with transformers' apply_chat_template for the Rust chat test.

Only needs the tokenizer files (no torch). Run via `make chat-ref`.
"""

import argparse
import json

from transformers import AutoTokenizer

CONVERSATIONS = [
    [{"role": "user", "content": "Hello!"}],
    [{"role": "system", "content": "  You are terse.  "}, {"role": "user", "content": "Why is the sky blue?"}],
    [
        {"role": "user", "content": "Name a prime."},
        {"role": "assistant", "content": "7"},
        {"role": "user", "content": "Another, with emoji 🎉 and ünïcödé."},
    ],
    [{"role": "system", "content": ""}, {"role": "user", "content": "\nleading and trailing newlines\n"}],
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    cases = []
    for messages in CONVERSATIONS:
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, date_string="26 Jul 2024"
        )
        ids = tok(text, add_special_tokens=False).input_ids
        cases.append({"messages": messages, "text": text, "ids": ids})
    json.dump(cases, open(args.out, "w"), indent=1, ensure_ascii=False)
    print("wrote", args.out, len(cases), "cases")


if __name__ == "__main__":
    main()
