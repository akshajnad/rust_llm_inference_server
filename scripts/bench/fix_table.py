"""Markdown table of `rlis fix` results (the <task>.json files scripts/bench/fix-tasks.sh writes).

pass@1 is the unbiased estimate from n samples, c / n; pass@n is whether any attempt passed.
Usage: python -I scripts/bench/fix_table.py results/mac/fix
"""

import json
import pathlib
import sys


def main() -> None:
    out = pathlib.Path(sys.argv[1])
    runs = []
    for path in sorted(out.glob("*.json")):
        r = json.loads(path.read_text())
        if "attempts" in r:
            runs.append((path.stem, r))
    n = runs[0][1]["n"]
    print(
        f"| task | applied | passed | pass@1 | pass@{n} | prompt tok | output tok "
        "| generate (s) | tests (s) | total (s) |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, r in runs:
        s = r["secs"]
        print(
            f"| {name} | {r['applied']}/{r['n']} | {r['passed']}/{r['n']} "
            f"| {r['passed'] / r['n']:.2f} | {int(r['passed'] > 0)} | {r['prompt_tokens']} "
            f"| {r['completion_tokens']} | {s['generate']:.1f} | {s['tests']:.1f} "
            f"| {s['total']:.1f} |"
        )
    k = len(runs)
    pass1 = sum(r["passed"] / r["n"] for _, r in runs) / k
    passn = sum(r["passed"] > 0 for _, r in runs)
    applied = sum(r["applied"] for _, r in runs)
    total = sum(r["secs"]["total"] for _, r in runs)
    gen = sum(r["secs"]["generate"] for _, r in runs)
    print(
        f"| **all {k}** | {applied}/{k * n} | {sum(r['passed'] for _, r in runs)}/{k * n} "
        f"| **{pass1:.2f}** | **{passn}/{k}** | | "
        f"| {gen:.1f} | | {total:.1f} |"
    )


if __name__ == "__main__":
    main()
