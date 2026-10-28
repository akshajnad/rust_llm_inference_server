"""Record one streamed chat completion from a running server and render it as an animated SVG
(docs/streaming.svg) that reveals each chunk at the time it actually arrived.

Usage: python3 scripts/bench/streaming_svg.py [port] [out]
"""

import html
import json
import sys
import textwrap
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8000"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/streaming.svg"
QUESTION = "In three sentences, why does a paged KV cache help an LLM server serve more users?"

body = json.dumps({
    "messages": [{"role": "user", "content": QUESTION}],
    "max_tokens": 120,
    "temperature": 0,
    "stream": True,
}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body,
                             {"content-type": "application/json"})
start = time.time()
chunks = []
with urllib.request.urlopen(req) as r:
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        delta = json.loads(line[6:])["choices"][0].get("delta", {}).get("content")
        if delta:
            chunks.append((time.time() - start, delta))

# Lay the text out in wrapped lines, remembering when each character arrived.
WIDTH, LINE_H, COLS = 760, 22, 78
text = "".join(c for _, c in chunks)
arrival = []
for t, c in chunks:
    arrival.extend([t] * len(c))
lines = textwrap.wrap(text, COLS, drop_whitespace=False)
total = chunks[-1][0] + 2.5 if chunks else 1.0

rows = []
pos = 0
for i, ln in enumerate(lines):
    # One <tspan> per chunk-run so each appears at its arrival time.
    y = 96 + i * LINE_H
    spans = []
    j = 0
    while j < len(ln):
        t = arrival[pos + j]
        k = j
        while k < len(ln) and arrival[pos + k] == t:
            k += 1
        frac = t / total
        spans.append(
            f'<tspan opacity="0"><animate attributeName="opacity" values="0;0;1;1" '
            f'keyTimes="0;{frac:.4f};{min(frac + 0.001, 0.999):.4f};1" dur="{total:.2f}s" '
            f'repeatCount="indefinite"/>{html.escape(ln[j:k])}</tspan>')
        j = k
    pos += len(ln)
    rows.append(f'<text x="24" y="{y}" xml:space="preserve">{"".join(spans)}</text>')

height = 96 + len(lines) * LINE_H + 40
tps = len(chunks) / (chunks[-1][0] - chunks[0][0]) if len(chunks) > 1 else 0
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}" font-family="ui-monospace, Menlo, monospace" font-size="14">
<rect width="{WIDTH}" height="{height}" rx="10" fill="#0f172a"/>
<text x="24" y="34" fill="#94a3b8">$ curl -N localhost:8000/v1/chat/completions  (stream: true)</text>
<text x="24" y="62" fill="#7dd3fc" xml:space="preserve">&gt; {html.escape(QUESTION)}</text>
<g fill="#e2e8f0">
{chr(10).join(rows)}
</g>
<text x="24" y="{height - 16}" fill="#64748b">recorded from rlis, Llama 3.2 1B Q8_0 on an M4 Pro: first token {chunks[0][0]*1000:.0f} ms, {tps:.0f} chunks/s, played back at real speed</text>
</svg>
'''
open(OUT, "w").write(svg)
print(f"wrote {OUT}: {len(chunks)} chunks over {chunks[-1][0]:.2f}s")
