"""Record a real streamed chat completion and render it as an animated SVG (docs/streaming.svg).

Each token appears at the moment it actually arrived from the server, so the animation shows
true time-to-first-token and decode speed. Standard library only.

Usage: python3 scripts/bench/streaming_demo.py [port] [out.svg]
"""

import html
import json
import sys
import textwrap
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8000"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/streaming.svg"
QUESTION = "Why is the sky blue? Answer in three sentences."


def record():
    body = json.dumps({
        "messages": [{"role": "user", "content": QUESTION}],
        "max_tokens": 120, "temperature": 0, "stream": True,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body,
                                 {"content-type": "application/json"})
    start, pieces = time.time(), []
    with urllib.request.urlopen(req) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            text = json.loads(line[6:])["choices"][0]["delta"].get("content")
            if text:
                pieces.append((time.time() - start, text))
    return pieces


def layout(text, cols):
    """Greedy word wrap; returns the lines and, for each character, its (line, column)."""
    lines, pos, cur = [], [], ""
    for word in text.replace("\n", " \n ").split(" "):
        if word == "\n":
            lines.append(cur)
            cur = ""
            continue
        sep = " " if cur else ""
        if len(cur) + len(sep) + len(word) > cols and cur:
            lines.append(cur)
            cur, sep = "", ""
        cur += sep + word
    lines.append(cur)
    # Map every character of the original text onto the wrapped layout.
    li, col = 0, 0
    for ch in text:
        while li < len(lines) - 1 and col >= len(lines[li]):
            li, col = li + 1, 0
            if ch == " ":
                break
        if ch in " \n" and col == 0:
            pos.append((li, 0))
            continue
        col += 1
        pos.append((li, col))
    return lines, pos


def render(pieces):
    width, line_h, cols, cw = 760, 22, 80, 8.43
    text = "".join(t for _, t in pieces)
    lines, pos = layout(text, cols)
    # When each line reaches each column: one keyframe per streamed chunk.
    frames = [[(0.0, 0)] for _ in lines]
    end = 0
    for t, piece in pieces:
        end += len(piece)
        li, col = pos[end - 1]
        for k in range(li):
            if frames[k][-1][1] < len(lines[k]):
                frames[k].append((t, len(lines[k])))
        frames[li].append((t, col))
    total = pieces[-1][0] if pieces else 1.0
    loop = total + 3.0
    height = 108 + line_h * len(lines) + 34
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="14">',
        f'<rect width="{width}" height="{height}" rx="10" fill="#0f172a"/>',
        '<circle cx="22" cy="20" r="6" fill="#ef4444"/><circle cx="42" cy="20" r="6" fill="#f59e0b"/>'
        '<circle cx="62" cy="20" r="6" fill="#22c55e"/>',
        '<text x="20" y="56" fill="#94a3b8">$ curl -N localhost:8000/v1/chat/completions  # "stream": true</text>',
        f'<text x="20" y="80" fill="#e2e8f0">&gt; {html.escape(QUESTION)}</text>',
    ]
    for i, line in enumerate(lines):
        y = 108 + i * line_h
        kf = frames[i]
        times = ";".join(f"{min(t / loop, 1):.4f}" for t, _ in kf) + ";1"
        values = ";".join(f"{c * cw:.1f}" for _, c in kf) + f";{kf[-1][1] * cw:.1f}"
        out.append(f'<clipPath id="l{i}"><rect x="20" y="{y - 15}" height="{line_h}" width="0">'
                   f'<animate attributeName="width" dur="{loop:.2f}s" repeatCount="indefinite" '
                   f'calcMode="discrete" keyTimes="{times}" values="{values}"/></rect></clipPath>')
        out.append(f'<text x="20" y="{y}" fill="#e2e8f0" clip-path="url(#l{i})" '
                   f'xml:space="preserve">{html.escape(line)}</text>')
    ttft = pieces[0][0] if pieces else 0
    rate = (len(pieces) - 1) / (total - ttft) if len(pieces) > 1 and total > ttft else 0
    out.append(f'<text x="20" y="{height - 14}" fill="#64748b">real timing: first token '
               f'{ttft * 1000:.0f} ms, {rate:.0f} tokens/s · rlis, Llama 3.2 1B Q8_0, M4 Pro</text>')
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    pieces = record()
    with open(OUT, "w") as f:
        f.write(render(pieces))
    print(f"wrote {OUT}: {len(pieces)} chunks over {pieces[-1][0]:.2f}s")
