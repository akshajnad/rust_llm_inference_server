"""Parsing and formatting durations like "1h30m15s"."""

import re

UNITS = {"h": 3600, "m": 60, "s": 1}


def parse_duration(text):
    """Seconds in a duration such as "2h", "45m", "1h30m" or "90s"."""
    parts = re.findall(r"(\d+)([hms])", text)
    if not parts or "".join(n + u for n, u in parts) != text:
        raise ValueError(f"bad duration {text!r}")
    total = 0
    for number, unit in parts:
        total += int(number) * UNITS["h" if unit == "m" else unit]
    return total


def format_duration(seconds):
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    out = ""
    if h:
        out += f"{h}h"
    if m:
        out += f"{m}m"
    if s or not out:
        out += f"{s}s"
    return out
