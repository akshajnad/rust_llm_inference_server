"""Text helpers for URLs."""

import re


def slugify(title):
    """Lowercase, words joined by single dashes, no leading or trailing dash."""
    slug = title.lower()
    slug = re.sub(r"[^a-z0-9]", "-", slug)
    return slug


def truncate(text, width):
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"
