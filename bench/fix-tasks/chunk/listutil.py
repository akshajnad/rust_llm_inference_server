"""List utilities."""


def chunk(items, size):
    """Split items into consecutive lists of `size` (the last one may be shorter)."""
    if size < 1:
        raise ValueError("size must be positive")
    return [items[i : i + size] for i in range(0, len(items) - size + 1, size)]


def dedupe(items):
    """Remove duplicates, keeping the first occurrence of each item."""
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
