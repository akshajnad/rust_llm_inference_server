"""Helpers for nested lists."""


def flatten(items):
    """All non-list values of an arbitrarily nested list, in order."""
    out = []
    for x in items:
        if isinstance(x, list):
            out.extend(x)
        else:
            out.append(x)
    return out


def depth(items):
    if not isinstance(items, list):
        return 0
    return 1 + max((depth(x) for x in items), default=0)
