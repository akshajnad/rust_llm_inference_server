"""Interval arithmetic on (start, end) pairs."""


def merge(intervals):
    """Merge overlapping or touching intervals; the result is sorted by start."""
    out = []
    for start, end in intervals:
        if out and start < out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def total_length(intervals):
    return sum(end - start for start, end in merge(intervals))
