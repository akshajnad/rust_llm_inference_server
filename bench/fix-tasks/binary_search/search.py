"""Searching sorted sequences."""


def binary_search(items, target):
    """Index of target in the sorted list items, or -1 if it is absent."""
    lo, hi = 0, len(items) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if items[mid] == target:
            return mid
        if items[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def contains(items, target):
    return binary_search(items, target) != -1
