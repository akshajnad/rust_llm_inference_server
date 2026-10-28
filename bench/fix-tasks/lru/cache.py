"""A small least-recently-used cache."""

from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.items = OrderedDict()

    def get(self, key, default=None):
        if key not in self.items:
            return default
        return self.items[key]

    def put(self, key, value):
        if key in self.items:
            self.items.move_to_end(key)
        self.items[key] = value
        if len(self.items) > self.capacity:
            self.items.popitem(last=False)

    def __len__(self):
        return len(self.items)
