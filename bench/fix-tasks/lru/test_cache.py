import unittest

from cache import LRUCache


class LRUCacheTest(unittest.TestCase):
    def test_put_get(self):
        c = LRUCache(2)
        c.put("a", 1)
        self.assertEqual(c.get("a"), 1)
        self.assertIsNone(c.get("b"))

    def test_evicts_least_recently_put(self):
        c = LRUCache(2)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)
        self.assertIsNone(c.get("a"))
        self.assertEqual(len(c), 2)

    def test_get_counts_as_use(self):
        c = LRUCache(2)
        c.put("a", 1)
        c.put("b", 2)
        c.get("a")
        c.put("c", 3)
        self.assertEqual(c.get("a"), 1)
        self.assertIsNone(c.get("b"))


if __name__ == "__main__":
    unittest.main()
