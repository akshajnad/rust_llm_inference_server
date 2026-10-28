import unittest

from listutil import chunk, dedupe


class ListUtilTest(unittest.TestCase):
    def test_chunk_even(self):
        self.assertEqual(chunk([1, 2, 3, 4], 2), [[1, 2], [3, 4]])

    def test_chunk_remainder(self):
        self.assertEqual(chunk([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]])
        self.assertEqual(chunk([1, 2], 5), [[1, 2]])

    def test_chunk_empty(self):
        self.assertEqual(chunk([], 3), [])

    def test_dedupe(self):
        self.assertEqual(dedupe([3, 1, 3, 2, 1]), [3, 1, 2])


if __name__ == "__main__":
    unittest.main()
