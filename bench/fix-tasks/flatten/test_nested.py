import unittest

from nested import depth, flatten


class NestedTest(unittest.TestCase):
    def test_flat(self):
        self.assertEqual(flatten([1, 2, 3]), [1, 2, 3])

    def test_one_level(self):
        self.assertEqual(flatten([1, [2, 3], 4]), [1, 2, 3, 4])

    def test_deep(self):
        self.assertEqual(flatten([1, [2, [3, [4, []]]], 5]), [1, 2, 3, 4, 5])

    def test_depth(self):
        self.assertEqual(depth([1, [2, [3]]]), 3)


if __name__ == "__main__":
    unittest.main()
