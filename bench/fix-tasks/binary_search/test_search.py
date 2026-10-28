import unittest

from search import binary_search, contains


class SearchTest(unittest.TestCase):
    def test_finds_every_element(self):
        items = [1, 3, 5, 7, 9, 11]
        for i, x in enumerate(items):
            self.assertEqual(binary_search(items, x), i)

    def test_single_element(self):
        self.assertEqual(binary_search([4], 4), 0)

    def test_missing(self):
        self.assertEqual(binary_search([1, 3, 5], 4), -1)
        self.assertEqual(binary_search([], 1), -1)
        self.assertFalse(contains([1, 3, 5], 6))


if __name__ == "__main__":
    unittest.main()
