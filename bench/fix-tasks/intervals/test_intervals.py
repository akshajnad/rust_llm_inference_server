import unittest

from intervals import merge, total_length


class IntervalsTest(unittest.TestCase):
    def test_overlapping(self):
        self.assertEqual(merge([(1, 3), (2, 6), (8, 10)]), [(1, 6), (8, 10)])

    def test_unsorted_input(self):
        self.assertEqual(merge([(8, 10), (1, 3), (2, 6)]), [(1, 6), (8, 10)])

    def test_touching(self):
        self.assertEqual(merge([(1, 2), (2, 4)]), [(1, 4)])

    def test_total_length(self):
        self.assertEqual(total_length([(5, 7), (1, 4), (3, 5)]), 6)


if __name__ == "__main__":
    unittest.main()
