import unittest

from roman import from_roman, to_roman


class RomanTest(unittest.TestCase):
    def test_to_roman(self):
        self.assertEqual(to_roman(1994), "MCMXCIV")

    def test_from_roman_additive(self):
        self.assertEqual(from_roman("XVI"), 16)

    def test_from_roman_subtractive(self):
        self.assertEqual(from_roman("IV"), 4)
        self.assertEqual(from_roman("MCMXCIV"), 1994)

    def test_round_trip(self):
        for n in range(1, 400):
            self.assertEqual(from_roman(to_roman(n)), n)


if __name__ == "__main__":
    unittest.main()
