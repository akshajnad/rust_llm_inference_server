import unittest

from timefmt import format_duration, parse_duration


class DurationTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_duration("2h"), 7200)
        self.assertEqual(parse_duration("45m"), 2700)
        self.assertEqual(parse_duration("1h30m15s"), 5415)

    def test_bad(self):
        with self.assertRaises(ValueError):
            parse_duration("10x")

    def test_round_trip(self):
        for s in [0, 59, 61, 3600, 3725]:
            self.assertEqual(parse_duration(format_duration(s)), s)


if __name__ == "__main__":
    unittest.main()
