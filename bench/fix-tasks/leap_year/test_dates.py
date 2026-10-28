import unittest

from dates import day_of_year, days_in_month, is_leap_year


class DatesTest(unittest.TestCase):
    def test_leap_years(self):
        self.assertTrue(is_leap_year(2024))
        self.assertTrue(is_leap_year(2000))
        self.assertFalse(is_leap_year(2023))
        self.assertFalse(is_leap_year(1900))
        self.assertFalse(is_leap_year(2100))

    def test_february(self):
        self.assertEqual(days_in_month(2024, 2), 29)
        self.assertEqual(days_in_month(1900, 2), 28)

    def test_day_of_year(self):
        self.assertEqual(day_of_year(2023, 3, 1), 60)
        self.assertEqual(day_of_year(2024, 3, 1), 61)


if __name__ == "__main__":
    unittest.main()
