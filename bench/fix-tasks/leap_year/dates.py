"""Calendar helpers."""

DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def is_leap_year(year):
    return year % 4 == 0


def days_in_month(year, month):
    if month == 2 and is_leap_year(year):
        return 29
    return DAYS_IN_MONTH[month - 1]


def day_of_year(year, month, day):
    return sum(days_in_month(year, m) for m in range(1, month)) + day
