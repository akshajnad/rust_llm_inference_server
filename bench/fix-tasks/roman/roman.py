"""Roman numerals."""

VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

PAIRS = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
    (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]


def to_roman(n):
    out = ""
    for value, numeral in PAIRS:
        while n >= value:
            out += numeral
            n -= value
    return out


def from_roman(s):
    return sum(VALUES[c] for c in s)
