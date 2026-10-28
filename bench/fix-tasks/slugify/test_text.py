import unittest

from text import slugify, truncate


class TextTest(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(slugify("Hello World"), "hello-world")

    def test_punctuation_and_spaces(self):
        self.assertEqual(slugify("  Rust, Metal & You!  "), "rust-metal-you")
        self.assertEqual(slugify("a -- b"), "a-b")

    def test_truncate(self):
        self.assertEqual(truncate("abcdef", 4), "abc…")
        self.assertEqual(truncate("abc", 4), "abc")


if __name__ == "__main__":
    unittest.main()
