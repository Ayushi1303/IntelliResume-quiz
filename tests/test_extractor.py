import unittest

from utils.extractor import extract_name


class TestExtractName(unittest.TestCase):
    def test_labelled_name(self):
        text = """
        Name: Alice B. Smith
        Email: alice@example.com
        """.strip()
        self.assertEqual(extract_name(text), "Alice B. Smith")

    def test_header_with_separators(self):
        text = """
        JANE DOE | Software Engineer | Bangalore
        jane.doe@example.com | +91 99999 99999 | github.com/janedoe
        """.strip()
        self.assertEqual(extract_name(text), "Jane Doe")

    def test_two_capitalized_words_fallback(self):
        text = """
        Portfolio
        John Doe
        Email: john@example.com
        """.strip()
        self.assertEqual(extract_name(text), "John Doe")

    def test_single_token_name(self):
        text = """
        PRINCE
        Email: prince@example.com
        """.strip()
        self.assertEqual(extract_name(text), "Prince")

    def test_unknown_when_absent(self):
        text = """
        Resume
        Software Engineer
        Email: se@example.com
        """.strip()
        self.assertEqual(extract_name(text), "Unknown")


if __name__ == "__main__":
    unittest.main()
