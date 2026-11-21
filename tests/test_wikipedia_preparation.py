import unittest

from scripts.prepare_wikipedia_corpus import _matches, _split_for, normalize_text, redact_sensitive


class WikipediaPreparationTests(unittest.TestCase):
    def test_normalizes_unicode_and_collapses_whitespace(self):
        self.assertEqual(normalize_text("  فارسی\n  متن  "), "فارسی متن")

    def test_redacts_emails_and_persian_mobile_numbers(self):
        source = "ایمیل test@example.com و شماره تماس ۰۹۱۲۳۴۵۶۷۸۹"
        cleaned, counts = redact_sensitive(source)
        self.assertNotIn("test@example.com", cleaned)
        self.assertNotIn("۰۹۱۲۳۴۵۶۷۸۹", cleaned)
        self.assertEqual(cleaned.count("[REDACTED-PERSONAL-INFO]"), 2)
        self.assertGreaterEqual(counts.get("email", 0), 1)
        self.assertEqual(_matches(cleaned), [])

    def test_redacts_labelled_financial_and_national_identifiers(self):
        source = "شماره کارت: 1234 5678 9012 3456، کد ملی: 1234567890"
        cleaned, counts = redact_sensitive(source)
        self.assertNotIn("1234 5678 9012 3456", cleaned)
        self.assertNotIn("1234567890", cleaned)
        self.assertIn("labelled_card_number", counts)
        self.assertIn("labelled_national_id", counts)
        self.assertEqual(_matches(cleaned), [])

    def test_redacts_iranian_iban(self):
        cleaned, counts = redact_sensitive("شبا IR123456789012345678901234")
        self.assertIn("iran_iban", counts)
        self.assertEqual(_matches(cleaned), [])

    def test_keeps_ordinary_persian_text(self):
        source = "ویکی‌پدیا یک دانشنامهٔ آزاد است."
        cleaned, counts = redact_sensitive(source)
        self.assertEqual(cleaned, source)
        self.assertEqual(counts, {})

    def test_numeric_title_and_body_boundary_does_not_form_false_phone_match(self):
        title, _ = redact_sensitive("90125")
        body, _ = redact_sensitive("90125 نام یک آلبوم موسیقی است.")
        document = normalize_text(f"عنوان: {title}; {body}")
        self.assertEqual(_matches(document), [])

    def test_document_split_is_stable_and_uses_only_three_splits(self):
        urls = [f"https://fa.wikipedia.org/wiki/Article_{index}" for index in range(500)]
        first = [_split_for(url) for url in urls]
        second = [_split_for(url) for url in urls]
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"train", "validation", "test"})


if __name__ == "__main__":
    unittest.main()
