import unittest

from src.language_detection import FASTTEXT_LID_SHA256, fasttext_language_evidence
from src.preprocessing import detect_email_language_details


class TestLanguageDetection(unittest.TestCase):
    def test_fasttext_model_is_verified_and_routes_accented_vietnamese(self):
        evidence = fasttext_language_evidence(
            "Vui lòng xác minh tài khoản ngân hàng của bạn ngay lập tức để tránh bị khóa."
        )
        self.assertEqual(evidence["state"], "AVAILABLE")
        self.assertEqual(evidence["top_language"], "vi")
        self.assertGreater(evidence["top_probability"], 0.90)
        self.assertIn("not a phishing decision", evidence["detail"])

    def test_language_model_rejects_an_untrusted_checksum(self):
        evidence = fasttext_language_evidence(
            "Please review the operational planning document before the next meeting.",
            expected_sha256="0" * len(FASTTEXT_LID_SHA256),
        )
        self.assertEqual(evidence["state"], "UNAVAILABLE")
        self.assertIn("checksum", evidence["detail"])

    def test_rules_keep_unaccented_vietnamese_and_spanish_safe(self):
        unaccented = detect_email_language_details(
            "Chung toi thong bao ve lich lam viec va ke hoach hop noi bo tuan nay."
        )
        self.assertEqual(unaccented["language"], "vi")
        self.assertGreaterEqual(unaccented["vietnamese_rule_score"], 4.0)
        spanish = detect_email_language_details(
            "Atención: su cuenta ha sido bloqueada. Verifique ahora para continuar con el servicio."
        )
        self.assertNotEqual(spanish["language"], "vi")
        self.assertEqual(spanish["language_model"]["top_language"], "es")

    def test_long_english_has_explainable_model_evidence(self):
        details = detect_email_language_details(
            "This correspondence contains procedural information regarding the requested allocation and availability."
        )
        self.assertEqual(details["language"], "en")
        self.assertEqual(details["language_model"]["top_language"], "en")
        self.assertEqual(details["model_english_support"], 4.0)


if __name__ == "__main__":
    unittest.main()
