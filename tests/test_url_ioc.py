import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.phishvn_url_benchmark import (
    EXPECTED_CORE_COUNTS,
    load_phishvn_core,
    verify_phishvn_source,
)
from src.preprocessing import analyze_url_risk
from src.url_ioc_model import (
    URLIocArtifactError,
    URL_FEATURE_COLUMNS,
    extract_url_features,
    load_url_ioc_detector,
    score_url_ioc,
)


class TestPhishVNURLIOC(unittest.TestCase):
    def test_feature_contract_matches_every_released_core_row(self):
        splits, _ = load_phishvn_core()
        for frame in splits.values():
            for row in frame.itertuples(index=False):
                actual = extract_url_features(row.url)
                for column in URL_FEATURE_COLUMNS:
                    self.assertAlmostEqual(actual[column], float(getattr(row, column)), places=10)
        self.assertNotIn("is_https", URL_FEATURE_COLUMNS)

    def test_source_integrity_and_official_group_splits(self):
        fingerprint = verify_phishvn_source()
        self.assertIn("data/vn_compphish.csv", fingerprint["files"])
        splits, _ = load_phishvn_core()
        for name, expected in EXPECTED_CORE_COUNTS.items():
            self.assertEqual(len(splits[name]), expected["samples"])
            self.assertEqual(int(splits[name]["label"].sum()), expected["phishing"])
        self.assertFalse(set(splits["train"]["dom"]).intersection(splits["val"]["dom"]))
        self.assertFalse(set(splits["train"]["dom"]).intersection(splits["test"]["dom"]))
        self.assertFalse(set(splits["val"]["dom"]).intersection(splits["test"]["dom"]))

    def test_detector_fails_closed_without_a_pointer(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(URLIocArtifactError):
                load_url_ioc_detector(results_root=root)
        evidence = score_url_ioc("https://ordinary-example.org/login")
        self.assertIn(evidence["state"], {"UNAVAILABLE", "AVAILABLE"})
        self.assertIn("not an email classifier", evidence["scope"])
        if evidence["state"] == "UNAVAILABLE":
            self.assertIsNone(evidence["alert"])

    def test_legacy_url_score_is_not_fused_with_url_ioc(self):
        analyze_url_risk.cache_clear()
        result = analyze_url_risk("https://ordinary-example.org/path", check_activity=False)
        self.assertIn("risk_score", result)
        self.assertIn(result["lexical_state"], {"NO_INDICATOR", "RISK"})
        self.assertIn("url_ml_ioc", result)
        self.assertIn(result["url_ml_ioc"]["state"], {"UNAVAILABLE", "AVAILABLE", "NOT_APPLICABLE"})


if __name__ == "__main__":
    unittest.main()
