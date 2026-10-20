"""Public-clone checks requiring synthetic fixtures and NumPy only."""

import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from src.security_matrix import build_12_layer_matrix

ROOT = Path(__file__).resolve().parents[1]


def example(name="synthetic_phishing.eml"):
    return (ROOT / "examples" / name).read_text(encoding="utf-8")


def by_key(raw):
    return {item["key"]: item for item in build_12_layer_matrix(raw)}


class OfflineEvidenceTests(unittest.TestCase):
    def test_reports_all_twelve_layers(self):
        layers = build_12_layer_matrix(example())
        self.assertEqual([item["layer"] for item in layers], list(range(1, 13)))
        self.assertTrue(all(item["state"] in {"RISK", "NO_INDICATOR", "UNKNOWN", "MODEL"} for item in layers))

    def test_missing_models_stay_unknown(self):
        layers = by_key(example())
        for key in ("ml_detector", "llm_review"):
            self.assertEqual(layers[key]["state"], "UNKNOWN")
            self.assertIsNone(layers[key]["risk"])

    def test_reply_to_mismatch_is_an_indicator(self):
        layers = by_key(example())
        self.assertEqual(layers["header_consistency"]["state"], "RISK")
        self.assertIn("differs", layers["header_consistency"]["detail"])

    def test_urgency_and_credentials_are_observed(self):
        layers = by_key(example())
        self.assertEqual(layers["social_engineering"]["state"], "RISK")
        self.assertGreater(layers["social_engineering"]["risk"], 0)

    def test_pasted_auth_pass_is_not_trusted_by_default(self):
        raw = "Authentication-Results: gateway.example; spf=pass; dkim=pass; dmarc=pass\n" + example()
        self.assertEqual(by_key(raw)["authentication"]["state"], "UNKNOWN")

    def test_no_mismatch_does_not_supply_model_verdict(self):
        layers = by_key(example("synthetic_benign.eml"))
        self.assertEqual(layers["header_consistency"]["state"], "NO_INDICATOR")
        self.assertEqual(layers["ml_detector"]["state"], "UNKNOWN")

    def test_extraction_makes_no_network_requests(self):
        with patch("socket.getaddrinfo", side_effect=AssertionError("DNS forbidden")), \
             patch("socket.create_connection", side_effect=AssertionError("Network forbidden")), \
             patch("urllib.request.urlopen", side_effect=AssertionError("HTTP forbidden")):
            self.assertEqual(len(build_12_layer_matrix(example())), 12)

    def test_cli_outputs_parseable_evidence_without_a_verdict(self):
        result = subprocess.run(
            [sys.executable, "tools/demo_indicators.py", "--json"],
            cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8",
        )
        payload = json.loads(result.stdout)
        self.assertIsNone(payload["final_verdict"])
        self.assertFalse(payload["ml_available"])
        self.assertFalse(payload["llm_available"])
        self.assertEqual(len(payload["layers"]), 12)


if __name__ == "__main__":
    unittest.main()
