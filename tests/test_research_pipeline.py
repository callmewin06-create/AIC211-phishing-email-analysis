import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.research_benchmark import (
    CHARACTER_FEATURE_TEXT_MAX_CHARS,
    _character_feature_view,
    _deployment_gate,
    _selection_cross_validation,
)
from src.research_data import (
    DEFAULT_REGISTRY,
    MODEL_DOCUMENT_MAX_CHARS,
    RESEARCH_TEXT_MAX_CHARS,
    _label_strings,
    _read_source,
    build_research_document,
    normalize_research_text,
    registry_project_root,
    source_cache_fingerprint,
    source_governance_record,
    split_primary_corpora,
)
from src.research_deployment import load_research_detector, prepare_research_document
from src.major_split_experiment import _partition_integrity, split_historical, split_major
from src.research_metrics import classification_metrics, grouped_bootstrap_ci
from src.security_matrix import build_12_layer_matrix, technical_matrix_score, url_risk_score
from src.llm_benchmark import PROMPT_VERSION, _extract_json, _prompt, _score
from src.preprocessing import decode_eml_upload, evaluate_12_layers, extract_llm_label, parse_raw_email


class TestResearchPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.external, cls.manifest = split_primary_corpora()

    def test_registry_has_explicit_spam_phishing_policy(self):
        registry = json.loads(Path(DEFAULT_REGISTRY).read_text(encoding="utf-8"))
        self.assertEqual(registry["task_definition"]["positive_class"], "phishing")
        self.assertIn("spam", registry["task_definition"]["negative_classes"])
        self.assertIn("other_fraud", registry["task_definition"]["excluded_from_primary"])
        # No historical spam source may silently provide positive labels.
        spam = self.train.loc[self.train["threat_type"].eq("spam")]
        self.assertTrue((spam["binary_label"] == 0).all())

    def test_research_normalization_masks_volatile_identifiers(self):
        value = normalize_research_text("From a@b.example in 2004: https://evil.example/x?token=12345 $500")
        self.assertNotIn("2004", value)
        self.assertNotIn("a@b.example", value)
        self.assertNotIn("https://evil.example", value)
        self.assertIn("<year>", value)
        self.assertIn("<url>", value)

    def test_research_normalization_bounds_large_email_text(self):
        value = normalize_research_text("x" * (RESEARCH_TEXT_MAX_CHARS + 5000))
        self.assertEqual(len(value), RESEARCH_TEXT_MAX_CHARS)

    def test_character_branch_is_bounded_but_retains_tail_security_tokens(self):
        document = "a" * (CHARACTER_FEATURE_TEXT_MAX_CHARS + 2000) + " credential_request"
        value = _character_feature_view(document)
        self.assertLessEqual(len(value), CHARACTER_FEATURE_TEXT_MAX_CHARS + 64)
        self.assertTrue(value.endswith("credential_request"))

    def test_model_document_bounds_long_mail_and_retains_security_tail(self):
        value = build_research_document("a" * (MODEL_DOCUMENT_MAX_CHARS + 500) + " Verify password")
        self.assertIn("truncated_for_model", value)
        self.assertIn("credential_request", value)

    def test_dashboard_deployment_loads_one_frozen_research_chain(self):
        vectorizer, model, manifest = load_research_detector()
        self.assertEqual(manifest["mode"], "research")
        self.assertEqual(manifest["model"], "logistic_regression")
        self.assertEqual(manifest["preprocessing"], "normalize_research_text")
        self.assertIn("phishing vs non-phishing", manifest["task"])
        self.assertNotEqual(manifest["threshold"], 0.5)

        document = prepare_research_document(
            "From: Security <notice@example.com> Verify your password at https://example.com/login"
        )
        score = float(model.predict_proba(vectorizer.transform([document]))[0][1])
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_dashboard_source_uses_research_preprocessing_and_frozen_threshold(self):
        source = Path("app.py").read_text(encoding="utf-8")
        self.assertIn("cleaned_text = prepare_research_document(raw_text_to_clean)", source)
        self.assertIn("ml_threshold=ml_threshold", source)
        self.assertIn('probability >= ml_threshold', source)
        self.assertNotIn('probability >= 0.5', source)

    def test_matrix_ui_exposes_checked_unknown_risk_and_model_states(self):
        raw = (
            "From: alerts@example.com\n"
            "To: user@example.com\n"
            "Subject: Verify account immediately\n\n"
            "Please verify your password at https://login.example.com/account"
        )
        parsed = parse_raw_email(raw)
        layers = evaluate_12_layers(raw, parsed, ml_probability=0.72, evidence_mode="research")
        statuses = {layer["status"] for layer in layers}
        self.assertTrue(any("CHECKED" in status for status in statuses))
        self.assertTrue(any("UNKNOWN" in status for status in statuses))
        self.assertTrue(any("MODEL" in status for status in statuses))
        ml_layer = next(layer for layer in layers if layer["layer"].startswith("11."))
        self.assertIn("72.00%", ml_layer["detail"])

    def test_manifest_preserves_external_and_new_training_dataset_counts(self):
        self.assertGreater(len(self.external["spaphish_v5"]), 0)
        self.assertGreater(len(self.external["uot_validation_2024"]), 0)
        self.assertGreater(len(self.external["major_corpus_2026"]), 0)
        self.assertNotIn("trec_2007", self.external)
        self.assertNotIn("hf_phishing_email", self.external)
        self.assertNotIn("trec_2005", self.manifest["train"]["source_counts"])
        self.assertNotIn("trec_2006", self.manifest["train"]["source_counts"])
        self.assertNotIn("major_corpus_2026", self.manifest["train"]["source_counts"])
        self.assertLess(self.external["uot_validation_2024"]["template_group"].nunique(), 100)
        self.assertIn("cross_role_exact_duplicates_removed", self.manifest["external"])

    def test_training_manifest_records_licence_governance_for_promotion(self):
        governance = {
            record["id"]: record
            for record in self.manifest["train"]["source_governance"]
        }
        self.assertTrue(governance["enron_spam"]["license_approved"])
        self.assertNotIn("trec_2005", governance)
        self.assertTrue(governance["enron_spam"]["sha256_declared"])
        unknown = source_governance_record({"id": "unreviewed", "license": "unknown"})
        self.assertFalse(unknown["license_approved"])
        self.assertIsNone(unknown["provenance_url"])

    def test_unknown_training_or_evaluation_licences_block_promotion(self):
        gate = _deployment_gate(
            {"selected_model": "logistic_regression", "fallback_used": False},
            [],
            self.manifest,
        )
        checks = {check["name"]: check for check in gate["checks"]}
        self.assertTrue(checks["training_data_governance"]["passed"])
        self.assertTrue(checks["evaluation_data_governance"]["passed"])

        manifest_with_unknown = json.loads(json.dumps(self.manifest))
        unknown = source_governance_record({"id": "unreviewed", "license": "unknown"})
        manifest_with_unknown["train"]["source_governance"].append(unknown)
        manifest_with_unknown["external"]["source_governance"].append(unknown)
        blocked = _deployment_gate(
            {"selected_model": "logistic_regression", "fallback_used": False},
            [],
            manifest_with_unknown,
        )
        blocked_checks = {check["name"]: check for check in blocked["checks"]}
        self.assertFalse(blocked_checks["training_data_governance"]["passed"])
        self.assertIn("unreviewed", blocked_checks["training_data_governance"]["detail"])
        self.assertFalse(blocked_checks["evaluation_data_governance"]["passed"])
        self.assertIn("unreviewed", blocked_checks["evaluation_data_governance"]["detail"])

    def test_numeric_csv_labels_are_stable_registry_keys(self):
        values = _label_strings(pd.Series([0.0, 1.0, "Safe Email"]))
        self.assertEqual(values.tolist(), ["0", "1", "Safe Email"])

    def test_compact_cache_projection_does_not_rebuild_raw_source(self):
        source = {
            "id": "cache_fixture",
            "path": "fixture.csv",
            "role": "train",
            "text_column": "body",
            "label_column": "label",
            "original_labels": {"0": "legitimate", "1": "phishing"},
            "research_binary": {"0": 0, "1": 1},
        }
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pd.DataFrame(
                {"body": ["Routine team update", "Verify your account now"], "label": [0, 1]}
            ).to_csv(root / "fixture.csv", index=False)
            original = _read_source(source, root, include_raw=True)
            self.assertIn("raw_text", original.columns)
            # The benchmark requests this compact view. If it rebuilt the raw
            # CSV, this patched reader would fail. Patching a preprocessing
            # callable is intentionally not used because code drift must
            # invalidate the cache fingerprint.
            with patch(
                "src.research_data.pd.read_csv",
                side_effect=AssertionError("cache projection rebuilt source"),
            ):
                compact = _read_source(source, root, include_raw=False)
            self.assertNotIn("raw_text", compact.columns)
            self.assertEqual(len(compact), len(original))

    def test_model_selection_includes_production_hybrid_without_external_data(self):
        rows = []
        for index in range(60):
            label = index % 2
            rows.append(
                {
                    "research_text": (
                        f"verify account credential token{index}"
                        if label
                        else f"normal project meeting token{index}"
                    ),
                    "binary_label": label,
                    "template_group": f"group-{index}",
                }
            )
        _, selection = _selection_cross_validation(
            pd.DataFrame(rows),
            ["logistic_regression", "hybrid_logistic_nb"],
            folds=3,
        )
        self.assertIn(selection["selected_model"], {"logistic_regression", "hybrid_logistic_nb"})
        self.assertFalse(selection["external_data_consulted"])

    def test_major_remediation_keeps_independent_grouped_holdouts(self):
        historical_train, historical_calibration, historical_holdout = split_historical(self.train)
        major_train, major_calibration, major_holdout = split_major(
            self.external["major_corpus_2026"]
        )
        historical_integrity = _partition_integrity(
            historical_train,
            historical_calibration,
            historical_holdout,
        )
        major_integrity = _partition_integrity(major_train, major_calibration, major_holdout)
        self.assertGreater(len(historical_holdout), 0)
        self.assertGreater(len(major_holdout), 0)
        self.assertTrue(all(value == 0 for value in historical_integrity.values()))
        self.assertTrue(all(value == 0 for value in major_integrity.values()))

    def test_dashboard_fails_closed_instead_of_silently_switching_models(self):
        source = Path("app.py").read_text(encoding="utf-8")
        self.assertIn("AIC211_ALLOW_LEGACY_FALLBACK", source)
        self.assertIn('"mode": "unavailable"', source)
        self.assertIn("return None, None, fallback_info", source)

    def test_portable_dataset_helpers_do_not_hardcode_an_old_drive(self):
        for script in ("download_dataset_huggingface.py", "read_file_parquet.py"):
            source = Path(script).read_text(encoding="utf-8")
            self.assertNotIn(r"E:\Project AIC211", source)
            self.assertIn("Path(__file__).resolve()", source)

    def test_runtime_registry_resolves_dataset_paths_from_immutable_project_root(self):
        runtime_registry = Path("data/runtime/selected_research_datasets.ui.json")
        self.assertEqual(registry_project_root(runtime_registry), Path(DEFAULT_REGISTRY).parents[1])
        app_source = Path("app.py").read_text(encoding="utf-8")
        self.assertIn("write_runtime_registry(selected_registry)", app_source)
        self.assertNotIn('Path("config") / "selected_research_datasets.ui.json"', app_source)

    def test_matrix_has_model_layers_separate_from_technical_layers(self):
        raw = (
            "From: CEO <ceo@company.com>\n"
            "Reply-To: payroll@evil.xyz\n"
            "Authentication-Results: mx; spf=pass dkim=pass dmarc=pass\n"
            "Subject: URGENT wire transfer\n\nPlease make a wire transfer."
        )
        layers = build_12_layer_matrix(raw, ml_probability=0.2, llm_probability=None)
        self.assertEqual(len(layers), 12)
        self.assertEqual([layer["layer"] for layer in layers[-2:]], [11, 12])
        self.assertEqual(layers[2]["state"], "UNKNOWN")
        self.assertIn("trusted gateway", layers[2]["detail"])
        trusted_layers = build_12_layer_matrix(
            raw, ml_probability=0.2, llm_probability=None, trusted_gateway_auth=True
        )
        self.assertEqual(trusted_layers[2]["state"], "NO_INDICATOR")
        self.assertIn("not proof", trusted_layers[2]["detail"])
        self.assertEqual(layers[-1]["state"], "UNKNOWN")
        self.assertGreater(technical_matrix_score(raw), 0.0)

    def test_url_risk_score_is_available_for_ablation(self):
        risky = "Please verify at http://login-account-reset.xyz"
        safe = "Project update without any link."
        self.assertGreater(url_risk_score(risky), 0.0)
        self.assertEqual(url_risk_score(safe), 0.0)

    def test_unknown_matrix_evidence_is_not_reported_as_safe(self):
        layers = build_12_layer_matrix("A plain message without trusted headers.", 0.1, None)
        self.assertEqual(layers[0]["state"], "UNKNOWN")
        self.assertEqual(layers[2]["state"], "UNKNOWN")
        self.assertNotEqual(layers[0]["state"], "NO_INDICATOR")

    def test_metrics_report_pr_auc_and_grouped_ci(self):
        y = np.array([0, 0, 1, 1, 1, 0])
        scores = np.array([0.1, 0.2, 0.7, 0.8, 0.9, 0.3])
        metrics = classification_metrics(y, scores, 0.5)
        self.assertAlmostEqual(metrics["average_precision"], 1.0)
        ci = grouped_bootstrap_ci(y, scores, ["a", "a", "b", "b", "c", "c"], 0.5, iterations=20)
        self.assertIn("average_precision", ci)
        self.assertEqual(len(ci["average_precision"]), 2)
        self.assertEqual(ci["requested_iterations"], 20)
        self.assertEqual(ci["valid_iterations"] + ci["skipped_one_class"], 20)
        self.assertEqual(ci["intervals"]["average_precision"], ci["average_precision"])

    def test_cache_fingerprint_binds_registry_label_semantics(self):
        source = {
            "id": "demo",
            "path": "data/demo.csv",
            "role": "training",
            "text_column": "text",
            "label_column": "label",
            "original_labels": {"0": "legitimate", "1": "phishing"},
            "research_binary": {"legitimate": 0, "phishing": 1},
        }
        before, _ = source_cache_fingerprint(source, "a" * 64)
        reversed_source = dict(source)
        reversed_source["research_binary"] = {"legitimate": 1, "phishing": 0}
        after, _ = source_cache_fingerprint(reversed_source, "a" * 64)
        self.assertNotEqual(before, after)

    def test_registry_quarantines_new_composite_and_vietnamese_sms(self):
        registry = json.loads(Path(DEFAULT_REGISTRY).read_text(encoding="utf-8"))
        excluded_ids = {row["id"] for row in registry["excluded_sources"]}
        self.assertIn("cybersectony_phishingemail_v2_train", excluded_ids)
        self.assertIn("vietnamese_sms_candidate", excluded_ids)
        url_ids = {row["id"] for row in registry["url_submodule_sources"]}
        self.assertIn("phiusiil_url_2024", url_ids)

    def test_llm_cache_requires_structured_label_and_probability(self):
        parsed = _extract_json('```json\n{"label":"phishing","confidence":0.8,"evidence":["verify"]}\n```')
        label, score, evidence = _score(parsed)
        self.assertEqual(label, "phishing")
        self.assertAlmostEqual(score, 0.8)
        self.assertEqual(evidence, ["verify"])
        parsed_safe = _extract_json('{"label":"non_phishing","confidence":0.9}')
        self.assertAlmostEqual(_score(parsed_safe)[1], 0.1)

    def test_llm_prompt_disables_hidden_reasoning_and_bounds_email_length(self):
        messages = _prompt("x" * 10000, max_chars=6000)
        self.assertIn("/no_think", messages[0]["content"])
        self.assertLess(len(messages[1]["content"]), 6100)
        self.assertIn("no-think-json-schema", PROMPT_VERSION)

    def test_llm_label_parser_accepts_safe_wrappers_but_not_buried_labels(self):
        self.assertEqual(extract_llm_label("```json\n{\"label\": \"phishing\"}\n```") , "PHISHING")
        self.assertEqual(extract_llm_label("assistant:\n**LABEL: SAFE**\nRoutine notice."), "SAFE")
        self.assertEqual(extract_llm_label("[NHÃN: LỪA ĐẢO]\nCó link giả."), "PHISHING")
        self.assertEqual(extract_llm_label("Phân tích: [NHÃN: AN TOÀN]"), "UNKNOWN")

    def test_large_eml_is_parsed_before_ml_length_limit(self):
        raw = (
            b"From: sender@example.com\r\n"
            b"To: user@example.com\r\n"
            b"Subject: Complete MIME test\r\n"
            b"MIME-Version: 1.0\r\n"
            b"Content-Type: multipart/mixed; boundary=BOUNDARY\r\n\r\n"
            b"--BOUNDARY\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n"
            + (b"A" * 20000)
            + b"\r\nTAIL_MARKER\r\n"
            b"--BOUNDARY\r\n"
            b"Content-Type: application/octet-stream; name=invoice.exe\r\n"
            b"Content-Disposition: attachment; filename=invoice.exe\r\n"
            b"Content-Transfer-Encoding: base64\r\n\r\nWA==\r\n"
            b"--BOUNDARY--\r\n"
        )
        decoded = decode_eml_upload(raw)
        self.assertGreater(len(decoded), 15000)
        parsed = parse_raw_email(decoded)
        self.assertIn("TAIL_MARKER", parsed["body"])
        self.assertEqual(parsed["attachments"][0]["filename"], "invoice.exe")

    def test_folded_auth_and_responsive_css_do_not_create_false_html_risk(self):
        raw = (
            "From: Service <service@example.com>\n"
            "To: user@example.com\n"
            "Subject: Normal notification\n"
            "MIME-Version: 1.0\n"
            "Content-Type: text/html\n"
            "Authentication-Results: mx.example;\n"
            " spf=pass smtp.mailfrom=example.com;\n"
            " dkim=pass header.d=example.com;\n"
            " dmarc=pass header.from=example.com\n\n"
            "<html><style>.preview{display:none;font-size:0}</style><body>Hello</body></html>"
        )
        layers = build_12_layer_matrix(raw, ml_probability=0.1, trusted_gateway_auth=True)
        self.assertIn("passes: 3", layers[2]["detail"])
        self.assertEqual(layers[7]["state"], "NO_INDICATOR")
        self.assertIn("responsive CSS", layers[7]["detail"])

    def test_attachment_layer_uses_mime_filename_not_url_suffix(self):
        raw = (
            "From: sender@example.com\nTo: user@example.com\nSubject: Files\n"
            "MIME-Version: 1.0\nContent-Type: multipart/mixed; boundary=x\n\n"
            "--x\nContent-Type: text/plain\n\nSee https://example.com/app.js\n"
            "--x\nContent-Type: application/octet-stream; name=payload.exe\n"
            "Content-Disposition: attachment; filename=payload.exe\n\nX\n--x--\n"
        )
        layers = build_12_layer_matrix(raw)
        self.assertEqual(layers[9]["state"], "RISK")
        self.assertIn("exe", layers[9]["detail"])

    def test_zero_width_is_risk_only_when_it_hides_sensitive_text(self):
        normal = build_12_layer_matrix("Hello\u200b\u200b\u200b world")
        evasion = build_12_layer_matrix("Please lo\u200bgin now")
        self.assertEqual(normal[6]["state"], "NO_INDICATOR")
        self.assertEqual(evasion[6]["state"], "RISK")


if __name__ == "__main__":
    unittest.main()
