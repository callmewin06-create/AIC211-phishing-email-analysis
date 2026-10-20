import json
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.deployment_validator import (
    DeploymentValidationError,
    resolve_run_pointer,
    sha256_file,
    validate_deployment_run,
)
from src.explainability import linear_feature_contributions
from src.fusion_model import HybridFusionModel, build_fusion_features
from src.llm_detector import parse_llm_detection
from src.hybrid_benchmark import run_hybrid_benchmark
from src.ml_detector import fit_ml_detector
from src.preprocessing import analyze_url_risk
from src.research_deployment import load_research_detector
from src.research_splits import partition_integrity, split_hybrid_partitions
from src.paper_cohort import build_matched_llm_cohort, cohort_integrity
from src.paper_protocol import load_paper_protocol


def synthetic_corpus(rows=280):
    values = []
    for index in range(rows):
        label = index % 2
        values.append(
            {
                "message_id": f"m-{index}",
                "content_hash": f"content-{index}",
                "template_group": f"template-{index}",
                "source": "synthetic",
                "binary_label": label,
                "research_text": (
                    f"verify account credential case {index}"
                    if label
                    else f"normal project meeting case {index}"
                ),
                "raw_text": (
                    "Verify account credential at https://login.example.com"
                    if label
                    else "Normal project meeting tomorrow"
                ),
            }
        )
    return pd.DataFrame(values)


class TestHybridPaperSystem(unittest.TestCase):
    def test_paper_v2_protocol_is_executable_and_offline(self):
        protocol = load_paper_protocol()
        self.assertEqual(protocol["schema_version"], 2)
        self.assertEqual(protocol["llm"]["mode"], "offline_local_only")
        self.assertEqual(len(protocol["config_sha256"]), 64)

    def test_matched_llm_cohort_is_deterministic_and_group_unique(self):
        corpus = synthetic_corpus(280)
        partitions = split_hybrid_partitions(corpus).as_dict()
        external = synthetic_corpus(80).copy()
        external["content_hash"] = "external-" + external["content_hash"]
        external["template_group"] = "external-" + external["template_group"]
        external["source"] = "external_demo"
        first = build_matched_llm_cohort(
            partitions, {"external_demo": external}, target_size=120, seed=20260715
        )
        second = build_matched_llm_cohort(
            partitions, {"external_demo": external}, target_size=120, seed=20260715
        )
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 120)
        self.assertFalse(first["content_hash"].duplicated().any())
        self.assertFalse(first["template_group"].duplicated().any())
        self.assertTrue(cohort_integrity(first, target_size=120)["passed"])

    def test_four_part_split_has_no_template_or_content_overlap(self):
        partitions = split_hybrid_partitions(synthetic_corpus()).as_dict()
        integrity = partition_integrity(partitions)
        self.assertTrue(integrity["passed"])
        self.assertEqual(set(partitions), {
            "ml_train", "fusion_train", "threshold_selection", "locked_internal_test"
        })
        self.assertTrue(all(value == 0 for pair in integrity["pairwise_overlaps"].values() for value in pair.values()))

    def test_llm_invalid_or_conflicting_output_abstains(self):
        valid = parse_llm_detection(
            {
                "label": "PHISHING",
                "phishing_probability": 0.82,
                "confidence": 0.77,
                "abstain": False,
                "signals": ["urgency"],
                "reason": "Credential request",
            }
        )
        self.assertTrue(valid.available)
        self.assertEqual(valid.label, "PHISHING")
        invalid = parse_llm_detection(
            {"label": "SAFE", "phishing_probability": 0.82, "confidence": 0.77}
        )
        self.assertTrue(invalid.abstain)
        self.assertIsNone(invalid.phishing_probability)

    def test_llm_structured_vietnamese_labels_remain_strict_and_usable(self):
        phishing = parse_llm_detection(
            {
                "label": "Lừa đảo",
                "phishing_probability": 0.82,
                "confidence": 0.77,
                "abstain": False,
                "signals": ["xác minh tài khoản"],
                "reason": "Yêu cầu cung cấp thông tin đăng nhập",
            }
        )
        safe = parse_llm_detection(
            {
                "label": "An toàn",
                "phishing_probability": 0.18,
                "confidence": 0.81,
                "abstain": False,
            }
        )
        prose = parse_llm_detection(
            {"label": "Email này có vẻ an toàn", "phishing_probability": 0.18, "abstain": False}
        )
        self.assertEqual(phishing.label, "PHISHING")
        self.assertEqual(safe.label, "SAFE")
        self.assertTrue(phishing.available)
        self.assertTrue(safe.available)
        self.assertTrue(prose.abstain)

    def test_fusion_uses_neutral_missing_llm_and_explicit_fallback(self):
        frame = synthetic_corpus(80)
        labels = frame["binary_label"].to_numpy()
        ml = np.where(labels == 1, 0.8, 0.2)
        llm = np.where(labels == 1, 0.9, 0.1).astype(object)
        features = build_fusion_features(frame["raw_text"], ml, llm, [0.8] * len(frame))
        model = HybridFusionModel().fit(features, labels)
        missing = build_fusion_features([frame.iloc[0]["raw_text"]], [0.2], [None], [None])
        self.assertEqual(float(missing.iloc[0]["llm_probability"]), 0.5)
        self.assertEqual(float(missing.iloc[0]["llm_available"]), 0.0)
        self.assertEqual(model.modes(missing).tolist(), ["fallback_ml_matrix"])

    def test_explanation_keeps_bigram_contribution(self):
        frame = synthetic_corpus(100)
        vectorizer, model = fit_ml_detector(frame["research_text"], frame["binary_label"], "word")
        result = linear_feature_contributions(vectorizer, model, "verify account credential", top_k=20)
        self.assertTrue(any(" " in feature for feature in result["feature"]))
        self.assertTrue(np.allclose(result["contribution"], result["tfidf"] * result["coefficient"]))

    def test_url_not_probed_is_not_reported_offline_or_safe(self):
        analyze_url_risk.cache_clear()
        result = analyze_url_risk("https://ordinary-example.org/path", check_activity=False)
        self.assertEqual(result["activity_status"]["reachability_state"], "NOT_CHECKED")
        self.assertIn(result["lexical_state"], {"NO_INDICATOR", "RISK"})

    def test_promoted_bundle_requires_gate_and_matching_hash(self):
        with TemporaryDirectory() as directory:
            run_dir = Path(directory) / "research_results" / "run-1"
            run_dir.mkdir(parents=True)
            (run_dir / "model.joblib").write_bytes(b"model")
            summary = {
                "schema_version": 2,
                "run_id": "run-1",
                "deployment_status": "promoted",
                "deployment_gate": {"passed": True},
            }
            (run_dir / "hybrid_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            manifest = {
                "artifacts": {"model.joblib": sha256_file(run_dir / "model.joblib")}
            }
            (run_dir / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            validated = validate_deployment_run(
                run_dir, required_artifacts=("model.joblib",), require_promoted=True
            )
            self.assertIn("model.joblib", validated["checked_hashes"])
            (run_dir / "model.joblib").write_bytes(b"tampered")
            with self.assertRaises(DeploymentValidationError):
                validate_deployment_run(
                    run_dir, required_artifacts=("model.joblib",), require_promoted=True
                )

    def test_pointer_cannot_escape_results_root(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "research_results"
            outside = root / "outside"
            results.mkdir()
            outside.mkdir()
            pointer = results / "latest.json"
            pointer.write_text(
                json.dumps({"run_id": "outside", "path": str(outside)}), encoding="utf-8"
            )
            with self.assertRaises(DeploymentValidationError):
                resolve_run_pointer(pointer, results)

    def test_live_dashboard_requires_promoted_bundle(self):
        with self.assertRaises(DeploymentValidationError):
            load_research_detector(require_promoted=True)
        source = Path("app.py").read_text(encoding="utf-8")
        self.assertIn("load_research_detector(require_promoted=True)", source)
        self.assertNotIn("highlighted_text =", source)

    def test_hybrid_benchmark_writes_complete_paper_bundle(self):
        corpus = synthetic_corpus(280)
        external = synthetic_corpus(60).copy()
        external["message_id"] = "external-" + external["message_id"]
        external["content_hash"] = "external-" + external["content_hash"]
        external["template_group"] = "external-" + external["template_group"]
        external["source"] = "external_demo"
        cache = pd.concat(
            [corpus[["content_hash", "binary_label"]], external[["content_hash", "binary_label"]]],
            ignore_index=True,
        )
        cache["llm_probability"] = np.where(cache["binary_label"].eq(1), 0.9, 0.1)
        cache["llm_confidence"] = 0.8
        cache["model"] = "local-test-model"
        cache["prompt_version"] = "hybrid-test-v1"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cache_path = root / "llm.csv"
            cache.to_csv(cache_path, index=False)
            with (
                patch(
                    "src.hybrid_benchmark.split_primary_corpora",
                    return_value=(corpus, {"external_demo": external}, {"synthetic": True}),
                ),
                patch(
                    "src.hybrid_benchmark.load_registry",
                    return_value={"schema_version": 1, "sources": []},
                ),
            ):
                summary = run_hybrid_benchmark(
                    llm_cache_path=cache_path,
                    output_root=root,
                    ml_variants=("word",),
                    selection_folds=2,
                    bootstrap=0,
                )
            run_dir = root / summary["run_id"]
            self.assertTrue(summary["protocol_gate"]["passed"])
            self.assertEqual(summary["configurations"]["Full_ML_Matrix_LLM"]["status"], "COMPLETED")
            for name in (
                "hybrid_summary.json",
                "partition_manifest.csv",
                "benchmark_metrics.csv",
                "artifact_manifest.json",
                "ml_model.joblib",
                "fallback_fusion.joblib",
                "full_fusion.joblib",
            ):
                self.assertTrue((run_dir / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
