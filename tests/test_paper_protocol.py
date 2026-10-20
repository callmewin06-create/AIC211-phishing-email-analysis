import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.paper_cohort import build_matched_llm_cohort, cohort_integrity
from src.paper_protocol import load_paper_protocol
from src.prepare_paper_run import _governance_gate
from src.llm_benchmark import _balanced_pilot
from src.hybrid_benchmark import _cohort_frames
from src.research_data import DEFAULT_REGISTRY, source_cache_fingerprint
from src.research_metrics import grouped_bootstrap_ci, grouped_paired_bootstrap_delta_ci
from src.research_splits import split_hybrid_partitions


def _synthetic_corpus(rows: int = 280) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "message_id": [f"m-{index}" for index in range(rows)],
            "content_hash": [f"content-{index}" for index in range(rows)],
            "template_group": [f"template-{index}" for index in range(rows)],
            "source": ["synthetic"] * rows,
            "binary_label": [index % 2 for index in range(rows)],
            "language": ["en"] * rows,
            "research_text": [
                f"{'credential lure' if index % 2 else 'project meeting'} {index}"
                for index in range(rows)
            ],
            "raw_text": [
                "Verify your credential" if index % 2 else "Project meeting tomorrow"
                for index in range(rows)
            ],
        }
    )


class TestPaperProtocolV2(unittest.TestCase):
    def test_protocol_is_executable_and_offline(self):
        protocol = load_paper_protocol()
        self.assertEqual(protocol["schema_version"], 2)
        self.assertEqual(protocol["llm"]["mode"], "offline_local_only")
        self.assertEqual(len(protocol["config_sha256"]), 64)

    def test_matched_cohort_is_deterministic_and_group_unique(self):
        corpus = _synthetic_corpus()
        partitions = split_hybrid_partitions(corpus).as_dict()
        external = _synthetic_corpus(80)
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
        self.assertFalse(first["content_hash"].duplicated().any())
        self.assertFalse(first["template_group"].duplicated().any())
        self.assertTrue(cohort_integrity(first, target_size=120)["passed"])

    def test_cache_fingerprint_changes_with_label_mapping(self):
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
        source["research_binary"] = {"legitimate": 1, "phishing": 0}
        after, _ = source_cache_fingerprint(source, "a" * 64)
        self.assertNotEqual(before, after)

    def test_bootstrap_reports_effective_replicates(self):
        y = np.array([0, 0, 1, 1, 1, 0])
        scores = np.array([0.1, 0.2, 0.7, 0.8, 0.9, 0.3])
        ci = grouped_bootstrap_ci(y, scores, ["a", "a", "b", "b", "c", "c"], 0.5, 20)
        self.assertEqual(ci["requested_iterations"], 20)
        self.assertEqual(ci["valid_iterations"] + ci["skipped_one_class"], 20)
        self.assertEqual(ci["intervals"]["average_precision"], ci["average_precision"])

    def test_paired_bootstrap_reports_candidate_minus_reference(self):
        y = np.array([0, 0, 1, 1, 1, 0])
        reference = np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6])
        candidate = np.array([0.1, 0.2, 0.75, 0.8, 0.9, 0.1])
        result = grouped_paired_bootstrap_delta_ci(
            y, reference, candidate, ["a", "a", "b", "b", "c", "c"], 0.5, 0.5, 40
        )
        self.assertEqual(result["requested_iterations"], 40)
        self.assertEqual(result["valid_iterations"] + result["skipped_one_class"], 40)
        self.assertGreater(result["intervals"]["f1"][0], 0.0)

    def test_new_data_roles_are_explicit(self):
        registry = json.loads(Path(DEFAULT_REGISTRY).read_text(encoding="utf-8"))
        excluded = {row["id"] for row in registry["excluded_sources"]}
        self.assertIn("cybersectony_phishingemail_v2_train", excluded)
        self.assertIn("vietnamese_sms_candidate", excluded)
        self.assertIn(
            "phiusiil_url_2024", {row["id"] for row in registry["url_submodule_sources"]}
        )

    def test_paper_governance_gate_is_separate_from_performance(self):
        clean = {
            scope: {
                "source_governance": [
                    {
                        "id": f"{scope}-demo",
                        "license_approved": True,
                        "provenance_url": "https://example.invalid/source",
                        "sha256_declared": True,
                    }
                ]
            }
            for scope in ("train", "external")
        }
        self.assertTrue(_governance_gate(clean)["passed"])
        clean["external"]["source_governance"][0]["license_approved"] = False
        result = _governance_gate(clean)
        self.assertFalse(result["passed"])
        self.assertNotIn("accuracy", json.dumps(result))

    def test_llm_pilot_spans_partitions_sources_and_labels(self):
        rows = []
        for partition in ("fusion_train", "threshold_selection", "locked_internal_test", "external"):
            for source in ("a", "b"):
                for label in (0, 1):
                    for item in range(5):
                        rows.append(
                            {
                                "evaluation_partition": partition,
                                "source": source,
                                "binary_label": label,
                                "content_hash": f"{partition}-{source}-{label}-{item}",
                            }
                        )
        frame = pd.DataFrame(rows)
        pilot = _balanced_pilot(frame, 32)
        self.assertEqual(len(pilot), 32)
        self.assertEqual(set(pilot["binary_label"]), {0, 1})
        self.assertEqual(pilot["evaluation_partition"].nunique(), 4)
        self.assertEqual(pilot["source"].nunique(), 2)

    def test_hybrid_matched_cohort_removes_external_content_duplicates(self):
        base = _synthetic_corpus(12)
        duplicate = pd.concat([base, base.iloc[[0]]], ignore_index=True)
        frames = {"locked_internal_test": duplicate}
        cohort = pd.DataFrame(
            {
                "evaluation_partition": ["locked_internal_test"] * 4,
                "content_hash": base["content_hash"].iloc[:4],
                "cohort_order": [1, 2, 3, 4],
            }
        )
        selected, gate = _cohort_frames(frames, cohort)
        self.assertTrue(gate["passed"])
        self.assertEqual(len(selected["locked_internal_test"]), 4)
        self.assertFalse(selected["locked_internal_test"]["content_hash"].duplicated().any())


if __name__ == "__main__":
    unittest.main()
