# Research protocol: Phishing detection, matrix semantics, and ablation

This document is the reproducibility contract for the research extension. The
Streamlit demo and its governed feedback registry remain usable, but paper
numbers must come from `src/hybrid_benchmark.py` under protocol
`hybrid-paper-v2-frozen-matched-cohort`. Operational and deployment pointers
are not paper citations.

## Problem definition and labels

The primary task is binary phishing detection:

* `phishing` = intentional deception to obtain credentials, money, access, or
  to deliver a malicious payload;
* `legitimate` and `spam` = negative class for the primary detector;
* `other_fraud` = a boundary-audit class (for example advance-fee Nigerian
  fraud), excluded from primary training rather than silently relabelled.

Spam is therefore not treated as a subset of phishing. It is retained as a
hard-negative subtype so that advertising language does not become evidence of
phishing. The old derived files (`combined_data_*.csv`, `phishing_email.csv`,
and feature derivatives) are not authoritative research inputs because they
collapse these distinctions.

## Data provenance

The exact source mapping, label policy, checksums, DOI and license are recorded
in [`config/research_datasets.json`](config/research_datasets.json). The
historical local sources are used only after source-aware relabelling:

* Enron, CEAS-08, Ling-Spam and SpamAssassin label-1 examples are `spam`;
* Nazario is `phishing`;
* Nigerian Fraud is `other_fraud` and is not used as a primary positive;
* SpaPhish v5 is an external primary test set;
* University of Twente/Zenodo 2024 is a secondary robustness set. It mixes
  real and artificial messages and is highly templated, so both all-row and
  unique-template metrics are emitted;
* TREC-05, TREC-06, TREC-07 and the local Hugging Face composites are excluded
  from paper-v2 because the local copies do not have sufficient reusable
  publication terms/provenance, or because their mappings collapse spam and
  phishing. They remain visible under `excluded_sources` rather than silently
  disappearing.
* `major_corpus_2026` is the MeAJOR Corpus external test source from Zenodo DOI
  `10.5281/zenodo.18471483`, downloaded as
  `data/external/major_corpus/meajor_cleaned_preprocessed.parquet.gzip`.
  The raw archive contains 108,685 anonymized emails and engineered features;
  the research loader drops one row with a missing label, leaving 108,684
  external test examples. It is kept out of model fitting in the confirmatory
  cross-dataset benchmark. A separate remediation experiment may use a grouped
  MeAJOR training partition, but its MeAJOR holdout remains untouched.

`data/external/` contains the pinned files. Their SHA-256 values are checked at
load time. The loader masks URLs, email addresses, IPs, amounts, long numeric
identifiers and years such as 2004/2005 to prevent source/time memorisation,
and groups repeated normalized templates before the internal calibration split.

## Frozen paper-v2 protocol

Install research-only dependencies once:

```powershell
python -m pip install -r requirements-research.txt
```

Freeze the registry, source bundle, grouped partitions, data manifest and
pre-registered 5,000-message Local-LLM cohort:

```powershell
python -m src.prepare_paper_run
```

The prepared-input gate requires publication-eligible sources, uncapped data,
zero cross-partition content/template overlap, and an exact-size, group-unique
cohort. The frozen `source_bundle.zip` permits code/config recovery when Git
metadata is unavailable. The paper does not use a chronological `latest_*`
pointer to choose its evidence.

Metrics include Accuracy, Balanced Accuracy, Precision, Recall, F1, F2,
Specificity, MCC, PR-AUC (Average Precision), ROC-AUC, Brier score, ECE and
95% grouped-bootstrap intervals. Paper tables use 1,000 iterations and report
requested, valid and one-class-skipped replicates. Grouped bootstrap samples
campaign/template groups, not individual duplicated rows.

## Meaning of the 12-layer matrix

The matrix is evidence reporting, not twelve independent classifiers:

* Layers 1–10: transport/IP, sender/domain, authentication evidence, routing,
  header consistency, social-engineering language, obfuscation, HTML deception,
  URL intelligence and attachment evidence.
* Layer 11: ML detector output.
* Layer 12: Local-LLM semantic review.

Each layer has `RISK`, `NO_INDICATOR`, `UNKNOWN`, or `MODEL` state. `UNKNOWN`
does not mean safe. SPF/DKIM/DMARC passes establish an authentication signal,
not benign intent; a compromised internal CEO account can still send phishing.
URL reachability is displayed separately and is not a maliciousness label.
Dashboard HTTP probing is hard-disabled. DNS-based evidence can still send
domain metadata to the configured resolver, so the system is offline-first,
not absolutely zero-egress. Any future live reachability or detonation must run
in an isolated egress-scanner service and remain separate from the verdict.

The Streamlit app uses this evidence contract when rendering a scan. The old
legacy function path remains only for compatibility tests and old demos.

## Matched ablation protocol

After the pre-registered 48-message feasibility pilot and 12-message
repeatability check pass, run the complete digest-pinned Bonsai cache:

```powershell
./scripts/run_bonsai_paper_v2.ps1 -PreparedRun research_results/paper-prep-v2-<timestamp>
```

The report emits:

* `ML_only`;
* `Technical_Matrix_only`;
* `ML_plus_Matrix`;
* `LLM_only`;
* `ML_plus_LLM`;
* `Matrix_plus_LLM`;
* `Full_ML_Matrix_LLM`.

All seven named configurations fit, threshold and report on the same frozen
cohort. `ML_only_full_data` and `ML_plus_Matrix_full_data` are secondary
full-partition estimates and are never used as unmatched evidence against the
Local LLM. Fusion uses only `fusion_train`; F2 tuning uses only
`threshold_selection`; locked and external labels remain untouched until
evaluation. Missing or abstained Local-LLM scores are never filled from ML or
the matrix.

Run the confirmatory benchmark:

```powershell
python -m src.hybrid_benchmark `
  --prepared-run research_results/paper-prep-v2-<timestamp> `
  --llm-cache research_results/paper-prep-v2-<timestamp>/llm_cache_bonsai_full.csv `
  --bootstrap 1000
```

Every candidate is compared with ML-only using a paired template-group
bootstrap. A predictive-improvement claim requires the candidate-minus-ML
interval to exclude zero in the positive direction.

These rows directly answer the supervisor's component questions: ML baseline,
ML plus URL risk, ML plus the 12-layer evidence matrix (implemented as layers
1-10 because layer 11 is ML and layer 12 is LLM), ML plus LLM, and the full
system.

## MeAJOR domain-shift remediation

The frozen benchmark model is first evaluated on the complete MeAJOR corpus as
an external source. Its low recall is reported as a failed transfer result, not
hidden. The remediation experiment is then run separately:

```powershell
python -m src.major_split_experiment --models all
```

The experiment uses normalized template groups to create:

* historical train, threshold-calibration, and untouched historical holdout;
* MeAJOR train, threshold-calibration, and untouched MeAJOR holdout;
* unchanged SpaPhish, UoT, and other registered robustness sets.

The vectorizer and model use only the two training partitions. F2 threshold
tuning uses only the two calibration partitions. Both holdouts remain
untouched, and the runner fails if any template group overlaps train,
calibration, or holdout. Results are written to a versioned
`major-split-*` directory and `latest_major_split.json`, with deployment status
`research_only_not_promoted`. The dashboard benchmark pointer is not changed.

## Local-LLM identity and privacy

The executable identity is `prism-ml/bonsai-27b`, file
`Bonsai-27B-Q1_0.gguf`, SHA-256
`71d6871836886267f39f1ac69b1b1dd8a783970708d7a189960112ed02978901`.
The prompt treats email text as untrusted data, fixes temperature to zero and
requires JSON (`label`, `confidence`, `attack_type`, `evidence`,
`recommended_action`). The cache rejects mixed paper-run IDs, model digests,
prompt hashes, runtime versions, backend versions or inference
configurations. The endpoint must be loopback; no hidden chain-of-thought is
requested or stored.

This is an offline benchmark statement, not an absolute no-egress claim for
the complete dashboard. Optional runtime DNS resolution can disclose queried
domain metadata to the configured resolver.

## Interpretation rule

The benchmark is deliberately allowed to show source shift. A high
internal PR-AUC does not justify a deployment claim if SpaPhish performance is
poor. Report paired percentage-point deltas versus `ML_only`, confidence
intervals, false negatives and the all-row/unique-template distinction. Do not
describe a heuristic layer as a learned contribution until the ablation and
external test support that claim.
