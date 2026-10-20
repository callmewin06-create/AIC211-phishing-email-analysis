> Original full-workspace documentation, preserved for architectural context.
> This portfolio export excludes datasets, model artifacts, manuscripts and machine-specific scripts.
> For the runnable synthetic-only demo and current public-clone scope, start with the repository-root README.

# AIC211 - Hybrid Phishing Email Analysis Dashboard
### An on-premise, offline-first research and analyst-assistance platform

---

## Project Overview
AIC211 analyzes pasted email or uploaded EML for phishing indicators. It is not an SMTP/mail-gateway interceptor and does not independently verify DKIM. By default it keeps analysis local and never opens URLs embedded in email; production deployment still requires authentication, encrypted feedback storage, model signing, Windows ACL hardening, TLS proxying, and a signed model that has passed its gate.

The system utilizes a Hybrid Architecture that combines the efficiency of classical Machine Learning (Logistic Regression + TF-IDF) for rapid, high-throughput filtering with the cognitive reasoning capabilities of a Local Large Language Model (LLM) queried via LM Studio.

---

## Core Features

### 1. Three-Tier Security Architecture
*   **Tier 1: Machine Learning Layer:** Uses a frozen Logistic Regression + TF-IDF bundle for analyst assistance. Throughput and detection efficacy must be measured in the target deployment; they are not guaranteed by this repository.
*   **Tier 2: Visual Explainability Layer:** Extracts regression coefficients from the Tier 1 model to dynamically highlight key tokens on the Streamlit interface (red background for phishing indicators, green background for safe indicators), allowing Security Operations Center (SOC) analysts to quickly verify system decisions.
*   **Tier 3: Deep Semantic Analysis Layer:** Invokes a local LLM through LM Studio to identify social-engineering intent and return a concise structured analyst report. Private reasoning fields and leaked `<think>` blocks are discarded and are never displayed or written to feedback logs.

### 2. 12-Layer Security Matrix
Emails are inspected across a 12-layer security checklist categorized into three primary security groups:

| Security Group | Security Layer | Description |
| :--- | :--- | :--- |
| **Group 1: Network & Technical Infrastructure** | Layer 1: Network IP | Extracts the originating IP from Received: headers and matches it against an offline IP blacklist. |
| | Layer 3: Authentication Protocols | Reports trusted gateway SPF/DKIM/DMARC evidence when available. A DNS key or a pasted header is not cryptographic proof of benign intent. |
| | Layer 4: MTA Routing | Analyzes transfer agent hops (MTA Hops) to flag anomalous routing patterns. |
| **Group 2: Identity & Domain Reputation** | Layer 2: Domain Reputation | Evaluates the sender's top-level domain against high-risk TLDs (.xyz, .top, .ml, etc.). |
| | Layer 5: Header Metadata | Flags Reply-To mismatch anomalies and Full-Caps subject headers. |
| | Layer 8: HTML & Visual | Scans for hidden CSS elements (display:none, font-size:0px) and injected scripts. |
| | Layer 9: URL & Link Intelligence | Performs offline lookalike, domain and lexical checks. Reachability is reported separately and is never treated as a maliciousness label. |
| | Layer 10: Attachment Metadata Risk | Inspects MIME filenames and dangerous extensions. It does not open, execute, or detonate attachments, so this is not a malware sandbox. |
| **Group 3: Statistical & Semantic Analysis** | Layer 6: Body Text Content | Scans for urgency, high-pressure vocabulary, and social engineering templates. |
| | Layer 7: Adversarial Defense | Decodes obfuscated character patterns (Leetspeak) and truncates payloads to 15k characters to prevent ReDoS. |
| | Layer 11: Statistical & ML | Reports the ML detector estimate as a model output; it is not sender authentication. |
| | Layer 12: Deep Semantic Analysis | Reports the separately evaluated Local LLM estimate. An unavailable/invalid response remains `UNKNOWN`, not safe. |

*Note: For plain-text inputs lacking EML headers, layers that require trusted transport/header evidence are `UNKNOWN`; they are not scored as safe. The evidence contract uses `RISK`, `NO_INDICATOR`, `UNKNOWN` and `MODEL` states so the UI distinguishes missing evidence from a benign finding.*

### 3. Intelligent Link Inspection System
*   **Typosquatting & Lookalike Detection:** Inspects extracted URLs and matches domain segments against official corporate and institutional domains (such as google.com, shopee.vn, vietcombank.com.vn, hust.edu.vn) using Levenshtein distance metrics to flag spoofed domains (e.g., shopee-secure.vn mimicking shopee.vn).
*   **No dashboard HTTP reachability:** The dashboard intentionally never opens email URLs. If live reachability or detonation is required, delegate it to an isolated egress-scanner service; the result must remain separate from the phishing verdict.
*   **Reputation Check:** Flags links utilizing free hosting platforms, dynamic DNS tunnels, or suspicious TLD configurations.

### 4. Active Learning & Feedback Loop v2
*   **Feedback registry có kiểm soát:** Feedback từ màn hình quét luôn bắt đầu ở trạng thái `pending`; analyst phải chấp nhận hoặc từ chối trong **Duyệt feedback**. Hệ thống phân biệt ca ML sai với ca chỉ Local LLM/Final Verdict sai, chống trùng bằng hash nội dung và cách ly nhãn mâu thuẫn trong `data/feedback_registry.sqlite3`. Chỉ feedback đã được chấp nhận, có lỗi ML và không bị leakage mới có thể vào huấn luyện ML.
*   **LLM phân tích, không giả vờ “tự huấn luyện”:** Local LLM không thay đổi trọng số từ feedback. Feedback chỉ sửa Local LLM, sau khi được duyệt, là bài học RAG cho email tương tự ở lần quét sau; nó không bao giờ là dữ liệu huấn luyện classifier ML. Email và lịch sử đều được truyền như dữ liệu không tin cậy, không phải chỉ dẫn.
*   **Candidate + quality gate có bằng chứng:** Mỗi nút huấn luyện tạo một candidate mới từ đầu, ghi lại CSV đã chọn, SHA-256, fingerprint, dữ liệu bị loại, benchmark, feedback thực sự dùng và so sánh với model trước. Candidate chỉ tự triển khai nếu không tăng False Negative, không suy giảm recall/precision/F1 vượt ngưỡng. Candidate trượt gate không thể ghi đè model đang chạy.
*   **Audit model/version:** **Nhật ký ML** hiển thị candidate, điều kiện gate, dataset fingerprint và các version đã triển khai. Thông báo sau test luôn nêu rõ `ĐÃ CẬP NHẬT ML` hoặc `KHÔNG CẬP NHẬT ML`, thay vì chỉ báo thành công chung chung.

### 5. Bilingual (Vietnamese-English) Support
*   **Class Balancing:** Mitigates vocabulary imbalances via oversampling Vietnamese training instances (synthetic_vn) to ensure representative performance across both languages.
*   **Legacy multilingual vectorizer:** The optional active-learning fallback uses word/character n-grams and bilingual routing. The default dashboard research model instead uses the frozen 30,000-feature research TF-IDF contract.

---

## Project Structure
```text
.
├── config/
│   ├── research_datasets.json          # Authoritative source registry
│   └── research_datasets_pre2020.json  # Strict row-level year < 2020 registry
├── data/
│   ├── external/                       # External validation sources
│   ├── filtered_pre2020/               # Filtered data, manifest and QA summary
│   └── *.csv / *.parquet               # Authoritative source datasets only
├── research_results/
│   ├── paper-prep-v2-*/                # Frozen registry, partitions, code and LLM cohort
│   ├── hybrid-paper-v2-*/              # Artifact-bound confirmatory paper results
│   ├── latest_benchmark.json           # Deployment pointer, not a paper citation
│   └── latest_hybrid.json              # Updated only by a passing deployment gate
├── src/                                # Research and system implementation
├── scripts/                            # Reproducible research/paper utilities
├── tests/                              # Automated integrity tests
├── paper/                              # Preserved source, revision, figures and QA
├── app.py                              # Demonstration dashboard
├── RESEARCH_PROTOCOL.md
└── README.md
```

---

## Research workspace policy

Paper results are identified by the `run_id` inside `hybrid_summary.json`, not
by whichever directory is newest. A paper-v2 input bundle freezes the registry,
protocol, source bundle, partition map, Local-LLM cohort and SHA-256 manifest.
Generated research caches are reusable only when their data, registry semantics
and preprocessing fingerprint match.

---

## Installation & Setup

### Step 1: Environment Setup
Ensure your local system runs **Python 3.12**. The checked-in hash locks were
generated and verified on Python 3.12; the pinned NumPy/SciPy/XGBoost releases
do not support the older Python 3.9 baseline.

1. Navigate to the project root directory and install the hash-locked runtime dependencies:
   On Windows, enable UTF-8 output first when the project path contains
   non-ASCII characters:
   ```powershell
   $env:PYTHONUTF8 = "1"
   $env:PIP_PROGRESS_BAR = "off"
   ```

   ```bash
   pip install --require-hashes -r requirements.lock
   ```

   For a reproducible research or CI environment use
   `requirements-research.lock` or `requirements-dev.lock` respectively. The
   short `.txt` files are compatibility entry points only; do not replace the
   locks with unpinned package versions.

   To regenerate the locks, use the pinned `pip==26.1.2` and
   `pip-tools==7.6.0` from `requirements-dev.in`, and pass `--allow-unsafe` so
   build frontends required by hashed installs are also pinned.

2. Verify or create a local IP Blacklist JSON at `data/ip_blacklist.json` for Network IP analysis:
   ```json
   [
     "198.51.100.10",
     "203.0.113.5",
     "185.220.101.5"
   ]
   ```

### Step 2: Machine Learning Model Training
The dashboard loads one frozen research chain (vectorizer, Logistic Regression
artifact, preprocessing contract, and F2-tuned threshold) from
`research_results/latest_benchmark.json`. If that chain is incomplete,
inference fails closed.

To reproduce the default research chain:
```bash
python -m src.research_benchmark --models all --bootstrap 200 --save-models
```

To rebuild the strict temporal subset:
```bash
python scripts/filter_datasets_pre2020.py --replace-output
```

### Step 3: Local LLM Configuration (LM Studio)
To enable the **Tier 3 (Deep Semantic Analysis Layer)** final analyst report:

1. Download and install [LM Studio](https://lmstudio.ai/).
2. Load `prism-ml/bonsai-27b`, file `Bonsai-27B-Q1_0.gguf`.
3. Verify the model-file SHA-256 is
   `71d6871836886267f39f1ac69b1b1dd8a783970708d7a189960112ed02978901`.
4. Click the **Local Server** tab, set the port to `1234`, and click
   **Start Server**.
5. Keep the endpoint on loopback (`127.0.0.1`) and do not configure an
   external LLM API. The frozen paper-v2 evaluation used this exact Bonsai
   file through LM Studio in offline, text-only mode.

### Step 4: Start Streamlit Dashboard
On Windows, start the dashboard through the project launcher:

```powershell
.\run.bat
```

Then:

1. Press **T** for **Local Test**. This mode loads the frozen research model
   for local analyst testing; it does not retrain or promote a model.
2. Wait for the launcher to start the server and open
   `http://127.0.0.1:8501` in the default browser. If the browser does not open,
   enter that address manually.
3. Copy the random string printed immediately below
   `Enter this one-time access token in the sidebar:`.
4. Paste the complete string into the dashboard's **Access token** field.
5. Keep the terminal window open while using the dashboard. Press `Ctrl+C` in
   that terminal to stop the app.

The Local Test access token is process-scoped: it stays local to this machine,
costs nothing, and is replaced whenever the launcher is restarted. It prevents
other local users or processes from opening the scanner without authorization.

Press **P** only for a production deployment. Production is intentionally
fail-closed and will not start unless authentication, encryption, release
signing, Windows ACLs, and a signed promoted model all pass the preflight gate.
See `docs/PRODUCTION_READINESS.md` for those requirements.

Do not use `python -m streamlit run app.py` as the normal Local Test command:
without the launcher-provided environment and temporary identity, the app
defaults to the production security path.

---

## Automated Verification
Run the comprehensive suite. It fails on any failure, skipped test, or expected
failure:
```bash
python run_tests.py
```
Also run the release supply-chain checks:

```bash
python -m pip check
python -m pip_audit -r requirements.lock
python -m pip_audit -r requirements.lock --format cyclonedx-json --output sbom.cdx.json
python -m ruff check src app.py
python -m mypy src/auth.py src/artifact_signing.py src/deployment_validator.py src/local_llm.py src/release_integrity.py src/operation_guard.py src/production_preflight.py
```

---

## Performance Metrics

Metrics displayed by the dashboard are measured on the immutable benchmark for the active dataset. A new candidate must meet the safety gate against the current active model; therefore old numbers from a random split are not used to claim improvement.

## Research benchmark and ablation

The paper-grade protocol is separate from the production retraining command.
Read [`RESEARCH_PROTOCOL.md`](RESEARCH_PROTOCOL.md) for the source-aware
spam/phishing label policy, external dataset checksums, cross-dataset test,
five-model benchmark, grouped MeAJOR remediation, and
ML/technical-matrix/Local-LLM ablation commands.

## Hybrid paper-v2 pipeline

The paper-grade hybrid path combines three independently measured signals:

* a base TF-IDF Logistic Regression detector;
* technical evidence from matrix layers 1–10;
* a structured, failure-aware Local-LLM probability.

A learned logistic fusion model combines the signals. Four template-grouped
partitions keep ML fitting, fusion fitting, threshold selection, and locked
internal reporting separate. A deterministic 5,000-message cohort makes every
LLM comparison feasible and matched on local hardware. Full-data ML baselines
are reported separately. Missing LLM output is never treated as safe.

Freeze the paper inputs, then run the digest-pinned offline Local LLM:

```powershell
python -m src.prepare_paper_run
./scripts/run_bonsai_paper_v2.ps1 -PreparedRun research_results/paper-prep-v2-<timestamp>
```

The paper integrity gate and deployment performance gate are separate: an
honest low metric may be frozen for publication, but it does not promote a
dashboard model.

---

## Documentation & Report
The preserved source and generated revision are under `paper/`. The revision
builder refuses to write performance claims until the frozen paper gate passes
and the registered Local-LLM cohort is complete.
