import base64
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat

from src.auth import (
    Actor,
    AuthenticationError,
    ROLE_ANALYST,
    ROLE_MODEL_OPERATOR,
    authenticate,
    authentication_is_configured,
    require_role,
)
from src.artifact_signing import ArtifactSignatureError, sign_manifest, verify_manifest_signature
from src.deployment_validator import (
    DeploymentValidationError,
    resolve_run_pointer,
    sha256_file,
    validate_deployment_run,
)
from src.feedback_store import FeedbackStore, MAX_FEEDBACK_BODY_CHARS
from src.llm_benchmark import _is_allowed_local_endpoint
from src.llm_detector import parse_llm_detection
from src.local_llm import canonical_local_llm_endpoint, is_safe_local_llm_endpoint, request_local_llm
from src.operation_guard import OperationGuard
from src.preprocessing import build_ml_document, is_public_ip, parse_raw_email
from src.preprocessing import escape_csv_formula
from src.production_preflight import _windows_acl_checks, run_preflight
from src.release_integrity import (
    RELEASE_PRIVATE_SIGNING_ENV,
    RELEASE_PUBLIC_VERIFICATION_ENV,
    ReleaseIntegrityError,
    release_public_key_from_private,
    verify_release_manifest,
    write_release_manifest,
)
from src.research_benchmark import run_benchmark
from src.security_matrix import build_12_layer_matrix


class TestProductionHardening(unittest.TestCase):
    @staticmethod
    def _release_key_environment() -> dict[str, str]:
        private_key = Ed25519PrivateKey.generate()
        private_bytes = private_key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
        public_bytes = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        return {
            RELEASE_PRIVATE_SIGNING_ENV: base64.b64encode(private_bytes).decode("ascii"),
            RELEASE_PUBLIC_VERIFICATION_ENV: base64.b64encode(public_bytes).decode("ascii"),
        }

    def test_token_authentication_requires_configured_identity_and_role(self):
        configuration = (
            '[{"subject":"analyst-1","token_env":"AIC211_TEST_ANALYST_TOKEN",'
            '"roles":["viewer","analyst"]}]'
        )
        with patch.dict(
            os.environ,
            {
                "AIC211_AUTH_IDENTITIES_JSON": configuration,
                "AIC211_TEST_ANALYST_TOKEN": "correct horse battery staple",
            },
            clear=False,
        ):
            self.assertTrue(authentication_is_configured())
            actor = authenticate("correct horse battery staple")
            self.assertEqual(actor.subject, "analyst-1")
            require_role(actor, ROLE_ANALYST)
            with self.assertRaises(AuthenticationError):
                require_role(actor, ROLE_MODEL_OPERATOR)
            with self.assertRaises(AuthenticationError):
                authenticate("wrong token")

    def test_feedback_storage_rejects_unbounded_or_unknown_input(self):
        with TemporaryDirectory() as directory:
            store = FeedbackStore(Path(directory) / "feedback.sqlite3")
            with self.assertRaises(ValueError):
                store.submit(body="x" * (MAX_FEEDBACK_BODY_CHARS + 1), label=1, source="user_confirmed")
            with self.assertRaises(ValueError):
                store.submit(body="valid body", label=1, source="untrusted_source")

    def test_production_feedback_is_encrypted_and_requires_a_key(self):
        with TemporaryDirectory() as directory:
            db_path = Path(directory) / "feedback.sqlite3"
            with patch.dict(os.environ, {"AIC211_ENV": "production"}, clear=False):
                with self.assertRaises(RuntimeError):
                    FeedbackStore(db_path)
            key = Fernet.generate_key().decode("ascii")
            with patch.dict(
                os.environ,
                {"AIC211_ENV": "production", "AIC211_FEEDBACK_ENCRYPTION_KEY": key},
                clear=False,
            ):
                store = FeedbackStore(db_path)
                with self.assertRaises(PermissionError):
                    store.submit(body="sensitive email payload", label=1, source="user_confirmed")
                store.submit(
                    body="sensitive email payload",
                    label=1,
                    source="user_confirmed",
                    actor=Actor("test-analyst", frozenset({ROLE_ANALYST})),
                )
                self.assertIn("sensitive email payload", store.review_queue(limit=1)[0]["body"])
            raw_database = db_path.read_bytes()
            self.assertNotIn(b"sensitive email payload", raw_database)

    def test_unknown_mime_charset_keeps_decodable_phishing_content(self):
        payload = base64.b64encode(b"Verify your account at https://evil.example/login now").decode()
        raw = (
            "From: alerts@example.com\nSubject: Verify\nMIME-Version: 1.0\n"
            "Content-Type: text/plain; charset=x-unknown\n"
            "Content-Transfer-Encoding: base64\n\n" + payload
        )
        parsed = parse_raw_email(raw)
        self.assertIn("https://evil.example/login", parsed["body"])
        self.assertIn("https://evil.example/login", parsed["links"])

    def test_multipart_html_cannot_be_hidden_by_benign_plain_part(self):
        raw = (
            "From: service@example.com\nSubject: Agenda\nMIME-Version: 1.0\n"
            "Content-Type: multipart/alternative; boundary=part\n\n"
            "--part\nContent-Type: text/plain; charset=utf-8\n\nWeekly agenda only.\n"
            "--part\nContent-Type: text/html; charset=utf-8\n\n"
            "<html><body><a href='https://evil.example/login'>Verify account</a></body></html>\n"
            "--part--\n"
        )
        parsed = parse_raw_email(raw)
        self.assertIn("Verify account", parsed["body"])
        self.assertIn("https://evil.example/login", parsed["links"])
        self.assertIn("credential_request", build_ml_document(parsed["clean_text"]))

    def test_forged_authentication_header_is_unknown_without_gateway_attestation(self):
        raw = (
            "From: attacker@example.com\nAuthentication-Results: attacker.invalid; "
            "spf=pass dkim=pass dmarc=pass\n\nVerify your password"
        )
        untrusted = build_12_layer_matrix(raw)
        trusted = build_12_layer_matrix(raw, trusted_gateway_auth=True)
        self.assertEqual(untrusted[2]["state"], "UNKNOWN")
        self.assertEqual(trusted[2]["state"], "NO_INDICATOR")

    def test_non_global_ranges_cannot_be_treated_as_public(self):
        for value in ("192.0.2.1", "198.51.100.7", "203.0.113.9", "100.64.0.1", "224.0.0.1", "::1"):
            self.assertFalse(is_public_ip(value), value)
        self.assertTrue(is_public_ip("8.8.8.8"))

    def test_local_llm_transport_rejects_host_network_aliases_bad_ports_and_proxy_routes(self):
        self.assertTrue(is_safe_local_llm_endpoint("http://127.0.0.1:1234", require_base_url=True))
        self.assertTrue(is_safe_local_llm_endpoint("http://[::1]:1234", require_base_url=True))
        self.assertEqual(
            canonical_local_llm_endpoint("http://localhost:1234", require_base_url=True),
            "http://127.0.0.1:1234",
        )
        for endpoint in (
            "http://host.docker.internal:1234",
            "http://127.0.0.1:1235",
            "http://user:secret@127.0.0.1:1234",
            "http://example.invalid:1234",
            "http://127.0.0.1:1234/admin",
        ):
            self.assertFalse(is_safe_local_llm_endpoint(endpoint, require_base_url=True), endpoint)
        self.assertTrue(_is_allowed_local_endpoint("http://127.0.0.1:1234/v1/chat/completions"))
        for endpoint in (
            "http://host.docker.internal:1234/v1/chat/completions",
            "http://127.0.0.1:1234/api/v1/models",
            "http://127.0.0.1:1234/v1/chat/completions?proxy=1",
        ):
            self.assertFalse(_is_allowed_local_endpoint(endpoint), endpoint)

    def test_local_llm_transport_disables_environment_proxy_and_canonicalizes_localhost(self):
        response = MagicMock()
        session = MagicMock()
        session.request.return_value = response
        with patch("src.local_llm.requests.Session") as session_class:
            session_class.return_value.__enter__.return_value = session
            actual = request_local_llm("GET", "http://localhost:1234/v1/models", timeout=3)
        self.assertIs(actual, response)
        self.assertFalse(session.trust_env)
        session.request.assert_called_once_with(
            method="GET",
            url="http://127.0.0.1:1234/v1/models",
            timeout=3,
        )

    def test_string_boolean_in_llm_response_abstains(self):
        detection = parse_llm_detection(
            '{"label":"PHISHING","phishing_probability":0.9,"confidence":0.9,"abstain":"false"}'
        )
        self.assertTrue(detection.abstain)
        self.assertFalse(detection.available)

    def test_production_artifact_manifest_must_be_signed_and_cannot_be_rewritten(self):
        with patch.dict(
            os.environ,
            {"AIC211_ENV": "production", "AIC211_ARTIFACT_SIGNING_KEY": "test-signing-secret"},
            clear=False,
        ):
            manifest = sign_manifest({"schema_version": 1, "artifacts": {"model.joblib": "abc"}})
            verify_manifest_signature(manifest)
            manifest["artifacts"]["model.joblib"] = "attacker-replaced-checksum"
            with self.assertRaises(ArtifactSignatureError):
                verify_manifest_signature(manifest)

    def test_model_and_source_manifests_use_distinct_signing_domains(self):
        release_keys = self._release_key_environment()
        with TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "AIC211_ENV": "production",
                "AIC211_ARTIFACT_SIGNING_KEY": "model-builder-key",
                **release_keys,
            },
            clear=False,
        ):
            self.assertEqual(release_public_key_from_private(), release_keys[RELEASE_PUBLIC_VERIFICATION_ENV])
            artifact_manifest = sign_manifest({"schema_version": 1, "artifacts": {"model.joblib": "abc"}})
            self.assertEqual(artifact_manifest["signature"]["key_id"], "AIC211_ARTIFACT_SIGNING_KEY")
            root = Path(directory)
            (root / "app.py").write_text("print('trusted')\n", encoding="utf-8")
            release_path = write_release_manifest(root)
            release = json.loads(release_path.read_text(encoding="utf-8"))
            self.assertEqual(release["signature"]["algorithm"], "ed25519")
            self.assertNotEqual(release["signature"]["key_id"], artifact_manifest["signature"]["key_id"])
            with patch.dict(os.environ, {RELEASE_PRIVATE_SIGNING_ENV: ""}, clear=False):
                verify_release_manifest(root)
            other_public_key = self._release_key_environment()[RELEASE_PUBLIC_VERIFICATION_ENV]
            with patch.dict(
                os.environ,
                {
                    RELEASE_PRIVATE_SIGNING_ENV: "",
                    RELEASE_PUBLIC_VERIFICATION_ENV: other_public_key,
                },
                clear=False,
            ):
                with self.assertRaises(ReleaseIntegrityError):
                    verify_release_manifest(root)

        with TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"AIC211_ENV": "production", "AIC211_ARTIFACT_SIGNING_KEY": "model-builder-key"},
            clear=True,
        ):
            root = Path(directory)
            (root / "app.py").write_text("print('trusted')\n", encoding="utf-8")
            with self.assertRaises(ReleaseIntegrityError):
                write_release_manifest(root)

    def test_production_deployment_binds_signed_pointer_summary_and_artifacts(self):
        with TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"AIC211_ENV": "production", "AIC211_ARTIFACT_SIGNING_KEY": "test-signing-secret"},
            clear=False,
        ):
            root = Path(directory)
            results = root / "research_results"
            run_dir = results / "run-1"
            run_dir.mkdir(parents=True)
            summary_path = run_dir / "benchmark_summary.json"
            summary_path.write_text(
                '{"run_id":"run-1","deployment_status":"promoted","deployment_gate":{"passed":true}}',
                encoding="utf-8",
            )
            artifact_path = run_dir / "logistic_regression.joblib"
            artifact_path.write_bytes(b"trusted-model")
            (run_dir / "tfidf_vectorizer.joblib").write_bytes(b"trusted-vectorizer")
            from src.deployment_validator import sha256_file

            manifest = sign_manifest(
                {
                    "run_id": "run-1",
                    "artifacts": {
                        "benchmark_summary.json": sha256_file(summary_path),
                        "logistic_regression.joblib": sha256_file(artifact_path),
                        "tfidf_vectorizer.joblib": sha256_file(run_dir / "tfidf_vectorizer.joblib"),
                    },
                }
            )
            (run_dir / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            pointer = results / "latest_benchmark.json"
            pointer.write_text(json.dumps(sign_manifest({"run_id": "run-1", "path": "research_results/run-1"})), encoding="utf-8")
            resolved, _ = resolve_run_pointer(pointer, results)
            validate_deployment_run(
                resolved,
                required_artifacts=("tfidf_vectorizer.joblib", "logistic_regression.joblib"),
            )
            summary_path.write_text("{}", encoding="utf-8")
            with self.assertRaises(DeploymentValidationError):
                validate_deployment_run(
                    resolved,
                    required_artifacts=("tfidf_vectorizer.joblib", "logistic_regression.joblib"),
                )

    def test_production_build_refuses_to_start_without_artifact_signing_key(self):
        with patch.dict(os.environ, {"AIC211_ENV": "production"}, clear=True):
            with self.assertRaises(ArtifactSignatureError):
                run_benchmark(["logistic_regression"], save_models=True)

    def test_rate_budget_is_shared_across_guard_instances(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "operations.sqlite3"
            first = OperationGuard(path, now=lambda: 1000.0)
            second = OperationGuard(path, now=lambda: 1000.0)
            self.assertEqual(first.reserve("analyst", "scan", limit=2, window_seconds=60), (True, 0))
            self.assertEqual(second.reserve("analyst", "scan", limit=2, window_seconds=60), (True, 0))
            allowed, retry = first.reserve("analyst", "scan", limit=2, window_seconds=60)
            self.assertFalse(allowed)
            self.assertGreater(retry, 0)

    def test_llm_log_fields_are_escaped_before_csv_export(self):
        self.assertEqual(escape_csv_formula("=HYPERLINK(\"https://evil\")"), "'=HYPERLINK(\"https://evil\")")
        app_source = Path("app.py").read_text(encoding="utf-8")
        self.assertIn("new_log[column] = new_log[column].map(escape_csv_formula)", app_source)

    def test_preflight_reports_missing_production_prerequisites(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            result = run_preflight(directory)
        self.assertFalse(result["ready"])
        failures = {item["name"] for item in result["checks"] if not item["passed"]}
        self.assertIn("authentication", failures)
        self.assertIn("signed_promoted_model", failures)

    def test_preflight_accepts_a_complete_signed_production_release(self):
        identity_config = json.dumps(
            [
                {
                    "subject": "release-operator",
                    "token_env": "AIC211_TEST_RELEASE_TOKEN",
                    "roles": ["viewer", "analyst", "model_operator"],
                }
            ]
        )
        environment = {
            "AIC211_ENV": "production",
            "AIC211_ARTIFACT_SIGNING_KEY": "preflight-integration-signing-key",
            "AIC211_FEEDBACK_ENCRYPTION_KEY": Fernet.generate_key().decode("ascii"),
            "AIC211_AUTH_IDENTITIES_JSON": identity_config,
            "AIC211_TEST_RELEASE_TOKEN": "preflight-integration-token",
        }
        environment.update(self._release_key_environment())
        with TemporaryDirectory() as directory, patch.dict(os.environ, environment, clear=False):
            root = Path(directory)
            (root / "app.py").write_text("print('trusted release')\n", encoding="utf-8")
            streamlit = root / ".streamlit"
            streamlit.mkdir()
            (streamlit / "config.toml").write_text('[server]\naddress = "127.0.0.1"\n', encoding="utf-8")
            run_dir = root / "research_results" / "run-1"
            run_dir.mkdir(parents=True)
            summary_path = run_dir / "benchmark_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "run_id": "run-1",
                        "deployment_status": "promoted",
                        "deployment_gate": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )
            vectorizer = run_dir / "tfidf_vectorizer.joblib"
            model = run_dir / "logistic_regression.joblib"
            vectorizer.write_bytes(b"trusted-vectorizer")
            model.write_bytes(b"trusted-model")
            artifact_manifest = sign_manifest(
                {
                    "run_id": "run-1",
                    "artifacts": {
                        "benchmark_summary.json": sha256_file(summary_path),
                        "tfidf_vectorizer.joblib": sha256_file(vectorizer),
                        "logistic_regression.joblib": sha256_file(model),
                    },
                }
            )
            (run_dir / "artifact_manifest.json").write_text(
                json.dumps(artifact_manifest), encoding="utf-8"
            )
            pointer = sign_manifest({"run_id": "run-1", "path": "research_results/run-1"})
            (root / "research_results" / "latest_benchmark.json").write_text(
                json.dumps(pointer), encoding="utf-8"
            )
            write_release_manifest(root)
            passing_acl = [
                {"name": "windows_code_acl", "passed": True, "critical": True, "detail": "test"},
                {"name": "windows_data_acl", "passed": True, "critical": True, "detail": "test"},
            ]
            with patch.dict(os.environ, {RELEASE_PRIVATE_SIGNING_ENV: ""}, clear=False), patch(
                "src.production_preflight._windows_acl_checks", return_value=passing_acl
            ):
                result = run_preflight(root)
        self.assertTrue(result["ready"], result)
        self.assertTrue(all(item["passed"] for item in result["checks"] if item["critical"]))

    def test_windows_acl_preflight_requires_read_only_code_and_writable_runtime_data(self):
        with TemporaryDirectory() as directory, patch.dict(
            os.environ, {"AIC211_SERVICE_ACCOUNT": "DOMAIN\\svc-aic211"}, clear=False
        ):
            root = Path(directory)
            for name in ("data", "research_results", "models"):
                (root / name).mkdir()

            def safe_acl(arguments, **_kwargs):
                target = Path(arguments[1])
                permissions = "DOMAIN\\svc-aic211:(RX)" if target == root else "DOMAIN\\svc-aic211:(M)"
                return SimpleNamespace(returncode=0, stdout=permissions, stderr="")

            with patch("src.production_preflight.subprocess.run", side_effect=safe_acl):
                passing = {item["name"]: item for item in _windows_acl_checks(root)}
            self.assertTrue(passing["windows_code_acl"]["passed"])
            self.assertTrue(passing["windows_data_acl"]["passed"])

            def unsafe_root_acl(arguments, **_kwargs):
                target = Path(arguments[1])
                permissions = (
                    "NT AUTHORITY\\Authenticated Users:(M)\nDOMAIN\\svc-aic211:(RX)"
                    if target == root
                    else "DOMAIN\\svc-aic211:(M)"
                )
                return SimpleNamespace(returncode=0, stdout=permissions, stderr="")

            with patch("src.production_preflight.subprocess.run", side_effect=unsafe_root_acl):
                failing = {item["name"]: item for item in _windows_acl_checks(root)}
            self.assertFalse(failing["windows_code_acl"]["passed"])
            self.assertTrue(failing["windows_data_acl"]["passed"])

    def test_acl_script_hardens_project_root_as_immutable(self):
        source = Path("scripts/harden_windows_acl.ps1").read_text(encoding="utf-8")
        self.assertIn("Make code and configuration read-only", source)
        self.assertIn("${ServiceAccount}:(OI)(CI)(RX)", source)
        self.assertIn("${ServiceAccount}:(OI)(CI)(M)", source)
        self.assertIn("Remove-BroadWriteGrants $ResolvedRoot", source)

    def test_signed_release_manifest_rejects_source_replacement(self):
        release_keys = self._release_key_environment()
        with TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"AIC211_ENV": "production", **release_keys},
            clear=False,
        ):
            root = Path(directory)
            (root / "src").mkdir()
            (root / "config").mkdir()
            (root / "app.py").write_text("print('trusted')\n", encoding="utf-8")
            (root / "src" / "worker.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "config" / "runtime.json").write_text("{}\n", encoding="utf-8")
            write_release_manifest(root)
            verified = verify_release_manifest(root)
            self.assertIn("app.py", verified["files"])
            (root / "src" / "worker.py").write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaises(ReleaseIntegrityError):
                verify_release_manifest(root)

    def test_production_launcher_runs_preflight_before_streamlit(self):
        source = Path("scripts/start_production.ps1").read_text(encoding="utf-8")
        self.assertIn("src.production_preflight", source)
        self.assertIn("Production preflight failed", source)
        self.assertIn("-m streamlit run app.py", source)


if __name__ == "__main__":
    unittest.main()
