"""Streamlit runtime smoke tests for fail-closed production startup."""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from cryptography.fernet import Fernet


class TestDashboardRuntime(unittest.TestCase):
    def test_production_session_fails_closed_without_crashing_when_preflight_is_unready(self):
        from streamlit.testing.v1 import AppTest

        token = "dashboard-test-token"
        configuration = json.dumps(
            [
                {
                    "subject": "test-operator",
                    "token_env": "AIC211_DASHBOARD_TEST_TOKEN",
                    "roles": ["viewer", "analyst", "model_operator"],
                }
            ]
        )
        with patch.dict(
            os.environ,
            {
                "AIC211_ENV": "production",
                "AIC211_ARTIFACT_SIGNING_KEY": "dashboard-test-signing-key",
                "AIC211_FEEDBACK_ENCRYPTION_KEY": Fernet.generate_key().decode("ascii"),
                "AIC211_DASHBOARD_TEST_TOKEN": token,
                "AIC211_AUTH_IDENTITIES_JSON": configuration,
            },
            clear=False,
        ):
            app = AppTest.from_file("app.py")
            app.run(timeout=60)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.sidebar.text_input), 1)
            app.sidebar.text_input[0].input(token).run(timeout=60)
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(
                any("Production readiness gate blocked the dashboard." in str(error.value) for error in app.error),
                [str(error.value) for error in app.error],
            )


if __name__ == "__main__":
    unittest.main()
