from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from src.codex_oauth import login


class CodexOAuthLoginCliTests(unittest.TestCase):
    def test_help_exits_without_starting_device_auth(self) -> None:
        with (
            redirect_stdout(io.StringIO()) as stdout,
            patch.object(login, "CodexAuthStore") as auth_store,
            patch.object(login, "CodexDeviceAuthClient") as auth_client,
            self.assertRaises(SystemExit) as exit_context,
        ):
            login.main(["--help"])

        self.assertEqual(exit_context.exception.code, 0)
        self.assertIn("usage: codex-oauth-login", stdout.getvalue())
        auth_store.assert_not_called()
        auth_client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
