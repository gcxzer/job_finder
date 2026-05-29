from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.messages import HumanMessage, ToolMessage

from src.codex_oauth.auth import CodexAuthError, CodexAuthStore
from src.codex_oauth.chat_model import (
    CodexOAuthChatModel,
    _collect_stream_response,
    _messages_to_responses_input,
    _raise_for_response_error,
    _response_to_ai_message,
)
from src.codex_oauth.types import CodexCredentials


class CodexOAuthChatModelTests(unittest.TestCase):
    def test_failed_stream_response_raises_auth_error(self) -> None:
        response = _collect_stream_response(
            iter(
                [
                    "event: response.failed",
                    'data: {"response": {"id": "resp_123", "status": "failed", "error": {"message": "model failed"}}}',
                    "",
                ]
            )
        )

        with self.assertRaises(CodexAuthError) as context:
            _raise_for_response_error(response)

        self.assertEqual(context.exception.code, "codex_response_failed")
        self.assertIn("model failed", str(context.exception))

    def test_completed_stream_response_does_not_raise(self) -> None:
        _raise_for_response_error({"id": "resp_123", "status": "completed", "output": []})

    def test_completed_stream_with_empty_terminal_output_preserves_items(self) -> None:
        response = _collect_stream_response(
            iter(
                [
                    "event: response.output_item.done",
                    'data: {"item": {"type": "message", "content": [{"type": "output_text", "text": "hello"}]}}',
                    "",
                    "event: response.completed",
                    'data: {"response": {"id": "resp_123", "status": "completed", "output": []}}',
                    "",
                ]
            )
        )

        self.assertEqual(_response_to_ai_message(response).content, "hello")

    def test_payload_requests_encrypted_reasoning_for_stateless_turns(self) -> None:
        model = CodexOAuthChatModel()

        payload = model._build_payload([HumanMessage(content="Find this job.")])

        self.assertFalse(payload["store"])
        self.assertIn("reasoning.encrypted_content", payload["include"])

    def test_response_output_items_are_sanitized_for_stateless_tool_turns(self) -> None:
        function_call_item = {
            "type": "function_call",
            "id": "fc_123",
            "call_id": "call_123",
            "name": "lookup_job",
            "arguments": '{"url": "https://example.com/jobs/1"}',
            "phase": "tool_calling",
        }
        response = {
            "id": "resp_123",
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "id": "rs_123",
                    "summary": [],
                    "phase": "reasoning",
                },
                function_call_item,
            ],
        }
        ai_message = _response_to_ai_message(response)

        _instructions, input_items = _messages_to_responses_input(
            [
                HumanMessage(content="Find this job."),
                ai_message,
                ToolMessage(content='{"success": true}', tool_call_id="call_123"),
            ]
        )

        self.assertEqual(
            input_items[1],
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "lookup_job",
                "arguments": '{"url": "https://example.com/jobs/1"}',
            },
        )
        self.assertEqual(
            input_items[2],
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": '{"success": true}',
            },
        )

    def test_encrypted_reasoning_items_are_replayed_for_stateless_tool_turns(self) -> None:
        response = {
            "id": "resp_123",
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "id": "rs_123",
                    "summary": [],
                    "encrypted_content": "encrypted",
                    "phase": "reasoning",
                },
                {
                    "type": "function_call",
                    "id": "fc_123",
                    "call_id": "call_123",
                    "name": "lookup_job",
                    "arguments": '{"url": "https://example.com/jobs/1"}',
                },
            ],
        }
        ai_message = _response_to_ai_message(response)

        _instructions, input_items = _messages_to_responses_input(
            [
                HumanMessage(content="Find this job."),
                ai_message,
                ToolMessage(content='{"success": true}', tool_call_id="call_123"),
            ]
        )

        self.assertEqual(
            input_items[1],
            {
                "type": "reasoning",
                "id": "rs_123",
                "summary": [],
                "encrypted_content": "encrypted",
            },
        )
        self.assertEqual(input_items[2]["type"], "function_call")

    def test_runtime_credentials_refreshes_when_only_refresh_token_is_stored(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store = CodexAuthStore(Path(temp_dir) / "codex.json")
            store.write_credentials(CodexCredentials(refresh_token="refresh-token"))

            refreshed = CodexCredentials(access_token="access-token", refresh_token="refresh-token")
            with patch("src.codex_oauth.auth.refresh_codex_credentials", return_value=refreshed) as refresh:
                credentials = store.runtime_credentials()

        refresh.assert_called_once()
        self.assertEqual(credentials.access_token, "access-token")


if __name__ == "__main__":
    unittest.main()
