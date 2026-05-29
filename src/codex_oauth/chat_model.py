from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from typing import Any, Iterator, Sequence

import httpx
from langchain_core.language_models.base import LangSmithParams
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, PrivateAttr

from src.codex_oauth.auth import (
    DEFAULT_CODEX_BASE_URL,
    CodexAuthError,
    CodexAuthStore,
    codex_default_headers,
)


class CodexOAuthChatModel(BaseChatModel):
    """LangChain chat model backed by the ChatGPT Codex OAuth backend.

    This adapter is intentionally project-local. It gives DeepAgents a normal
    LangChain chat model while keeping Codex OAuth token handling outside the
    deepagents package.
    """

    model_name: str = Field(default="gpt-5.5")
    auth_store_path: str | None = None
    base_url: str = DEFAULT_CODEX_BASE_URL
    timeout: float | httpx.Timeout = 300.0
    max_retries: int = 2
    request_options: dict[str, Any] = Field(default_factory=dict)

    _auth_store: CodexAuthStore = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._auth_store = CodexAuthStore(self.auth_store_path)

    @property
    def _llm_type(self) -> str:
        return "codex-oauth"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "auth_store_path": self.auth_store_path,
        }

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        params = LangSmithParams(
            ls_provider="codex_oauth",
            ls_model_type="chat",
            ls_model_name=str(kwargs.get("model") or self.model_name),
        )
        if stop:
            params["ls_stop"] = stop
        return params

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | BaseTool | Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice:
            kwargs["tool_choice"] = _normalize_tool_choice(tool_choice)
        return self.bind(tools=formatted_tools, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._build_payload(messages, stop=stop, **kwargs)
        response = self._create_response(payload)
        message = _response_to_ai_message(response)
        return ChatResult(
            generations=[ChatGeneration(message=message)],
            llm_output={
                "model_name": self.model_name,
                "response_id": response.get("id"),
                "status": response.get("status"),
            },
        )

    def _build_payload(
        self,
        messages: list[BaseMessage],
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        instructions, input_items = _messages_to_responses_input(messages)
        payload: dict[str, Any] = {
            "model": str(kwargs.get("model") or self.model_name),
            "input": input_items,
            "instructions": instructions or "",
            "store": False,
            "stream": True,
        }

        tools = _responses_tools(kwargs.get("tools") or [])
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = kwargs.get("tool_choice") or "auto"

        # The Codex Responses surface used by this OAuth backend is stricter
        # than the public API; unsupported request fields are intentionally not
        # forwarded here.

        payload.update({key: value for key, value in self.request_options.items() if value is not None})
        return _strip_none(payload)

    def _create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        retryable_errors = (
            httpx.ConnectError,
            httpx.ReadError,
            httpx.RemoteProtocolError,
            httpx.TimeoutException,
        )
        for attempt in range(self.max_retries + 1):
            try:
                return self._create_response_once(payload)
            except retryable_errors as error:
                if attempt >= self.max_retries:
                    raise CodexAuthError(
                        f"Codex response request failed after {self.max_retries + 1} attempts: {error}",
                        code="codex_response_transport_error",
                    ) from error
                time.sleep(min(2**attempt, 5))

        raise CodexAuthError(
            "Codex response request failed before receiving a response.",
            code="codex_response_transport_error",
        )

    def _create_response_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        credentials = self._auth_store.runtime_credentials()
        if not credentials.access_token:
            raise CodexAuthError(
                "Codex OAuth is not connected. Run `uv run codex-oauth-login` first.",
                code="codex_not_connected",
            )

        base_url = (credentials.base_url or self.base_url or DEFAULT_CODEX_BASE_URL).rstrip("/")
        headers = {
            **codex_default_headers(credentials.access_token),
            "Authorization": f"Bearer {credentials.access_token}",
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout, headers=headers) as client:
            with client.stream("POST", f"{base_url}/responses", json=payload) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as error:
                    detail = response.read().decode("utf-8", errors="replace")
                    raise CodexAuthError(
                        f"Codex response request failed with status {response.status_code}: {detail[:500]}",
                        code="codex_response_error",
                    ) from error
                result = _collect_stream_response(response.iter_lines())
                _raise_for_response_error(result)
                return result


def _messages_to_responses_input(messages: list[BaseMessage]) -> tuple[str | None, list[dict[str, Any]]]:
    instructions: list[str] = []
    input_items: list[dict[str, Any]] = []

    for message in messages:
        if isinstance(message, SystemMessage):
            text = _content_text(message.content)
            if text:
                instructions.append(text)
            continue

        if isinstance(message, ToolMessage):
            tool_call_id = str(message.tool_call_id or "").strip()
            if tool_call_id:
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": _content_text(message.content),
                    }
                )
            continue

        if isinstance(message, AIMessage):
            raw_output_items = _response_output_items_from_message(message)
            if raw_output_items:
                input_items.extend(raw_output_items)
                continue

            text = _content_text(message.content)
            if text and not message.tool_calls:
                input_items.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }
                )
            for index, tool_call in enumerate(message.tool_calls or []):
                name = str(tool_call.get("name") or "").strip()
                if not name:
                    continue
                args = tool_call.get("args") or {}
                arguments = json.dumps(args, ensure_ascii=False) if isinstance(args, dict | list) else str(args or "{}")
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": str(tool_call.get("id") or _deterministic_call_id(name, arguments, index)),
                        "name": name,
                        "arguments": arguments,
                    }
                )
            continue

        if isinstance(message, HumanMessage):
            input_items.append(
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": _content_text(message.content)}],
                }
            )
            continue

        text = _content_text(message.content)
        if text:
            input_items.append({"role": "user", "content": [{"type": "input_text", "text": text}]})

    return "\n\n".join(instructions) or None, input_items


def _responses_tools(tools: Sequence[Any]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") in _BUILT_IN_RESPONSE_TOOLS:
            converted.append(dict(tool))
            continue
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            function = tool["function"]
            name = function.get("name")
            if not name:
                continue
            next_tool = {
                "type": "function",
                "name": str(name),
                "description": _content_text(function.get("description", "")),
                "parameters": function.get("parameters") or {"type": "object", "properties": {}},
            }
            if "strict" in function:
                next_tool["strict"] = bool(function["strict"])
            converted.append(next_tool)
            continue
        if tool.get("type") == "function" and tool.get("name"):
            next_tool = dict(tool)
            next_tool.setdefault("parameters", {"type": "object", "properties": {}})
            converted.append(next_tool)
    return converted


_BUILT_IN_RESPONSE_TOOLS = {
    "web_search",
    "web_search_2025_08_26",
    "web_search_preview",
    "web_search_preview_2025_03_11",
}


def _collect_stream_response(lines: Iterator[str]) -> dict[str, Any]:
    event_type: str | None = None
    data_lines: list[str] = []
    terminal_response: dict[str, Any] | None = None
    output_items: list[dict[str, Any]] = []

    for line in lines:
        line = line.rstrip("\n")
        if not line:
            event = _decode_sse_event(event_type, data_lines)
            event_type = None
            data_lines = []
            if event:
                terminal_response = _apply_stream_event(event, output_items, terminal_response)
            continue
        if line.startswith("event:"):
            event_type = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())

    event = _decode_sse_event(event_type, data_lines)
    if event:
        terminal_response = _apply_stream_event(event, output_items, terminal_response)

    response = terminal_response or {"status": "completed", "output": output_items}
    response_output = response.get("output")
    if output_items and (not isinstance(response_output, list) or not response_output):
        response = {**response, "output": output_items}
    return response


def _decode_sse_event(event_type: str | None, data_lines: list[str]) -> dict[str, Any] | None:
    if not data_lines:
        return None
    data = "\n".join(data_lines).strip()
    if not data or data == "[DONE]":
        return None
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    payload.setdefault("type", event_type)
    return payload


def _apply_stream_event(
    event: dict[str, Any],
    output_items: list[dict[str, Any]],
    terminal_response: dict[str, Any] | None,
) -> dict[str, Any] | None:
    event_type = str(event.get("type") or "")
    if event_type == "response.output_item.done":
        item = event.get("item")
        if isinstance(item, dict):
            output_items.append(item)
    elif event_type in {"response.completed", "response.incomplete", "response.failed", "response.cancelled"}:
        response = event.get("response")
        if isinstance(response, dict):
            terminal_response = response
    return terminal_response


def _raise_for_response_error(response: dict[str, Any]) -> None:
    status = str(response.get("status") or "").lower()
    if status not in {"failed", "cancelled"}:
        return
    detail = _response_error_detail(response)
    raise CodexAuthError(
        f"Codex response {status}: {detail}",
        code=f"codex_response_{status}",
    )


def _response_error_detail(response: dict[str, Any]) -> str:
    error = response.get("error")
    if isinstance(error, dict):
        message = _content_text(error.get("message") or error.get("code") or error.get("type"))
        if message:
            return message[:500]
        return json.dumps(error, ensure_ascii=False, sort_keys=True)[:500]
    if error:
        return _content_text(error)[:500]
    response_id = _content_text(response.get("id"))
    return f"response_id={response_id or 'unknown'}"


def _response_to_ai_message(response: dict[str, Any]) -> AIMessage:
    content_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        if item_type == "message":
            content_parts.extend(_message_output_text(item))
        elif item_type in {"function_call", "custom_tool_call"}:
            tool_call = _tool_call_from_response_item(item, index=len(tool_calls))
            if tool_call:
                tool_calls.append(tool_call)

    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    usage_metadata = None
    if usage:
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": int(usage.get("total_tokens") or input_tokens + output_tokens),
        }

    return AIMessage(
        content="\n".join(part for part in content_parts if part).strip(),
        additional_kwargs={"response_output_items": deepcopy(response.get("output") or [])},
        tool_calls=tool_calls,
        response_metadata={
            "id": response.get("id"),
            "status": response.get("status"),
            "model": response.get("model"),
            "finish_reason": "tool_calls" if tool_calls else _finish_reason(response),
        },
        usage_metadata=usage_metadata,
    )


def _response_output_items_from_message(message: AIMessage) -> list[dict[str, Any]]:
    output_items = message.additional_kwargs.get("response_output_items")
    if not isinstance(output_items, list):
        return []
    return [deepcopy(item) for item in output_items if _is_response_output_item(item)]


def _is_response_output_item(item: Any) -> bool:
    return isinstance(item, dict) and isinstance(item.get("type"), str)


def _message_output_text(item: dict[str, Any]) -> list[str]:
    content = item.get("content")
    if not isinstance(content, list):
        return []
    chunks: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in {"output_text", "text"} and isinstance(part.get("text"), str):
            chunks.append(part["text"])
    return chunks


def _tool_call_from_response_item(item: dict[str, Any], *, index: int) -> dict[str, Any] | None:
    name = str(item.get("name") or "").strip()
    if not name:
        return None
    raw_arguments = item.get("input", "{}") if item.get("type") == "custom_tool_call" else item.get("arguments", "{}")
    arguments_text = json.dumps(raw_arguments, ensure_ascii=False) if isinstance(raw_arguments, dict | list) else str(raw_arguments or "{}")
    try:
        args = json.loads(arguments_text)
        if not isinstance(args, dict):
            args = {"value": args}
    except json.JSONDecodeError:
        args = {"_raw": arguments_text}
    call_id = str(item.get("call_id") or _call_id_from_item_id(item.get("id")) or _deterministic_call_id(name, arguments_text, index))
    return {"name": name, "args": args, "id": call_id}


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("content")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunk for chunk in chunks if chunk)
    return str(content)


def _normalize_tool_choice(tool_choice: str) -> str:
    if tool_choice == "any":
        return "required"
    return tool_choice


def _call_id_from_item_id(value: Any) -> str:
    item_id = str(value or "").strip()
    if item_id.startswith("fc_") and len(item_id) > 3:
        return f"call_{item_id[3:]}"
    return item_id


def _deterministic_call_id(function_name: str, arguments: str, index: int = 0) -> str:
    seed = f"{function_name}:{arguments}:{index}"
    digest = hashlib.sha256(seed.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"call_{digest}"


def _finish_reason(response: dict[str, Any]) -> str:
    status = str(response.get("status") or "").lower()
    if status in {"failed", "cancelled"}:
        return status
    if status == "incomplete":
        details = response.get("incomplete_details")
        reason = details.get("reason") if isinstance(details, dict) else ""
        return "length" if reason == "max_output_tokens" else "incomplete"
    return "stop"


def _strip_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}
