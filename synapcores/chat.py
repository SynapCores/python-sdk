"""
AI chat client for SynapCores Python SDK (v1.5.0-ce).

Wraps:
  /v1/ai/sessions, /v1/ai/sessions/:id, /v1/ai/sessions/:id/messages
  /v1/ai/chat, /v1/ai/chat/stream
  /v1/ai/suggestions, /v1/ai/models, /v1/ai/system-prompts
  /v1/ai/tools, /v1/ai/tools/execute, /v1/ai/tools/sql
  /v1/ai/cache/stats, /v1/ai/cache/clear
"""

from typing import Any, Dict, Iterator, List, Optional, TYPE_CHECKING
import json as _json

if TYPE_CHECKING:
    from .client import SynapCores


class _ChatSessions:
    def __init__(self, client: "SynapCores") -> None:
        self.client = client

    def create(
        self,
        title: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {}
        if title:
            body["title"] = title
        if model:
            body["model"] = model
        if system_prompt:
            body["system_prompt"] = system_prompt
        if tools is not None:
            body["tools"] = tools
        if metadata is not None:
            body["metadata"] = metadata
        response = self.client._client.post("/ai/sessions", json=body)
        return self.client._handle_response(response)

    def list(
        self, limit: Optional[int] = None, offset: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        params: Dict[str, str] = {}
        if limit is not None:
            params["limit"] = str(limit)
        if offset is not None:
            params["offset"] = str(offset)
        response = self.client._client.get("/ai/sessions", params=params)
        data = self.client._handle_response(response)
        return data.get("sessions") or data or []

    def get(self, id: str) -> Dict[str, Any]:
        response = self.client._client.get(f"/ai/sessions/{id}")
        return self.client._handle_response(response)

    def delete(self, id: str) -> None:
        response = self.client._client.delete(f"/ai/sessions/{id}")
        self.client._handle_response(response)

    def messages(
        self,
        id: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, str] = {}
        if limit is not None:
            params["limit"] = str(limit)
        if offset is not None:
            params["offset"] = str(offset)
        response = self.client._client.get(
            f"/ai/sessions/{id}/messages", params=params
        )
        data = self.client._handle_response(response)
        return data.get("messages") or data or []


class _ChatTools:
    def __init__(self, client: "SynapCores") -> None:
        self.client = client

    def list(self) -> List[Dict[str, Any]]:
        response = self.client._client.get("/ai/tools")
        data = self.client._handle_response(response)
        return data.get("tools") or data or []

    def execute(self, name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        response = self.client._client.post(
            "/ai/tools/execute",
            json={"tool": name, "arguments": args or {}},
        )
        return self.client._handle_response(response)

    def sql(self, sql: str, parameters: Optional[List[Any]] = None) -> Dict[str, Any]:
        response = self.client._client.post(
            "/ai/tools/sql",
            json={"sql": sql, "parameters": parameters or []},
        )
        return self.client._handle_response(response)


class _ChatCache:
    def __init__(self, client: "SynapCores") -> None:
        self.client = client

    def stats(self) -> Dict[str, Any]:
        response = self.client._client.get("/ai/cache/stats")
        return self.client._handle_response(response)

    def clear(self) -> None:
        response = self.client._client.post("/ai/cache/clear", json={})
        self.client._handle_response(response)


class ChatClient:
    """High-level wrapper around the v1.5.0-ce AI chat API."""

    def __init__(self, client: "SynapCores") -> None:
        self.client = client
        self.sessions = _ChatSessions(client)
        self.tools = _ChatTools(client)
        self.cache = _ChatCache(client)

    def send(
        self,
        session_id: str,
        content: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[str]] = None,
        attachments: Optional[List[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"session_id": session_id, "content": content}
        if model:
            body["model"] = model
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if tools is not None:
            body["tools"] = tools
        if attachments is not None:
            body["attachments"] = attachments
        if metadata is not None:
            body["metadata"] = metadata
        response = self.client._client.post("/ai/chat", json=body)
        return self.client._handle_response(response)

    def stream(
        self,
        session_id: str,
        content: str,
        **opts: Any,
    ) -> Iterator[Dict[str, Any]]:
        """Stream a chat completion as an iterator over chunks.

        Each chunk is the parsed SSE payload (usually a dict with a
        ``delta`` or ``content`` field). The ``done`` flag is set on the
        terminal frame.
        """
        body = {"session_id": session_id, "content": content, **opts}
        with self.client._client.stream(
            "POST", "/ai/chat/stream", json=body
        ) as response:
            buffer = ""
            for chunk in response.iter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    frame, buffer = buffer.split("\n\n", 1)
                    frame = frame.strip()
                    if not frame:
                        continue
                    line = frame[5:].strip() if frame.startswith("data:") else frame
                    if line == "[DONE]":
                        yield {"done": True}
                        return
                    try:
                        yield _json.loads(line)
                    except Exception:
                        yield {"delta": line}

    def suggestions(
        self,
        count: Optional[int] = None,
        context: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        body: Dict[str, Any] = {}
        if count is not None:
            body["count"] = count
        if context:
            body["context"] = context
        if category:
            body["category"] = category
        response = self.client._client.post("/ai/suggestions", json=body)
        data = self.client._handle_response(response)
        return data.get("suggestions") or data or []

    def models(self) -> List[Dict[str, Any]]:
        response = self.client._client.get("/ai/models")
        data = self.client._handle_response(response)
        return data.get("models") or data or []

    def system_prompts(self) -> List[Dict[str, Any]]:
        response = self.client._client.get("/ai/system-prompts")
        data = self.client._handle_response(response)
        return data.get("prompts") or data.get("system_prompts") or data or []
