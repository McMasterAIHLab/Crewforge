"""
Cloud LLM Provider — Abstraction Layer for Cloud-Based LLM APIs

This module provides a cloud-based alternative to ollama.chat() that routes
LLM calls to cloud providers (currently Vertex AI / Gemini). It returns
responses in the same shape as ollama.chat() so that SingleLLMSystem and
MultiLLMSystem can use it as a drop-in replacement with zero changes to
their parsing logic.

Usage:
    from cloud_llm_provider import create_cloud_chat_fn

    cloud_chat = create_cloud_chat_fn(provider="vertex")
    # cloud_chat(**kwargs) has the same signature/return shape as ollama.chat()

Supported Providers:
    - "vertex": Google Vertex AI (Gemini models)
"""

from __future__ import annotations

import datetime
import fcntl
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

@dataclass
class _ToolFunction:
    """Mimics ollama ToolCall.function"""
    name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _ToolCall:
    """Mimics ollama ToolCall"""
    function: _ToolFunction = field(default_factory=_ToolFunction)


@dataclass
class _Message:
    """Mimics ollama response.message with .content, .thinking, .tool_calls"""
    content: str = ""
    thinking: str = ""
    tool_calls: Optional[List[_ToolCall]] = None


@dataclass
class CloudChatResponse:
    """Drop-in replacement for ollama.ChatResponse.

    Supports both attribute access (response.message.content) and dict access
    (response["message"]["content"]) so that the existing extraction logic
    in single_system.py and multi_system.py works without changes.
    """
    message: _Message = field(default_factory=_Message)
    prompt_eval_count: int = 0
    eval_count: int = 0
    thoughts_token_count: int = 0  # reasoning tokens (already folded into eval_count)

    # --- dict-style access for legacy code paths ---
    def __getitem__(self, key: str):
        if key == "message":
            return {
                "content": self.message.content,
                "thinking": self.message.thinking,
                "tool_calls": self.message.tool_calls,
            }
        if key == "prompt_eval_count":
            return self.prompt_eval_count
        if key == "eval_count":
            return self.eval_count
        if key == "thoughts_token_count":
            return self.thoughts_token_count
        raise KeyError(key)

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str):
        return key in ("message", "prompt_eval_count", "eval_count", "thoughts_token_count")


# ---------------------------------------------------------------------------
# Vertex AI (Gemini) provider
# ---------------------------------------------------------------------------

# Lazy-loaded module-level client to avoid import cost when cloud mode is off
_vertex_client = None
_vertex_types = None
_vertex_exceptions = None


def _ensure_vertex_sdk():
    """Lazy-import and cache the google-genai SDK components."""
    global _vertex_client, _vertex_types, _vertex_exceptions
    if _vertex_client is not None:
        return

    from google import genai
    from google.genai import types
    from google.api_core import exceptions

    _vertex_client = genai.Client(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION"),
    )
    _vertex_types = types
    _vertex_exceptions = exceptions


def _is_resource_exhausted(exc: Exception) -> bool:
    """Check if an exception is a 429 Resource Exhausted error.

    The google-genai SDK may raise this as either:
      - google.api_core.exceptions.ResourceExhausted
      - google.genai.errors.ClientError with '429' or 'RESOURCE_EXHAUSTED' in message
      - Any exception whose string contains '429' and 'RESOURCE_EXHAUSTED'
    """
    # Direct type check for the canonical exception
    if isinstance(exc, _vertex_exceptions.ResourceExhausted):
        return True
    # The unified google-genai SDK wraps quota errors in ClientError
    exc_str = str(exc)
    if "429" in exc_str and "RESOURCE_EXHAUSTED" in exc_str.upper():
        return True
    return False


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception should trigger a retry.

    Retryable conditions:
      - 429 Resource Exhausted (rate limiting)
      - RemoteProtocolError / server disconnect (Gemini API ~60s non-streaming timeout)
    """
    if _is_resource_exhausted(exc):
        return True
    # httpx / httpcore RemoteProtocolError raised when the server closes the
    # connection without a response — this is the Gemini 60s non-streaming timeout.
    exc_type_name = type(exc).__name__
    if "RemoteProtocolError" in exc_type_name:
        return True
    exc_str = str(exc)
    if "RemoteProtocolError" in exc_str:
        return True
    if "Server disconnected without sending a response" in exc_str:
        return True
    if "server disconnected" in exc_str.lower():
        return True
    return False


def _vertex_chat_stream_with_retry(
    client,
    call_kwargs: Dict[str, Any],
    max_retries: int = 100,
):
    """Stream a Gemini generate_content_stream call and collect the full response.

    Uses streaming to bypass the Gemini API's ~60-second HTTP timeout that
    applies to non-streaming full-response requests.  The stream keeps the
    HTTP connection alive via chunked transfer encoding, so even heavy
    reasoning models (e.g. gemini-*-pro-*) that take >60s to respond
    complete successfully.

    The entire stream iteration is wrapped in the retry loop so that any
    retryable error — whether raised when initiating the stream or mid-
    iteration — causes the whole request to restart cleanly.

    Returns:
        Tuple of (content_text, thinking_text, tool_calls_list,
                  prompt_tokens, completion_tokens, thoughts_tokens, total_tokens)

        completion_tokens here is the model's visible response tokens
        (candidates_token_count); thoughts_tokens is the reasoning/thinking
        token count reported separately by Gemini. total_tokens is the API's
        authoritative total_token_count when available.
    """
    MAX_DURATION_SECONDS = 1200
    RETRY_DELAY_SECONDS = 3

    start_time = datetime.datetime.now()
    attempt = 0

    while True:
        try:
            stream = client.models.generate_content_stream(**call_kwargs)

            content_text = ""
            thinking_text = ""
            tool_calls_list: Optional[List[_ToolCall]] = None
            prompt_tokens = 0
            completion_tokens = 0
            thoughts_tokens = 0
            total_tokens = 0

            for chunk in stream:
                # Token usage — accumulate the latest non-zero values. Gemini
                # reports cumulative counts on each chunk's usage_metadata, so
                # the last non-zero value is the authoritative final count.
                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    um = chunk.usage_metadata
                    pt = getattr(um, "prompt_token_count", 0) or 0
                    ct = getattr(um, "candidates_token_count", 0) or 0
                    tt = getattr(um, "thoughts_token_count", 0) or 0
                    tot = getattr(um, "total_token_count", 0) or 0
                    if pt:
                        prompt_tokens = pt
                    if ct:
                        completion_tokens = ct
                    if tt:
                        thoughts_tokens = tt
                    if tot:
                        total_tokens = tot

                if not chunk.candidates:
                    continue
                candidate = chunk.candidates[0]
                if not (
                    hasattr(candidate, "content")
                    and candidate.content
                    and candidate.content.parts
                ):
                    continue

                for part in candidate.content.parts:
                    if getattr(part, "thought", False):
                        # Thinking / reasoning token
                        thinking_text += part.text or ""
                    elif hasattr(part, "function_call") and part.function_call:
                        # Tool / function call
                        fc = part.function_call
                        if tool_calls_list is None:
                            tool_calls_list = []
                        tool_calls_list.append(
                            _ToolCall(
                                function=_ToolFunction(
                                    name=getattr(fc, "name", ""),
                                    arguments=(
                                        dict(getattr(fc, "args", {}))
                                        if getattr(fc, "args", None)
                                        else {}
                                    ),
                                )
                            )
                        )
                    elif hasattr(part, "text") and part.text:
                        content_text += part.text

            # Stream consumed successfully — return collected results
            return (content_text, thinking_text, tool_calls_list,
                    prompt_tokens, completion_tokens, thoughts_tokens, total_tokens)

        except Exception as e:
            if not _is_retryable(e):
                raise  # Non-retryable — propagate immediately
            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            attempt += 1
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Vertex streaming API failed after {attempt} retries. Original Error: {e}"
                )
            if elapsed >= MAX_DURATION_SECONDS:
                raise RuntimeError(
                    f"Vertex streaming API timed out after {elapsed:.0f}s. Original Error: {e}"
                )
            reason = "429 Resource Exhausted" if _is_resource_exhausted(e) else "Server Disconnect"
            print(
                f"    [CLOUD] {reason} — retry {attempt}/{max_retries} in {RETRY_DELAY_SECONDS}s "
                f"(elapsed {elapsed:.0f}s) ..."
            )
            time.sleep(RETRY_DELAY_SECONDS)


def _convert_tools_to_vertex(ollama_tools: Optional[List[Dict]]) -> Optional[List]:
    """Convert Ollama-style tool schemas to Vertex AI function declarations.

    Ollama tool format:
        {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}

    Vertex expects google.genai.types.Tool with FunctionDeclaration objects.
    """
    if not ollama_tools:
        return None

    _ensure_vertex_sdk()
    types = _vertex_types

    declarations = []
    for tool in ollama_tools:
        func = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = func.get("name", "")
        description = func.get("description", "")
        params = func.get("parameters", {})
        if not name:
            continue

        declarations.append(
            types.FunctionDeclaration(
                name=name,
                description=description,
                parameters=params if params else None,
            )
        )

    if not declarations:
        return None

    return [types.Tool(function_declarations=declarations)]


def _vertex_chat(**kwargs) -> CloudChatResponse:
    """Vertex AI implementation of the chat function.

    Accepts the same kwargs as ollama.chat() (model, messages, options,
    format, tools, think, keep_alive) and returns a CloudChatResponse
    with the same shape as an ollama ChatResponse.
    """
    _ensure_vertex_sdk()
    client = _vertex_client
    types = _vertex_types

    model_id = kwargs.get("model", "")
    messages = kwargs.get("messages", [])
    options = kwargs.get("options", {})
    fmt = kwargs.get("format", None)
    tools_ollama = kwargs.get("tools", None)
    think_enabled = kwargs.get("think", False)
    # Optional explicit thinking level ("low"/"high") for Gemini 3 models.
    # When provided, a ThinkingConfig is attached so reasoning depth is
    # controlled and thinking text is returned. When None, behaviour is
    # unchanged (model default) — preserving backward compatibility.
    thinking_level = kwargs.get("thinking_level", None)

    # --- Build Vertex config from ollama options ---
    temperature = options.get("temperature", 1.0)
    max_output_tokens = options.get("num_predict", 3000)

    config_kwargs: Dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }

    # Thinking configuration — set reasoning level and request thought text.
    if thinking_level is not None:
        try:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=thinking_level,
                include_thoughts=True,
            )
        except Exception:
            # Model/SDK may not accept thinking_level; fall back to requesting
            # thoughts only, and finally to no thinking config at all.
            try:
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    include_thoughts=True,
                )
            except Exception:
                pass

    # JSON mode
    if fmt == "json":
        config_kwargs["response_mime_type"] = "application/json"

    # System instruction (extract from messages)
    system_instruction = None
    user_contents = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system_instruction = content
        elif role == "user":
            user_contents.append(content)

    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction

    # Tools conversion
    vertex_tools = _convert_tools_to_vertex(tools_ollama)

    config = types.GenerateContentConfig(**config_kwargs)

    # Combine all user messages into a single prompt (Gemini uses flat content)
    prompt = "\n\n".join(user_contents) if user_contents else ""

    # --- Make the API call (streaming to bypass Gemini's ~60s non-streaming timeout) ---
    call_kwargs: Dict[str, Any] = {
        "model": model_id,
        "contents": prompt,
        "config": config,
    }
    if vertex_tools:
        call_kwargs["config"] = types.GenerateContentConfig(
            **config_kwargs,
            tools=vertex_tools,
        )

    # Use streaming so the HTTP connection stays alive for the full generation,
    # bypassing the Gemini API's ~60-second timeout on non-streaming requests.
    # The helper collects all chunks and returns the same data we'd get from a
    # single non-streaming response, so the rest of this function is unchanged.
    (content_text, thinking_text, tool_calls_list,
     prompt_tokens, candidate_tokens, thoughts_tokens, total_tokens) = (
        _vertex_chat_stream_with_retry(client, call_kwargs)
    )

    # Report completion tokens INCLUSIVE of thinking tokens so that
    # prompt_eval_count + eval_count equals the true billed total. Gemini
    # reports candidates_token_count (visible output) and thoughts_token_count
    # separately; total_token_count is authoritative when present.
    if total_tokens and total_tokens >= prompt_tokens:
        eval_count = total_tokens - prompt_tokens
    else:
        eval_count = candidate_tokens + thoughts_tokens

    return CloudChatResponse(
        message=_Message(
            content=content_text,
            thinking=thinking_text,
            tool_calls=tool_calls_list,
        ),
        prompt_eval_count=prompt_tokens,
        eval_count=eval_count,
        thoughts_token_count=thoughts_tokens,
    )


# ---------------------------------------------------------------------------
# Provider registry & factory
# ---------------------------------------------------------------------------

_PROVIDERS: Dict[str, Callable] = {
    "vertex": _vertex_chat,
}


def create_cloud_chat_fn(provider: str = "vertex", thinking_level: Optional[str] = None) -> Callable:
    """Create and return a cloud chat function for the specified provider.

    The returned callable has the same signature as ollama.chat():
        response = chat_fn(model=..., messages=..., options=..., ...)

    Args:
        provider: Cloud provider name. Currently supported: "vertex".
        thinking_level: Optional reasoning level ("low"/"high") injected into
            every call as a ThinkingConfig. When None (default), no thinking
            config is added and behaviour is unchanged.

    Returns:
        A callable that can be used as a drop-in replacement for ollama.chat().

    Raises:
        ValueError: If the provider is not supported.
    """
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unsupported cloud provider '{provider}'. "
            f"Supported: {list(_PROVIDERS.keys())}"
        )
    base_fn = _PROVIDERS[provider]
    if thinking_level is None:
        return base_fn

    def _chat_with_thinking(**kwargs):
        kwargs.setdefault("thinking_level", thinking_level)
        return base_fn(**kwargs)

    return _chat_with_thinking


# ---------------------------------------------------------------------------
# Cloud model context limits
# ---------------------------------------------------------------------------

CLOUD_MODEL_CONTEXT_LIMITS: Dict[str, tuple] = {
    # Gemini models — (max_context_tokens, max_output_tokens)
    "gemini-3.1-pro-preview":       (1048576, 65536),
    "gemini-3.5-flash":             (1048576, 65535),
    "gemini-3.1-flash-lite":        (1048576, 65536),
}

CLOUD_DEFAULT_CONTEXT_LIMIT = (1048576, 65535)


def get_cloud_model_limits(model_name: str) -> tuple:
    """Get (max_context_tokens, max_output_tokens) for a cloud model.

    Exact match is tried first, then prefix matching for versioned model IDs.
    """
    if model_name in CLOUD_MODEL_CONTEXT_LIMITS:
        return CLOUD_MODEL_CONTEXT_LIMITS[model_name]

    # Try prefix matching (e.g., "gemini-2.0-flash-001" matches "gemini-2.0-flash")
    for prefix in sorted(CLOUD_MODEL_CONTEXT_LIMITS.keys(), key=len, reverse=True):
        if model_name.startswith(prefix):
            return CLOUD_MODEL_CONTEXT_LIMITS[prefix]

    return CLOUD_DEFAULT_CONTEXT_LIMIT


# ---------------------------------------------------------------------------
# Persistent Cloud Token Tracker
# ---------------------------------------------------------------------------

# File path for the persistent token usage JSON
CLOUD_TOKEN_USAGE_FILE = Path(__file__).parent / "cloud_token_usage.json"


def _load_token_usage() -> Dict[str, Any]:
    """Load token usage data from the persistent JSON file."""
    if not CLOUD_TOKEN_USAGE_FILE.exists():
        return {}
    try:
        with open(CLOUD_TOKEN_USAGE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_token_usage(data: Dict[str, Any]) -> None:
    """Save token usage data atomically."""
    temp = CLOUD_TOKEN_USAGE_FILE.with_suffix(".tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    temp.replace(CLOUD_TOKEN_USAGE_FILE)


def record_cloud_token_usage(model_name: str, input_tokens: int, output_tokens: int) -> None:
    """Atomically update the persistent cloud token usage file.

    Uses file-level locking (fcntl.flock) to prevent race conditions
    when multiple processes or experiments run concurrently.

    Args:
        model_name: The cloud model ID (e.g., "gemini-2.0-flash").
        input_tokens: Number of input/prompt tokens used in this call batch.
        output_tokens: Number of output/completion tokens used in this call batch.
    """
    if input_tokens == 0 and output_tokens == 0:
        return  # nothing to record

    lock_path = CLOUD_TOKEN_USAGE_FILE.with_suffix(".lock")
    try:
        with open(lock_path, "w") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)  # Exclusive lock
            try:
                data = _load_token_usage()

                if model_name not in data:
                    data[model_name] = {
                        "total_input_tokens": 0,
                        "total_output_tokens": 0,
                        "last_updated": "",
                    }

                data[model_name]["total_input_tokens"] += input_tokens
                data[model_name]["total_output_tokens"] += output_tokens
                data[model_name]["last_updated"] = datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(timespec="milliseconds")

                _save_token_usage(data)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
    except Exception as e:
        print(f"    [CLOUD TOKEN TRACKER] WARNING: Failed to record tokens: {e}")


def get_cloud_token_usage() -> Dict[str, Any]:
    """Read the current cloud token usage data (for display/reporting)."""
    return _load_token_usage()
