import json
import time
import concurrent.futures
from typing import List, Optional, Tuple
from datetime import datetime, timezone

import ollama

from agent_config import LLMConfig
from system import System, ChatCallRecord, SystemMoveRecord, illegal_move_notification
from environment import EnvStatePrompt, EnvIntroPrompt

# Per-call timeout for ollama.chat()
OLLAMA_CALL_TIMEOUT_SECONDS = 3600


class SingleLLMSystem(System):
    """
    A system that uses a single LLM to decide on the next move, bypassing intermediate engines to call ollama.chat directly. 
    Adapts behavior based on whether the environment provides strict legal actions (Chess) or requires open-ended generation (EvalPlus).
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        env_intro_prompt: EnvIntroPrompt,
        max_retries: int = 3,
        chat_fn=None,
        symmetric_invalid_move: bool = False,
    ):
        self._config = llm_config
        self._env_intro = env_intro_prompt
        self._max_retries = max_retries
        self._symmetric_invalid_move = symmetric_invalid_move
        self._chat_fn = chat_fn or ollama.chat  # Default to ollama.chat when no cloud fn provided
        self._current_calls: List[ChatCallRecord] = []

        self._json_behavior_prompt = (
            "REQUIRED OUTPUT FORMAT: Return ONLY valid JSON with exactly these two fields:\n"
            '1. "move": One exact action string from the provided legal actions list\n'
            '2. "rationale_brief": A brief explanation (12 words maximum)\n\n'
            "CRITICAL: The action MUST be chosen from the provided legal actions list. "
            "Do not invent actions. Ensure the casing and format matches exactly."
        )

    def next_move(self, env_state_prompt: EnvStatePrompt) -> str:
        self._current_calls = []
        legal = env_state_prompt.legal_actions

        # Dynamically build prompts based on environment constraints
        if legal is not None:
            system_prompt = f"{self._env_intro.domain_prompt}\n\n{self._json_behavior_prompt}"
            user_prompt = (
                f"{env_state_prompt.prompt}\n"
                "Choose your move exactly from the legal actions list above "
                "and respond strictly with JSON."
                f"\n\nStructured data:\n{json.dumps({'legal_actions': legal})}"
            )
        else:
            # Open-ended mode for EvalPlus
            system_prompt = self._env_intro.domain_prompt
            user_prompt = env_state_prompt.prompt

        last_error = None
        for attempt in range(1, self._max_retries + 1):
            action, call_record = self._try_call(
                system_prompt, user_prompt, legal)
            self._current_calls.append(call_record)

            if call_record.error is None and action is not None:
                return action
            last_error = call_record.error
            if self._symmetric_invalid_move and last_error == f"'{call_record.response}' not in legal actions":
                user_prompt += "\n\n" + illegal_move_notification(call_record.response, legal)

        raise RuntimeError(
            f"SingleLLMSystem failed after {self._max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def call_ollama(self, user_prompt: str, system_prompt: str, llm_config: LLMConfig, format: Optional[str] = None) -> ChatCallRecord:
        record: ChatCallRecord

        timestamp = datetime.now(timezone.utc).isoformat(
            timespec="milliseconds")
        t0 = time.time()

        # llm_config parameter unused here, we only have one config, and that's the self._config.
        kwargs = self._config.create_kwargs(
            user_prompt=user_prompt, system_prompt=system_prompt, fmt=format)

        # try just getting the response with a timeout guard
        # NOTE: We intentionally avoid the `with` context manager here.
        # Using `with ThreadPoolExecutor(...) as pool:` would call
        # pool.shutdown(wait=True) on exit, which blocks forever if the
        # worker thread is stuck in ollama.chat() after a timeout.
        # Instead we manage the pool manually with shutdown(wait=False).
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(self._chat_fn, **kwargs)
            response = future.result(timeout=OLLAMA_CALL_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            pool.shutdown(wait=False, cancel_futures=True)
            latency = int((time.time() - t0) * 1000)
            record = ChatCallRecord(
                latency_ms=latency, timestamp=timestamp,
                system_prompt=system_prompt, user_prompt=user_prompt,
                error=f"TimeoutError: ollama.chat exceeded {OLLAMA_CALL_TIMEOUT_SECONDS}s limit")
            return record
        except Exception as e:
            pool.shutdown(wait=False, cancel_futures=True)
            latency = int((time.time() - t0) * 1000)
            # return the record if this failed
            record = ChatCallRecord(
                latency_ms=latency, timestamp=timestamp,
                system_prompt=system_prompt, user_prompt=user_prompt, error=f"{e!r}")
            return record
        else:
            pool.shutdown(wait=False)

        # capture the latency
        latency = int((time.time() - t0) * 1000)

        # extract message contents robustly (supports both dict and object notations based on ollama version)
        raw = ""
        thinking = ""
        if hasattr(response, "message"):
            raw = getattr(response.message, "content", "")
            thinking = getattr(response.message, "thinking", "")
        elif isinstance(response, dict) and "message" in response:
            raw = response["message"].get("content", "")
            thinking = response["message"].get("thinking", "")

        # extract tokens robustly
        prompt_toks = getattr(response, "prompt_eval_count", 0)
        completion_toks = getattr(response, "eval_count", 0)
        thoughts_toks = getattr(response, "thoughts_token_count", 0) or 0
        if isinstance(response, dict):
            prompt_toks = prompt_toks or response.get("prompt_eval_count", 0)
            completion_toks = completion_toks or response.get("eval_count", 0)
            thoughts_toks = thoughts_toks or response.get("thoughts_token_count", 0)

        record = ChatCallRecord(
            latency_ms=latency, timestamp=timestamp,
            prompt_tokens=prompt_toks, completion_tokens=completion_toks,
            thoughts_tokens=thoughts_toks,
            thinking=thinking,
            raw_response=raw,
            system_prompt=system_prompt, user_prompt=user_prompt,
        )

        return record

    def _try_call(
        self,
        system_prompt: str,
        user_prompt: str,
        legal: Optional[List[str]],
    ) -> Tuple[Optional[str], ChatCallRecord]:
        """
        Single ollama.chat() attempt, including parsing of the output.

        Returns:
            A tuple containing the action as a string (or None if error) and the ChatCallRecord object.
        """
        call_record = self.call_ollama(user_prompt=user_prompt, system_prompt=system_prompt,
                                       llm_config=self._config, format=("json" if legal is not None else None))

        # an error occuring while calling chat(), no point in proceeding with parsing, return now
        if call_record.error is not None:
            return None, call_record

        # Mode 1: Constrained choices (Chess) requires JSON parsing & Verification
        if legal is not None:
            try:
                data = json.loads(call_record.raw_response)
                action = str(data.get("move") or "").strip()
                reason = str(data.get("rationale_brief") or "").strip()
            except json.JSONDecodeError as e:
                call_record.error = f"{e!r}"
                return None, call_record

            call_record.response = action

            if action in legal:
                return action, call_record

            call_record.error = f"'{action}' not in legal actions"
            return None, call_record

        # Mode 2: Open-ended (EvalPlus) accepts raw completion directly
        else:
            action = call_record.raw_response.strip()
            return action, call_record

    def get_move_record(self) -> SystemMoveRecord:
        """Returns the full log history for the recent `next_move` call."""
        return SystemMoveRecord(calls=self._current_calls)

    def get_name(self) -> str:
        """Returns the name of the system."""
        return "single_llm"

    def close(self) -> None:
        """Clean-up open resources."""
        pass
