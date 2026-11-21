"""
The agent config. Contains variables, functions, and classes related
to configurations of agents (both meta and sub)
"""
from dataclasses import dataclass
import random
import string
import itertools
from typing import Dict, Any, Optional, Tuple, List
from ollama import ChatResponse


AUID_LEN = 3


def gen_auid(count: int, id_length: int = AUID_LEN) -> list[str]:
    """
    Generates unique alphanumeric IDs to be assigned to agents.

    Args:
        count: the number of IDs to generate
        id_length: the length of each ID (defaults to AUID_LEN)

    Returns:
        a list containing the IDs
    """
    chars = string.ascii_uppercase + string.digits

    # create iterator of all possible combinations
    all_combos = itertools.product(chars, repeat=id_length)

    # take count random samples, convert to strings
    samples = random.sample(list(all_combos), count)
    return [''.join(combo) for combo in samples]


@dataclass
class LLMConfig:
    """
    Configuration for LLM parameters used across agents.

    Initial values default to qwen3:8b with lightweight parameters 
    for prototyping convenience.

    model_name:
        Name of the Ollama model to use (e.g., "qwen3:8b").
    temperature:
        Sampling temperature for the LLM (higher = more random).
    max_context_tokens:
        Context window size for the model.
    max_token:
        Maximum tokens to generate.
    think:
        Whether to enable "thinking" output (if the model supports it).
    seed:
        Random seed for reproducibility.
    """
    model_name: str = "qwen3:8b"
    temperature: float = 0.2
    max_context_tokens: int = 8192
    max_token: int = 96
    think: bool = False
    keep_alive: str = "4m"
    seed: int = 42
    tools: Optional[List[Dict]] = None

    @property
    def options(self) -> dict:
        """Return options dict for ollama.chat()"""
        return {
            "num_ctx": self.max_context_tokens,
            "temperature": self.temperature,
            "num_predict": self.max_token,
            "seed": self.seed,
        }

    def create_kwargs(self, user_prompt: str, system_prompt: str, fmt: Optional[str]) -> Dict[str, Any]:
        kwargs = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": self.options,
            "keep_alive": self.keep_alive,
            "think": self.think,
        }

        if fmt is not None:
            kwargs["format"] = fmt

        if self.tools is not None:
            kwargs["tools"] = self.tools

        return kwargs


@dataclass
class SystemPrompt:
    """
    Contains editable fields within a system prompt.
    """
    intro: str = ""
    tools: str = ""
    live_state: Optional[str] = ""
    move_ids: Optional[str] = ""

    @property
    def prompt(self) -> str:
        """
        Calling prompt() returns a concatenated string of all not-None
        fields.
        """
        parts = [self.intro, self.tools, self.live_state, self.move_ids]
        return ''.join(part for part in parts if part)


@dataclass
class UserPrompt:
    """
    Contains the user prompt.
    """
    # TODO: add direct messages and common space messages as variables?
    prompt: str = ""
