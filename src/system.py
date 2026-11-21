from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Dict, Any, List, Optional
from environment import EnvStatePrompt
from agent_config import LLMConfig


@dataclass
class ChatCallRecord:
    """Hold statistics related to a single call to chat().
    """
    latency_ms: int
    timestamp: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thoughts_tokens: int = 0  # reasoning/thinking tokens (subset of completion for billing)
    response: str = ""
    thinking: str = ""
    raw_response: str = ""
    # Exact prompts sent to the model for this call (captured for full chat reconstruction)
    system_prompt: str = ""
    user_prompt: str = ""
    error: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        if self.error is not None:
            # -1 symbolizes error --- NOTE: or would we want prompt_tokens here?
            return (-1)
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # inject computed aggregates so they survive serialization
        d["total_tokens"] = self.total_tokens
        return d

# NOTE: The MAS implementation would have this dataclass to add extra fields to collection
# @dataclass
# class MASCallRecord(ChatCallRecord):
#     """
#     """
#     pass
# and the Single system implementation would have a symmetrical dataclass


@dataclass
class SystemMoveRecord:
    """Contains information related to all chat() calls used in a single next_move() call

    See the ChatCallRecord dataclass.
    """
    calls: List[ChatCallRecord]  # the list of chat() call information

    @property
    def prompt_tokens(self) -> int:
        """Returns:
            the total of all prompt tokens used in a single next_move() call
        """
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def completion_tokens(self) -> int:
        """Returns:
            the total of all completion tokens used in a single next_move() call
        """
        return sum(c.completion_tokens for c in self.calls)

    @property
    def total_tokens(self) -> int:
        """Returns:
            the total of all tokens used in a single next_move() call
        """
        return self.prompt_tokens + self.completion_tokens

    @property
    def total_latency_ms(self) -> int:
        """Returns:
            the total latency of a single next_move() call
        """
        return sum(c.latency_ms for c in self.calls)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # inject computed aggregates so they survive serialization
        d["prompt_tokens"] = self.prompt_tokens
        d["completion_tokens"] = self.completion_tokens
        d["total_tokens"] = self.total_tokens
        d["total_latency_ms"] = self.total_latency_ms
        d["call_count"] = len(self.calls)

        return d


def illegal_move_notification(illegal_move: str, legal_moves: List[str]) -> str:
    return (
        f"ILLEGAL MOVE ALERT: The proposed move '{illegal_move}' is not legal in the current position. "
        f"Valid moves are: {', '.join(legal_moves)}. "
        f"Please continue deliberation and propose a VALID move from this list."
    )


class System(ABC):
    """Abstract Base Class for guiding implementation of systems (i.e. single and multi).
    """

    @abstractmethod
    def call_ollama(
        self,
        user_prompt: str,
        system_prompt: str,
        llm_config: LLMConfig,
        format: Optional[str] = None,
    ) -> ChatCallRecord:
        """Call ollama.chat() and return the ChatCallRecord.

        Args:
            user_prompt: the user prompt string.
            system_prompt: the system prompt string.
            llm_config: the LLMConfig to make the call with.
            format: whether the response should be in JSON (format="json") or not (format=None)

        Returns:
            the resulting ChatCallRecord. All relevant fields are filled out (including error if 
            an error occurred)
        """
        ...

    @abstractmethod
    def next_move(self, env_state_prompt: EnvStatePrompt) -> str:
        """Prompt the System with the given environment prompt containing updated state information.

        Args:
            env_state_prompt: the most current EnvStatePrompt given by the environment's get_env_state_prompt().

        Returns:
            A string containing the System's response. 

        Note:
            This method is responsible making sure the response is legal in the environment (the 
            legal responses can be checked in the env_state_prompt's legal_move member).
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Returns:
            the name of the system as a string.
        """
        ...

    @abstractmethod
    def get_move_record(self) -> SystemMoveRecord:
        """Returns:
            A SystemMoveRecord of the most current move made using next_move().
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Clean-up any open system resources (such as files, etc...)."""
        ...
