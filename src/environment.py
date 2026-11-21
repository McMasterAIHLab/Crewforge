from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class EnvStatePrompt:
    """Houses the prompt string for all environment specific state info, including legal actions and move id
    """
    prompt: str  # the raw prompt built by get_prompt(), contains env state info & passed directly to LLMs
    move_id: int  # move id for the MAS and tracking purposes
    # flexible metadata for use by the environment (as necessary, not critical)
    metadata: Optional[Dict[str, str]]
    # legal actions for verification by next_move() in Single/MAS system
    legal_actions: Optional[List[str]]


@dataclass
class EnvIntroPrompt:
    """Houses the prompt string for any environment specific introductory info (e.g. for System prompts)
    """
    domain_prompt: str          # "You are playing chess against Stockfish..."
    notation_rules: str         # "ALL moves MUST be in UCI notation..."
    consensus_schema: Dict[str, Any]  # field_name -> description
    # the field that contains the actual action in the consensus_schema
    consensus_move_field: str
    reflection: Optional[str]  # reflection step instructions
    reply: Optional[str]    # instructions when replying
    # consensus_example: str      # example JSON for the consensus call


@dataclass
class EnvResult:
    """Generic container for environment result information, for use in visualization, metrics, results, etc...
    """
    # The concrete environment implementation will provide the members here

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EnvState:
    """Generic container for environment state information, similar to EnvResult but for states
    """
    # The concrete environment implementation will provide the members here

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Environment(ABC):
    """Abstract Base Class for guiding implementation of environments (e.g. Chess, EvalPlus, etc...) """

    def __init__(self):
        # requires all implementers to call super().__init__()
        # True if the environment has an opponent (e.g. the chess engine in the chess environment)
        self.has_opponent: bool
        # the opponent name if has_opponent is True (e.g. "Stockfish" in the chess environment)
        self.opponent_name: str

    @abstractmethod
    def is_active(self) -> bool:
        """Returns True if the environment is still active (e.g. in the chess game, this 
        would be True as long as the game is still in play --- no checkmate yet).

        Returns:
            True if the environment is active, False otherwise      
        """
        ...

    @abstractmethod
    def get_env_state_prompt(self) -> EnvStatePrompt:
        """Populates an EnvStatePrompt with the current state of the environment, and returns it.

        Returns:
            An EnvStatePrompt, built from the current EnvState.
        """
        ...

    @abstractmethod
    def get_env_intro_prompt(self) -> EnvIntroPrompt:
        """Populates an EnvIntroPrompt with information about the environment (e.g. rules, notations, etc...) 
        and returns it.

        Returns:
            The EnvIntroPrompt for the environment. 
        """
        ...

    @abstractmethod
    def give_move(self, action: str) -> EnvState:
        """Takes the action of a System (Single/MAS) and updates in the environment to reflect the action. For example,
        in a chess game, this would take the generated chess move and play it in the board.

        Returns:
            A current EnvState with state information after the move  

        Note:
            No move legality checking happens here, that should happen in the System's next_move() based off of the 
            supplied legal_moves in the EnvStatePrompt.
        """
        ...

    @abstractmethod
    def opponent_move(self) -> EnvState:
        """If the environment has an opponent (has_opponent = True), then this method will get the opponent to make
        a move in the environment.

        Returns:
            A current EnvState with state information after the move  
        """
        ...

    @abstractmethod
    def get_env_name(self) -> str:
        """Returns:
            the name of the environment as a string.
        """
        ...

    @abstractmethod
    def get_result(self) -> EnvResult:
        """Once is_active becomes False, this would return the final result of the environment.

        Returns:
            The environment's implementation of the EnvResult dataclass.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Clean-up any open environment resources (such as engines, files, etc...)."""
        ...
