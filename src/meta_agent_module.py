"""
The Meta Agent module. Contains the class defining the structure of a 
meta agent.
"""

from __future__ import annotations

from agent_config import SystemPrompt, UserPrompt, LLMConfig, AUID_LEN


class MetaAgent():
    """
    Class for the meta agent LLM.
    """

    def __init__(
        self,
        system_prompt: SystemPrompt,
        user_prompt: UserPrompt,
        llm_config: LLMConfig,
        name: str = "Meta",
        agent_id: str = ("0"*AUID_LEN),  # Hardcoded meta agent ID of all zeros
    ):
        """
        Initialize an instance of the meta agent.

        Args:
            system_prompt: the system prompt for creating the crew (will change after creation)
            user_prompt: the user prompt for creating the crew (will change after creation)
            llm_config: the LLM model and params to use
            name: hardcoded to Meta
            agent_id: hardcoded to repeating zeros for the length of AUID_LEN

        Note:
            After initialization, the agent will also have the current_move_id field.
        """
        self.name = name
        self.id = agent_id
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        # NOTE: this is a change from the previous str version.
        self.current_move_id: int
        self.llm_config = llm_config
