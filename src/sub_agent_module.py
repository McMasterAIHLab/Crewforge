"""
The Sub Agent module. Contains the class defining the structure of a 
sub agent.
"""

from __future__ import annotations

from agent_config import SystemPrompt, UserPrompt, LLMConfig
from typing import Optional


class SubAgent:
    """
    Class for the sub agent LLMs.
    """

    def __init__(
        self,
        name: str,
        agent_id: str,
        llm_config: LLMConfig,
        responsibility: str,
        provides: str,
        talksto_strs: list,
        # optional since normally can't have this without initializing all agents
        talksto: Optional[list] = None,
    ):
        """
        Initialize a Sub Agent.

        Args:
            name: agent's display name
            agent_id: agent unique identifier (AUID)
            system_prompt: system prompt configuration
            user_prompt: user prompt configuration
            responsibility: description of agent's responsibilities
            provides: brief rationale for agent's role
            talkto_strs: list of agent names this agent can communicate with
            talkto: list of agent IDs this agent can communicate with
            llm_config: LLM configuration parameters
        """
        self.name = name
        self.id = agent_id
        self.system_prompt: SystemPrompt = SystemPrompt()
        self.user_prompt: UserPrompt = UserPrompt()
        self.responsibility = responsibility
        self.provides = provides
        self.talksto_strs = talksto_strs
        self.talksto = talksto if talksto is not None else []
        self.current_move_id: int  # NOTE: changed from the previous str version
        self.llm_config = llm_config
        self.last_inbox_check: str = ""
