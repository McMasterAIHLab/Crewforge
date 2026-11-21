
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from environment import EnvIntroPrompt, EnvStatePrompt
from memory_module import InboxItem
from multi_system import ContextWindowExceededError, MultiLLMSystem
from sub_agent_module import SubAgent

from . import prompts
from .unit_spec import UnitSpec


class UnitCommsMultiLLMSystem(MultiLLMSystem):
    """A MultiLLMSystem whose agents communicate only in the environment's unit."""

    #: Corrective retries allowed per call before the agent contributes nothing.
    #: Matches the baseline's MAX_SILENT_STRIP_RETRIES.
    MAX_UNIT_RETRIES = 3

    def __init__(self, *args, unit_spec: UnitSpec, **kwargs):
        """
        Args:
            unit_spec: The communication unit for the environment being run.
            *args, **kwargs: Passed through to MultiLLMSystem.

        Note:
            unit_spec is assigned BEFORE super().__init__(). That call builds the
            meta agent and the crew's tools and runs _mas_init, all of which go
            through overrides that read self.unit_spec. This ordering is
            load-bearing, not stylistic.
        """
        self.unit_spec = unit_spec
        super().__init__(*args, **kwargs)

    # -- prompt / tool overrides -------------------------------------------

    def _crew_creation_sys_prompt(self, intro_prompt: EnvIntroPrompt) -> str:
        return prompts.crew_creation_sys_prompt(intro_prompt)

    def _crew_creation_user_prompt(self, intro_prompt: EnvIntroPrompt) -> str:
        return prompts.crew_creation_user_prompt(intro_prompt)

    def _build_agent_tools(self) -> List[Dict]:
        """The talk / add_to_common_space tools, carrying a `move` rather than
        free text. The parameter name and description are themselves part of the
        contract: a schema field called `text` invites prose."""
        unit = self.unit_spec
        move_desc = (f"Exactly one {unit.description}, or '{unit.abstain_token}' "
                     f"to put nothing forward. Nothing else. "
                     f"Examples: {', '.join(unit.examples)}")

        talk_tool = {
            'type': 'function',
            'function': {
                'name': 'talk',
                'description': 'Send one move to another team member',
                'parameters': {
                    'type': 'object',
                    'required': ['member', 'move'],
                    'properties': {
                        'member': {'type': 'string', 'description': 'The name of the team member to send the move to'},
                        'move': {'type': 'string', 'description': move_desc},
                    },
                },
            },
        }

        add_to_common_space_tool = {
            'type': 'function',
            'function': {
                'name': 'add_to_common_space',
                'description': 'Send one move to every team member',
                'parameters': {
                    'type': 'object',
                    'required': ['move'],
                    'properties': {
                        'move': {'type': 'string', 'description': move_desc},
                    },
                },
            },
        }

        return [talk_tool, add_to_common_space_tool]

    def _ready_meta(self, intro_prompt: EnvIntroPrompt):
        """As baseline, with the unit-aware post-init system prompt."""
        self.meta_agent.system_prompt.intro = prompts.post_init_sys_prompt(intro_prompt)
        self.meta_agent.user_prompt.prompt = ""

    def _ready_crew(self, intro_prompt: EnvIntroPrompt):
        """As baseline, with the unit-aware communication section.

        The role half of the prompt is unchanged; only the tools blurb differs,
        because the baseline's tells the agent to explain its reasoning.
        """
        from agent_config import SystemPrompt

        for agent in self.memory.registry.all_agents():
            agent.talksto = self.memory.registry.ids_from_names(agent.talksto_strs)
            agent.current_move_id = -1  # pre-game
            agent.system_prompt = SystemPrompt(
                intro=prompts.agent_intro(agent, intro_prompt),
                tools=prompts.agent_tools_blurb(self.unit_spec),
                live_state=None, move_ids=None)

    def _unit_reminder(self) -> str:
        """The standing format reminder appended to every agent user prompt."""
        return prompts.unit_reminder(self.unit_spec)

    # -- user prompts -------------------------------------------------------

    def _handle_regular_agent_inbox(self, agent: SubAgent, inbox_item: InboxItem,
                                    cycle: int, env_state_prompt: EnvStatePrompt):
        """As baseline: same blocks, same order, same headers. Only the guidance
        paragraph differs, and the environment's `reply` line is dropped — it
        asks the agent to phrase things, which is no longer possible."""
        msg_display = inbox_item.format()

        if inbox_item.source == "direct":
            msg_type_intro = f"You are receiving a DIRECT MESSAGE from {inbox_item.other_agent_name}."
        else:
            msg_type_intro = f"You are receiving a COMMON SPACE message (sent to all team members) from {inbox_item.other_agent_name}."

        agent.user_prompt.prompt = (
            f"{env_state_prompt.prompt}\n\n"
            f"{msg_type_intro}\n\n"
            f"=====MESSAGE=====\n"
            f"{msg_display}\n"
            f"=====END MESSAGE=====\n\n"
            f"{prompts.agent_inbox_guidance(inbox_item, self.unit_spec)}\n"
        )

        self._call_agent(agent, cycle, call_type="deliberation")

    def _handle_regular_agent_no_inbox(self, agent: SubAgent, cycle: int,
                                       env_state_prompt: EnvStatePrompt,
                                       is_reflection: bool):
        """As baseline, including the reflection step and both transcript
        blocks. Only the guidance paragraph differs."""
        direct_msgs = self.memory.view_fmt_direct(
            agent=agent, current_move_id=env_state_prompt.move_id)
        common_msgs = self.memory.view_fmt_common(
            current_move_id=env_state_prompt.move_id)

        guidance = prompts.agent_no_inbox_guidance(is_reflection, self.unit_spec)

        if not is_reflection:
            agent.user_prompt.prompt = (
                f"{env_state_prompt.prompt}\n\n"
                f"{guidance}\n"
                f"Review your conversation history below. Remember: when you see \"{agent.name}\", that's you!\n\n"
                f"====={agent.name} (You) - DIRECT MESSAGES=====\n"
                f"{direct_msgs}\n\n"
                f"=====COMMON SPACE MESSAGES=====\n"
                f"{common_msgs}\n"
            )
        else:
            agent.user_prompt.prompt = (
                f"{env_state_prompt.prompt}"
                f"{guidance}\n"
                f"Review the current state of all conversations:\n\n"
                f"====={agent.name} (You) - DIRECT MESSAGES=====\n"
                f"{direct_msgs}\n\n"
                f"=====COMMON SPACE MESSAGES=====\n"
                f"{common_msgs}\n"
            )

        self._call_agent(agent=agent, cycle=cycle, call_type="deliberation")

    def _handle_meta_consensus_check(self, env_state_prompt: EnvStatePrompt, cycle: int,
                                     cycles_remaining: int,
                                     is_countdown_phase: bool) -> Optional[str]:
        """As baseline in every respect except one added sentence in the prompt
        explaining what a message is. The urgency block, the majority threshold,
        the legality check and the illegal-move notification are unchanged."""
        urgency_msg = ""
        if is_countdown_phase:
            if cycles_remaining > 1:
                urgency_msg = f"\nCOUNTDOWN WARNING: {cycles_remaining} cycles remaining. MUST converge on final consensus soon.\n"
            else:
                urgency_msg = "\nFINAL ROUND - DECIDE NOW.\nThis is your LAST chance. Call consensus tool with agreed action RIGHT NOW.\n"

        all_direct_msgs = self.memory.get_all_direct_messages_formatted(
            self.meta_agent.current_move_id)
        all_common_msgs = self.memory.view_fmt_common(
            self.meta_agent.current_move_id)

        self.meta_agent.user_prompt.prompt = prompts.meta_consensus_user_prompt(
            env_state_prompt=env_state_prompt,
            num_agents=len(self.memory.registry.all_agents()),
            current_move_id=self.meta_agent.current_move_id,
            urgency_msg=urgency_msg,
            all_direct_msgs=all_direct_msgs,
            all_common_msgs=all_common_msgs,
        )

        self._call_meta(cycle=cycle, call_type="consensus_check")

        move = None
        if self.consensus_move is not None:
            if env_state_prompt.legal_actions is not None:
                move = self.consensus_move
                if move not in env_state_prompt.legal_actions:
                    self._notify_legal_move(
                        move, env_state_prompt.legal_actions, env_state_prompt.move_id)
                    self.consensus_move = None
                    move = None
            else:
                move = self.consensus_move
        return move

    # -- the channel: validate, correct, retry ------------------------------

    def _extract(self, record, agent: SubAgent) -> Tuple[List[Tuple[str, str, Optional[str]]],
                                                         List[str]]:
        """Pull the outgoing message(s) out of a response.

        Returns a list so a response carrying several tool calls needs no
        special case. Every non-unit failure mode (unknown tool, missing or
        unresolvable member, message to self) keeps its baseline handling and
        does NOT trigger a unit retry — those are not format problems.

        Args:
            record: The MASCallRecord just returned.
            agent: The calling agent.

        Returns:
            (outgoing, tool_issues) where outgoing is a list of
            (candidate, channel, recipient_id).
        """
        outgoing: List[Tuple[str, str, Optional[str]]] = []
        tool_issues: List[str] = []

        if not record.tool_calls:
            # Baseline fallback: an agent that answered without calling a tool
            # is treated as having spoken to the common space. Preserved, but
            # the content is now validated rather than stored raw.
            raw = (record.raw_response or "").strip()
            if raw:
                outgoing.append((raw, "common", None))
            return outgoing, tool_issues

        for tc in record.tool_calls:
            name = tc.get("name", "")
            args = tc.get("arguments", {}) or {}

            if name == "talk":
                move = args.get("move", "")
                member = args.get("member", "")
                if not move:
                    tool_issues.append("Tool talk() called without 'move' argument!")
                    continue
                if not member:
                    tool_issues.append("Tool talk() called without 'member' argument!")
                    outgoing.append((move, "common", None))   # baseline fallback
                    continue
                recipient_id = self.memory.registry.id_from_name(member)
                if recipient_id is None:
                    tool_issues.append(
                        f"Tool talk() called with unmatchable 'member' argument: {member}!")
                    continue
                if recipient_id == agent.id:
                    tool_issues.append("Tool talk() called with 'member' argument set to self!")
                    continue
                outgoing.append((move, "direct", recipient_id))

            elif name == "add_to_common_space":
                move = args.get("move", "")
                if not move:
                    tool_issues.append(
                        "Tool add_to_common_space() called without 'move' argument!")
                    continue
                outgoing.append((move, "common", None))

            else:
                tool_issues.append(f"Tool {name} not found!")

        return outgoing, tool_issues

    def _call_agent(self, agent: SubAgent, cycle: int, call_type: str):
        """Call the model for one agent, then enforce the unit contract.

        A response whose message is not a valid unit is retried with a format
        correction appended, up to MAX_UNIT_RETRIES times. Every attempt is
        appended to _current_calls, so retry tokens are counted against the
        prototype in every summary rather than hidden. After the budget is
        spent the agent contributes nothing to the channel for this call and
        the last record carries a UnitViolation error, which is what makes the
        violation rate recoverable post-hoc.

        Args:
            agent: The agent being called; its prompts must already be built.
            cycle: The deliberation cycle this call belongs to.
            call_type: "deliberation" | "init".

        Raises:
            ContextWindowExceededError: If the agent's prompts are estimated to
                exceed the configured context limit (baseline behaviour).
        """
        base_user_prompt = agent.user_prompt.prompt + self._unit_reminder()

        within_limit, estimated = self.check_context_limit(
            agent.system_prompt.prompt, agent.user_prompt.prompt,
            agent.llm_config.max_context_tokens)
        if not within_limit:
            raise ContextWindowExceededError(
                f"Context window exceeded for agent {agent.name}: estimated {estimated} tokens > max {agent.llm_config.max_context_tokens}"
            )

        corrective = ""
        # (channel, recipient_id, unit) already committed on THIS call, so a
        # retry that resends an accepted message cannot store it twice.
        stored: set = set()

        for attempt in range(1 + self.MAX_UNIT_RETRIES):
            record = self._impl_call_ollama(
                user_prompt=base_user_prompt + corrective,
                system_prompt=agent.system_prompt.prompt,
                llm_config=agent.llm_config,
                silent_test=True,
                include_content=True,
                is_meta=False,
            )

            record.caller_id = agent.id
            record.caller_name = agent.name
            record.call_type = call_type
            record.cycle = cycle
            self._current_calls.append(record)

            print(record, end='\n\n')

            if record.error is not None:
                return  # transport error or exhausted silent-strip: baseline behaviour

            outgoing, tool_issues = self._extract(record, agent)

            rejected: List[str] = []
            for candidate, channel, recipient_id in outgoing:
                ok, unit, _is_abstain = self.unit_spec.validate(candidate)
                if not ok:
                    rejected.append(candidate)
                    continue
                key = (channel, recipient_id, unit)
                if key in stored:
                    continue
                # An abstention is stored exactly like a move: it is a message
                # in the channel, visible to teammates and to the meta agent.
                if channel == "direct" and recipient_id is not None:
                    self.memory.store_direct(agent.id, recipient_id, unit,
                                             agent.current_move_id)
                else:
                    self.memory.store_common(agent.id, unit, agent.current_move_id)
                stored.add(key)

            if tool_issues:
                record.error = ' AND '.join(tool_issues)

            if not rejected:
                return

            if attempt == self.MAX_UNIT_RETRIES:
                violation = (
                    f"UnitViolation: {len(rejected)} message(s) were not a valid "
                    f"{self.unit_spec.name} after {self.MAX_UNIT_RETRIES + 1} attempt(s); "
                    f"last rejected: {rejected[0][:120]!r}")
                record.error = (f"{record.error} AND {violation}"
                                if record.error else violation)
                return

            corrective = self.unit_spec.corrective(
                bad=rejected[0], attempt=attempt + 1,
                final=(attempt + 1 == self.MAX_UNIT_RETRIES))
