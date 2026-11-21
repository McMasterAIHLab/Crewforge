
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from environment import EnvIntroPrompt, EnvStatePrompt
from memory_module import InboxItem
from multi_system import ContextWindowExceededError, MultiLLMSystem
from sub_agent_module import SubAgent

from . import prompts
from .unit_spec import CHALLENGE_CORRECTIVE, MAX_REASON_WORDS, UnitSpec

#: Separator between the move and its reason in the stored message. Chosen so
#: that the move remains the first whitespace-delimited token of the stored
#: string, which keeps post-hoc parsing of proposals trivial.
REASON_SEP = " | "


class ChallengeMultiLLMSystem(MultiLLMSystem):
    """A MultiLLMSystem in which disagreement must be raised as a challenge.

    Messages carry a move plus a short reason, as in the move-reason prototype.
    The addition here is enforcement of the channel contract: an agent that
    receives a proposal and backs a different move must both post its own
    standing position to the common space AND challenge the sender directly.
    Instructing this was not sufficient - agents posted the disagreement to the
    common space alone, because that is where the meta agent counts positions -
    so a response that disagrees without challenging is rejected and retried.
    """

    #: Corrective retries allowed per call before the agent contributes nothing.
    MAX_UNIT_RETRIES = 3

    #: The post-inbox reflection call. Disabled: in logged runs it produced
    #: unmotivated position changes — agents switched moves with no new
    #: information on the channel since their previous call — while never
    #: producing an improvement. It also accounts for roughly a third of all
    #: agent calls. Set True to restore it.
    ENABLE_REFLECTION = False

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
        """The talk / add_to_common_space tools, each carrying a move and a reason.

        Both fields are marked required in the schema. The reason's description
        states the word cap, so a compliant model does not need the corrective
        path to discover it.
        """
        unit = self.unit_spec
        move_desc = (f"Exactly one {unit.description}, or '{unit.abstain_token}' "
                     f"to put nothing forward. Nothing else. "
                     f"Examples: {', '.join(unit.examples)}")
        reason_desc = (f"Why this move, judged by your own specialism, in at most "
                       f"{MAX_REASON_WORDS} words. One or two sentences, no line breaks.")

        talk_tool = {
            'type': 'function',
            'function': {
                'name': 'talk',
                'description': 'Send one move and its reason to another team member',
                'parameters': {
                    'type': 'object',
                    'required': ['member', 'move', 'reason'],
                    'properties': {
                        'member': {'type': 'string', 'description': 'The name of the team member to send the move to'},
                        'move': {'type': 'string', 'description': move_desc},
                        'reason': {'type': 'string', 'description': reason_desc},
                    },
                },
            },
        }

        add_to_common_space_tool = {
            'type': 'function',
            'function': {
                'name': 'add_to_common_space',
                'description': 'Send one move and its reason to every team member',
                'parameters': {
                    'type': 'object',
                    'required': ['move', 'reason'],
                    'properties': {
                        'move': {'type': 'string', 'description': move_desc},
                        'reason': {'type': 'string', 'description': reason_desc},
                    },
                },
            },
        }

        return [talk_tool, add_to_common_space_tool]

    def _ready_meta(self, intro_prompt: EnvIntroPrompt):
        """As baseline, with the move-and-reason post-init system prompt."""
        self.meta_agent.system_prompt.intro = prompts.post_init_sys_prompt(intro_prompt)
        self.meta_agent.user_prompt.prompt = ""

    def _ready_crew(self, intro_prompt: EnvIntroPrompt):
        """As baseline, with the move-and-reason communication section."""
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
        paragraph differs."""
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

        # The challenge rule only applies to a proposal from a named teammate.
        # SYSTEM notices carry no move and are not answerable.
        #
        # These attributes are read without a default on purpose: if InboxItem
        # is renamed, this must break loudly. An earlier version used getattr
        # with a "" fallback and silently disabled the whole rule for a full
        # sweep, which looked like agents complying rather than a dead check.
        sender_name = inbox_item.other_agent_name
        sender_id = inbox_item.other_agent_id
        sender_move = self._move_of(inbox_item.new_message.text)

        if sender_name == "SYSTEM":
            self._challenge_context = None
        elif not sender_id or not sender_move:
            raise RuntimeError(
                f"Cannot build challenge context for a message from "
                f"{sender_name!r}: sender_id={sender_id!r}, "
                f"parsed move={sender_move!r} from "
                f"{inbox_item.new_message.text!r}")
        else:
            self._challenge_context = (sender_name, sender_id, sender_move)

        try:
            self._call_agent(agent, cycle, call_type="deliberation")
        finally:
            self._challenge_context = None

    def _handle_regular_agent_no_inbox(self, agent: SubAgent, cycle: int,
                                       env_state_prompt: EnvStatePrompt,
                                       is_reflection: bool):
        """As baseline, except that the reflection call is skipped when
        ENABLE_REFLECTION is False.

        A skipped reflection contributes nothing to the channel and makes no
        model call, so the agent's standing position from earlier in the cycle
        is what the meta agent sees.
        """
        if is_reflection and not self.ENABLE_REFLECTION:
            return

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
        """As baseline. The urgency block, the legality check and the
        illegal-move notification are unchanged."""
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

    def _pending_sender(self) -> Optional[Tuple[str, str, str]]:
        """The proposal this call is answering, if any.

        Returns:
            (sender_name, sender_id, sender_move) when the current call was
            triggered by a teammate's proposal, else None.
        """
        return getattr(self, "_challenge_context", None)

    @staticmethod
    def _move_of(message: str) -> str:
        """Recover the move from a stored 'move | reason' message.

        Args:
            message: The stored message body.

        Returns:
            The leading move token, or "" when the message is empty.
        """
        return (message or "").strip().split(REASON_SEP)[0].strip().split()[0] \
            if (message or "").strip() else ""

    def _check_reason(self, reason: str) -> Tuple[bool, str, str]:
        """Validate the reason half of a message.

        Args:
            reason: The raw reason string from the tool call.

        Returns:
            (ok, cleaned, complaint) where cleaned is the whitespace-normalised
            reason and complaint describes the failure when ok is False.
        """
        cleaned = " ".join((reason or "").split())
        if not cleaned:
            return False, "", "the reason was empty"
        word_count = len(cleaned.split())
        if word_count > MAX_REASON_WORDS:
            return False, cleaned, (f"the reason was {word_count} words, over the "
                                    f"{MAX_REASON_WORDS}-word limit")
        return True, cleaned, ""

    def _extract(self, record, agent: SubAgent) -> Tuple[
            List[Tuple[str, str, str, Optional[str]]], List[str]]:
        """Pull the outgoing message(s) out of a response.

        Every non-format failure mode (unknown tool, missing or unresolvable
        member, message to self) keeps its baseline handling and does NOT
        trigger a retry — those are not format problems. A missing reason IS a
        format problem and is passed through as an empty string for
        _check_reason to reject.

        Args:
            record: The MASCallRecord just returned.
            agent: The calling agent.

        Returns:
            (outgoing, tool_issues) where outgoing is a list of
            (move, reason, channel, recipient_id).
        """
        outgoing: List[Tuple[str, str, str, Optional[str]]] = []
        tool_issues: List[str] = []

        if not record.tool_calls:
            # Baseline fallback: an agent that answered without calling a tool
            # is treated as having spoken to the common space. The raw text
            # cannot be split into move and reason, so it is passed through as
            # a move and will be rejected by the unit check.
            raw = (record.raw_response or "").strip()
            if raw:
                outgoing.append((raw, "", "common", None))
            return outgoing, tool_issues

        for tc in record.tool_calls:
            name = tc.get("name", "")
            args = tc.get("arguments", {}) or {}

            if name == "talk":
                move = args.get("move", "")
                reason = args.get("reason", "")
                member = args.get("member", "")
                if not move:
                    tool_issues.append("Tool talk() called without 'move' argument!")
                    continue
                if not member:
                    tool_issues.append("Tool talk() called without 'member' argument!")
                    outgoing.append((move, reason, "common", None))   # baseline fallback
                    continue
                recipient_id = self.memory.registry.id_from_name(member)
                if recipient_id is None:
                    tool_issues.append(
                        f"Tool talk() called with unmatchable 'member' argument: {member}!")
                    continue
                if recipient_id == agent.id:
                    tool_issues.append("Tool talk() called with 'member' argument set to self!")
                    continue
                outgoing.append((move, reason, "direct", recipient_id))

            elif name == "add_to_common_space":
                move = args.get("move", "")
                reason = args.get("reason", "")
                if not move:
                    tool_issues.append(
                        "Tool add_to_common_space() called without 'move' argument!")
                    continue
                outgoing.append((move, reason, "common", None))

            else:
                tool_issues.append(f"Tool {name} not found!")

        return outgoing, tool_issues

    def _call_agent(self, agent: SubAgent, cycle: int, call_type: str):
        """Call the model for one agent, then enforce the move-and-reason contract.

        A response whose move is not a valid unit, or whose reason is missing or
        over the word cap, is retried with a format correction appended, up to
        MAX_UNIT_RETRIES times. Every attempt is appended to _current_calls, so
        retry tokens are counted against the prototype rather than hidden. After
        the budget is spent the agent contributes nothing to the channel for
        this call and the last record carries a UnitViolation error, which is
        what makes the violation rate recoverable post-hoc.

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
        # (channel, recipient_id, move) already committed on THIS call, so a
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
            for move_raw, reason_raw, channel, recipient_id in outgoing:
                ok_move, move, _is_abstain = self.unit_spec.validate(move_raw)
                if not ok_move:
                    rejected.append(f"Your move was: {move_raw!r} - that is not a valid move.")
                    continue

                ok_reason, reason, complaint = self._check_reason(reason_raw)
                if not ok_reason:
                    rejected.append(f"Your move {move!r} was valid, but {complaint}.")
                    continue

                key = (channel, recipient_id, move)
                if key in stored:
                    continue

                message = f"{move}{REASON_SEP}{reason}"
                # An abstention is stored exactly like a move: it is a message
                # in the channel, visible to teammates and to the meta agent.
                if channel == "direct" and recipient_id is not None:
                    self.memory.store_direct(agent.id, recipient_id, message,
                                             agent.current_move_id)
                else:
                    self.memory.store_common(agent.id, message, agent.current_move_id)
                stored.add(key)

            if tool_issues:
                record.error = ' AND '.join(tool_issues)

            # --- the challenge rule ---------------------------------------
            # Enforced only when nothing was rejected on format grounds, so a
            # malformed response is corrected for format first and for the
            # missing challenge on a later attempt. This keeps the corrective
            # the agent sees to one problem at a time.
            challenge_complaint = ""
            ctx = self._pending_sender()
            if not rejected and ctx is not None:
                sender_name, sender_id, sender_move = ctx
                posted = [(ch, rid, mv) for (ch, rid, mv) in stored]
                own_moves = {mv for (ch, rid, mv) in posted if ch == "common"}
                challenged = any(ch == "direct" and rid == sender_id
                                 for (ch, rid, mv) in posted)
                disagreed = bool(own_moves) and sender_move not in own_moves

                if disagreed and not challenged:
                    agent_move = sorted(own_moves)[0]
                    challenge_complaint = CHALLENGE_CORRECTIVE.format(
                        sender=sender_name, sender_move=sender_move,
                        agent_move=agent_move)

            if not rejected and not challenge_complaint:
                return

            if attempt == self.MAX_UNIT_RETRIES:
                if rejected:
                    violation = (
                        f"UnitViolation: {len(rejected)} message(s) were not a valid "
                        f"{self.unit_spec.name} after {self.MAX_UNIT_RETRIES + 1} attempt(s); "
                        f"last rejected: {rejected[0][:160]!r}")
                else:
                    # The standing position was stored; only the challenge is
                    # missing. Tagged distinctly so the rate at which agents
                    # refuse to challenge is recoverable from the logs.
                    violation = (
                        f"ChallengeViolation: agent disagreed with "
                        f"{self._pending_sender()[0]} but sent no challenge after "
                        f"{self.MAX_UNIT_RETRIES + 1} attempt(s)")
                record.error = (f"{record.error} AND {violation}"
                                if record.error else violation)
                return

            if rejected:
                corrective = self.unit_spec.corrective(
                    bad=rejected[0], attempt=attempt + 1,
                    final=(attempt + 1 == self.MAX_UNIT_RETRIES))
            else:
                corrective = "\n\n" + challenge_complaint
