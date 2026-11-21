from __future__ import annotations

from environment import EnvIntroPrompt

from .unit_spec import ABSTAIN


# ---------------------------------------------------------------------------
# Crew creation — the meta agent designing the team
# ---------------------------------------------------------------------------

def crew_creation_sys_prompt(intro_prompt: EnvIntroPrompt) -> str:
    """The meta agent's system prompt for designing the crew.

    Args:
        intro_prompt: The EnvIntroPrompt given by the environment.

    Returns:
        The crew-creation system prompt.
    """
    return (
        f"You are the meta-agent: you design and supervise a team of LLM agents "
        f"whose task is to{intro_prompt.domain_prompt}\n\n"

        "HOW YOUR TEAM COMMUNICATES:\n"
        "Agents exchange moves, not words. Every message an agent sends is one move in UCI "
        f"notation, or '{ABSTAIN}' to put nothing forward.\n"
        "Write each responsibility so it can be carried out under that constraint. A role must "
        "never require an agent to explain, justify, or report anything to a teammate, because it "
        "has no way to do so. An agent's analysis is private; its move is what the team sees.\n\n"

        "TEAM DESIGN:\n"
        "Give each agent one clear evaluative angle that would change which move is played. "
        "Three or four sharply distinct angles beat seven overlapping ones.\n"
        "Ask yourself: if this agent were removed, would the team ever play a different move? "
        "If not, do not create it.\n\n"

        "REQUIRED OUTPUT FORMAT: Return ONLY a valid JSON object with the following structure:\n"
        "{\n"
        "  \"team\": [\n"
        "    {\n"
        "      \"agent\": \"<role name>\",\n"
        "      \"talks\": [\"<list of other agent names this agent can talk to>\"],\n"
        "      \"responsibility\": \"<one or two sentence explanation of responsibilities>\",\n"
        "      \"rationale_brief\": \"<12 words max explaining why this agent is needed>\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"

        "CRITICAL RULES:\n"
        "1. Do NOT include a consensus agent - YOU determine consensus\n"
        "2. The 'talks' field must only reference agents you have created\n"
        "3. Create 3-5 agents\n"
        "4. Each agent must judge the position by a criterion the others do not use"
    )


def crew_creation_user_prompt(intro_prompt: EnvIntroPrompt) -> str:
    """The meta agent's user prompt for designing the crew.

    Args:
        intro_prompt: The EnvIntroPrompt given by the environment.

    Returns:
        The crew-creation user prompt.
    """
    return (
        f"Design your team for this task: {intro_prompt.domain_prompt}\n\n"
        "Each agent evaluates the position through one specialist lens and proposes the move "
        "that lens favours. You will see their proposals and decide the move actually played.\n\n"
        "Return the team JSON now."
    )


# ---------------------------------------------------------------------------
# The meta agent, post-initialization
# ---------------------------------------------------------------------------

def post_init_sys_prompt(intro_prompt: EnvIntroPrompt) -> str:
    """The meta agent's system prompt once the crew exists.

    Args:
        intro_prompt: The EnvIntroPrompt given by the environment.

    Returns:
        The meta agent's post-initialization system prompt.
    """
    return (
        f"You are the meta-agent supervising a team of specialist LLM agents whose task is "
        f"to{intro_prompt.domain_prompt}\n\n"

        "Each agent evaluates the position through its own specialist lens and proposes the move "
        "that lens favours. Every message you see from an agent is one move in UCI notation, or "
        f"'{ABSTAIN}', meaning that agent is putting nothing forward. Agents send no words.\n\n"

        "You see all messages: direct ones between agents and those on the common space.\n\n"

        "YOUR RESPONSIBILITY: after each cycle, decide whether the team has settled on a move. "
        "When it has, call the 'consensus' tool with that move. You choose the move that is "
        "played; the agents advise.\n\n"

        f"{intro_prompt.notation_rules}"
    )


def meta_consensus_user_prompt(env_state_prompt, num_agents: int, current_move_id: int,
                               urgency_msg: str, all_direct_msgs: str,
                               all_common_msgs: str) -> str:
    """The meta agent's per-cycle consensus-check prompt.

    Args:
        env_state_prompt: The current EnvStatePrompt.
        num_agents: Number of agents in the crew.
        current_move_id: The move under deliberation.
        urgency_msg: The countdown block, or "" outside the countdown phase.
        all_direct_msgs: Formatted direct-message transcript.
        all_common_msgs: Formatted common-space transcript.

    Returns:
        The consensus-check user prompt.
    """
    threshold = (num_agents // 2) + 1
    return (
        f"{env_state_prompt.prompt}\n\n"

        f"=====DIRECT MESSAGES=====\n"
        f"{all_direct_msgs}\n\n"
        f"=====COMMON SPACE=====\n"
        f"{all_common_msgs}\n\n"

        f"These are proposals for move {current_move_id} from your {num_agents} agents. "
        f"Ignore any that refer to earlier positions.\n\n"

        f"Call consensus once at least {threshold} agents have converged on the same move. "
        f"Abstentions do not block this, and silent agents do not block this. If the team has "
        f"converged, further cycles will not improve the move.\n\n"

        f"If proposals remain genuinely split across different moves, let deliberation continue.\n"
        f"{urgency_msg}"
    )


# ---------------------------------------------------------------------------
# Sub-agents
# ---------------------------------------------------------------------------

def agent_tools_blurb(unit_spec) -> str:
    """The 'how you communicate' half of a sub-agent's system prompt.

    Args:
        unit_spec: The environment's UnitSpec.

    Returns:
        The communication section of the agent system prompt.
    """
    return (
        "HOW YOU COMMUNICATE\n"
        f"{unit_spec.format_rules}\n\n"

        "Your tools:\n"
        "- 'talk': send one move to a specific team member\n"
        "- 'add_to_common_space': send one move to every team member\n\n"

        "A move from a teammate is the move their specialism favours. You cannot ask them why, "
        f"so judge the move itself. A '{
            unit_spec.abstain_token}' means they are putting nothing "
        "forward.\n\n"

        "When you send a move, send the one your own specialism favours after considering what "
        "you have seen. If a teammate's move is better by your criterion than anything you had, "
        "send theirs. If yours still stands, send yours again."
    )


def agent_intro(agent, intro_prompt: EnvIntroPrompt) -> str:
    """The role half of a sub-agent's system prompt.

    Args:
        agent: The SubAgent being readied.
        intro_prompt: The EnvIntroPrompt given by the environment.

    Returns:
        The role section of the agent system prompt.
    """
    return (
        f"You are part of a team whose responsibility is to {
            intro_prompt.domain_prompt}\n\n"

        f"You are \"{
            agent.name}\". Your responsibility is to {
            agent.responsibility} "
        f"In short, your role {agent.provides}.\n\n"

        f"Judge every position by your own criterion and no one else's. Teammates cover the "
        f"angles you do not; you are not responsible for theirs. A move that is wrong by your "
        f"criterion stays wrong however many teammates propose it.\n\n"

        f"You may direct message only the {
            ', the '.join(
                name for name in agent.talksto_strs)} "
        f"team members. The common space reaches everyone.\n\n"

        f"The meta-agent sees every message and decides which move is played. Your task is to "
        f"give it the best move your specialism can find.\n\n"

        f"{intro_prompt.notation_rules}"
    )


def unit_reminder(unit_spec) -> str:
    """The standing reminder appended to every agent user prompt.

    Args:
        unit_spec: The environment's UnitSpec.

    Returns:
        The reminder line.
    """
    return (f"\n\nSend exactly one move in UCI notation, or "
            f"'{unit_spec.abstain_token}'. Nothing else.")


def agent_inbox_guidance(inbox_item, unit_spec) -> str:
    """What an agent is told when it has received a message.

    Args:
        inbox_item: The InboxItem being delivered.
        unit_spec: The environment's UnitSpec.

    Returns:
        The guidance paragraph.
    """
    sender = inbox_item.other_agent_name
    if sender == "SYSTEM":
        return ("This is a SYSTEM message that you cannot reply to. "
                "Make sure to heed what it says.")
    return (
        f"{sender} proposes the move above. Judge it by your own criterion, then send the move "
        f"you now favour."
    )


def agent_no_inbox_guidance(is_reflection: bool, unit_spec) -> str:
    """What an agent is told when it has no new messages.

    Args:
        is_reflection: True for the post-inbox reflection call.
        unit_spec: The environment's UnitSpec.

    Returns:
        The guidance paragraph.
    """
    if is_reflection:
        return (
            "=====REFLECTION STEP=====\n"
            "Look at the position once more, along with any moves on the table. "
            "Send the move you favour.\n"
        )
    return "Analyse the position and send the move you favour.\n"