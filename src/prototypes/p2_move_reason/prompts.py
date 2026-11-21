from __future__ import annotations

from environment import EnvIntroPrompt

from .unit_spec import ABSTAIN, MAX_REASON_WORDS


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
        "Every message an agent sends is one move in UCI notation, plus a reason of at most "
        f"{MAX_REASON_WORDS} words for that move. An agent may instead send '{ABSTAIN}' to put "
        "nothing forward.\n"
        "Agents have two channels. The common space is where an agent posts the move it "
        "currently backs. A direct message is for challenging one teammate's move on a specific "
        "point. The 'talks' field decides who can challenge whom, so give an agent direct "
        "channels to the teammates whose blind spots it is placed to catch.\n"
        "Write each responsibility so it can be carried out in that space. A role must not "
        "require an agent to produce a long argument, a list, or a report, because it has room "
        "for one or two sentences and no more.\n\n"

        "TEAM DESIGN:\n"
        "Give each agent one clear evaluative angle that would change which move is played. "
        "Three or four sharply distinct angles beat seven overlapping ones.\n"
        "Ask yourself: if this agent were removed, would the team ever play a different move? "
        "If not, do not create it.\n"
        "The reason field is what lets one agent change another's mind, so favour angles whose "
        "verdicts can be stated compactly and checked by a teammate.\n\n"

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
        "that lens favours, with a short reason. You will see their proposals and decide the "
        "move actually played.\n\n"
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

        "Each agent evaluates the position through its own specialist lens. Every message you "
        "see from an agent is one move in UCI notation together with a short reason for it, or "
        f"'{ABSTAIN}', meaning that agent is putting nothing forward.\n\n"

        "You see all messages: direct ones between agents and those on the common space.\n\n"

        "YOUR RESPONSIBILITY: after each cycle, decide whether the team has settled on a move. "
        "When it has, call the 'consensus' tool with that move. Weigh the reasons given, not "
        "only the count of proposals: a single agent with a concrete refutation may be right "
        "against three who agree. You choose the move that is played; the agents advise.\n\n"

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

        f"These are proposals for move {current_move_id} from your {num_agents} agents, each "
        f"with the reason its author gave. Ignore any that refer to earlier positions.\n\n"

        f"Call consensus once at least {threshold} agents have converged on the same move and "
        f"no unanswered objection stands against it. Abstentions and silent agents do not block "
        f"this. If the team has converged, further cycles will not improve the move.\n\n"

        f"If proposals remain genuinely split, or an agent has raised a concrete problem with "
        f"the leading move that no one has addressed, let deliberation continue.\n"
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

        "Your two channels do different jobs:\n"
        "- 'add_to_common_space': post the move you currently back, for the whole team and the "
        "meta-agent. This is your standing position.\n"
        "- 'talk': challenge one teammate's move on a specific point, or answer a challenge to "
        "yours. Use this whenever you disagree with a move a teammate has proposed: the "
        "disagreement is between the two of you and is settled there, not broadcast.\n\n"

        "Do not direct message to agree. If a teammate's move is right by your criterion, post "
        "that move to the common space as your own position - that is how agreement is "
        "registered. A direct message that only signals assent adds nothing the common space "
        "does not already carry.\n\n"

        "A message from a teammate is the move their specialism favours and why. Judge both: "
        "a reason that does not survive your own criterion is grounds to keep your move, and a "
        f"reason that defeats your move is grounds to change it. A '{unit_spec.abstain_token}' "
        "means they are putting nothing forward.\n\n"

        "Spend your reason on what a teammate could act on: the concrete thing you saw that "
        "they may not have. Do not restate your role, and do not describe the move in words."
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
        f"You are part of a team whose responsibility is to {intro_prompt.domain_prompt}\n\n"

        f"You are \"{agent.name}\". Your responsibility is to {agent.responsibility} "
        f"In short, your role {agent.provides}.\n\n"

        f"Judge every position by your own criterion and no one else's. Teammates cover the "
        f"angles you do not; you are not responsible for theirs. A move that is wrong by your "
        f"criterion stays wrong however many teammates propose it.\n\n"

        f"Hold your move until something defeats it. Change it when a teammate gives a reason "
        f"that shows it fails by your own criterion - not because opinion has moved, not "
        f"because a move has been repeated, and not because a later cycle feels more informed "
        f"than an earlier one. Your first reading of the position is evidence too. If you "
        f"cannot name what defeated your previous move, you have no reason to abandon it.\n\n"

        f"You may direct message only the {', the '.join(name for name in agent.talksto_strs)} "
        f"team members. The common space reaches everyone.\n\n"

        f"The meta-agent sees every message and decides which move is played. Your task is to "
        f"give it the best move your specialism can find, and the one thing it should know "
        f"about that move.\n\n"

        f"{intro_prompt.notation_rules}"
    )


def unit_reminder(unit_spec) -> str:
    """The standing reminder appended to every agent user prompt.

    Args:
        unit_spec: The environment's UnitSpec.

    Returns:
        The reminder line.
    """
    return (f"\n\nSend one move in UCI notation (or '{unit_spec.abstain_token}') "
            f"and a reason of at most {MAX_REASON_WORDS} words.")


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
        f"{sender} proposes the move above, for the reason given. Test that reason against your "
        f"own criterion, then send one message:\n"
        f"- If their reason defeats the move you were backing, post their move to the common "
        f"space as your own position, with what defeated yours.\n"
        f"- If it does not, use 'talk' to send {sender} the move you back and the specific point "
        f"their reason fails on. Take the disagreement to them directly rather than to the "
        f"whole team."
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
            "Look at the position once more, along with the moves and reasons on the table. "
            "Send the move you back. If it differs from the one you backed last time, your "
            "reason must name what defeated the earlier move; otherwise send that move again.\n"
        )
    return ("Analyse the position and post the move you back to the common space, "
            "with your reason.\n")