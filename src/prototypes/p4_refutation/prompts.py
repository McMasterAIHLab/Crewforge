
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

        "Every responsibility must say when the agent has nothing to offer. An angle that "
        f"applies to every position is not an angle; it will produce an endorsement of whatever "
        f"is already on the table. Write each role so that '{ABSTAIN}' is the correct output "
        "whenever the position falls outside what that agent can judge, and say so in the "
        "responsibility itself.\n\n"

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

        f"Convergence is not by itself evidence that the move is right. Agents see each other's "
        f"proposals, so a move can accumulate endorsements without any of them having been "
        f"reached independently. Before committing, check that at least one reason names a "
        f"concrete feature of this position rather than restating an earlier agent's reason in "
        f"other words. Where every reason is a paraphrase of the first, the team has agreed "
        f"without having checked anything.\n\n"

        f"If proposals remain genuinely split, or a challenge in the direct messages has gone "
        f"unanswered, let deliberation continue. Challenges are where disagreements are argued, "
        f"so read them before counting the common space.\n"
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

        "Your two channels do different jobs, and they are not alternatives:\n"
        "- 'add_to_common_space': the move you currently back. This is your standing position, "
        "and it is what the meta-agent counts. Post it every time you are asked.\n"
        "- 'talk': a challenge to one named teammate, saying where their reason fails. This "
        "does not replace your standing position; it is sent in addition to it.\n\n"

        "So: when you agree with the teammate who wrote to you, send one tool call - post their "
        "move to the common space as your own position. When you disagree, send two - post your "
        "move to the common space AND talk to that teammate with the point their reason fails "
        "on. Disagreeing without challenging leaves them free to keep a move you believe is "
        "wrong, and leaves the team with two positions and no argument between them.\n\n"

        "When a move reaches you, do not ask whether you agree with it. Ask what refutes it. "
        "Look for the concrete failure your criterion is placed to find, and state it if you "
        "find one. Endorse only when you have looked for a refutation and cannot construct "
        "one - and then say what you checked, not that the move is good.\n\n"

        "How many teammates back a move is not evidence about the move. A move endorsed by "
        "everyone before you is exactly as strong as it was when it was first proposed, and "
        "you are the only one who can apply your criterion to it. Treat each move on the board "
        f"in front of you. A '{unit_spec.abstain_token}' from a teammate means they are putting "
        "nothing forward, and it is a legitimate answer from you as well: if your criterion "
        "has nothing to say about this position, say nothing rather than endorsing to fill "
        "the space.\n\n"

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
        f"The move above is on the table. Apply your criterion to it and look for a refutation - "
        f"not for whether you agree with {sender}, but for whether the move survives.\n\n"
        f"IT SURVIVES - you looked for a refutation and could not construct one. One tool call: "
        f"add_to_common_space with that move, and say what you checked.\n\n"
        f"IT FAILS - you found the concrete point on which it breaks. Two tool calls, both "
        f"required:\n"
        f"  1. add_to_common_space with the move you back instead, so your position is on "
        f"record.\n"
        f"  2. talk to {sender} with that move and the point on which theirs fails.\n\n"
        f"A refutation posted only to the common space is not a challenge: {sender} cannot "
        f"answer a point that was never put to them."
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
