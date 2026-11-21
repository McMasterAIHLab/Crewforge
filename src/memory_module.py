"""
Memory Module

This module manages shared memory and communication between agents in the multi-agent system.
It handles internal agent thoughts, direct agent-to-agent communication, and common space
messaging with proper timestamping and move-id tracking.

The memory is structured using XML-like tags to organize different communication channels:
- <agent-{id}-internal>: Private thoughts for each agent
- <agent-{id}-to-{id}>: Direct communication channels between agents
- <common-space>: Shared communication space accessible to all agents

All messages are prepended with metadata: [Agent ID][Timestamp][Move ID]
"""

# memory_module.py

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

from typing import Dict, List, Optional

from sub_agent_module import SubAgent


@dataclass
class InboxMessage:
    """Represents one message in a chain of communications (represented by InboxItem)
    """
    sender_id: str
    sender_name: str
    text: str
    timestamp: str
    move_id: int
    is_from_me: bool


@dataclass
class InboxItem:
    """Represents a chain of communication, built with InboxMessages as the individual messages"""
    other_agent_id: str
    other_agent_name: str
    source: str
    new_message: InboxMessage
    history: List[InboxMessage]

    def format(self) -> str:
        """Format an inbox item into a readable message display.
        """
        msg_display = f"From {self.other_agent_name}:\n{self.new_message.text}\n"

        if self.history:
            msg_display += f"\nConversation history:\n"

            current_move = None
            for msg in self.history:
                if msg.move_id != current_move:
                    current_move = msg.move_id
                    label = "Pre-game (Move (-1))" if current_move == - \
                        1 else f"Move {current_move}"
                    msg_display += f"--- {label} ---\n"
                sender = "You" if msg.is_from_me else msg.sender_name
                msg_display += f"  [{sender}]: {msg.text}\n"

        return msg_display


class AgentRegistry:
    """
    Registry for storing and retrieving sub-agents by name or ID.

    This class provides a central lookup mechanism for agents in the multi-agent
    system, allowing efficient access by either unique ID or human-readable name.

    Attributes:
        _agents: Dictionary mapping agent IDs to SubAgent objects
        _name_to_id: Dictionary mapping agent names to agent IDs
    """

    def __init__(self):
        """
        Initialize an empty agent registry.
        """
        self._agents = {}  # id -> agent
        self._name_to_id = {}  # name -> id

    def register(self, agent: SubAgent):
        """
        Register a new agent in the registry.

        Args:
            agent: SubAgent object to register
        """
        self._agents[agent.id] = agent
        self._name_to_id[agent.name] = agent.id

    def get_by_id(self, agent_id: str) -> Optional[SubAgent]:
        """
        Retrieve an agent by their unique ID.

        Args:
            agent_id: The unique identifier of the agent

        Returns:
            SubAgent object if found, None otherwise
        """
        return self._agents.get(agent_id)

    def get_by_name(self, name: str) -> Optional[SubAgent]:
        """
        Retrieve an agent by their name.

        Args:
            name: The human-readable name of the agent

        Returns:
            SubAgent object if found, None otherwise
        """
        agent_id = self._name_to_id.get(name)
        return self._agents.get(agent_id) if agent_id else None

    def id_from_name(self, name: str) -> Optional[str]:
        """
        Get an agent's ID from their name.

        Args:
            name: The human-readable name of the agent

        Returns:
            Agent ID string if found, None otherwise
        """
        return self._name_to_id.get(name)

    def name_from_id(self, agent_id: str) -> Optional[str]:
        """
        Get an agent's name from their ID.

        Args:
            agent_id: The unique identifier of the agent

        Returns:
            Agent name string if found, None otherwise
        """
        agent = self._agents.get(agent_id)
        return agent.name if agent else None

    def ids_from_names(self, names: list[str]) -> list[str]:
        """
        Convert a list of agent names to their corresponding IDs.

        Args:
            names: List of agent names to convert

        Returns:
            List of agent IDs (excludes names that weren't found -- they are None)
        """
        return [self.id_from_name(name) for name in names if self.id_from_name(name)]  # type: ignore

    def all_agents(self) -> list[SubAgent]:
        """
        Get all registered agents.

        Returns:
            List of all SubAgent objects in the registry
        """
        return list(self._agents.values())


class Memory:
    """Contains the XML memory for communications between agents, as well as operations 
    to store and retrieve.
    """

    def __init__(self, xml_path: Optional[str], every_other: bool, limit: int):
        """Initialize a fresh memory. This must be called before initializing any other
        MAS component (i.e. it should be one of the first things in the MAS init).

        Args:
            xml_path: Optional path string for saving the XML to file. Defaults 
                      to ./unified_memory.xml (local dir.).
            every_other: if the environment this system is operating in has an opponent
                         that makes moves every other turn (specifically ODD turns).
            limit: limit the number of entries returned by the view...() functions. Some 
                   functions have this parameter as well, if not supplied there, the value
                   supplied here acts as a default
        """
        # store given path if it exists
        self.xml_path = xml_path if xml_path is not None else "unified_memory.xml"

        # initialize the XML tree. These shouldn't be accessed directly
        self._ROOT = ET.Element("unified-memory")

        # add the common communication section so that we have a node to make the root
        ET.SubElement(self._ROOT, "common-comms")
        self._TREE = ET.ElementTree(self._ROOT)

        # initialize clean registery. This must be done before
        self.registry = AgentRegistry()

        self.every_other = every_other

        self.MOVE_HISTORY_LIMIT = limit

    def _utc_ts(self) -> str:
        """
        Get the current UTC timestamp in ISO format.

        Returns:
            ISO-formatted timestamp string with second precision (e.g., '2024-01-15T10:30:45+00:00')
        """
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _get_recent_system_move_ids(self, current_move_id: int, limit: Optional[int] = None) -> set:
        """
        Get the set of recent system move IDs to include in conversation history.

        Returns the last `limit` system moves up to and including current_move_id.

        Args:
            current_move_id: the current move being deliberated
            limit: how many system moves to include, defaults to MOVE_HISTORY_LIMIT.

        Returns:
            Set of move IDs to include (always includes -1 for pre-game)
        """
        # Always include pre-game messages
        allowed_moves = {-1}

        if limit is None:
            limit = self.MOVE_HISTORY_LIMIT

        # Find the last N system moves (even numbers if every_other=True) up to current_move_id
        # Current move counts as one of the N, unless every_other=True and it's not even
        sys_moves = []

        if self.every_other:
            move = current_move_id if current_move_id % 2 == 0 else current_move_id - 1
        else:
            move = current_move_id

        while move >= 0 and len(sys_moves) < limit:
            sys_moves.append(move)
            move -= (2 if self.every_other else 1)

        allowed_moves.update(sys_moves)
        return allowed_moves

    def _get_tree(self) -> ET.ElementTree:
        """Getter for access to the XML object.

        Returns:
            ElementTree object containing the memory structure
        """
        return self._TREE

    def store_internal(self, agent_id: str, text: str, move_id: int):
        """
        Store a private internal thought for an agent.

        Internal thoughts are stored in agent-specific XML tags and are only
        visible to that agent. They are useful for tracking an agent's reasoning
        process without sharing it with other agents.

        Args:
            agent_id: Unique identifier of the agent storing the thought
            text: The thought content to store
            move_id: move ID to associate with this thought

        Raises:
            ValueError: If agent_id or text is empty or not a string
        """
        if not agent_id or not isinstance(agent_id, str) or len(agent_id) == 0:
            raise ValueError("agent_id must be a non-empty string")
        if not text or not isinstance(text, str) or len(text) == 0:
            raise ValueError("text must be a non-empty string")

        tree = self._get_tree()
        root = tree.getroot()

        # if there is no root element in the tree, error out
        if root is None:
            raise ValueError("memory root not found")

        tag_name = f"internal-thought-agent-{agent_id}"
        agent_tag = root.find(tag_name)

        # if we couldn't find the tag for this agent's internal thoughts then
        # this must be the first time calling this for this agent, so create it
        if agent_tag is None:
            agent_tag = ET.SubElement(root, tag_name)

        # add the actual text entry
        entry = ET.SubElement(agent_tag, "entry", {"ts": self._utc_ts()})
        if move_id:
            entry.set("move_id", str(move_id))
        entry.text = text

    def store_direct(self, agent_id_from: str, agent_id_to: str, text: str, move_id: int):
        """
        Send a direct message between two agents.

        Direct messages are stored in a shared XML tag that is accessible to both
        participating agents. The tag name is deterministically generated from both
        agent IDs (sorted alphabetically) to ensure A->B and B->A messages appear
        in the same conversation thread.

        Args:
            agent_id_from: ID of the sending agent
            agent_id_to: ID of the receiving agent
            text: Message content to send
            move_id: move ID to associate with this message

        Raises:
            ValueError: If either agent ID is empty, if both IDs are the same,
                        or if text is empty or not a string
        """
        # basic validation
        if not agent_id_from or not agent_id_to or len(agent_id_to) == 0:
            raise ValueError("agent IDs must be non-empty")
        if agent_id_from == agent_id_to:
            raise ValueError("direct comms require two different agents")
        if not text or not isinstance(text, str) or len(agent_id_from) == 0:
            raise ValueError("text must be a non-empty string")

        tree = self._get_tree()
        root = tree.getroot()

        # if there is no root element in the tree, error out
        if root is None:
            raise ValueError("memory root not found")

        # ensure stable ordering so A-B and B-A map to the same tag
        a, b = sorted([str(agent_id_from), str(agent_id_to)])
        tag_name = f"comms-agent-{a}-and-{b}"

        # find or create the shared comms tag
        comms_tag = root.find(tag_name)
        if comms_tag is None:
            comms_tag = ET.SubElement(root, tag_name)

        # append message entry
        entry = ET.SubElement(
            comms_tag,
            "entry",
            {
                "from": str(agent_id_from),
                "ts": self._utc_ts(),
            },
        )

        entry.set("move_id", str(move_id))

        entry.text = text

    def store_common(self, agent_id_from: str, text: str, move_id: int):
        """
        Broadcast a message to the common space visible to all agents.

        Common space messages are stored in the <common-comms> XML tag and are
        visible to all agents in the system. This is useful for team-wide
        announcements or discussions.

        Args:
            agent_id_from: ID of the sending agent (or "SYSTEM" for system messages)
            text: Message content to broadcast
            move_id: move ID to associate with this message

        Raises:
            ValueError: If agent_id_from or text is empty or not a string
        """
        if not agent_id_from or not isinstance(agent_id_from, str):
            raise ValueError("agent_id_from must be a non-empty string")
        if not text or not isinstance(text, str):
            raise ValueError("text must be a non-empty string")

        tree = self._get_tree()
        root = tree.getroot()

        # if there is no root element in the tree, error out
        if root is None:
            raise ValueError("memory root not found")

        # common-comms should exist (created in init), so something
        # has definitely gone wrong here. NOTE: this is a change in
        # behaviour from before, where originally this would've just
        # create the tag anew
        common_tag = root.find("common-comms")
        if common_tag is None:
            raise ValueError("common-comms doesn't exist")

        entry = ET.SubElement(
            common_tag,
            "entry",
            {
                "from": agent_id_from,
                "ts": self._utc_ts(),
            },
        )

        entry.set("move_id", str(move_id))

        entry.text = text

    def view_internal(self, agent_id: str) -> str:
        """
        Retrieve an agent's internal thoughts as XML string.

        Args:
            agent_id: The unique identifier of the agent

        Returns:
            XML string containing all internal thoughts for this agent,
            or an empty tag if no thoughts exist

        Raises:
            ValueError: If agent_id is empty or not a string
        """
        if not agent_id or not isinstance(agent_id, str):
            raise ValueError("agent_id must be a non-empty string")

        tree = self._get_tree()
        root = tree.getroot()

        # if there is no root element in the tree, error out
        if root is None:
            raise ValueError("memory root not found")

        tag_name = f"internal-thought-agent-{agent_id}"
        elem = root.find(tag_name)

        # if the tag doesn't exist yet, return an empty tag (consistent output)
        # TODO: Not sure if we want this behaviour ....
        if elem is None:
            elem = ET.Element(tag_name)

        return ET.tostring(elem, encoding="unicode")

    def view_inbox(self, agent: SubAgent, cur_move_id: int, last_ts: str) -> List[InboxItem]:
        """
        Get new messages for an agent since their last inbox check.

        Scans both direct message channels and the common space for messages
        that were sent after the agent's last check timestamp. Returns inbox
        items with the new message and conversation history for context.

        Args:
            agent: The SubAgent checking their inbox
            cur_move_id: Current move ID being deliberated
            last_ts: ISO timestamp of agent's last inbox check (messages after
                    this time are considered new)

        Returns:
            List of inbox items

        Raises:
            ValueError: if XML tree is missing elements
        """
        tree = self._get_tree()
        root = tree.getroot()

        # if there is no root element in the tree, error out
        if root is None:
            raise ValueError("memory root not found")

        inbox_items = []

        # --- 1. Process Direct Messages (Private Channels) ---
        needle = f"-{agent.id}-and-"
        needle2 = f"-and-{agent.id}"

        for child in list(root):
            if not isinstance(child.tag, str) or not child.tag.startswith("comms-agent-"):
                continue
            if not (needle in child.tag or child.tag.endswith(needle2)):
                continue

            # Identify the specific person on the other end of this private line
            parts = child.tag.replace("comms-agent-", "").split("-and-")
            other_id = parts[1] if parts[0] == agent.id else parts[0]
            other_name = self.registry.name_from_id(
                other_id) or other_id  # Fallback to ID if name not found

            inbox_items.extend(self._process_comm_element(
                child, agent.id, cur_move_id, last_ts,
                other_id=other_id, other_name=other_name, source="direct"
            ))

        # --- 2. Process Common Space (Public Channel) ---
        common_elem = root.find("common-comms")
        if common_elem is not None:
            # In the common space, 'other_id' is just a label for the channel,
            # but the 'from_id' inside msg_data will still be the actual sender.
            inbox_items.extend(self._process_comm_element(
                common_elem, agent.id, cur_move_id, last_ts,
                other_id="COMMON", other_name="Common Space", source="common"
            ))

        inbox_items.sort(key=lambda item: item.new_message.timestamp)
        return inbox_items

    def _process_comm_element(self, elem, agent_id: str, cur_move_id: int, last_ts: str,
                              other_id: str, other_name: str, source: str) -> List[InboxItem]:
        """
        Process a communication element (direct or common) to extract inbox items.

        Identifies new messages (from current move, after last_ts, not from self)
        and builds conversation history grouped by move ID. History is filtered
        to only include recent system moves per MOVE_HISTORY_LIMIT.

        Args:
            elem: XML element containing the conversation entries
            agent_id: ID of the agent checking their inbox
            cur_move_id: Current move ID being deliberated
            last_ts: Timestamp of agent's last inbox check
            other_id: ID of conversation partner (or channel identifier)
            other_name: Display name for the conversation partner
            source: 'direct' for DMs, 'common' for common space

        Returns:
            List of inbox item dicts, one per new message found
        """
        # Messages in allowed moves (for history)
        all_messages: List[InboxMessage] = []
        new_messages: List[InboxMessage] = []

        # Get allowed moves for history filtering
        allowed_moves = self._get_recent_system_move_ids(cur_move_id)

        # Collect messages from this conversation element
        for entry in elem.findall('entry'):
            msg_move_id = int(entry.get('move_id', -1))

            # This correctly captures the actual author from the XML attribute
            sender_id = entry.get('from')
            sender_name = self.registry.name_from_id(
                sender_id) or sender_id  # Fallback to ID if name not found
            timestamp = entry.get('ts', '')
            text = entry.text or ""

            msg = InboxMessage(
                sender_id=sender_id,
                sender_name=sender_name,
                text=text,
                timestamp=timestamp,
                move_id=msg_move_id,
                is_from_me=(sender_id == agent_id),
            )

            # Only include messages from allowed moves in history
            if msg_move_id in allowed_moves:
                all_messages.append(msg)

            # Identify if this message should trigger an inbox notification
            # (only messages from current move that are new)
            if msg_move_id == cur_move_id and sender_id != agent_id and (not last_ts or timestamp > last_ts):
                new_messages.append(msg)

        all_messages.sort(key=lambda m: m.timestamp)

        # For common space: only notify about the LATEST new message
        # (previous messages will be in the history, avoiding compounding notifications)
        # For direct messages: notify about each new message separately
        if source == "common" and len(new_messages) > 1:
            new_messages.sort(key=lambda m: m.timestamp)
            new_messages = [new_messages[-1]]  # Keep only the latest

        inbox_items: List[InboxItem] = []
        for new_msg in new_messages:
            # Build conversation history: all messages EXCEPT the current new_message
            # Group by move_id with labels, similar to view_fmt_direct/common

            history = [
                msg for msg in all_messages if not (msg.timestamp == new_msg.timestamp and msg.sender_id == new_msg.sender_id)
            ]

            # Determine the display name for the notification
            # If it's the common space, show who actually sent it.
            # Otherwise, use the fixed partner name from the direct message thread.
            display_name = new_msg.sender_name if source == "common" else other_name
            display_id = new_msg.sender_id if source == "common" else other_id

            inbox_items.append(InboxItem(
                other_agent_id=display_id,
                other_agent_name=display_name,
                source=source,
                new_message=new_msg,
                history=history,
            ))

        return inbox_items

    def view_direct(self, agent_id: str) -> str:
        """
        Retrieve all direct message conversations involving an agent as XML.

        Returns all <comms-agent-A-and-B> elements where the agent is either
        A or B, wrapped in a container element for valid XML output.

        Args:
            agent_id: The unique identifier of the agent

        Returns:
            XML string containing all direct message conversations

        Raises:
            ValueError: If agent_id is empty or not a string
        """
        if not agent_id or not isinstance(agent_id, str):
            raise ValueError("agent_id must be a non-empty string")

        tree = self._get_tree()
        root = tree.getroot()

        # if there is no root element in the tree, error out
        if root is None:
            raise ValueError("memory root not found")

        # Wrap multiple comms tags in a container so the result is valid XML
        container = ET.Element("direct-comms-view")
        needle = f"-{agent_id}-and-"  # matches: comms-agent-{agent_id}-and-{X}
        needle2 = f"-and-{agent_id}"  # matches: comms-agent-{X}-and-{agent_id}

        for child in list(root):
            if not isinstance(child.tag, str):
                continue
            if child.tag.startswith("comms-agent-") and (needle in child.tag or child.tag.endswith(needle2)):
                # Append a copy so callers can't mutate the main tree
                container.append(ET.fromstring(
                    ET.tostring(child, encoding="utf-8")))

        return ET.tostring(container, encoding="unicode")

    def view_fmt_direct(self, agent: SubAgent, current_move_id: Optional[int] = None) -> str:
        """
        Get formatted direct message conversations for an agent.

        Returns a human-readable string of all direct message conversations
        involving this agent, grouped by conversation partner and move ID.
        Messages are formatted as "[You]:" or "[OtherName]:" for clarity.

        Args:
            agent: The SubAgent whose conversations to retrieve
            current_move_id: If provided, only shows messages from the last
                            MOVE_HISTORY_LIMIT White moves (to limit prompt size)

        Returns:
            Formatted string of all direct message conversations, or "(empty)"
            if no conversations exist
        """
        tree = self._get_tree()
        root = tree.getroot()

        # if there is no root element in the tree, error out
        if root is None:
            raise ValueError("memory root not found")

        needle = f"-{agent.id}-and-"
        needle2 = f"-and-{agent.id}"

        # Determine which moves to include
        if current_move_id is not None:
            allowed_moves = self._get_recent_system_move_ids(
                current_move_id=current_move_id)
        else:
            allowed_moves = None  # No filtering

        output = []

        for child in list(root):
            if not isinstance(child.tag, str):
                continue
            if child.tag.startswith("comms-agent-") and (needle in child.tag or child.tag.endswith(needle2)):
                parts = child.tag.replace("comms-agent-", "").split("-and-")
                other_id = parts[1] if parts[0] == agent.id else parts[0]
                other_name = self.registry.name_from_id(
                    other_id) or other_id  # Fallback to ID if name not found

                # Group entries by move_id
                by_move = {}
                for entry in child.findall('entry'):
                    move_id = int(entry.get('move_id', 0))
                    # Filter by allowed moves if specified
                    if allowed_moves is not None and move_id not in allowed_moves:
                        continue
                    if move_id not in by_move:
                        by_move[move_id] = []
                    by_move[move_id].append(entry)

                # Only add conversation header if there are messages after filtering
                if by_move:
                    output.append(f"Conversation with {other_name}:")
                    # Format by move (sorted numerically)
                    for move_id in sorted(by_move.keys()):
                        label = "Pre-game (Move (-1))" if move_id == - \
                            1 else f"Move {move_id}"
                        output.append(f"--- {label} ---")
                        for entry in by_move[move_id]:
                            from_id = entry.get('from')
                            from_name = self.registry.name_from_id(
                                from_id) or from_id
                            text = entry.text or ""
                            # Consistent format: [You] or [OtherName]
                            sender = "You" if from_id == agent.id else from_name
                            output.append(f"  [{sender}]: {text}")

        if len(output) == 0:
            output.append("(empty)")
        return '\n'.join(output)

    def view_common(self) -> str:
        """
        Retrieve the common space messages as XML.

        Returns:
            XML string containing the <common-comms> element with all
            broadcast messages, or an empty element if none exist

        Raises:
            ValueError: if missing elements from the XML tree
        """
        tree = self._get_tree()
        root = tree.getroot()

        # if there is no root element in the tree, error out
        if root is None:
            raise ValueError("memory root not found")

        elem = root.find("common-comms")
        if elem is None:
            raise ValueError("common-comms tag not found")

        return ET.tostring(elem, encoding="unicode")

    def view_fmt_common(self, current_move_id: Optional[int] = None) -> str:
        """
        Get formatted common space messages.

        Returns a human-readable string of all common space (broadcast) messages,
        grouped by move ID. Messages are formatted as "[SenderName]: message".

        Args:
            current_move_id: If provided, only shows messages from the last
                            MOVE_HISTORY_LIMIT White moves (to limit prompt size)

        Returns:
            Formatted string of common space messages, or "(empty)" if none exist

        Raises:
            ValueError: if missing elements from XML tree
        """
        tree = self._get_tree()
        root = tree.getroot()

        # if there is no root element in the tree, error out
        if root is None:
            raise ValueError("memory root not found")

        elem = root.find("common-comms")
        if elem is None:
            raise ValueError("common-comms tag not found")

        # Determine which moves to include
        if current_move_id is not None:
            allowed_moves = self._get_recent_system_move_ids(current_move_id)
        else:
            allowed_moves = None  # No filtering

        output = []

        # Group entries by move_id
        by_move = {}
        for entry in elem.findall('entry'):
            move_id = int(entry.get('move_id', 0))
            # Filter by allowed moves if specified
            if allowed_moves is not None and move_id not in allowed_moves:
                continue
            if move_id not in by_move:
                by_move[move_id] = []
            by_move[move_id].append(entry)

        # Format by move (sorted numerically)
        for move_id in sorted(by_move.keys()):
            label = "Pre-game (Move (-1))" if move_id == - \
                1 else f"Move {move_id}"
            output.append(f"--- {label} ---")
            for entry in by_move[move_id]:
                from_id = entry.get('from')
                from_name = self.registry.name_from_id(from_id) or from_id
                text = entry.text or ""
                output.append(f"  [{from_name}]: {text}")

        if len(output) == 0:
            output.append("(empty)")
        return '\n'.join(output)

    def export_direct_messages(self) -> list:
        """
        Export all direct messages for data capture and analysis.

        Extracts all direct message entries from the XML memory and returns
        them as a list of dictionaries suitable for JSON serialization or
        data analysis.

        Returns:
            List of dicts, each containing:
            - channel: XML tag name of the conversation
            - from_agent_id: ID of the sending agent
            - from_agent_name: Name of the sending agent
            - to_agent_id: ID of the receiving agent
            - to_agent_name: Name of the receiving agent
            - timestamp: ISO timestamp of the message
            - move_id: Move ID when message was sent
            - text: Message content

        Raises:
            ValueError: if missing elements from XML tree
        """
        tree = self._get_tree()
        root = tree.getroot()

        # if there is no root element in the tree, error out
        if root is None:
            raise ValueError("memory root not found")

        messages = []

        for child in list(root):
            if not isinstance(child.tag, str):
                continue
            if not child.tag.startswith("comms-agent-"):
                continue

            # Parse channel name to get agent IDs
            # Format: comms-agent-{A}-and-{B}
            parts = child.tag.replace("comms-agent-", "").split("-and-")
            if len(parts) != 2:
                continue

            agent_a, agent_b = parts

            for entry in child.findall('entry'):
                from_id = entry.get('from')
                to_id = agent_b if from_id == agent_a else agent_a

                if from_id is not None:
                    messages.append({
                        "channel": child.tag,
                        "from_agent_id": from_id,
                        "from_agent_name": self.registry.name_from_id(from_id),
                        "to_agent_id": to_id,
                        "to_agent_name": self.registry.name_from_id(to_id),
                        "timestamp": entry.get('ts'),
                        "move_id": entry.get('move_id'),
                        "text": entry.text or "",
                    })

        return messages

    def export_common_space(self) -> list:
        """
        Export all common space messages for data capture and analysis.

        Extracts all broadcast messages from the common space and returns
        them as a list of dictionaries suitable for JSON serialization.

        Returns:
            List of dicts, each containing:
            - from_agent_id: ID of the sending agent
            - from_agent_name: Name of the sending agent
            - timestamp: ISO timestamp of the message
            - move_id: Move ID when message was sent
            - text: Message content

        Raises:
            ValueError: if missing elements in the XML tree
        """
        tree = self._get_tree()
        root = tree.getroot()

        # if there is no root element in the tree, error out
        if root is None:
            raise ValueError("memory root not found")

        elem = root.find("common-comms")
        if elem is None:
            raise ValueError("common-comms tag not found")

        messages = []

        for entry in elem.findall('entry'):
            from_id = entry.get('from')
            if from_id is not None:  # NOTE: Not sure when this would happen, doing it for type-safety
                messages.append({
                    "from_agent_id": from_id,
                    "from_agent_name": self.registry.name_from_id(from_id),
                    "timestamp": entry.get('ts'),
                    "move_id": entry.get('move_id'),
                    "text": entry.text or "",
                })

        return messages

    def export_internal_thoughts(self) -> dict:
        """
        Export all agents' internal thoughts for data capture and analysis.

        Extracts all internal thought entries and organizes them by agent ID.

        Returns:
            Dict mapping agent_id to:
            - agent_name: Human-readable name of the agent
            - entries: List of thought dicts with timestamp, move_id, text

        Raises:
            ValueError: if missing elements in XML tree
        """
        tree = self._get_tree()
        root = tree.getroot()

        # if there is no root element in the tree, error out
        if root is None:
            raise ValueError("memory root not found")

        thoughts = {}

        for child in list(root):
            if not isinstance(child.tag, str):
                continue
            if not child.tag.startswith("internal-thought-agent-"):
                continue

            # Parse agent ID from tag
            agent_id = child.tag.replace("internal-thought-agent-", "")

            agent_thoughts = []
            for entry in child.findall('entry'):
                agent_thoughts.append({
                    "timestamp": entry.get('ts'),
                    "move_id": entry.get('move_id'),
                    "text": entry.text or "",
                })

            thoughts[agent_id] = {
                "agent_name": self.registry.name_from_id(agent_id),
                "entries": agent_thoughts,
            }

        return thoughts

    def export_full_memory_state(self) -> dict:
        """
        Export complete memory module state for audit trail and analysis.

        Provides a comprehensive snapshot of the entire memory system including
        all agents, their configurations, and all communications. Useful for
        debugging, data analysis, and experiment logging.

        Returns:
            Dict containing:
            - agents: List of agent info dicts (id, name, responsibility, etc.)
            - direct_messages: All direct messages (from export_direct_messages)
            - common_space: All broadcast messages (from export_common_space)
            - internal_thoughts: All internal thoughts (from export_internal_thoughts)
            - message_counts: Summary counts of direct and common messages
        """
        return {
            "agents": [
                {
                    "id": agent.id,
                    "name": agent.name,
                    "responsibility": agent.responsibility,
                    "provides": agent.provides,
                    "talks_to_ids": agent.talksto,
                    "talks_to_names": agent.talksto_strs,
                }
                for agent in self.registry.all_agents()
            ],
            "direct_messages": self.export_direct_messages(),
            "common_space": self.export_common_space(),
            "internal_thoughts": self.export_internal_thoughts(),
            "message_counts": {
                "direct": len(self.export_direct_messages()),
                "common": len(self.export_common_space()),
            }
        }

    def get_all_direct_messages_formatted(self, current_move_id: int) -> str:
        """Get ALL direct messages between ALL agents, formatted for visibility.
        Filtered to recent moves based on the configured move history limit.

        Args:
            current_move_id: the current move id for filtering.

        Returns:
            A formatted string of all direct messages grouped by move.
        """
        all_dms = self.export_direct_messages()

        if not all_dms:
            return "(No direct messages yet)"

        allowed_moves = self._get_recent_system_move_ids(current_move_id)

        # format is [move_id, dms list for that move]
        by_move: Dict[int, list] = {}
        for dm in all_dms:
            raw_move_id = dm.get("move_id")
            if raw_move_id is None:  # NOTE: this is a change from before, which would default to -1
                continue

            move_id = int(raw_move_id)
            if move_id not in allowed_moves:
                continue

            if move_id not in by_move:
                by_move[move_id] = []

            by_move[move_id].append(dm)

        if not by_move:
            return "(No direct messages in recent moves)"

        formatted_lines = []
        for move_id in sorted(by_move.keys()):
            label = f"Pre-game (Move ({move_id}))" if move_id < 0 else f"Move {move_id}"
            formatted_lines.append(f"--- {label} ---")

            for dm in by_move[move_id]:
                sender = dm.get("from_agent_name") or dm.get("from_id")
                recipient = dm.get("to_agent_name") or dm.get("to_id")
                text = dm.get("text")

                # NOTE: this is another change from before, which would default to "Unknown"
                if sender is None or recipient is None or text is None:
                    continue

                formatted_lines.append(f"  [{sender} -> {recipient}]: {text}")

        if not formatted_lines:
            return "(No direct messages in recent moves)"

        return "\n".join(formatted_lines)

    def _indent(self, elem, level=0):
        """
        Add indentation to XML elements for pretty-printing.

        Recursively adds newlines and spaces to XML elements to produce
        human-readable formatted output when saved.

        Args:
            elem: XML Element to indent
            level: Current indentation level (default 0 for root)
        """
        pad = "\n" + ("  " * level)
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = pad + "  "
            for child in elem:
                self._indent(child, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = pad
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = pad

    def _save(self, tree):
        """
        Save the XML tree to disk with pretty-printing.

        Args:
            tree: ElementTree object to save
        """
        root = tree.getroot()
        self._indent(root)
        tree.write(self.xml_path, encoding="utf-8", xml_declaration=True)
