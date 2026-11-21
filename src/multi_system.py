import json
from pathlib import Path
import time
import concurrent.futures
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from memory_module import InboxItem, Memory

import ollama

# Per-call timeout for ollama.chat() — prevents a single stalled inference
# from hanging the worker process forever.  Set to 60 minutes.
OLLAMA_CALL_TIMEOUT_SECONDS = 3600

# malformed syntax retries
MAX_SILENT_STRIP_RETRIES = 3

from sub_agent_module import SubAgent

from agent_config import LLMConfig, SystemPrompt, UserPrompt, gen_auid
from meta_agent_module import MetaAgent
from system import System, ChatCallRecord, SystemMoveRecord, illegal_move_notification
from environment import EnvStatePrompt, EnvIntroPrompt


class AgentCreationError(Exception):
    """Raised when any issue occurred with creating an agent (e.g. missing fields)"""
    pass


class ContextWindowExceededError(Exception):
    """Raised when a call to Ollama is estimated to have exceeded the context limit set."""
    pass


class ConsensusFailureError(Exception):
    """Raised when consensus couldn't be reached after exhausting all cycles."""
    pass


def get_crew_creation_sys_prompt(intro_prompt: EnvIntroPrompt):
    return (
        f"You are the manager/supervisor of a team of LLMs like you designated to{intro_prompt.domain_prompt}. You may alternately be referred to as the meta-agent.\n\n"

        "TEAM DESIGN PHILOSOPHY:\n"
        "Your goal is to create a team where the COMBINED intelligence exceeds what any single perspective could achieve. Think about:\n"
        # "- What strategic tensions exist in chess? (e.g., safety vs aggression, material vs position, short-term vs long-term)\n"
        "- How can agents with opposing viewpoints debate to find actions that balance these tensions?\n"
        "- Which perspectives, when they DISAGREE, would force the team to discover non-obvious solutions?\n"
        "- Avoid redundant viewpoints - each agent should bring something the others would naturally overlook\n\n"

        "The sub-agents give recommendations on moves based on their unique perspective. They do not make final decisions - YOU (the meta-agent) determine when consensus is reached and make the final move.\n\n"

        "REQUIRED OUTPUT FORMAT: Return ONLY a valid JSON object with the following structure:\n"
        "{\n"
        "  \"team\": [\n"
        "    {\n"
        "      \"agent\": \"<role name>\",\n"
        "      \"talks\": [\"<list of other agent names this agent can talk to>\"],\n"
        "      \"responsibility\": \"<one or two sentence explanation of responsibilities>\",\n"
        "      \"rationale_brief\": \"<12 words max explaining why this agent is needed>\",\n"
        "      \"natural_tension_with\": \"<which other agent(s) this one might naturally disagree with, and why>\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"

        "CRITICAL RULES:\n"
        "1. Do NOT include a consensus agent - YOU (the meta-agent) will handle consensus determination\n"
        "2. The 'talks' field must only reference agents you have created\n"
        "3. Create 3-7 agents total\n"
        "4. Each agent must bring a UNIQUE PERSPECTIVE that others lack. Roles must be genuinely different and non-redundant\n"
        "5. At least two agents should have naturally opposing viewpoints to ensure productive debate"
    )


def get_crew_creation_user_prompt(intro_prompt: EnvIntroPrompt):
    return (
        f"You are the principal investigator overseeing a research group of reasoning agents whose domain is to {intro_prompt.domain_prompt}\n\n"
        f"Each agent functions as a specialist exploring the problem from a particular analytical perspective relevant to this domain.\n\n"
        f"Design a team where each member contributes a unique viewpoint that advances collective understanding. "
        f"Encourage constructive disagreement so that weak ideas are challenged and stronger alternatives emerge.\n\n"
        f"Your responsibility is to guide the research process, evaluate competing arguments, and integrate the most robust insights into a final decision."
    )


def get_post_init_sys_prompt(intro_prompt: EnvIntroPrompt):
    return (
        f"You are the manager/supervisor of a team of LLMs like you designated to{intro_prompt.domain_prompt}. You may alternately be referred to as the meta-agent.\n\n"

        "Each of your sub-agents is a specialist in what they do. "
        # NOTE: remove this line below until this is actually a feature
        # "The number of sub-agents (3-7) is adaptive based on the specialized roles needed.\n\n"

        "The sub-agents' task is to focus on their role/specialization and give recommendations on moves. "
        "They are not the ones to make a move - they suggest the best way to proceed.\n\n"

        "The sub-agents may communicate with each other privately (direct messages), and also put their thoughts on a public channel (common space). "
        "You have FULL VISIBILITY into ALL communications - both private direct messages between agents AND public common space messages. "
        "This complete view allows you to make informed decisions about when true consensus has been reached.\n\n"

        "YOUR CONSENSUS RESPONSIBILITY: At the end of each deliberation cycle, you will review ALL team communications and determine if consensus has been reached on a move to take. "
        "When you see clear agreement among the team, use the 'consensus' tool to declare the final move. "
        "Take your time - don't rush to declare consensus if agents are still actively debating or asking questions. Wait for genuine agreement.\n\n"

        f"{intro_prompt.notation_rules}"
    ) + (
        "Review the communications happening between the team members and decide for yourself if consensus has been reached. "
        "When you determine the team has agreed on a move, use the 'consensus' tool to declare it. "
        "Do not feel pressured to call consensus early - wait for genuine agreement.\n"
        # TODO: may need to add a consensus tools outline / example usage imported from env
    )


@dataclass
class MASCallRecord(ChatCallRecord):
    """Appends MAS-specific info to a ChatCallRecord, which records information
    about a singular call to ollama.chat()
    """
    caller_id: str = ""  # id of the agent calling
    caller_name: str = ""  # name of the agent calling
    call_type: str = ""  # "deliberation" | "consensus_check" | "init"
    # the cycle of a deliberation this call took place on (if any)
    cycle: int = 0
    # stores tool calls exactly as Ollama returns them
    tool_calls: List[Dict] = field(default_factory=list)


class MultiLLMSystem(System):
    """An implementation of the System ABC for a multi-agent capabilities. The flow begins in _mas_init() (which is
    called upon initialization of this class), wherein the crew gets created and system prompts are loaded. Then, in the 
    same fashion as the Single System, the next_move() can be called to get the crew's move in a given environment.
    """

    def _detect_silent_strip(self, raw: str, raw_tool_calls, completion_toks: int, include_content: bool) -> bool:
        """Detect Ollama's "silent content stripping" condition.

        When a model attempts a malformed tool call, Ollama's internal parser
        may silently discard both the content AND the tool_calls, yielding a
        response with non-zero completion tokens but empty content and no
        parsed tools.  This helper identifies that specific condition.

        Args:
            raw: the extracted content string from the response.
            raw_tool_calls: the raw tool_calls attribute (None or list).
            completion_toks: number of completion tokens reported by Ollama.
            include_content: whether or not to take into account the content field being empty. This
                             option should be turned on for the Meta agent because when it doesn't use 
                             the consensus tool it will just say something without calling tools. That is
                             normal. What isn't normal is if it doesn't say anything AND doesn't use any tools.
                             For the sub agents, this can be turned off (if the subagents don't use any tools 
                             then most likely they are speaking to the void without realizing).

        Returns:
            True if the response matches the silent-strip pattern.
        """
        content_empty = not raw or not raw.strip()
        tools_empty = raw_tool_calls is None or (isinstance(raw_tool_calls, list) and len(raw_tool_calls) == 0)
        model_produced_tokens = completion_toks > 0
        
        if include_content: # if this is set, include content_empty in the boolean formula
            return content_empty and tools_empty and model_produced_tokens
        else:
            return tools_empty and model_produced_tokens

    @staticmethod
    def _build_corrective_suffix(llm_config: LLMConfig, is_meta: bool) -> str:
        """Build a corrective prompt suffix that restates expected tool-call syntax.

        This is injected into the system or user prompt on retry after a silent strip is
        detected, giving the model an explicit reminder of the exact tool schema
        it must conform to.

        Args:
            llm_config: the LLMConfig whose tools list defines the available tools.
            is_meta: True if the caller is the meta agent. (The corrective suffix is different for the meta agent)

        Returns:
            A string containing strict syntax instructions for the available tools.
        """
        if not llm_config.tools:
            return ""

        tool_descriptions: List[str] = []
        for tool_schema in llm_config.tools:
            func = tool_schema.get("function", {}) if isinstance(tool_schema, dict) else {}
            name = func.get("name", "unknown")
            params = func.get("parameters", {}).get("properties", {})
            required = func.get("parameters", {}).get("required", [])
            param_strs = []
            for pname, pinfo in params.items():
                req_marker = " [REQUIRED]" if pname in required else ""
                param_strs.append(f"    - {pname} ({pinfo.get('type', 'any')}){req_marker}")
            tool_descriptions.append(f"  Tool '{name}':\n" + "\n".join(param_strs))

        tools_block = "\n".join(tool_descriptions)
        if is_meta:
            return (
                "\n\n=== CORRECTIVE INSTRUCTION (PREVIOUS RESPONSE WAS MALFORMED) ===\n"
                "Your previous response was completely empty! Your response or tool call was malformed and rejected.\n"
                "If you intended to call the consensus tool, try again, with EXACTLY the parameters listed.\n Do NOT embed tool calls in plain text. Use the tool-calling interface ONLY.\n"
                "If you did not intend to call the consensus tool, then reply with a reason why the consensus has not yet been met.\n"
                f"AVAILABLE TOOLS:\n{tools_block}\n"
                "=== END CORRECTIVE INSTRUCTION ===\n"
            )    
        else:
            return (
                "\n\n=== CORRECTIVE INSTRUCTION (PREVIOUS RESPONSE WAS MALFORMED) ===\n"
                "Your previous response was EMPTY because your tool call was malformed and rejected.\n"
                "You MUST use ONE of the following tools with EXACTLY the parameters listed.\n"
                "Do NOT embed tool calls in plain text. Use the tool-calling interface ONLY.\n"
                "If you have nothing to say via a tool, respond with plain text analysis instead.\n\n"
                f"AVAILABLE TOOLS:\n{tools_block}\n"
                "=== END CORRECTIVE INSTRUCTION ===\n"
            )

    def _impl_call_ollama(self, user_prompt: str, system_prompt: str, llm_config: LLMConfig, silent_test: bool, include_content: bool, is_meta: bool, format: Optional[str] = None) -> MASCallRecord:
        """Make a call to ollama.chat() with the given parameters and returns the
        MASCallRecord representing the call. Note that the MAS-specific fields of the 
        returned MASCallRecord must be filled out by the caller.

        Includes automatic retry with corrective prompt injection when Ollama's
        "silent content stripping" is detected (non-zero completion tokens but
        empty content and no parsed tool_calls).

        Args:
            user_prompt: the user prompt to use.
            system_prompt: the system prompt.
            llm_config: the LLMConfig with the options to use.
            silent_test: whether or not to test for silent Ollama stripping (should be not used with the meta agent).
            include_content: if the silent_test flag is True, then this decides if the content field should be looked
                             at to decide if the silent strip has occurred. See the _detect_silent_strip() function.
            is_meta: True if the caller is the meta agent. This is for the purposes of the _detect_silent_strip().
            format: should either be "json" or None.

        Returns:
            The MASCallRecord. If any errors occurred, the error field will be set.
        """
        # We may retry up to MAX_SILENT_STRIP_RETRIES times on silent-strip.
        # On each retry the user prompt is augmented with a corrective suffix.
        # effective_system_prompt = system_prompt
        effective_user_prompt = user_prompt

        for attempt in range(1 + MAX_SILENT_STRIP_RETRIES):
            timestamp = datetime.now(timezone.utc).isoformat(
                timespec="milliseconds")
            t0 = time.time()

            kwargs = llm_config.create_kwargs(
                user_prompt=effective_user_prompt, system_prompt=system_prompt, fmt=format)

            # attempt the call with a timeout guard
            # NOTE: We intentionally avoid the `with` context manager here.
            # Using `with ThreadPoolExecutor(...) as pool:` would call
            # pool.shutdown(wait=True) on exit, which blocks forever if the
            # worker thread is stuck in ollama.chat() after a timeout.
            # Instead we manage the pool manually with shutdown(wait=False).
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                future = pool.submit(self._chat_fn, **kwargs)
                response = future.result(timeout=OLLAMA_CALL_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                pool.shutdown(wait=False, cancel_futures=True)
                latency = int((time.time() - t0) * 1000)
                return MASCallRecord(
                    latency_ms=latency, timestamp=timestamp,
                    system_prompt=system_prompt, user_prompt=effective_user_prompt,
                    error=f"TimeoutError: ollama.chat exceeded {OLLAMA_CALL_TIMEOUT_SECONDS}s limit")
            except Exception as e:
                pool.shutdown(wait=False, cancel_futures=True)
                latency = int((time.time() - t0) * 1000)
                return MASCallRecord(
                    latency_ms=latency, timestamp=timestamp,
                    system_prompt=system_prompt, user_prompt=effective_user_prompt, error=f"{e!r}")
            else:
                pool.shutdown(wait=False)

            latency = int((time.time() - t0) * 1000)

            # extract message contents robustly
            raw = ""
            thinking = ""
            raw_tool_calls = None

            if hasattr(response, "message"):
                raw = getattr(response.message, "content", "") or ""
                thinking = getattr(response.message, "thinking", "") or ""
                raw_tool_calls = getattr(response.message, "tool_calls", None)
            elif isinstance(response, dict) and "message" in response:
                raw = response["message"].get("content", "") or ""
                thinking = response["message"].get("thinking", "") or ""
                raw_tool_calls = response["message"].get("tool_calls", None)

            # extract tokens robustly
            prompt_toks = getattr(response, "prompt_eval_count", 0) or 0
            completion_toks = getattr(response, "eval_count", 0) or 0
            thoughts_toks = getattr(response, "thoughts_token_count", 0) or 0
            if isinstance(response, dict):
                prompt_toks = prompt_toks or response.get("prompt_eval_count", 0)
                completion_toks = completion_toks or response.get("eval_count", 0)
                thoughts_toks = thoughts_toks or response.get("thoughts_token_count", 0)

            # When Ollama silently strips a malformed tool call, the model
            # produced tokens (completion_toks > 0) but the response has no
            # content and no tool_calls.  Retry with a corrective prompt that
            # restates the expected tool syntax in strict terms.
            if silent_test:
                if self._detect_silent_strip(raw=raw, raw_tool_calls=raw_tool_calls, completion_toks=completion_toks, include_content=include_content):
                    is_last_attempt = (attempt == MAX_SILENT_STRIP_RETRIES)
                    if not is_last_attempt:
                        # Inject corrective suffix and retry
                        # effective_system_prompt = self._build_corrective_suffix(llm_config) + system_prompt
                        effective_user_prompt = user_prompt + self._build_corrective_suffix(llm_config, is_meta=is_meta)
                        print(f"\n\nTAKING CORRECTIVE ACTION FOR EMPTY TOOL CALL LIST!!!! Raw response: {raw}\n\n")
                        continue  # retry the call
                    # else: fall through — return what we have with an error annotation
                    # so callers know this was a stripped response
                    record = MASCallRecord(
                        latency_ms=latency, timestamp=timestamp,
                        prompt_tokens=prompt_toks, completion_tokens=completion_toks,
                        thoughts_tokens=thoughts_toks,
                        thinking=thinking, raw_response=raw,
                        system_prompt=system_prompt, user_prompt=effective_user_prompt,
                        error=f"SilentStrip: model produced {completion_toks} tokens but content and tool_calls were empty after {MAX_SILENT_STRIP_RETRIES + 1} attempt(s)"
                    )
                    return record

            # Normal path: build and return the record
            record = MASCallRecord(latency_ms=latency, timestamp=timestamp,
                                   prompt_tokens=prompt_toks, completion_tokens=completion_toks,
                                   thoughts_tokens=thoughts_toks,
                                   thinking=thinking, raw_response=raw,
                                   system_prompt=system_prompt, user_prompt=effective_user_prompt)

            if raw_tool_calls is not None:
                # Ollama returns ToolCall pydantic objects, not plain dicts.
                # Convert to the {"name": ..., "arguments": ...} format that
                # _call_agent() and _call_meta() expect.
                normalised: List[Dict] = []
                for tc in raw_tool_calls:
                    if hasattr(tc, "function"):
                        # ToolCall(function=Function(name=..., arguments=...))
                        fn = tc.function
                        normalised.append({
                            "name": getattr(fn, "name", ""),
                            "arguments": getattr(fn, "arguments", {}),
                        })
                    elif isinstance(tc, dict):
                        # Already a plain dict (legacy / future-proof)
                        normalised.append(tc)
                    else:
                        normalised.append({"name": "", "arguments": {}})
                record.tool_calls = normalised

            return record

        # Should not be reached, but safety net
        return MASCallRecord(
            latency_ms=0, timestamp=timestamp,
            error="Unexpected: call_ollama retry loop exited without returning")

    def call_ollama(self, user_prompt: str, system_prompt: str, llm_config: LLMConfig, format: Optional[str] = None) -> MASCallRecord:
        # don't call this version outside of crew creation, call _impl_call_ollama instead as that configurably tests for the silent strip conditions
        return self._impl_call_ollama(user_prompt=user_prompt, system_prompt=system_prompt, llm_config=llm_config, silent_test=False, include_content=False, is_meta=False, format=format)

    def _ready_meta(self, intro_prompt: EnvIntroPrompt):
        """Assigns the meta agent it's post-intialization system prompt.

        Args:
            intro_prompt: the EnvIntroPrompt given by the environment.
        """
        self.meta_agent.system_prompt.intro = get_post_init_sys_prompt(
            intro_prompt)

        # empty for now, this will be populated after the first cycle of deliberation
        self.meta_agent.user_prompt.prompt = ""

    def _ready_crew(self, intro_prompt: EnvIntroPrompt):
        """Loops through all crew members and sets their system prompt to the basic MAS outline 
        (e.g. "your responsibility is ..., your tools are ..., etc...).

        Args:
            intro_prompt: the EnvIntroPrompt passed by the environment.

        Note:
            Before the refactor, this used to also do a initial call to Ollama to get confirmation (this
            was move_id -1 (pre-game)). This has been removed here in the interest of being closer to the 
            Single System which does not have this pre-game phase.
        """
        for agent in self.memory.registry.all_agents():
            agent.talksto = self.memory.registry.ids_from_names(
                agent.talksto_strs)
            intro = (
                f"You are part of a team whose responsibility is to {intro_prompt.domain_prompt}\n\n"

                f"You are the \"{agent.name}\" member and your responsibility is to {agent.responsibility} "
                f"Remember, your name is \"{agent.name}\" and will show up as such in any conversation logs. "
                f"In short, your role {agent.provides}.\n\n"

                f"Stick closely to your role - other members are in charge of considering the problem from different angles than you, "
                f"so you only need to stay in your area of expertise.\n\n"

                f"The team has several members, and you are allowed to direct message ONLY the {', the '.join(name for name in agent.talksto_strs)} team members. "
                f"You can alternatively message all team members via the common space, but this should only be used for team-wide messages.\n\n"

                f"You must eventually come in agreement with the other members. "
                f"The meta-agent (your boss) monitors all communications and will determine when consensus has been reached. "
                f"Focus on your job and communicating clearly with your teammates.\n\n"

                f"{intro_prompt.notation_rules}"
            )

            tools = (
                "You have tools available to you to contribute to the team:\n"
                "- 'talk': Send a direct message to a specific team member\n"
                "- 'add_to_common_space': Broadcast a message to all team members\n\n"
                "If you do not use these tools, your output will be stored privately. "
                "Whenever you make a call to these tools, make sure to explain your reasoning.\n\n"
                "When another team member messages you, you will receive the message plus conversation history. "
                "You will also receive messages sent to the common space - reply to these as you see fit.\n"
            )

            # keep this line from the original version of MAS, but don't think it's necessary
            agent.current_move_id = -1  # pre-game
            agent.system_prompt = SystemPrompt(
                intro=intro, tools=tools, live_state=None, move_ids=None)
            # NOTE: originally, this went on to do a call to each agent with an intro message,
            # I've removed that here in the interest of keeping the MAS closer to the Single System

    def _mas_init(self, env_intro_prompt: EnvIntroPrompt):
        """Initializes the MAS. If self.preset_crew_name is None or self.preset_crew_path is None, the meta agent
        is prompted to create a crew, the output of which is parsed to initialize all 
        the agents and assign their responsibilities. After this is called, the MAS is ready
        for calls to next_move(). If the above two variables are not None, then this function reads the preset crew
        with the self.preset_crew_name namee specified in the self.preset_crew_path JSON file.

        Args:
            env_intro_prompt: An EnvIntroPrompt given by the environment which contains prompting information to 
                              be given to the agents + the consensus tool.

        Raises:
            AgentCreationError: if crew creation failed after self.max_retries attempts. Check the error field of all
                                the calls in self._current_calls for info on the chain of errors if this happens. Alternatively,
                                if self.preset_crew_name is not None and self.preset_crew_path is not None, the error field
                                will contain the reason why (creation will only be attempted once in this case)
            File I/O Errors: if in hardcoded mode and reading self.preset_crew_path encounters issues being read.
        """
        if self.preset_crew_name is not None and self.preset_crew_path is not None:
            self._parse_crew(self._load_crew()) # if any I/O fails, this will raise errors
            self._ready_meta(env_intro_prompt)
            self._ready_crew(env_intro_prompt)
        else:
            call_record: MASCallRecord  # worry about conversion to MASCallRecord later.

            passed = False

            # stores the previous error for the purposes of notifying what happened last after
            # all tries have failed
            last_error: str = ""

            # try up to max_retries times to generate the crew
            for attempt in range(0, self.max_retries):
                call_record = self.call_ollama(user_prompt=self.meta_agent.user_prompt.prompt,
                                            system_prompt=self.meta_agent.system_prompt.prompt, llm_config=self.meta_agent.llm_config, format="json")

                if call_record.error is not None:
                    last_error = call_record.error
                    self._current_calls.append(call_record)
                    continue

                # now that we have a valid response, let's try parsing the crew:
                # beggining with parsing the JSON:
                try:
                    crew_json = json.loads(call_record.raw_response)
                except json.JSONDecodeError as e:
                    # json decoding failed, continue to next attempt
                    call_record.error = f"{e!r}"
                    last_error = f"{e!r}"
                    self._current_calls.append(call_record)
                    continue

                # next, try parsing a crew member (subagent)
                try:
                    self._parse_crew(crew_json)
                except AgentCreationError as e:
                    call_record.error = f"{e!r}"
                    last_error = f"{e!r}"
                    self._current_calls.append(call_record)
                    continue

                self._ready_meta(env_intro_prompt)
                self._ready_crew(env_intro_prompt)

                # if we made it this far, then crew creation was successful
                passed = True
                break

            # raise an error if we didn't pass across all attempts
            if passed == False:
                raise AgentCreationError(f"Meta agent failed to create valid crew after {self.max_retries} attempts. Last error: {last_error}")

        self.meta_agent.llm_config.tools = [env_intro_prompt.consensus_schema]

        # Extract the consensus tool name from the environment's schema so
        # _call_meta() does not need to hardcode any particular name.
        _schema = env_intro_prompt.consensus_schema
        _func = _schema.get("function") if isinstance(_schema, dict) else None
        self._consensus_tool_name = (
            _func.get("name", "consensus") if isinstance(_func, dict) else "consensus"
        )

    def _validate_agent_json(self, agent_json: Dict, auid: str) -> SubAgent:
        """Parse JSON defining an agent, from the meta's crew initialization output.

        Args:
            agent_json: the JSON dict extracted from the meta's output.
            auid: if we are able to successfully parse an agent, it will have this id.

        Returns:
            the SubAgent representing the parsed agent. Raises AgentCreationError otherwise.
        """
        def _extract_field(candidates: List[str], json: Dict, expected_type1: type, expected_type2: Optional[type]) -> Tuple[Optional[str], Optional[str]]:
            """Searches the given json for any field in the given candidates.

            Args:
                candidates: a list of strings which are field names to check.
                json: the json to search inside of.
                expected_type1: the expected type of the field.
                expected_type2: if the field can be expected to have a type other than expected_type1, this contains it.

            Returns:
                (field, candidate which matched) if found, None otherwise
            """
            # loop through all potential field names
            for attempt in candidates:
                if attempt in json:
                    ret = json.get(attempt)
                    # if we were given two types, create a tuple
                    if expected_type2 is not None:
                        if ret is not None and isinstance(ret, (expected_type1, expected_type2)):
                            return ret, attempt
                    else:
                        if ret is not None and isinstance(ret, expected_type1):
                            return ret, attempt
            # no luck
            return None, None

        # go through all required fields, checking possibilities that the
        # meta could've used for each, and raise error if missing

        name_fields = ["agent", "name", "role",
                       "role name", "role_name", "member"]
        name, name_field_used = _extract_field(
            name_fields, agent_json, str, None)
        if name is None:
            raise AgentCreationError(
                f"Agent missing required name field (tried: {name_fields})")

        talks_fields = ["talks", "talks_to", "talksto", "communicates_with"]
        talks, _ = _extract_field(talks_fields, agent_json, list, str)
        if talks is None:
            raise AgentCreationError(
                f"Agent missing required 'talks' field (tried: {talks_fields})")
        # NOTE: important change here, originally talks was allowed to be
        # either list or str --- however, if it was str, there was no conversion to
        # list performed, breaking SubAgent.talksto_strs (which expects a list). This
        # fixes that.
        if isinstance(talks, str):
            # convert the string to a list (assumes LLM generated "x, y, z" instead of "[x,y,z]")
            talks = [t.strip() for t in talks.split(",") if t.strip()]

        resp_fields = ["responsibility", "responsibilities",
                       "role", "description", "desc"]
        # if we matched the name field with "role", remove it from here (NOTE: in
        # the event that "role" was selected for name, but intended for this, then we'd
        # have an issue).
        resp_fields = [item for item in resp_fields if item != name_field_used]
        resp, _ = _extract_field(resp_fields, agent_json, str, None)
        if resp is None:
            raise AgentCreationError(
                f"Agent missing required 'talks' field (tried: {talks_fields})")

        # this one is optional, it is just a brief rationale behind the meta's thinking
        rationale_fields = ["rationale_brief",
                            "rationale", "brief", "summary", "provides"]
        rationale, _ = _extract_field(rationale_fields, agent_json, str, None)
        if not rationale:
            rationale = ""

        agent = SubAgent(name=name, agent_id=auid, llm_config=self.crew_llm_config,
                         talksto_strs=talks, responsibility=resp, provides=rationale)
        # set last inbox check time to creation time --- we didn't actually check the inbox, but this ensures that upon first call,
        # the inbox will definitely be checked.
        agent.last_inbox_check = self.memory._utc_ts()

        return agent
    
    
    def _load_crew(self) -> Dict:
        """Load a hardcoded crew preset by name. Uses the JSON file specified by 
        self.preset_crew_path and initializes the crew with the name specified by 
        self.preset_crew_name.

        Returns:
            The selected crew in the {"team": [...]} format that _parse_crew expects

        Raises:
            AgentCreationError if preset crew not found, or agents refer to invalid 
            agents in the talks field.
            The usual file I/O errors if anything goes wrong with opening the 
            self.preset_crew_path file.
        """

        # sanity check, if we got this far and either of these are None, raise Error
        if self.preset_crew_name is None or self.preset_crew_path is None:
            raise AgentCreationError("Preset crew name or preset crew path missing!")

        # read all crew presets
        with open(self.preset_crew_path, "r", encoding="utf-8") as f:
            presets = json.load(f)

        # match preset to wanted crew, raise error if not found
        if self.preset_crew_name not in presets:
            raise AgentCreationError(f"Unknown crew '{self.preset_crew_name}'. Available: {sorted(presets)}")
        team = presets[self.preset_crew_name]

        # check that the agents reference valid agents in their talks field
        valid_names = {agent["agent"] for agent in team}
        for agent in team:
            unknown = set(agent.get("talks", [])) - valid_names # this should be empty set, else error
            if unknown:
                raise AgentCreationError(
                    f"Crew '{self.preset_crew_name}', agent '{agent['agent']}': 'talks' references "
                    f"unknown members {sorted(unknown)}")

        # return the team in the same format that the meta agent would generate.
        return {"team": team}
    

    def _parse_crew(self, crew_json: Dict):
        """Given a generated JSON object containing crew members' information, try to parse and extract fields
        of each one. The actual parsing is done by self._validate_agent_json().

        Args:
            crew_json: the JSON object containing the crew information.

        Raises:
            AgentCreationError: if any call to self._validate_agent_json() failed. 
        """
        # try to access team field of JSON, which should contain the team JSON
        team = crew_json.get("team", None)
        if team is None:
            raise AgentCreationError(
                "Invalid crew JSON: missing or empty 'team'")

        team_size = len(team)
        team_auids = gen_auid(team_size)

        # loop through each agent in the team, validate them and gather any errors
        # along the way.
        agents = []
        validation_errors = []
        for idx, agent_json in enumerate(team):
            try:
                agent = self._validate_agent_json(agent_json, team_auids[idx])
            except AgentCreationError as e:
                # add to the list of errors if any occurred
                validation_errors.append(e)
                continue
            agents.append(agent)
            # Register the agent in the memory registry so it's accessible
            # during deliberation cycles via self.memory.registry.all_agents()
            self.memory.registry.register(agent)

        # if errors occured, throw them all into one error for debugging
        if len(validation_errors) > 0:
            raise AgentCreationError(
                f"Agent validation failed with {len(validation_errors)} error(s):\n" +
                "\n".join(f"  - {e}" for e in validation_errors)
            )

    def estimate_tokens(self, text: str) -> int:
        """Estimates token usage of a piece of text.

        Args
        """
        return int(len(text) * self.TOKENS_PER_CHAR_ESTIMATE)

    def check_context_limit(self, system_prompt: str, user_prompt: str, max_context_tokens: int) -> Tuple[bool, int]:
        """Checks if a given set of user and system prompts will exceed the given limit based on estimation.

        Args:
            system_prompt: the system prompt component of the prompt.
            user_prompt: the user prompt component of the prompt.
            max_context_tokens: the limit to check.

        Returns:
            A tuple of the following format: ((False if limit is respected, True otherwise), estimated amount)
        """
        estimated = self.estimate_tokens(
            system_prompt) + self.estimate_tokens(user_prompt)
        return estimated <= max_context_tokens, estimated

    def _call_agent(self, agent: SubAgent, cycle: int, call_type: str):
        """Call call_ollama for the given agent and fill out the MAS-specific fields of the 
        returned MASCallRecord. Appends the record to the _current_calls member. If the call resulted
        in any tools being called, this will execute them.

        Args:
            agent: the agent we are calling for. The user_prompt and system_prompt fields should be filled out.
            cycle: the cycle count this call takes place in.
            call_type: the reasoning for the call. Should be one of "deliberation", "init".

        Raises:
            ContextWindowExceededError: if the agent's prompts are estimated to exceed the configured limit.
        """
        # extra encourage to add to the end of the prompts
        reminder_str = "\n\n ----- CRITICAL: RESPONSES MUST BE SHORT IN LENGTH AND TO THE POINT. RESPONSES MUST BE WELL THOUGHT. ------"

        within_limit, estimated = self.check_context_limit(
            agent.system_prompt.prompt, agent.user_prompt.prompt, agent.llm_config.max_context_tokens)
        if not within_limit:
            raise ContextWindowExceededError(
                f"Context window exceeded for agent {agent.name}: estimated {estimated} tokens > max {agent.llm_config.max_context_tokens}"
            )

        # do the call
        record = self._impl_call_ollama(
            user_prompt=agent.user_prompt.prompt + reminder_str,
            system_prompt=agent.system_prompt.prompt + reminder_str,
            llm_config=agent.llm_config,
            silent_test=True,
            include_content=True, # set this to true now that we have the below fallback for taking the direct output to the common space
            is_meta=False
        )

        # stamp the MAS-specific fields
        record.caller_id = agent.id
        record.caller_name = agent.name
        record.call_type = call_type
        record.cycle = cycle

        self._current_calls.append(record)

        print(record, end='\n\n')

        if record.error is not None:
            return  # if there was any error, don't parse the tools, just return

        # gather all issues that occurred whilst parsing the tools
        tool_issues: List[str] = []

        if not record.tool_calls and record.raw_response and record.raw_response.strip():
            # Fallback for when subagents output code directly
            self.memory.store_common(agent.id, record.raw_response.strip(), agent.current_move_id)
            return

        for tc in record.tool_calls:
            name = tc.get("name", "")
            args = tc.get("arguments", {})

            if name == "talk":
                text = args.get("text", "")
                member = args.get("member", "")
                if not text:
                    tool_issues.append(
                        f"Tool talk() called without 'text' argument!")
                    continue

                if not member:
                    tool_issues.append(
                        f"Tool talk() called without 'member' argument!")
                    # fallback: no member specified, broadcast to common space instead
                    self.memory.store_common(
                        agent.id, text, agent.current_move_id)
                    continue

                recipient_id = self.memory.registry.id_from_name(member)
                if recipient_id is None:
                    tool_issues.append(
                        f"Tool talk() called with unmatchable 'member' argument: {member}!")
                    continue
                if recipient_id == agent.id:
                    tool_issues.append(
                        f"Tool talk() called with 'member' argument set to self!")
                    continue

                self.memory.store_direct(
                    agent.id, recipient_id, text, agent.current_move_id)

            elif name == "add_to_common_space":
                text = args.get("text", "")
                if not text:
                    tool_issues.append(
                        f"Tool add_to_common_space() called without 'text' argument!")
                    continue
                self.memory.store_common(agent.id, text, agent.current_move_id)
            else:
                # if the tool called doesn't match any the allowed tools
                tool_issues.append(f"Tool {name} not found!")

        # if any issues occurring in parsing, put them all into the error field.
        if len(tool_issues) > 0:
            # error X AND error Y AND error Z etc....
            record.error = ' AND '.join(tool_issues)

    def _call_meta(self, cycle: int, call_type: str):
        """Same as call_agent() but for the Meta agent
        """
        record = self._impl_call_ollama(
            user_prompt=self.meta_agent.user_prompt.prompt,
            system_prompt=self.meta_agent.system_prompt.prompt,
            llm_config=self.meta_agent.llm_config,
            silent_test=True,
            include_content=True,
            is_meta=True
        )

        # stamp the MAS-specific fields
        record.caller_id = self.meta_agent.id
        record.caller_name = self.meta_agent.name
        record.call_type = call_type
        record.cycle = cycle

        self._current_calls.append(record)

        print(record, end='\n\n')

        if record.error is not None:
            return  # if there was any error, don't parse the tools, just return

        # gather all issues that occurred whilst parsing the tools
        tool_issues: List[str] = []

        for tc in record.tool_calls:
            name = tc.get("name", "")
            args = tc.get("arguments", {})

            if name == self._consensus_tool_name:
                # store the consensus result --- the required fields come from the env's schema
                if self.env_intro_prompt.consensus_move_field not in args:
                    tool_issues.append(f"Tool {self._consensus_tool_name}() called without the move field '{self.env_intro_prompt.consensus_move_field}' argument!")
                    continue

                self.consensus_move = args.get(
                    self.env_intro_prompt.consensus_move_field)
            else:
                tool_issues.append(f"Tool {name} not found!")

        # if any issues occurring in parsing, put them all into the error field.
        if len(tool_issues) > 0:
            # error X AND error Y AND error Z etc....
            record.error = ' AND '.join(tool_issues)

    def _handle_regular_agent_inbox(self, agent: SubAgent, inbox_item: InboxItem, cycle: int, env_state_prompt: EnvStatePrompt):
        """Give an agent a single inbox message. This involves building a user prompt combining
        the environment state + the actual message (and history, if any) + response guidance 
        (depending on the type of message).

        Args:
            agent: the agent to prompt with the message.
            inbox_item: the InboxItem containing the relevant message.
            cycle: the current deliberation cycle this takes place in.
            env_state_prompt: the current EnvStatePrompt given by the environment.
        """

        # get the message along with all relevant history
        msg_display = inbox_item.format()

        # add additional guidance beyond just pasting the message
        if inbox_item.source == "direct":
            msg_type_intro = f"You are receiving a DIRECT MESSAGE from {inbox_item.other_agent_name}."
        else:  # assume common space
            msg_type_intro = f"You are receiving a COMMON SPACE message (sent to all team members) from {inbox_item.other_agent_name}."

        if inbox_item.other_agent_name == "SYSTEM":
            response_guidance = "This is an important SYSTEM message that you cannot reply to. Make sure to heed what it says."
        else:
            response_guidance = (
                f"Consider whether to reply to {inbox_item.other_agent_name} using the 'talk' tool with your analysis. "
                f"The conversation history section (if present) shows previous messages in this thread."
            )
            if inbox_item.source == "common":
                response_guidance += "\nSince this was sent to everyone, you may not need to reply - use your judgment."

        # paste together the final prompt
        agent.user_prompt.prompt = (
            f"{env_state_prompt.prompt}\n\n"
            f"{msg_type_intro}\n\n"
            f"=====MESSAGE=====\n"
            f"{msg_display}\n"
            f"=====END MESSAGE=====\n\n"
            # add response guidance only if the message didn't come from the system
            f"{self.env_intro_prompt.reply if inbox_item.other_agent_name != 'SYSTEM' else ''}\n\n"
            f"{response_guidance}\n"
        )

        self._call_agent(agent, cycle, call_type="deliberation")

    def _handle_regular_agent_no_inbox(self, agent: SubAgent, cycle: int, env_state_prompt: EnvStatePrompt, is_reflection: bool):
        """Called when an agent has no new messages to respond to --- can be used in the 
        case that agent reflection is enabled, or just there are no new messages.

        Args:
            agent: the agent to prompt.
            cycle: the current deliberation cycle this takes place in.
            env_state_prompt: the current EnvStatePrompt given by the environment.
            is_reflection: whether or not this call is a reflection or not.
        """
        direct_msgs = self.memory.view_fmt_direct(
            agent=agent, current_move_id=env_state_prompt.move_id)
        common_msgs = self.memory.view_fmt_common(
            current_move_id=env_state_prompt.move_id)

        if not is_reflection:
            # if not reflecting, then let the agent know they have no new messages.
            agent.user_prompt.prompt = (
                f"{env_state_prompt.prompt}\n\n"
                f"You have no new messages to respond to.\n\n"
                f"If you haven't already shared your analysis with the team, consider using your tools:\n"
                f"- 'talk': Send a direct message to a teammate\n"
                f"- 'add_to_common_space': Broadcast to everyone\n\n"
                f"If you've already shared your thoughts and reached agreement, you may wait for new messages.\n\n"
                f"Review your conversation history below. Remember: when you see \"{agent.name}\", that's you!\n\n"
                f"====={agent.name} (You) - DIRECT MESSAGES=====\n"
                f"{direct_msgs}\n\n"
                f"=====COMMON SPACE MESSAGES=====\n"
                f"{common_msgs}\n"
            )
        else:
            # if reflecting, let the agent know they can reflect on the state of things.
            agent.user_prompt.prompt = (
                f"{env_state_prompt.prompt}"
                f"=====REFLECTION STEP=====\n"
                f"You've just finished responding to your messages. Now take a moment to reflect.\n\n"
                f"Consider:\n"
                f"- Is there anything important about the position that hasn't been discussed yet?\n"
                f"- Do you have concerns or ideas that weren't covered in your recent conversations?\n"
                f"- Based on your role as {agent.name}, is there analysis you should share proactively?\n\n"
                f"If you have something new to contribute, use your tools. If not, that's fine too.\n\n"
                f"Review the current state of all conversations:\n\n"
                f"====={agent.name} (You) - DIRECT MESSAGES=====\n"
                f"{direct_msgs}\n\n"
                f"=====COMMON SPACE MESSAGES=====\n"
                f"{common_msgs}\n"
            )

        self._call_agent(agent=agent, cycle=cycle, call_type="deliberation")

    def _handle_meta_consensus_check(self, env_state_prompt: EnvStatePrompt, cycle: int, cycles_remaining: int, is_countdown_phase: bool) -> Optional[str]:
        """Have the meta agent review all team communications and decide whether
        consensus has been reached.

        Builds a user prompt containing the environment state + all direct messages
        + all common space messages + urgency warnings (if in countdown phase),
        then calls the meta agent.

        Args:
            env_state_prompt: the current environment state prompt.
            cycle: current deliberation cycle.
            cycles_remaining: how many cycles are left.
            is_countdown_phase: True if we've entered the countdown phase.

        Returns:
            The consensus action string if consensus was reached and is valid,
            None otherwise.
        """
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

        num_agents = len(self.memory.registry.all_agents())

        self.meta_agent.user_prompt.prompt = (
            f"{env_state_prompt.prompt}\n\n"
            f"Review ALL team communications below and determine if consensus has been reached.\n"
            f"Your team has {num_agents} specialist agents. Look for agreement from the MAJORITY (at least {(num_agents // 2) + 1} agents).\n"
            f"Wait if members are still discussing. Only declare consensus when completely clear.\n"
            f"Be careful: agents may mention previous actions. Only declare consensus for the CURRENT move (Move {self.meta_agent.current_move_id}).\n"
            f"IMPORTANT: If multiple agents agree on the same action in the COMMON SPACE, that IS consensus - even if some agents are silent.\n"
            f"{urgency_msg}\n"
            f"=====ALL DIRECT MESSAGES (Private conversations between agents)=====\n"
            f"{all_direct_msgs}\n\n"
            f"=====COMMON SPACE (Public messages to all)=====\n"
            f"{all_common_msgs}\n\n"
            f"Based on the above communications, has the team reached consensus on an action? "
            f"If yes, use the consensus tool. If not, let deliberation continue.\n"
        )

        self._call_meta(cycle=cycle, call_type="consensus_check")

        move = None

        if self.consensus_move is not None:
            if env_state_prompt.legal_actions is not None:
                move = self.consensus_move
                if move not in env_state_prompt.legal_actions:
                    # Notify all agents that the proposed move is illegal
                    self._notify_legal_move(
                        move, env_state_prompt.legal_actions, env_state_prompt.move_id)
                    self.consensus_move = None
                    move = None  # reset — deliberation must continue
            else:
                move = self.consensus_move # no need to parse legal moves if not given, just use move directly
        return move

    def _notify_legal_move(self, illegal_move: str, legal_moves: List[str], move_id: int):
        """Notify all agents that an illegal move was proposed.

        Args:
            illegal_move: the illegal move that was made.
            legal_moves: the list of legal moves, given by the environment.
            move_id: the move that the illegal move took place on.
        """
        if self.symmetric_invalid_move:
            self._invalid_move_count += 1
            if self._invalid_move_count >= self.max_retries:
                raise RuntimeError(
                    f"MultiLLMSystem failed after {self.max_retries} attempts. "
                    f"Last error: '{illegal_move}' not in legal actions"
                )

        notification = illegal_move_notification(illegal_move, legal_moves)

        # send the notification to the common space so everyone gets it
        self.memory.store_common("SYSTEM", notification, move_id=move_id)

    # -----------------------------------------------------------------------
    # Extension points.
    #
    # The meta agent and the crew's tool schemas are built inside __init__, so a
    # subclass cannot substitute them after the fact. These three methods are the
    # seam that lets a communication prototype (see prototypes/) replace them
    # without duplicating __init__ and drifting from it. Their bodies here are
    # the system's real defaults, not placeholders: overriding one is what
    # changes behaviour, and leaving all three alone leaves the system exactly as
    # it was before the seam existed.
    # -----------------------------------------------------------------------

    def _crew_creation_sys_prompt(self, intro_prompt: EnvIntroPrompt) -> str:
        """The meta agent's system prompt for designing the crew.

        Args:
            intro_prompt: The EnvIntroPrompt given by the environment.

        Returns:
            The crew-creation system prompt.
        """
        return get_crew_creation_sys_prompt(intro_prompt)

    def _crew_creation_user_prompt(self, intro_prompt: EnvIntroPrompt) -> str:
        """The meta agent's user prompt for designing the crew.

        Args:
            intro_prompt: The EnvIntroPrompt given by the environment.

        Returns:
            The crew-creation user prompt.
        """
        return get_crew_creation_user_prompt(intro_prompt)

    def _build_agent_tools(self) -> List[Dict]:
        """The tools every sub-agent is given, i.e. how an agent may communicate.

        Returns:
            The Ollama-style tool schemas assigned to the crew's LLMConfig.
        """
        talk_tool = {
            'type': 'function',
            'function': {
                'name': 'talk',
                'description': 'Talk to another team member',
                'parameters': {
                    'type': 'object',
                    'required': ['member', 'text'],
                    'properties': {
                        'member': {'type': 'string', 'description': 'The name of the team member to talk to'},
                        'text': {'type': 'string', 'description': 'The text to say to the team member'},
                    },
                },
            },
        }

        add_to_common_space_tool = {
            'type': 'function',
            'function': {
                'name': 'add_to_common_space',
                'description': 'Add a thought to the common space',
                'parameters': {
                    'type': 'object',
                    'required': ['text'],
                    'properties': {
                        'text': {'type': 'string', 'description': 'The text to add to the common space'},
                    },
                },
            },
        }

        return [talk_tool, add_to_common_space_tool]

    def __init__(self, meta_llm_config: LLMConfig, crew_llm_config: LLMConfig, env_intro_prompt: EnvIntroPrompt, every_other: bool, max_retries: int = 3, chat_fn=None, preset_crew_path: Optional[Path] = None, preset_crew_name: Optional[str] = None, symmetric_invalid_move: bool = False):
        """Initialize the MAS, which will be ready for a call to next_move() after this.

        Args:
            meta_llm_config: the LLMConfig that the meta agent will use.
            crew_llm_config: the LLMConfig that the sub agents will use.
            env_intro_prompt: the EnvIntroPrompt containing prompt info and the consensus dict.
            every_other: if the environment has an opponent which logs moves every other move_id.
            max_retries: the max number of times to retry creating a crew.
            chat_fn: optional callable replacing ollama.chat() for cloud providers.
            preset_crew_path: Optional, only for hardcoded agent roles. Specifies the JSON file 
                              containing the agent roles.
            preset_crew_name: Optional, only for hardcoded agent roles. Specifies the crew name 
                              in the JSON file.

        Raises:
            AgentCreationError: if, after the max number of retries, the crew cannot be created.
                                Check the _current_calls() to see the specific chain of errors.

        Note:
            If preset_crew_path and preset_crew_name are not None, then the system will use 
            hardcoded agent roles, specified in the preset_crew_path JSON file (will parse the
            crew with the preset_crew_name name).
        """
        super().__init__()
        self.meta_llm_config = meta_llm_config
        self.crew_llm_config = crew_llm_config
        self._chat_fn = chat_fn or ollama.chat  # Default to ollama.chat when no cloud fn provided

        self.preset_crew_path = preset_crew_path
        self.preset_crew_name = preset_crew_name

        self.env_intro_prompt = env_intro_prompt

        self.meta_agent = MetaAgent(system_prompt=SystemPrompt(intro=self._crew_creation_sys_prompt(env_intro_prompt)), user_prompt=UserPrompt(
            # put sys prompt generic
            prompt=self._crew_creation_user_prompt(env_intro_prompt)), llm_config=meta_llm_config)

        # assign the tools to the crew config --- these won't change so may as well assign them here
        self.crew_llm_config.tools = self._build_agent_tools()

        self.max_retries = max_retries
        self.symmetric_invalid_move = symmetric_invalid_move

        self._current_calls: List[ChatCallRecord] = []

        # save some consts:
        self.MOVE_HISTORY_LIMIT = 4
        self.TOKENS_PER_CHAR_ESTIMATE = 0.25
        self.MAX_NORMAL_CYCLES = 10
        self.MAX_COUNTDOWN_CYCLES = 10
        self.TOTAL_MAX_CYCLES = (
            self.MAX_NORMAL_CYCLES + self.MAX_COUNTDOWN_CYCLES)
        self.ENABLE_AGENT_REFLECTION: bool = True

        # create a memory XML with agent registry
        self.memory = Memory(
            xml_path=None, every_other=every_other, limit=self.MOVE_HISTORY_LIMIT)

        self._mas_init(env_intro_prompt)

    def next_move(self, env_state_prompt: EnvStatePrompt) -> str:
        self.consensus_move = None
        self._invalid_move_count = 0
        self._current_calls = []
        move_id = env_state_prompt.move_id

        # first and foremost, make sure to update the internal move_id for the
        # ollama-calling functions
        self.meta_agent.current_move_id = move_id

        # loop through all allowed cycles
        for cycle in range(self.TOTAL_MAX_CYCLES):
            is_countdown_phase = cycle >= self.MAX_NORMAL_CYCLES
            cycles_remaining = self.TOTAL_MAX_CYCLES - cycle

            # loop through all agents
            for agent in self.memory.registry.all_agents():
                # update the internal move id to the current one.
                agent.current_move_id = move_id

                # get the current inbox and reset the timestamp to now
                inbox = self.memory.view_inbox(
                    agent=agent, cur_move_id=move_id, last_ts=agent.last_inbox_check)
                agent.last_inbox_check = self.memory._utc_ts()

                if inbox:
                    for inbox_item in inbox:
                        self._handle_regular_agent_inbox(
                            agent=agent,
                            inbox_item=inbox_item,
                            env_state_prompt=env_state_prompt,
                            cycle=cycle,
                        )

                    # after we've processed all inbox items, do reflection if enabled
                    if self.ENABLE_AGENT_REFLECTION:
                        self._handle_regular_agent_no_inbox(
                            agent=agent,
                            env_state_prompt=env_state_prompt,
                            cycle=cycle,
                            is_reflection=True
                        )
                else:
                    # No inbox messages — agent must still be called so they can
                    # proactively share analysis and use their tools (talk /
                    # add_to_common_space).  Without this, deliberation never
                    # starts because no agent ever speaks on the first cycle.
                    self._handle_regular_agent_no_inbox(
                        agent=agent,
                        env_state_prompt=env_state_prompt,
                        cycle=cycle,
                        is_reflection=False
                    )

            # after having looped through all agents, call meta to see if it wants
            # to call consensus
            consensus_reached = self._handle_meta_consensus_check(
                env_state_prompt=env_state_prompt, cycle=cycle, cycles_remaining=cycles_remaining, is_countdown_phase=is_countdown_phase)

            if consensus_reached is not None:
                return consensus_reached

        raise ConsensusFailureError(
            f"No consensus after {self.TOTAL_MAX_CYCLES} cycles for move {move_id}"
        )

    def get_name(self) -> str:
        return "multi_llm"

    def close(self) -> None:
        pass

    def get_move_record(self) -> SystemMoveRecord:
        return SystemMoveRecord(calls=self._current_calls)
