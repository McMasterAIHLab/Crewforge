
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

# A fenced block whose opening fence carries an optional language tag, e.g.
# ```json\n<payload>\n``` . Matched before the tag-less form so that the tag is
# not mistaken for the first characters of the payload.
_FENCE_MULTILINE = re.compile(r"^```[A-Za-z0-9_+-]*\r?\n(.*?)\r?\n?```$", re.DOTALL)
_FENCE_INLINE = re.compile(r"^```(.*?)```$", re.DOTALL)
_BACKTICK = re.compile(r"^`(.*?)`$", re.DOTALL)

# Trailing sentence punctuation a model may append to an otherwise clean unit.
_TRAILING_PUNCT = ".,;:!"


@dataclass(frozen=True)
class UnitSpec:
    """The atomic communication unit for one environment.

    A unit is the entire vocabulary an agent may use to communicate, plus one
    reserved token meaning "I am putting nothing forward" (see the plan §2.5 for
    why abstention needs a unit of its own rather than being expressible as a
    move or as silence).

    Attributes:
        name: Identifier for the unit kind, e.g. "uci_move".
        pattern: Regex the normalized unit must fullmatch.
        examples: Valid units, quoted into prompts and corrections.
        description: One-line description, quoted into prompts.
        format_rules: The strict format statement injected into system prompts.
        corrective_template: Retry message; formatted with `bad` and `examples`.
        abstain_token: The reserved "no proposal" unit, or None to disallow
            abstention entirely.
        check_legal: RESERVED. Not used by P1, which validates format only so
            that the baseline's consensus-time legality check remains the sole
            legality gate. See the plan §9.
    """

    name: str
    pattern: str
    examples: Tuple[str, ...]
    description: str
    format_rules: str
    corrective_template: str
    abstain_token: Optional[str] = "--"
    check_legal: bool = False

    def normalize(self, raw: str) -> str:
        """Strip the decoration models habitually add, without guessing.

        Tolerant of wrapping (fences, quotes, a single-key JSON object, trailing
        punctuation) because those are formatting noise around an intended unit.
        NOT tolerant of a unit embedded in prose: "I think e2e4 is best" must
        fail, because silently extracting the move from it would accept exactly
        the natural-language behaviour the prototype exists to remove.

        Args:
            raw: The candidate string as the model emitted it.

        Returns:
            The normalized candidate, which may still be invalid.
        """
        if raw is None:
            return ""
        text = raw.strip()

        # Unwrap layer by layer: a model may produce `"e2e4"` inside a fence.
        for _ in range(4):
            before = text
            for rx in (_FENCE_MULTILINE, _FENCE_INLINE, _BACKTICK):
                m = rx.match(text)
                if m:
                    text = m.group(1).strip()
                    break
            else:
                # A one-key JSON object, or a bare JSON string, is decoration
                # around a single value; anything richer is not a unit.
                if text[:1] in ('{', '"'):
                    try:
                        parsed = json.loads(text)
                    except (ValueError, TypeError):
                        parsed = None
                    if isinstance(parsed, str):
                        text = parsed.strip()
                    elif isinstance(parsed, dict) and len(parsed) == 1:
                        value = next(iter(parsed.values()))
                        if isinstance(value, (str, int, float)):
                            text = str(value).strip()
                text = text.strip("'\"").strip()
            if text == before:
                break

        text = text.rstrip(_TRAILING_PUNCT).strip()
        return text.lower()

    def validate(self, raw: str) -> Tuple[bool, str, bool]:
        """Check a candidate against the unit's format.

        Args:
            raw: The candidate string as the model emitted it.

        Returns:
            (ok, normalized, is_abstain). `is_abstain` is True only when the
            candidate is exactly the abstain token. Callers must not branch
            storage on it — an abstention is stored in the channel like any
            other message; the flag exists so post-hoc analysis can separate
            abstentions from proposals without re-parsing.
        """
        text = self.normalize(raw)
        if not text:
            return False, text, False
        if self.abstain_token is not None and text == self.abstain_token:
            return True, text, True
        if re.fullmatch(self.pattern, text):
            return True, text, False
        return False, text, False

    def corrective(self, bad: str, attempt: int, final: bool) -> str:
        """Build the correction injected into the prompt after a rejected unit.

        Args:
            bad: The rejected candidate, quoted back so the model can see what
                it actually sent.
            attempt: 1-based retry number.
            final: True on the last retry, which says so plainly.

        Returns:
            The correction block to append to the user prompt. It never
            suggests which unit to send, nor nudges toward or away from
            abstaining.
        """
        shown = (bad or "").strip()
        if len(shown) > 200:
            shown = shown[:200] + "…"
        body = self.corrective_template.format(
            bad=shown, examples="  ".join(self.examples),
            abstain=self.abstain_token or "")
        return ("\n\nFINAL ATTEMPT.\n" + body) if final else ("\n\n" + body)


@dataclass
class PrototypeSpec:
    """Everything the harness needs to run one prototype.

    Not frozen: `unit_specs` is a dict, and the spec is a registry record rather
    than a value that is ever hashed or compared.

    Attributes:
        id: Registry key, and the value recorded in every experiment document.
        description: One line, printed in run banners.
        multi_system_cls: The MultiLLMSystem subclass to build for multi-agent
            modes. Single-LLM modes are never affected by a prototype.
        unit_specs: env.get_env_name() -> UnitSpec.
        preset_crew_path: Replaces the harness' hardcoded-crew preset when set,
            so a prototype can ship a preset written for its own channel.
        preset_crew_name: The crew name within that file.
    """

    id: str
    description: str
    multi_system_cls: type
    unit_specs: Dict[str, UnitSpec] = field(default_factory=dict)
    preset_crew_path: Optional[str] = None
    preset_crew_name: Optional[str] = None
