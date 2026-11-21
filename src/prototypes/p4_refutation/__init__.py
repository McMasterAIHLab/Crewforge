
from __future__ import annotations

from pathlib import Path

from prototypes.base import PrototypeSpec

from .system import RefutationMultiLLMSystem
from .unit_spec import ABSTAIN, MAX_REASON_WORDS, MOVE_REASON

_HERE = Path(__file__).resolve().parent

SPEC = PrototypeSpec(
    id="p4_refutation",
    description=("as p4_challenge, with prompts that ask agents to refute "
                 "before endorsing and to disregard the endorsement count"),
    multi_system_cls=RefutationMultiLLMSystem,
    unit_specs={"Chess": MOVE_REASON, "ChessPuzzle": MOVE_REASON},
    preset_crew_path=str(_HERE.parent / "p4_refutation" / "hardcoded_crew_p4.json"),
    preset_crew_name="hardcoded_crew_p4",
)

__all__ = ["SPEC", "RefutationMultiLLMSystem", "MOVE_REASON", "ABSTAIN",
           "MAX_REASON_WORDS"]
