
from __future__ import annotations

from pathlib import Path

from prototypes.base import PrototypeSpec

from .system import MoveReasonMultiLLMSystem
from .unit_spec import ABSTAIN, MAX_REASON_WORDS, MOVE_REASON

_HERE = Path(__file__).resolve().parent

SPEC = PrototypeSpec(
    id="p2_move_reason",
    description=("agents communicate in one move plus a reason of at most "
                 f"{MAX_REASON_WORDS} words"),
    multi_system_cls=MoveReasonMultiLLMSystem,
    # Both chess environments speak the same unit. The mapping is keyed on
    # env.get_env_name(), and resolve_unit_spec() raises for anything absent
    # rather than guessing a default.
    unit_specs={"Chess": MOVE_REASON, "ChessPuzzle": MOVE_REASON},
    # The baseline preset's role briefs assume open-ended prose and the p1
    # preset assumes no prose at all; neither fits a capped reason field. This
    # is the translation, and both earlier presets are left untouched.
    preset_crew_path=str(_HERE / "hardcoded_crew_p2.json"),
    preset_crew_name="hardcoded_crew_p2",
)

__all__ = ["SPEC", "MoveReasonMultiLLMSystem", "MOVE_REASON", "ABSTAIN",
           "MAX_REASON_WORDS"]
