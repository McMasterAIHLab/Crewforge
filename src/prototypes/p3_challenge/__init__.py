
from __future__ import annotations

from pathlib import Path

from prototypes.base import PrototypeSpec

from .system import ChallengeMultiLLMSystem
from .unit_spec import ABSTAIN, MAX_REASON_WORDS, MOVE_REASON

_HERE = Path(__file__).resolve().parent

SPEC = PrototypeSpec(
    id="p3_challenge",
    description=("move plus reason, with disagreement enforced as a direct "
                 "challenge alongside the agent's standing position"),
    multi_system_cls=ChallengeMultiLLMSystem,
    unit_specs={"Chess": MOVE_REASON, "ChessPuzzle": MOVE_REASON},
    preset_crew_path=str(_HERE / "hardcoded_crew_p3.json"),
    preset_crew_name="hardcoded_crew_p3",
)

__all__ = ["SPEC", "ChallengeMultiLLMSystem", "MOVE_REASON", "ABSTAIN",
           "MAX_REASON_WORDS"]
