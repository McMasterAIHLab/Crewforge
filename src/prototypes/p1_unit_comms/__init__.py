
from __future__ import annotations

from pathlib import Path

from prototypes.base import PrototypeSpec

from .system import UnitCommsMultiLLMSystem
from .unit_spec import ABSTAIN, UCI_MOVE

_HERE = Path(__file__).resolve().parent

SPEC = PrototypeSpec(
    id="p1_unit_comms",
    description="agents communicate only in the environment's unit (chess: one UCI move)",
    multi_system_cls=UnitCommsMultiLLMSystem,
    # Both chess environments speak the same unit. The mapping is keyed on
    # env.get_env_name(), and resolve_unit_spec() raises for anything absent
    # rather than guessing a default.
    unit_specs={"Chess": UCI_MOVE, "ChessPuzzle": UCI_MOVE},
    # The baseline preset's role briefs instruct agents to name squares, say
    # "no pattern" and reply "safe" — none of which survive a unit channel. This
    # is the translation; hardcoded_crew_v1.json is left untouched for the
    # baseline campaign.
    preset_crew_path=str(_HERE / "hardcoded_crew_p1.json"),
    preset_crew_name="hardcoded_crew_p1",
)

__all__ = ["SPEC", "UnitCommsMultiLLMSystem", "UCI_MOVE", "ABSTAIN"]
