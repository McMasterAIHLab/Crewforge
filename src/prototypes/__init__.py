
from __future__ import annotations

import importlib
from typing import Any, List

from .base import PrototypeSpec, UnitSpec

# id -> module exporting SPEC. Imported on demand: pulling in a prototype drags
# in multi_system and the whole model stack, which a baseline run must not pay
# for merely because this package is importable.
_REGISTRY = {
    "p1_unit_comms": "prototypes.p1_unit_comms",
    "p2_move_reason" : "prototypes.p2_move_reason",
    "p3_challenge" : "prototypes.p3_challenge",
    "p4_refutation" : "prototypes.p4_refutation"
}

__all__ = ["PrototypeSpec", "UnitSpec", "get", "list_ids", "resolve_unit_spec"]


def list_ids() -> List[str]:
    """Every registered prototype id, sorted."""
    return sorted(_REGISTRY)


def get(prototype_id: str) -> PrototypeSpec:
    """Look up a prototype by id.

    Args:
        prototype_id: The id as set in a harness' PROTOTYPE_ID.

    Returns:
        The prototype's PrototypeSpec.

    Raises:
        KeyError: If the id is not registered. A typo'd id must never fall back
            to the baseline silently — that would produce a run labelled as a
            prototype whose data came from the unmodified system.
    """
    if prototype_id not in _REGISTRY:
        raise KeyError(
            f"unknown prototype id {prototype_id!r}; registered: {list_ids()}")
    module = importlib.import_module(_REGISTRY[prototype_id])
    spec = getattr(module, "SPEC", None)
    if not isinstance(spec, PrototypeSpec):
        raise KeyError(
            f"prototype module {_REGISTRY[prototype_id]!r} does not export a "
            f"PrototypeSpec named SPEC")
    if spec.id != prototype_id:
        raise KeyError(
            f"prototype module {_REGISTRY[prototype_id]!r} declares id "
            f"{spec.id!r}, which does not match its registry key "
            f"{prototype_id!r}")
    return spec


def resolve_unit_spec(spec: PrototypeSpec, env: Any) -> UnitSpec:
    """Find the communication unit for the environment being run.

    The unit belongs to the environment, not to the prototype, so an environment
    that declares its own wins. No environment does today; the hook exists so a
    future one can own its unit natively without touching this package.

    Args:
        spec: The prototype being run.
        env: The Environment instance.

    Returns:
        The UnitSpec for this environment.

    Raises:
        KeyError: If neither source defines a unit. Falling back to some default
            unit would run the experiment with the wrong communication contract,
            which is worse than not running it.
    """
    own = getattr(env, "get_comm_unit_spec", None)
    if callable(own):
        unit = own()
        if isinstance(unit, UnitSpec):
            return unit

    env_name = env.get_env_name()
    if env_name not in spec.unit_specs:
        raise KeyError(
            f"prototype {spec.id!r} defines no communication unit for "
            f"environment {env_name!r}; it knows: {sorted(spec.unit_specs)}")
    return spec.unit_specs[env_name]
