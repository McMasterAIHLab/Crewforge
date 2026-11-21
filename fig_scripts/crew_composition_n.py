"""Reconstruct generated-crew composition from logged deliberation calls.

The meta-agent's emitted roster and communication topology are not persisted as
structured fields, but both are recoverable from the per-call ``system_prompt``
that each sub-agent receives, and the configured crew size is stated in the
meta-agent's ``consensus_check`` prompt.

Role names are emitted free-form, so the same role may appear as
``Tactical_Calculator``, ``Tactical Calculator`` or ``Tactical-Calculator``
across runs. Names are therefore canonicalised on separators and case before
any cross-run comparison. Canonicalisation is deliberately formatting-only:
distinct wordings such as ``Dynamic Attacker`` and ``Dynamic Gambiteer`` are
never merged.
"""

import re
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple


META_CALLER_ID = "000"
META_CALLER_NAME = "Meta"
CONSENSUS_CALL_TYPE = "consensus_check"
CORE_ROLE_THRESHOLD = 0.5

_AGENT_NAME_RE = re.compile(r'You are the "([^"]+)" member')
_ROLE_RE = re.compile(
    r"your responsibility is to (.*?)\. Remember, your name", re.DOTALL)
_PERMITTED_RE = re.compile(
    r"direct message ONLY the (.*?) team members", re.DOTALL)
_TEAM_SIZE_RE = re.compile(r"Your team has (\d+) specialist agents")
_SEPARATORS_RE = re.compile(r"[-_\s]+")
_REFLECTION_MARKER = "=====REFLECTION STEP====="


def canonical_name(name: Optional[str]) -> Optional[str]:
    """Canonicalise an agent name for cross-run comparison.

    Collapses underscores, hyphens and whitespace to single spaces and lowers
    case, so ``Anti_Computer_Specialist`` and ``Anti-Computer Specialist``
    compare equal. Wording is never altered.

    Args:
        name: Raw agent name, or None.

    Returns:
        Canonical form, or None if the input was None.
    """
    if name is None:
        return None
    return _SEPARATORS_RE.sub(" ", name).strip().lower()


def display_name(canonical: str) -> str:
    """Render a canonical name for report output."""
    return canonical.title()


def family_of(canonical: str, role_map: Optional[Dict[str, str]],
              depth: int) -> str:
    """Resolve a canonical name to its family label.

    An explicit ``role_map`` takes precedence: it is an author-assigned
    grouping, published alongside results so a reader can disagree with any
    individual assignment. The leading-token fallback is a crude heuristic
    that does not track function reliably (``Chaos Provocateur`` and
    ``Anti Computer Specialist`` describe the same job but share no leading
    token), and is retained only for quick exploration.

    Args:
        canonical: Canonical agent name.
        role_map: Optional canonical-name to family-label mapping.
        depth: Leading-token fallback depth, used when no map is given.

    Returns:
        Family label. Unmapped names fall back to themselves.
    """
    if role_map:
        return role_map.get(canonical, canonical)
    return family_key(canonical, depth)


def family_key(canonical: str, depth: int) -> str:
    """Group a canonical name by its leading tokens.

    A coarser grouping than exact naming: at depth 2, ``anti computer
    specialist`` and ``anti computer pragmatist`` share the key ``anti
    computer``. This is a stated heuristic, not a semantic judgement, and is
    intended only as a sensitivity check alongside exact-name results.

    Args:
        canonical: Canonical agent name.
        depth: Number of leading tokens to retain.

    Returns:
        Family key, or the full name if it has fewer than ``depth`` tokens.
    """
    tokens = canonical.split()
    return " ".join(tokens[:depth]) if len(tokens) >= depth else canonical


def _parse_permitted(blob: str, normalize: bool) -> Set[str]:
    """Split the permitted-recipient clause into agent names."""
    names = set()
    for part in blob.split(","):
        part = part.strip()
        if part.startswith("the "):
            part = part[4:]
        part = part.strip().rstrip(".")
        if part:
            names.add(canonical_name(part) if normalize else part)
    return names


def _iter_calls(run: Dict[str, Any]
                ) -> Iterator[Tuple[str, int, Dict[str, Any]]]:
    """Yield every logged call in a run, in ply order.

    Args:
        run: A run document.

    Yields:
        (move key, zero-based call index within that move, call) triples. The
        locator lets any finding be traced to an exact call in the source
        file without scanning it.
    """
    moves = run.get("moves") or {}
    for move_key, move in sorted(moves.items()):
        calls = list(move.get("calls") or [])
        # Chunked moves store an empty inline list and spread the calls over
        # subcollection chunks; call_start gives each chunk's global offset.
        chunks = (move.get("__subcollection_calls") or {}).values()
        for chunk in sorted(chunks, key=lambda c: int(c.get("call_start", 0))):
            calls.extend(chunk.get("calls") or [])
        for index, call in enumerate(calls):
            yield move_key, index, call



def _is_meta(call: Dict[str, Any]) -> bool:
    """Return True if the call was made by the meta-agent."""
    return (call.get("caller_id") == META_CALLER_ID
            or call.get("caller_name") == META_CALLER_NAME
            or call.get("call_type") == CONSENSUS_CALL_TYPE)


def extract_crew(run: Dict[str, Any],
                 normalize: bool = True) -> Dict[str, Any]:
    """Reconstruct the roster and emitted topology for a single run.

    Args:
        run: A run document containing a ``moves`` map with ``calls``.
        normalize: Canonicalise agent names before comparison.

    Returns:
        Dict with ``agents`` (name -> {raw_name, role, permitted, caller_id}),
        ``declared_size``, ``observed_size``, ``has_reflection`` and
        ``calls_seen``.
    """
    agents: Dict[str, Dict[str, Any]] = {}
    declared_size: Optional[int] = None
    has_reflection = False
    calls_seen = 0

    for _, _, call in _iter_calls(run):
        calls_seen += 1
        prompt = call.get("system_prompt") or ""
        user_prompt = call.get("user_prompt") or ""

        if _REFLECTION_MARKER in user_prompt:
            has_reflection = True

        if _is_meta(call):
            if declared_size is None:
                match = (_TEAM_SIZE_RE.search(user_prompt)
                         or _TEAM_SIZE_RE.search(prompt))
                if match:
                    declared_size = int(match.group(1))
            continue

        name_match = _AGENT_NAME_RE.search(prompt)
        if not name_match:
            continue
        raw_name = name_match.group(1)
        key = canonical_name(raw_name) if normalize else raw_name
        if key in agents:
            continue

        role_match = _ROLE_RE.search(prompt)
        permitted_match = _PERMITTED_RE.search(prompt)
        agents[key] = { # type: ignore
            "raw_name": raw_name,
            "role": role_match.group(1).strip() if role_match else None,
            "permitted": (_parse_permitted(permitted_match.group(1), normalize)
                          if permitted_match else set()),
            "caller_id": call.get("caller_id"),
        }

    return {
        "agents": agents,
        "declared_size": declared_size,
        "observed_size": len(agents),
        "has_reflection": has_reflection,
        "calls_seen": calls_seen,
    }


def topology_adherence(run: Dict[str, Any],
                       crew: Optional[Dict[str, Any]] = None,
                       normalize: bool = True) -> Dict[str, Any]:
    """Classify every direct message against the sender's permitted edge set.

    Two failure modes are distinguished. An *unpermitted* message names a real
    teammate the sender was not granted an edge to, and is a topology
    violation. An *unknown* message names an agent absent from the roster
    entirely, and is an instruction-following error rather than a topology one.

    Args:
        run: A run document.
        crew: Optional pre-computed result of :func:`extract_crew`.
        normalize: Canonicalise agent names before comparison.

    Returns:
        Dict of counts plus an ``offenders`` list of dicts carrying the
        sender's permitted set and the full roster, for manual verification.
    """
    crew = crew or extract_crew(run, normalize)
    agents = crew["agents"]
    roster = set(agents)
    counts: Dict[str, Any] = {"total": 0, "permitted": 0, "unpermitted": 0,
                              "unknown": 0, "self_addressed": 0,
                              "undelivered": 0, "excluded": 0}
    offenders: List[Dict[str, Any]] = []

    for move_key, call_index, call in _iter_calls(run):
        if _is_meta(call):
            continue
        raw_sender = call.get("caller_name")
        sender = canonical_name(raw_sender) if normalize else raw_sender
        for tool_index, tool_call in enumerate(call.get("tool_calls") or []):
            if tool_call.get("name") != "talk":
                continue
            args = tool_call.get("arguments") or {}
            raw_recipient = args.get("member")
            # The transport reroutes a talk with no member to the common space
            # and drops one with no text, so neither is a direct message.
            if not raw_recipient or not args.get("text"):
                counts["excluded"] += 1
                continue
            recipient = (canonical_name(raw_recipient) if normalize
                         else raw_recipient)
            counts["total"] += 1
            allowed = agents.get(sender, {}).get("permitted", set())

            if recipient == sender:
                kind = "self_addressed"
            elif recipient in allowed:
                kind = "permitted"
            elif recipient in roster:
                kind = "unpermitted"
            else:
                kind = "unknown"

            counts[kind] += 1
            # The transport rejects unresolvable or self-addressed recipients
            # and logs the reason in the call's error field.
            error = call.get("error") or ""
            delivered = not (
                f"unmatchable 'member' argument: {raw_recipient}!" in error
                or (kind == "self_addressed"
                    and "'member' argument set to self!" in error))
            if not delivered:
                counts["undelivered"] += 1
            if kind != "permitted" or not delivered:
                offenders.append({
                    "sender": raw_sender,
                    "recipient": raw_recipient,
                    "kind": kind,
                    "delivered": delivered,
                    "move": move_key,
                    "call_index": call_index,
                    "tool_index": tool_index,
                    "caller_id": call.get("caller_id"),
                    "cycle": call.get("cycle"),
                    "permitted": sorted(allowed),
                    "roster": sorted(roster),
                })

    counts["offenders"] = offenders
    return counts


def roster_similarity(rosters: List[Set[str]]) -> Optional[float]:
    """Mean pairwise Jaccard similarity over crew rosters.

    Jaccard is shared agents divided by combined agents: 1.0 for identical
    rosters, 0.0 for fully disjoint ones. A value near 1.0 means the
    meta-agent converges on essentially one crew.

    Args:
        rosters: One set of agent names per run.

    Returns:
        Mean pairwise Jaccard index, or None if fewer than two rosters.
    """
    if len(rosters) < 2:
        return None
    total = 0.0
    pairs = 0
    for i in range(len(rosters)):
        for j in range(i + 1, len(rosters)):
            union = rosters[i] | rosters[j]
            if not union:
                continue
            total += len(rosters[i] & rosters[j]) / len(union)
            pairs += 1
    return total / pairs if pairs else None


def summarise(items: Iterable[Tuple[str, Dict[str, Any]]],
              normalize: bool = True,
              family_depth: int = 0,
              role_map: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Aggregate crew size, roster diversity and topology adherence.

    Args:
        items: Iterable of (source label, run document) pairs. The source
            label is retained against every statistic so each reported figure
            can be traced back to the files that produced it.
        normalize: Canonicalise agent names before comparison.
        family_depth: If non-zero, additionally report roster diversity with
            names grouped by their leading ``family_depth`` tokens, as a
            sensitivity check on the exact-name result.

    Returns:
        Dict of aggregate statistics, each accompanied by source provenance.
    """
    sizes: List[int] = []
    rosters: List[Set[str]] = []
    role_counts: Dict[str, int] = {}
    role_sources: Dict[str, List[str]] = {}
    roster_counts: Dict[str, int] = {}
    roster_sources: Dict[str, List[str]] = {}
    size_sources: Dict[int, List[str]] = {}
    name_variants: Dict[str, Set[str]] = {}
    role_texts: Dict[str, Set[str]] = {}
    family_rosters: List[Set[str]] = []
    family_counts: Dict[str, int] = {}
    family_run_counts: Dict[str, int] = {}
    family_names: Dict[str, Set[str]] = {}
    mismatch_sources: List[str] = []
    no_reflection_sources: List[str] = []
    per_run: List[Dict[str, Any]] = []
    offenders: List[Dict[str, Any]] = []
    n_runs = 0
    dm = {"total": 0, "permitted": 0, "unpermitted": 0,
          "unknown": 0, "self_addressed": 0,
          "undelivered": 0, "excluded": 0}

    for source, run in items:
        crew = extract_crew(run, normalize)
        if not crew["agents"]:
            continue
        n_runs += 1

        declared = crew["declared_size"]
        observed = crew["observed_size"]
        size = declared if declared is not None else observed
        if declared is not None and declared != observed:
            mismatch_sources.append(source)
        sizes.append(size)
        size_sources.setdefault(size, []).append(source)

        if not crew["has_reflection"]:
            no_reflection_sources.append(source)

        names = set(crew["agents"])
        rosters.append(names)
        key = " + ".join(display_name(n) for n in sorted(names))
        roster_counts[key] = roster_counts.get(key, 0) + 1
        roster_sources.setdefault(key, []).append(source)
        for name, info in crew["agents"].items():
            role_counts[name] = role_counts.get(name, 0) + 1
            role_sources.setdefault(name, []).append(source)
            name_variants.setdefault(name, set()).add(info["raw_name"])
            if info["role"]:
                role_texts.setdefault(name, set()).add(info["role"])

        if family_depth or role_map:
            fam = {family_of(n, role_map, family_depth) for n in names}
            family_rosters.append(fam)
            fam_key = " + ".join(display_name(f) for f in sorted(fam))
            family_counts[fam_key] = family_counts.get(fam_key, 0) + 1
            for f in fam:
                family_run_counts[f] = family_run_counts.get(f, 0) + 1
            for n in names:
                family_names.setdefault(
                    family_of(n, role_map, family_depth), set()).add(n)

        counts = topology_adherence(run, crew, normalize)
        for field in dm:
            dm[field] += counts[field]
        for offender in counts["offenders"]:
            offender["source"] = source
            offenders.append(offender)

        per_run.append({
            "source": source,
            "declared_size": declared,
            "observed_size": observed,
            "roster": sorted(display_name(n) for n in names),
            "raw_roster": sorted(i["raw_name"]
                                 for i in crew["agents"].values()),
            "has_reflection": crew["has_reflection"],
            "dm_total": counts["total"],
            "dm_permitted": counts["permitted"],
            "offenders": counts["offenders"],
        })

    adherence = (dm["permitted"] / dm["total"]) if dm["total"] else None
    core = {n: c for n, c in role_counts.items()
            if c >= CORE_ROLE_THRESHOLD * max(n_runs, 1)}
    merged = {display_name(n): sorted(v)
              for n, v in name_variants.items() if len(v) > 1}

    return {
        "n_runs": n_runs,
        "normalized": normalize,
        "size_mean": sum(sizes) / len(sizes) if sizes else None,
        "size_min": min(sizes) if sizes else None,
        "size_max": max(sizes) if sizes else None,
        "size_counts": {str(s): sizes.count(s) for s in sorted(set(sizes))},
        "size_sources": {str(k): v for k, v in sorted(size_sources.items())},
        "size_mismatch_sources": mismatch_sources,
        "no_reflection_sources": no_reflection_sources,
        "distinct_rosters": len(roster_counts),
        "roster_counts": dict(sorted(roster_counts.items(),
                                     key=lambda kv: -kv[1])),
        "roster_sources": roster_sources,
        "mean_roster_jaccard": roster_similarity(rosters),
        "role_counts": {display_name(n): c for n, c in
                        sorted(role_counts.items(), key=lambda kv: -kv[1])},
        "role_sources": {display_name(n): v for n, v in role_sources.items()},
        "core_roles": {display_name(n): c for n, c in
                       sorted(core.items(), key=lambda kv: -kv[1])},
        "merged_name_variants": merged,
        "role_descriptions": {display_name(n): sorted(v)
                              for n, v in sorted(role_texts.items())},
        "family_depth": family_depth,
        "family_mode": ("role-map" if role_map else
                        (f"leading {family_depth} token(s)"
                         if family_depth else None)),
        "unmapped_roles": (sorted(display_name(n) for n in role_counts
                                  if n not in role_map)
                           if role_map else []),
        "distinct_family_rosters": len(family_counts) or None,
        "family_roster_counts": dict(sorted(family_counts.items(),
                                            key=lambda kv: -kv[1])),
        "mean_family_jaccard": roster_similarity(family_rosters),
        "family_table": [
            {
                "family": display_name(f),
                "n_names": len(family_names.get(f, ())),
                "n_runs": c,
                "example": display_name(max(
                    family_names.get(f, {f}),
                    key=lambda n: role_counts.get(n, 0))),
            }
            for f, c in sorted(family_run_counts.items(),
                               key=lambda kv: (-kv[1], kv[0]))
        ],
        "dm_counts": dm,
        "adherence_rate": adherence,
        "offenders": offenders,
        "per_run": per_run,
    }

_ALERT_RE = re.compile(r"ILLEGAL MOVE ALERT: The proposed move '[^']*'"
                       r" is not legal in the current position\. "
                       r"Valid moves are: [^.]*\.")


def illegal_alerts(run: Dict[str, Any]) -> int:
    """Count distinct illegal-move alerts seen anywhere in a run's prompts.

    Args:
        run: A run document containing a ``moves`` map with ``calls``.

    Returns:
        Number of distinct alerts, i.e. rejected consensus proposals.
    """
    return len({m.group(0) for _, _, call in _iter_calls(run)
                for m in _ALERT_RE.finditer(call.get("user_prompt") or "")})