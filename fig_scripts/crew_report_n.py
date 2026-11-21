"""Report generated-crew composition over the dashboard's cached run JSON.

Usage::

    python crew_report.py path/to/cache/            # a directory of .json files
    python crew_report.py run1.json run2.json       # explicit files
    python crew_report.py cache/ --group-by difficulty
    python crew_report.py cache/ --json out.json
    python crew_report.py cache/ --role-map map.json # use a hand-assigned role map (template emitted by --emit-role-map)
"""

import argparse
import json
import os
import sys
from typing import (Any, Dict, Iterable, Iterator, List, Optional,
                    Tuple)

from crew_composition_n import (CORE_ROLE_THRESHOLD,
                              canonical_name,
                              summarise, illegal_alerts)


GROUP_CANDIDATES = (
    "experiment.hardness",
    "hardness",
    "experiment.config.hardness",
    "config.hardness",
    "experiment.stockfish_elo",
    "stockfish_elo",
    "difficulty",
    "difficulty_tier",
)

#: Paths probed to tell a handcrafted crew from a meta-agent-generated one.
#: A populated preset crew name or path means the roster was supplied rather
#: than generated at runtime.
PRESET_PATHS = (
    "experiment.config.preset_crew_name",
    "experiment.config.preset_crew_path",
    "config.preset_crew_name",
    "config.preset_crew_path",
    "preset_crew_name",
    "preset_crew_path",
)

#: Paths probed for the run mode, the primary crew-type discriminator.
MODE_PATHS = (
    "experiment.config.mode",
    "experiment.mode",
    "config.mode",
    "mode",
)

#: Known ``mode`` values mapped to crew type. ``hardcoded_multi`` supplies a
#: preset roster; ``multi`` has the meta-agent generate one at runtime.
MODE_CLASSES = {
    "single": "single",
    "single_agent": "single",
    "multi": "generated",
    "generated_multi": "generated",
    "hardcoded_multi": "handcrafted",
    "hardcoded": "handcrafted",
    "preset_multi": "handcrafted",
}

#: Paths probed for the run identifier.
ID_PATHS = (
    "experiment_id",
    "experiment.experiment_id",
    "experiment.config.experiment_id",
)

_MISSING = object()


def _get_path(doc: Dict[str, Any], path: str) -> Optional[Any]:
    """Resolve a dotted key path against a nested dict."""
    node: Any = doc
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _resolve(doc: Dict[str, Any],
             paths: Iterable[str]) -> Tuple[Optional[str], Any]:
    """Return the first path that exists in the document, and its value.

    Distinguishes an absent key from a key present with a null value, which
    is the difference between "cannot classify" and "generated crew".

    Args:
        doc: Run document.
        paths: Candidate dotted paths, in priority order.

    Returns:
        (matched path, value) or (None, _MISSING) if no path exists.
    """
    for path in paths:
        node: Any = doc
        ok = True
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                ok = False
                break
            node = node[part]
        if ok:
            return path, node
    return None, _MISSING


def classify_run(run: Dict[str, Any]) -> Dict[str, Any]:
    """Classify a run as handcrafted, generated, single or unknown.

    ``mode`` is the primary discriminator because it states the crew type
    explicitly. The preset crew name is evaluated independently as a
    cross-check, and any disagreement between the two is reported rather than
    silently resolved.

    Args:
        run: Run document.

    Returns:
        Dict with ``kind``, the raw ``mode`` and ``preset`` values, and a
        ``conflict`` flag set when the two discriminators disagree.
    """
    _, mode = _resolve(run, MODE_PATHS)
    mode_str = mode.lower() if isinstance(mode, str) else None
    by_mode = MODE_CLASSES.get(mode_str) if mode_str else None

    path, preset = _resolve(run, PRESET_PATHS)
    if path is None:
        by_preset = None
        preset_value = None
    else:
        preset_value = preset
        by_preset = ("generated" if preset in (None, "", "null")
                     else "handcrafted")

    kind = by_mode or by_preset or "unknown"
    conflict = (by_mode is not None and by_preset is not None
                and by_mode != by_preset and by_mode != "single")

    return {"kind": kind, "mode": mode if mode is not _MISSING else None,
            "preset": preset_value, "conflict": conflict,
            "by_mode": by_mode, "by_preset": by_preset}


def _find_runs(node: Any, depth: int = 0) -> Iterator[Dict[str, Any]]:
    """Yield every object containing a ``moves`` map, at any nesting depth."""
    if depth > 4:
        return
    if isinstance(node, dict):
        if isinstance(node.get("moves"), dict):
            yield node
            return
        for value in node.values():
            yield from _find_runs(value, depth + 1)
    elif isinstance(node, list):
        for item in node:
            yield from _find_runs(item, depth + 1)


def load_runs(paths: List[str],
              exclude: Optional[str] = None
              ) -> List[Tuple[str, Dict[str, Any]]]:
    """Load run documents from files or directories of JSON.

    Args:
        paths: File or directory paths.
        exclude: Absolute path to skip (typically this run's own output file,
            which would otherwise be picked up on a re-run).

    Returns:
        List of (source filename, run document) pairs.
    """
    skip = os.path.abspath(exclude) if exclude else None
    files: List[str] = []
    for path in paths:
        if os.path.isdir(path):
            files.extend(
                os.path.join(path, name)
                for name in sorted(os.listdir(path))
                if name.endswith(".json"))
        else:
            files.append(path)

    runs: List[Tuple[str, Dict[str, Any]]] = []
    for path in files:
        if skip and os.path.abspath(path) == skip:
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                doc = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  ! skipped {os.path.basename(path)}: {exc}",
                  file=sys.stderr)
            continue
        found = list(_find_runs(doc))
        if not found:
            print(f"  ! no 'moves' object in {os.path.basename(path)}",
                  file=sys.stderr)
        base = os.path.basename(path)
        for index, run in enumerate(found):
            label = base if len(found) == 1 else f"{base}#{index}"
            runs.append((label, run))
    return runs


def detect_group_field(runs: List[Tuple[str, Dict[str, Any]]]) -> Optional[str]:
    """Probe common field names for a difficulty-tier key.

    Retained for the optional independence check only; grouping is never
    applied automatically.
    """
    for candidate in GROUP_CANDIDATES:
        if any(_get_path(run, candidate) is not None for _, run in runs):
            return candidate
    return None


def _fmt(value: Any, spec: str = ".3f") -> str:
    """Format a value, tolerating None."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return format(value, spec)
    return str(value)


def print_report(label: str, stats: Dict[str, Any],
                 verbose: bool = False) -> None:
    """Print one group's statistics, optionally with per-run provenance.

    Args:
        label: Heading for this group.
        stats: Result of :func:`crew_composition.summarise`.
        verbose: Print the source file behind every figure.
    """
    print(f"\n=== {label} ===")
    print(f"runs parsed          : {stats['n_runs']}")
    if not stats["n_runs"]:
        return
    if not stats["normalized"]:
        print("  (name normalisation DISABLED: separator/case variants "
              "count as distinct roles)")

    print(f"crew size            : mean {_fmt(stats['size_mean'], '.2f')}, "
          f"range {stats['size_min']}-{stats['size_max']}")
    for size, count in stats["size_counts"].items():
        line = f"    size {size}: {count} run(s)"
        if verbose:
            line += "  <- " + ", ".join(stats["size_sources"].get(size, []))
        print(line)
    if stats["size_mismatch_sources"]:
        print(f"  ! declared vs observed size mismatch in "
              f"{len(stats['size_mismatch_sources'])} run(s): "
              f"{', '.join(stats['size_mismatch_sources'][:5])}")
    if stats["no_reflection_sources"]:
        print(f"  ! {len(stats['no_reflection_sources'])} run(s) lack the "
              f"reflection step (prototype leak?): "
              f"{', '.join(stats['no_reflection_sources'][:5])}")

    if stats["merged_name_variants"]:
        merged = stats["merged_name_variants"]
        print(f"name variants merged : {len(merged)} role(s)")
        shown = list(merged.items()) if verbose else list(merged.items())[:8]
        for name, variants in shown:
            print(f"    {name}: {variants}")
        if len(shown) < len(merged):
            print(f"    ... {len(merged) - len(shown)} more "
                  f"(use --verbose to list)")

    print(f"distinct rosters     : {stats['distinct_rosters']} "
          f"across {stats['n_runs']} runs")
    print(f"mean roster Jaccard  : {_fmt(stats['mean_roster_jaccard'])}"
          f"   (1.0 = every run the same crew, 0.0 = no shared agents)")
    limit = None if verbose else 5
    for roster, count in list(stats["roster_counts"].items())[:limit]:
        share = count / stats["n_runs"]
        print(f"    {count:3d} ({share:5.1%})  {roster}")
        if verbose:
            print(f"              <- "
                  f"{', '.join(stats['roster_sources'].get(roster, []))}")
    if not verbose and stats["distinct_rosters"] > 5:
        print(f"    ... {stats['distinct_rosters'] - 5} more "
              f"(use --verbose to list)")

    if stats.get("family_mode"):
        print(f"-- sensitivity: names grouped by {stats['family_mode']} --")
        if stats.get("unmapped_roles"):
            print(f"  ! {len(stats['unmapped_roles'])} name(s) absent from "
                  f"the role map, kept ungrouped: "
                  f"{', '.join(stats['unmapped_roles'][:6])}")
        print(f"distinct rosters     : {stats['distinct_family_rosters']} "
              f"across {stats['n_runs']} runs")
        print(f"mean roster Jaccard  : "
              f"{_fmt(stats['mean_family_jaccard'])}")
        fam_rosters = stats["family_roster_counts"]
        shown = (list(fam_rosters.items()) if verbose
                 else list(fam_rosters.items())[:5])
        for roster, count in shown:
            print(f"    {count:3d} ({count / stats['n_runs']:5.1%})  {roster}")
        if len(shown) < len(fam_rosters):
            print(f"    ... {len(fam_rosters) - len(shown)} more "
                  f"(use --verbose to list)")

        table = stats.get("family_table") or []
        if table:
            print("\n  per-family totals (appendix table):")
            print(f"    {'Family':<24} {'Example role name':<26} "
                  f"{'Names':>5} {'Runs':>13}")
            for row in table:
                pct = row["n_runs"] / stats["n_runs"]
                print(f"    {row['family']:<24} {row['example']:<26} "
                      f"{row['n_names']:>5} {row['n_runs']:>6} ({pct:5.1%})")
            print(f"    {'TOTAL':<24} {'':<26} "
                  f"{sum(r['n_names'] for r in table):>5}")
        print("-- end sensitivity --")

    print(f"core roles (>={int(CORE_ROLE_THRESHOLD * 100)}% of runs):")
    if not stats["core_roles"]:
        print("    (none)")
    for name, count in list(stats["core_roles"].items())[:12]:
        print(f"    {count:3d} ({count / stats['n_runs']:5.1%})  {name}")
        if verbose:
            print(f"              <- "
                  f"{', '.join(stats['role_sources'].get(name, []))}")

    dm = stats["dm_counts"]
    print(f"direct messages      : {dm['total']}")
    print(f"  permitted          : {dm['permitted']}")
    print(f"  unpermitted (real) : {dm['unpermitted']}")
    print(f"  unknown recipient  : {dm['unknown']}")
    print(f"  self-addressed     : {dm['self_addressed']}")
    print(f"  not delivered      : {dm['undelivered']}")
    print(f"excluded talk calls  : {dm['excluded']}  (no recipient or no text)")
    print(f"adherence rate       : {_fmt(stats['adherence_rate'], '.4f')}")

    if stats["offenders"]:
        shown = stats["offenders"] if verbose else stats["offenders"][:5]
        print(f"offenders ({len(stats['offenders'])} total):")
        for off in shown:
            cycle = ("" if off.get("cycle") is None
                     else f", cycle {off['cycle']}")
            print(f"    [{off['kind']}] {off['sender']} -> "
                  f"{off['recipient']}")
            print(f"        at: {off['source']}  "
                  f"moves.{off['move']}.calls[{off['call_index']}]"
                  f".tool_calls[{off['tool_index']}]  "
                  f"(caller_id {off.get('caller_id')}{cycle})")
            if verbose:
                print(f"        sender permitted: {off['permitted']}")
                print(f"        run roster      : {off['roster']}")
                print(f"        delivered       : {off['delivered']}")

    if verbose:
        print("\nper-run detail  "
              "(size = meta-declared/observed agents, "
              "dm = permitted/total direct messages):")
        for row in stats["per_run"]:
            size = (f"{row['declared_size']}/{row['observed_size']}"
                    if row["declared_size"] is not None
                    else f"?/{row['observed_size']}")
            print(f"    {row['source']}  size={size}  "
                  f"dm={row['dm_permitted']}/{row['dm_total']}  "
                  f"reflection={row['has_reflection']}")
            print(f"        raw roster: {row['raw_roster']}")


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+",
                        help="JSON files or directories of cached runs")
    parser.add_argument("--group-by", default=None,
                        help=("optional dotted field path to split on, e.g. "
                              "experiment.hardness. Off by default: crew "
                              "generation cannot depend on fields the "
                              "meta-agent never sees, so pooling is correct "
                              "for reporting. Useful as an independence "
                              "check."))
    parser.add_argument("--json", dest="json_out", default=None,
                        help="also write the raw statistics to this path")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show the source file behind every figure")
    parser.add_argument("--raw-names", action="store_true",
                        help="disable name canonicalisation (debug)")
    parser.add_argument("--family-depth", type=int, default=0,
                        help=("also report roster diversity with names grouped "
                              "by their leading N tokens, e.g. 2 groups "
                              "'anti computer specialist' with 'anti computer "
                              "pragmatist'. Sensitivity check only; 0 = off"))
    parser.add_argument("--role-map", default=None,
                        help=("JSON mapping canonical role name -> family "
                              "label. An author-assigned grouping, which "
                              "should be published with results. Overrides "
                              "--family-depth"))
    parser.add_argument("--emit-role-map", default=None,
                        help=("write a starter role-map skeleton (every "
                              "distinct name, with its descriptions as "
                              "hints) to this path and exit"))
    parser.add_argument("--dump-roles", action="store_true",
                        help=("print every distinct role name with its "
                              "verbatim description(s) and exit"))
    parser.add_argument("--crew-type", default="generated",
                        choices=("generated", "handcrafted", "single",
                                 "unknown", "all"),
                        help=("which runs to report on (default: generated)"))
    parser.add_argument("--illegal", action="store_true",
                        help=("count distinct illegal-move alerts (rejected "
                                "consensus proposals) per MAS crew type, and "
                                "exit"))
    args = parser.parse_args(argv)

    runs = load_runs(args.paths, exclude=args.json_out)
    if not runs:
        print("no runs found", file=sys.stderr)
        return 1
    print(f"loaded {len(runs)} run document(s)")

    buckets: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    modes: Dict[str, Dict[str, int]] = {}
    presets: Dict[str, Dict[str, int]] = {}
    conflicts: List[str] = []
    for source, run in runs:
        info = classify_run(run)
        kind = info["kind"]
        buckets.setdefault(kind, []).append((source, run))
        mode_key = str(info["mode"]) if info["mode"] is not None else "<absent>"
        preset_key = ("<null>" if info["preset"] in (None, "", "null")
                      else str(info["preset"]))
        modes.setdefault(kind, {})
        modes[kind][mode_key] = modes[kind].get(mode_key, 0) + 1
        presets.setdefault(kind, {})
        presets[kind][preset_key] = presets[kind].get(preset_key, 0) + 1
        if info["conflict"]:
            conflicts.append(f"{source} (mode->{info['by_mode']}, "
                             f"preset->{info['by_preset']})")

    print("crew-type breakdown (mode is primary, preset is cross-check):")
    for kind in sorted(buckets):
        mode_str = ", ".join(f"{v} x{c}" for v, c in
                             sorted(modes[kind].items(), key=lambda kv: -kv[1]))
        preset_str = ", ".join(
            f"{os.path.basename(str(v))} x{c}" for v, c in
            sorted(presets[kind].items(), key=lambda kv: -kv[1]))
        print(f"    {kind:12s} {len(buckets[kind]):4d} run(s)")
        print(f"        mode  : {mode_str}")
        print(f"        preset: {preset_str}")
    if conflicts:
        print(f"  ! {len(conflicts)} run(s) where mode and preset disagree:")
        for line in conflicts[:5]:
            print(f"      {line}")
    if "unknown" in buckets:
        print("  ! some runs have neither a recognised mode nor a preset "
              "field; excluded unless --crew-type all")

    if args.illegal:
        for kind in ("generated", "handcrafted"):
            counts = [(s, illegal_alerts(r)) for s, r in buckets.get(kind, [])]
            hit = [f"{s} ({n})" for s, n in counts if n]
            print(f"{kind}: {sum(n for _, n in counts)} alert(s) in "
                  f"{len(hit)}/{len(counts)} run(s)"
                  + (f"  <- {', '.join(hit)}" if hit else ""))
        return 0

    if args.crew_type != "all":
        runs = buckets.get(args.crew_type, [])
        print(f"filtered to crew-type '{args.crew_type}': {len(runs)} run(s)")
        if not runs:
            print("nothing to report after filtering", file=sys.stderr)
            return 1

    group_field = args.group_by
    if group_field:
        print(f"grouping by: {group_field}")

    role_map: Optional[Dict[str, str]] = None
    if args.role_map:
        with open(args.role_map, encoding="utf-8") as handle:
            raw_map = json.load(handle)
        role_map = {canonical_name(k): v for k, v in raw_map.items()
                    if v and not k.startswith("_")} # type: ignore
        print(f"loaded role map: {len(role_map)} name(s) -> " # type: ignore
              f"{len(set(role_map.values()))} family/families") # type: ignore

    if args.emit_role_map:
        stats = summarise(runs, normalize=not args.raw_names)
        skeleton = {
            "_comment": ("Fill in a family label for each role name, then "
                         "pass this file with --role-map. Publish it "
                         "alongside your results."),
            "_descriptions": stats["role_descriptions"],
        }
        for name in stats["role_counts"]:
            skeleton[name] = ""
        with open(args.emit_role_map, "w", encoding="utf-8") as handle:
            json.dump(skeleton, handle, indent=2)
        print(f"wrote skeleton for {len(stats['role_counts'])} role name(s) "
              f"to {args.emit_role_map}")
        return 0

   
    if args.dump_roles:
        stats = summarise(runs, normalize=not args.raw_names)
        counts = stats["role_counts"]
        print(f"\n{len(counts)} distinct role name(s), "
              f"grouped by leading token:\n")
        by_head: Dict[str, List[str]] = {}
        for name in counts:
            by_head.setdefault(name.split()[0].lower(), []).append(name)
        for head in sorted(by_head):
            group = sorted(by_head[head], key=lambda n: -counts[n])
            if len(group) > 1:
                print(f"[{head}] -- {len(group)} similar names")
            for name in group:
                print(f"  {counts[name]:3d} runs  {name}")
                for text in stats["role_descriptions"].get(name, []):
                    print(f"           \"{text}\"")
            print()
        return 0

    results: Dict[str, Any] = {"overall": summarise(runs, normalize=not args.raw_names,
                                   family_depth=args.family_depth,
                                   role_map=role_map)}
    print_report("ALL RUNS", results["overall"], args.verbose)

    if group_field:
        groups: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        for source, run in runs:
            key = str(_get_path(run, group_field))
            groups.setdefault(key, []).append((source, run))
        for key in sorted(groups):
            stats = summarise(groups[key],
                              normalize=not args.raw_names,
                              family_depth=args.family_depth,
                              role_map=role_map)
            results[key] = stats
            print_report(f"{group_field} = {key}", stats, args.verbose)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, default=str)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
