#!/usr/bin/env python3

#NOTE: ASSUMES THIS FILE IS IN THE figs_scripts directory.
"""
Pulls the ChessExperiments collection (experiment docs + moves subcollection +
chunked-calls sub-subcollections) from the named Firestore database into a local
cache folder.


Flags:
  --collection NAME   Firestore collection to analyse
                      (default: ChessExperiments)
  --prototype NAME    keep only runs whose prototype_id is NAME; omit to keep
                      the baseline runs, which carry no prototype_id
  --refresh           discard the cache and re-pull
  --cached            use the existing cache, skip the gcloud check

READ-ONLY GUARANTEE
-------------------
This script NEVER writes to Firestore. The only Firestore operations used are
`collection().stream()` and document reads. The only thing it writes is the
local cache folder (LOCAL_CACHE_DIR), which it owns and may clear/recreate.

Caching
-------
A successful pull writes a manifest into LOCAL_CACHE_DIR. On the next run, if
the manifest exists, is complete, and is younger than CACHE_MAX_AGE_HOURS, the
cached data is reused. Otherwise the folder is cleared and re-pulled in full.
Pulls favour reliability over speed: generous timeouts, per-experiment retries,
and one JSON file per experiment written as soon as it is fetched.

Prerequisites: `gcloud` logged in with ADC for the project configured as
GCLOUD_PROJECT in CrewForge/.env (the script sanity-checks this before pulling).
"""

from __future__ import annotations

import json
import math
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast
import numpy as np
import chess
import chess.engine

# =============================================================================
# CONFIGURATION
# =============================================================================

_ROOT = Path(__file__).resolve().parent
ENV_FILE = _ROOT / ".." / ".env"          # holds GCLOUD_PROJECT (same file run_gcc_experiments.py uses)

FIRESTORE_DATABASE = "experiment-data"           # named DB (NOT "(default)")

# Which sweep the script is about. Defaults to the baseline; override on the
# command line (see configure_sweep and the entry point at the bottom):
#
#   python3 gcc_dashboard.py --collection ChessExperiments_p4_refutation \
#                            --prototype p4_refutation
#
# Every path below is derived from the collection, so a prototype pull can never
# overwrite the baseline cache or focused set in place.
DEFAULT_COLLECTION = "ChessExperiments"
FIRESTORE_COLLECTION = DEFAULT_COLLECTION
# None keeps the baseline runs (prototype_id absent); a string keeps ONLY the
# runs carrying that prototype_id.
FOCUS_PROTOTYPE: Optional[str] = None

LOCAL_CACHE_DIR = _ROOT / "gcc_dashboard_data"   # owned by this script; safe to clear
CACHE_MAX_AGE_HOURS = 24.0
MANIFEST_NAME = "_manifest.json"

TB_MAX_CHARS = 20000          # traceback chars embedded per run (full copy stays in Firestore)

STREAM_TIMEOUT_S = 300        # generous per-stream timeout (reliability over speed)
PULL_MAX_RETRIES = 4          # per-experiment retry budget
PULL_BASE_DELAY_S = 2.0

# Eval clamp used ONLY for averaging position evals (mate is stored as +/-30000,
# which would otherwise dominate any mean). Clamped metrics are labelled as such.
EVAL_CLAMP_CP = 1500

# Substrings that deterministically identify a Firestore 1 MiB doc-limit failure.
FS_1MB_PATTERNS = ("exceeds the maximum allowed size", "1048576")

W = 100  # report width

# =============================================================================
# FILTERING CONSTANTS
# =============================================================================

FILTER_VERSION = "focus-1.0"          # bump when selection rules change

FOCUS_MODEL = "gemini-3.5-flash"
FOCUS_MODES = ("single", "multi", "hardcoded_multi")
# Difficulty is matched on the CONFIGURED ENGINE, not the label alone, because the
# label definitions changed over time. These are the current definitions.
FOCUS_DIFFICULTIES: Dict[str, Tuple[int, int]] = {   # hardness -> (elo, play_depth)
    "low": (1320, 1),
    "medium": (1400, 2),
    "high": (1500, 3),
}

FOCUSED_DIR = _ROOT / "gcc_focused_data"             # owned by this script
FOCUSED_RUNS_DIR = FOCUSED_DIR / "focused_runs"      # verbatim accepted payloads
FOCUSED_META_DIR = FOCUSED_DIR / "meta"              # selection report + manifest
FOCUSED_ANALYSIS_DIR = FOCUSED_DIR / "analysis_cache"  # survives rebuilds (stockfish results)
# The engine's best move is a pure function of the position, so every sweep
# shares one cache: a prototype run costs nothing for positions already analysed.
BEST_MOVES_CACHE = _ROOT / "gcc_shared_cache" / "best_moves_depth20.json"


def configure_sweep(collection: str, prototype: Optional[str] = None) -> None:
    """Point the script at one sweep, recomputing every path derived from it.

    Called once from the entry point before anything touches the cache. The
    baseline keeps its historical directory names so existing caches stay valid;
    any other collection gets a suffixed set of its own, which is what stops a
    prototype pull from overwriting the baseline in place.
    """
    global FIRESTORE_COLLECTION, FOCUS_PROTOTYPE
    global LOCAL_CACHE_DIR, FOCUSED_DIR, FOCUSED_RUNS_DIR
    global FOCUSED_META_DIR, FOCUSED_ANALYSIS_DIR

    FIRESTORE_COLLECTION = collection
    FOCUS_PROTOTYPE = prototype or None
    suffix = ("" if collection == DEFAULT_COLLECTION
              else "_" + collection.replace(DEFAULT_COLLECTION + "_", ""))
    LOCAL_CACHE_DIR = _ROOT / f"gcc_dashboard_data{suffix}"
    FOCUSED_DIR = _ROOT / f"gcc_focused_data{suffix}"
    FOCUSED_RUNS_DIR = FOCUSED_DIR / "focused_runs"
    FOCUSED_META_DIR = FOCUSED_DIR / "meta"
    FOCUSED_ANALYSIS_DIR = FOCUSED_DIR / "analysis_cache"
    # BEST_MOVES_CACHE is deliberately NOT suffixed: the engine's best move is a
    # pure function of the position, so every sweep shares one cache.


# Eval clamp for averaging/plotting (mate = +/-30000 would dominate any mean).
# Raw values are preserved in per-run data; clamped metrics are labelled.
EVAL_CLAMP_CP = 1000

# =============================================================================
# CAP CONSTANTS
# =============================================================================

CELL_CAP_N = 10
CELL_CAP_KEEP = "latest"      # "latest" | "earliest" — which end of the cell to keep

# =============================================================================
# WIN% AND LICHESS CONSTANTS
# =============================================================================
# Judgment cutoffs, in Win% POINTS SHED by a single move (Lichess Lila / the
# paper's Table 2). These replace the orchestrator's centipawn cutoffs
# (10/30/80/150 cp) for every quality and blunder statistic in the report.
INACCURACY_DWP = 10.0
MISTAKE_DWP = 20.0
BLUNDER_DWP = 30.0
WP_QUALITIES = ("ok", "inaccuracy", "mistake", "blunder")   # best -> worst

# Point of no return: the position is treated as lost once the LLM's winning
# chances sit below this and never recover. 20 Win% ~= -377 cp at the balance
# point, i.e. materially stricter than the -150 cp rule it replaces.
PONR_WIN_PCT = 20.0

# =============================================================================
# GAME ANALYSIS CONSTANTS
# =============================================================================
GAME_CATEGORIES = ("win", "loss", "draw", "ply_cap_incomplete")   # produced a full game record
DECIDED_CATEGORIES = ("win", "loss", "draw")                       # has a chess result
SCORE_MAP = {"win": 1.0, "draw": 0.5, "loss": 0.0}
# Phase definitions:
#   opening    = plies 1..OPENING_PLIES of the game
#   endgame    = positions where the side about to be evaluated has a board with
#                <= ENDGAME_PIECES total pieces (both sides, incl. kings/pawns)
#   middlegame = everything else
OPENING_PLIES = 20            # first 10 full moves
ENDGAME_PIECES = 12

# Paper-replication (RefMaterial/arXiv-2512.01992v1) settings — their Appendix values.
PAPER_SF_DEPTH = 20                  # their analysis depth
PAPER_SF_HASH_MB = 128
PAPER_SF_THREADS = 1
PAPER_MATE_CP = 1000                 # they map any mate score to +/-1000 cp
PAPER_WHITE_ADV_ELO = 35             # white-advantage correction; we SUBTRACT (our LLM is White)



# --- Win% scale (Lichess accuracy model; used by LLM Chess, arXiv:2512.01992)
# A centipawn score is not comparable across game stages: -300 cp in a balanced
# middlegame is a catastrophe, -300 cp in an already-lost endgame is noise. Win%
# maps cp onto the expected-score axis, where a fixed number of points shed means
# the same thing everywhere in the game. Every game-level "how badly is it going"
# and "when did it go wrong" metric in this report is computed on this scale.
#
#   Win% = 50 + 50 * (2 / (1 + exp(-K * cp)) - 1)      cp from the LLM's (White's) POV
#
WIN_PCT_K = 0.00368208        # Lichess' fitted constant
MATE_ABS_CP = 20000           # |cp| above this is a stored mate score (+/-30000)
MATE_EQUIV_CP = 1000          # mate maps to this cp before conversion (the paper's rule)


def _phase_of(ply: int, pieces_before: Optional[int]) -> str:
    if ply is not None and ply <= OPENING_PLIES:
        return "opening"
    if pieces_before is not None and pieces_before <= ENDGAME_PIECES:
        return "endgame"
    return "middlegame"

def lichess_win_pct(cp: float) -> float:
    """Expected score (0-100) for the side the cp score favours (White = LLM).

    Stored mate scores (+/-30000) are mapped to +/-MATE_EQUIV_CP first, matching
    the reference paper. Note this matters far less on the Win% scale than on the
    cp scale: 1000 cp is already 97.5%, so the entire mate-vs-huge-advantage
    ambiguity that plagues centipawn clamping is worth <= 2.5 points here.
    """
    if abs(cp) > MATE_ABS_CP:
        cp = MATE_EQUIV_CP if cp > 0 else -MATE_EQUIV_CP
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-WIN_PCT_K * cp)) - 1.0)


def delta_win_pct(cp_before: float,
                  cp_after: float) -> float:
    """Win% points SHED by a move: Win%(before) - Win%(after), floored at 0.

    Positive = the move gave winning chances away. Moves that improved the
    position score 0 (not a negative loss), so the quantity is a loss in the
    same sense centipawn loss was, and can be averaged/exceedance-plotted.
    """
    wb, wa = lichess_win_pct(cp_before), lichess_win_pct(cp_after)
    return max(0.0, wb - wa)


def win_pct_judgment(dwp: Optional[float]) -> Optional[str]:
    """Lichess Lila judgment from the Win% points a move shed."""
    if dwp is None:
        return None
    if dwp >= BLUNDER_DWP:
        return "blunder"
    if dwp >= MISTAKE_DWP:
        return "mistake"
    if dwp >= INACCURACY_DWP:
        return "inaccuracy"
    return "ok"

def clamp_mate_cp(cp: int) -> int:
    """Collapse stored mate scores to +/-MATE_EQUIV_CP, leaving real evals alone.

    Replaces the old blanket clamp to +/-EVAL_CLAMP_CP, which conflated a mate
    with a legitimately winning -1400 cp position (both became -1000).
    """
    if abs(cp) > MATE_ABS_CP:
        return MATE_EQUIV_CP if cp > 0 else -MATE_EQUIV_CP
    return int(cp)

def _percentiles(vals: List[float]) -> Optional[Dict[str, float]]:
    if not vals:
        return None
    arr = np.asarray(vals, dtype=float)
    q = np.percentile(arr, [25, 50, 75])
    return {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "p25": float(q[0]), "median": float(q[1]), "p75": float(q[2]),
            "min": float(arr.min()), "max": float(arr.max()), "n": len(vals)}


def dist_stats(vals: List[float]) -> Optional[Dict[str, Any]]:
    """Mean/sd/median/IQR/min/max"""
    s = _percentiles(vals)
    if s is None:
        return None
    return s

# =============================================================================
# Group aggregation + statistical comparisons
# =============================================================================

def summarize_group(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """All aggregate metrics for one group of analyzed runs (cells/margins)."""
    n = len(runs)
    cats = Counter(r["category"] for r in runs)
    game_runs = [r for r in runs if r["category"] in GAME_CATEGORIES]
    decided = [r for r in runs if r["category"] in DECIDED_CATEGORIES]
    wins = cats.get("win", 0)
    scores = [r["score"] for r in decided]

    acpls = [r["acpl_std"] for r in runs if r["acpl_std"] is not None]
    dwps = [r["avg_dwp"] for r in runs if r["avg_dwp"] is not None]
    blunder_rates = [r["blunder_rate"] for r in runs if r["blunder_rate"] is not None]
    pooled_sys = sum(r["n_sys"] for r in runs)
    pooled_blunders = sum(r["blunders"] for r in runs)
    quality = Counter()
    quality_wp = Counter()
    for r in runs:
        quality.update(r["quality"])
        quality_wp.update(r["quality_wp"])
    pooled_judged = sum(quality_wp.values())
    phase = {}
    for p in ("opening", "middlegame", "endgame"):
        # pooled per-move phase losses, on both scales
        tot_n = sum(r["phase"][p]["n"] for r in runs)
        tot_loss = sum((r["phase"][p]["acpl_std"] or 0) * r["phase"][p]["n"] for r in runs)
        tot_dwp = sum((r["phase"][p]["avg_dwp"] or 0) * r["phase"][p]["n"] for r in runs)
        tot_bl = sum(r["phase"][p]["blunders"] for r in runs)
        phase[p] = {"moves": tot_n,
                    "acpl_pooled": (tot_loss / tot_n) if tot_n else None,
                    "dwp_pooled": (tot_dwp / tot_n) if tot_n else None,
                    "blunder_rate_pooled": (tot_bl / tot_n) if tot_n else None}

    plies_completed = [r["n_plies"] for r in decided]
    tpm = [r["tokens_per_sys_move"] for r in runs if r["tokens_per_sys_move"] is not None]
    tpg = [r["total_tokens"] for r in runs if r["n_sys"] > 0]
    lpm = [r["lat_per_sys_move_s"] for r in runs if r["lat_per_sys_move_s"] is not None]
    cpm = [r["cycles_per_sys_move"] for r in runs if r["cycles_per_sys_move"] is not None]
    thinking = [r["thinking_share"] for r in runs if r["thinking_share"] is not None]

    return {
        "n": n,
        "categories": dict(cats),
        "n_game": len(game_runs), "n_decided": len(decided),
        "wins": wins, "losses": cats.get("loss", 0), "draws": cats.get("draw", 0),
        "ply_cap": cats.get("ply_cap_incomplete", 0),
        "consensus_failures": cats.get("consensus_failure", 0),
        "illegal_failures": cats.get("illegal_move_failure", 0),
        "win_rate": (wins / len(game_runs)) if game_runs else None,
        "score_mean": (sum(scores) / len(scores)) if scores else None,
        "score_pct": dist_stats(scores) if scores else None,
        "acpl": dist_stats(acpls),
        "dwp": dist_stats(dwps),
        "blunder_rate_per_game": dist_stats(blunder_rates),
        "blunder_rate_pooled": (pooled_blunders / pooled_sys) if pooled_sys else None,
        # The same denominator under the ORCHESTRATOR's centipawn definition
        # (loss >= 150 cp), for §8's side-by-side of the two blunder definitions.
        # Precomputed here rather than derived in the template: deriving it there is
        # exactly what let §8 pair a centipawn blunder COUNT with the Win% RATE.
        "blunder_rate_cp_pooled": ((quality.get("blunder", 0) / pooled_sys)
                                   if pooled_sys else None),
        "pooled_sys_moves": pooled_sys,
        "pooled_judged_moves": pooled_judged,
        "quality": dict(quality),
        "quality_wp": dict(quality_wp),
        "phase": phase,
        "plies_decided": dist_stats([float(x) for x in plies_completed]),
        "tokens_per_move": dist_stats(tpm),
        "tokens_per_game": dist_stats([float(x) for x in tpg]),
        "latency_per_move_s": dist_stats(lpm),
        "cycles_per_move": dist_stats(cpm),
        "thinking_share": dist_stats(thinking),
    }



# =============================================================================
# Cross-run pooled structures (trajectories, blunder timing, deliberation)
# =============================================================================

PLY_BUCKETS = [(1, 20), (21, 40), (41, 60), (61, 80), (81, 100),
               (101, 130), (131, 160), (161, 200)]


def _bucket_label(lo: int, hi: int) -> str:
    return f"{lo}–{hi}"


def eval_trajectory_bands(runs: List[Dict[str, Any]], max_ply: int = 200,
                          step: int = 2) -> List[Dict[str, Any]]:
    """Median + IQR of the position's Win% at sampled plies across a run group.

    Reported on the Win% scale rather than in centipawns: a cp median mixes
    early-game and late-game scores whose practical meaning differs by an order
    of magnitude, and the resulting band is dominated by a handful of
    already-decided games sitting at the clamp. Win% is bounded 0-100 and its
    quantiles are directly interpretable as "the median game's winning chances".

    Both scales are returned (cp for reference, Win% for plotting). Because Win%
    is a strictly monotone transform of cp, the Win% quantiles here are exactly
    the transformed cp quantiles — the change is one of axis, not of ranking.
    """
    import numpy as np
    out = []
    for ply in range(step, max_ply + 1, step):
        cps, wps = [], []
        for r in runs:
            # eval at the greatest stored ply <= target (game state at that time)
            best = None
            for p, cp, wp in r["evals"]:
                if p <= ply:
                    best = (cp, wp)
                else:
                    break
            if best is not None and r["evals"] and r["evals"][-1][0] >= ply:
                cps.append(best[0])
                wps.append(best[1])
        if len(wps) >= 3:
            qw = np.percentile(np.asarray(wps, dtype=float), [25, 50, 75])
            qc = np.percentile(np.asarray(cps, dtype=float), [25, 50, 75])
            out.append({"ply": ply, "n": len(wps),
                        "p25_wp": float(qw[0]), "median_wp": float(qw[1]),
                        "p75_wp": float(qw[2]),
                        "p25": float(qc[0]), "median": float(qc[1]),
                        "p75": float(qc[2])})
    return out


def blunder_timing(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Blunder RATE per ply bucket (pooled blunders / pooled system moves).

    A blunder is a system move shedding >= BLUNDER_DWP Win% points (move row
    column 13), not a fixed centipawn drop — otherwise late-game buckets are
    inflated by large cp swings in positions that were already resolved.
    """
    out = []
    for lo, hi in PLY_BUCKETS:
        moves_in = 0
        blunders_in = 0
        for r in runs:
            for row in r["moves"]:
                if row[2] == 1 and row[1] is not None and lo <= row[1] <= hi:
                    moves_in += 1
                    dwp = row[13]
                    if dwp is not None and dwp >= BLUNDER_DWP:
                        blunders_in += 1
        out.append({"bucket": _bucket_label(lo, hi), "moves": moves_in,
                    "blunders": blunders_in,
                    "rate": (blunders_in / moves_in) if moves_in else None})
    return out


def deliberation_by_quality(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """For multi-agent runs: mean cycles & tokens per move, split by the quality
    of the move that deliberation produced (pooled moves)."""
    acc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"cycles": [], "tokens": []})
    for r in runs:
        for row in r["moves"]:
            if row[2] == 1 and row[15]:          # 15 = Win% judgment
                acc[row[15]]["cycles"].append(row[11] or 0)
                acc[row[15]]["tokens"].append(row[9] or 0)
    return {q: {"n": len(v["cycles"]),
                "cycles_mean": (sum(v["cycles"]) / len(v["cycles"])) if v["cycles"] else None,
                "tokens_mean": (sum(v["tokens"]) / len(v["tokens"])) if v["tokens"] else None}
            for q, v in acc.items() if q in WP_QUALITIES}


def deliberation_by_bucket(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mean deliberation cycles per system move by ply bucket (consensus stress)."""
    out = []
    for lo, hi in PLY_BUCKETS:
        cyc = [row[11] or 0 for r in runs for row in r["moves"]
               if row[2] == 1 and row[1] is not None and lo <= row[1] <= hi]
        out.append({"bucket": _bucket_label(lo, hi), "n": len(cyc),
                    "cycles_mean": (sum(cyc) / len(cyc)) if cyc else None})
    return out


# =============================================================================
# First-blunder timing: survival (CDF), density (PDF) and hazard
# =============================================================================
#
# "When does a game first go badly wrong?" is a TIME-TO-EVENT question, not a
# plain histogram question: a game that ended (mate, resignation-equivalent,
# ply cap, aborted crew) before ever blundering is *right-censored* — we know
# only that its first blunder would have come later than the game lasted.
# Treating such a game as "no blunder" in a histogram silently deletes it; the
# Kaplan-Meier estimator keeps it in the risk set for exactly as long as it was
# actually exposed, and drops it afterwards.
#
# Timeline unit: PLY. The system plays White, so its moves land on odd plies
# (verified: all 5,597 system moves in the focused set are odd-ply). A game is
# "at risk" at ply t if it still had a system move at ply >= t.
#
# Exposure: a run that produced NO system move at all (crew never agreed on a
# first move) never had an opportunity to blunder and is excluded from the risk
# set entirely — reported separately as `excluded_no_exposure`.
#
# Event: the first system move that sheds >= BLUNDER_DWP Win% points — the same
# blunder threshold used everywhere else in this report. Defining the event on
# Win% rather than centipawns matters here specifically: under the old >= 150 cp
# rule a game could be recorded as "first blundering" at ply 90 for a swing
# inside an already-lost position, which put events late for exactly the games
# that had collapsed earliest.

FB_HAZARD_WINDOWS: List[Tuple[int, int]] = [
    (1, 10), (11, 20), (21, 30), (31, 40), (41, 60), (61, 200)]

# Log-spaced grid for the per-move loss exceedance curve, in Win% POINTS shed;
# the three judgment cutoffs (10/20/30) are members of the grid by construction.
LOSS_CCDF_GRID: List[float] = sorted(set(
    [0.1, 0.2, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 70, 85, 100]))

Z95 = 1.959963985


def _survival_input(runs: List[Dict[str, Any]],
                    field: str = "first_blunder_ply"
                    ) -> Tuple[List[Tuple[int, int]], int]:
    """[(time, event)] per run, plus the count of runs with no exposure at all.

    time  = ply of the event (event=1) or ply of the last system move the game
            ever played (event=0, right-censored).

    `field` is the per-run key holding the event ply: "first_blunder_ply" for
    when the system first errs badly, "ponr_ply" for when the position became
    unrecoverable. A run whose field is None did not have the event before the
    game ended, and is censored rather than dropped -- dropping biases the
    estimate early, because the games that never had the event are
    disproportionately the ones that lasted longest.
    """
    data: List[Tuple[int, int]] = []
    no_exposure = 0
    for r in runs:
        last = r.get("last_sys_ply")
        if last is None:
            no_exposure += 1
            continue
        at = r.get(field)
        if at is None:
            data.append((int(last), 0))
        else:
            data.append((int(at), 1))
    return data, no_exposure


def km_event(runs: List[Dict[str, Any]],
             field: str = "first_blunder_ply") -> Dict[str, Any]:
    """Kaplan-Meier cumulative incidence F(t) = P(event by ply t).

    Greenwood variance; 95% CI via the complementary log-log transform, which
    keeps the interval inside [0,1] and behaves at the extremes (the plain
    linear interval does not, and our curves run right up against 1).

    `field` selects the event: "first_blunder_ply" or "ponr_ply". Both are
    computed the same way so the two curves are directly comparable -- the
    horizontal offset between them is the lag from erring to being lost.
    """
    data, no_exposure = _survival_input(runs, field)
    n0 = len(data)
    out: Dict[str, Any] = {
        "n": n0,
        "n_events": sum(e for _, e in data),
        "n_censored": sum(1 for _, e in data if e == 0),
        "excluded_no_exposure": no_exposure,
        "points": [], "quartiles": {"p25": None, "median": None, "p75": None},
        "plateau": None, "censor_marks": [],
    }
    if not n0:
        return out
    times = sorted({t for t, e in data if e == 1})
    S = 1.0
    green = 0.0                      # running Greenwood sum
    pts = [{"ply": 0, "at_risk": n0, "events": 0, "censored": 0,
            "F": 0.0, "lo": 0.0, "hi": 0.0}]
    for t in times:
        at_risk = sum(1 for tt, _ in data if tt >= t)
        d = sum(1 for tt, e in data if tt == t and e == 1)
        c = sum(1 for tt, e in data if tt == t and e == 0)
        if at_risk <= 0:
            continue
        S *= (1.0 - d / at_risk)
        if at_risk > d:
            green += d / (at_risk * (at_risk - d))
        F = 1.0 - S
        if 0.0 < S < 1.0 and green > 0:
            # complementary log-log: CI on S is S**exp(±z*sqrt(green)/ln S)
            spread = Z95 * math.sqrt(green) / abs(math.log(S))
            s_lo = S ** math.exp(spread)
            s_hi = S ** math.exp(-spread)
        else:
            s_lo = s_hi = S
        pts.append({"ply": t, "at_risk": at_risk, "events": d, "censored": c,
                    "F": F, "lo": max(0.0, 1.0 - s_hi), "hi": min(1.0, 1.0 - s_lo)})
    out["points"] = pts
    out["plateau"] = pts[-1]["F"] if pts else None
    for q, key in ((0.25, "p25"), (0.5, "median"), (0.75, "p75")):
        hit = next((p["ply"] for p in pts if p["F"] >= q - 1e-12), None)
        out["quartiles"][key] = hit
    # Brookmeyer-Crowley CI for the median: the plies whose pointwise CI for F
    # still straddles 0.5.
    out["median_ci"] = [next((p["ply"] for p in pts if p["hi"] >= 0.5), None),
                        next((p["ply"] for p in pts if p["lo"] >= 0.5), None)]
    # censoring tick marks (ply, F at that moment) for the chart
    for t, e in sorted(data):
        if e == 0:
            f_here = 0.0
            for p in pts:
                if p["ply"] <= t:
                    f_here = p["F"]
            out["censor_marks"].append({"ply": t, "F": f_here})
    # Readout grid every 10 plies: F(t) is the last step at or before t; the risk
    # set at t is the number of games whose time (event or censoring) is >= t —
    # it hits 0 once the last game has left, which a "last event's at_risk" lookup
    # would wrongly report as 1.
    grid = []
    for t in range(10, 201, 10):
        f = 0.0
        lo = hi = 0.0
        for p in pts:
            if p["ply"] <= t:
                f, lo, hi = p["F"], p["lo"], p["hi"]
        grid.append({"ply": t, "at_risk": sum(1 for tt, _ in data if tt >= t),
                     "F": f, "lo": lo, "hi": hi})
    out["risk_grid"] = grid
    return out


def logrank_test(a: List[Dict[str, Any]], b: List[Dict[str, Any]],
                 field: str = "first_blunder_ply") -> Dict[str, Any]:
    """Two-sample log-rank test on event timing (chi-square, 1 df).

    Tests the null that the two groups share one hazard over ply, using every
    event time and the risk sets at it — the timing analogue of the rate
    comparisons in section 4. This, not the overlap of pointwise confidence
    bands, is the right test for whether two curves differ: bands are pointwise,
    so curves whose bands overlap can still separate significantly.
    """
    from scipy import stats
    da, _ = _survival_input(a, field)
    db, _ = _survival_input(b, field)
    if not da or not db:
        return {"chi2": None, "p": None, "n_a": len(da), "n_b": len(db)}
    all_times = sorted({t for t, e in (da + db) if e == 1})
    o_minus_e = 0.0
    var = 0.0
    for t in all_times:
        na = sum(1 for tt, _ in da if tt >= t)
        nb = sum(1 for tt, _ in db if tt >= t)
        n = na + nb
        if n < 2:
            continue
        d_a = sum(1 for tt, e in da if tt == t and e == 1)
        d = d_a + sum(1 for tt, e in db if tt == t and e == 1)
        if d == 0:
            continue
        exp_a = d * na / n
        o_minus_e += d_a - exp_a
        var += (d * na * nb * (n - d)) / (n * n * (n - 1))
    if var <= 0:
        return {"chi2": None, "p": None, "n_a": len(da), "n_b": len(db)}
    chi2 = (o_minus_e ** 2) / var
    return {"chi2": chi2, "p": float(stats.chi2.sf(chi2, 1)),
            "n_a": len(da), "n_b": len(db),
            "events_a": sum(e for _, e in da), "events_b": sum(e for _, e in db)}


def fb_density(runs: List[Dict[str, Any]], lo: int = 1, hi: int = 200,
               step: int = 1, h_fixed: Optional[float] = None) -> Dict[str, Any]:
    """Probability density of the first-blunder ply (censoring-aware KDE).

    Each event contributes the Kaplan-Meier jump in F at its ply (so censored
    games re-weight the events that follow them instead of being dropped), and
    the kernel is reflected at the left boundary `lo` — without reflection a
    Gaussian kernel leaks mass below ply 1, biasing the early peak downward.
    The area under the curve therefore equals F(infinity) = the share of games
    that ever blunder, NOT 1; that is stated on the chart.

    `h_fixed` forces a bandwidth. Curves that are meant to be COMPARED must share
    one: a per-group Silverman bandwidth makes the group with the tighter spread
    look taller purely because it was smoothed less, which is a plotting artefact,
    not a finding. Callers pass the pooled bandwidth for the comparison charts.
    """
    import numpy as np
    km = km_event(runs)
    jumps = [(p["ply"], p["F"] - prev["F"])
             for prev, p in zip(km["points"], km["points"][1:])]
    jumps = [(t, w) for t, w in jumps if w > 0]
    ev_times = [float(t) for t, _ in jumps]
    if len(ev_times) < 2:
        return {"grid": [], "h": None, "area": 0.0, "n_events": km["n_events"]}
    arr = np.asarray([float(t) for r in runs
                      if r.get("first_blunder_ply") is not None
                      for t in [r["first_blunder_ply"]]])
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 1.0
    q75, q25 = np.percentile(arr, [75, 25])
    iqr = float(q75 - q25)
    spread = min(x for x in (sd, iqr / 1.349) if x > 0) if (sd > 0 or iqr > 0) else 1.0
    h_own = max(0.9 * spread * (arr.size ** -0.2), 2.0)   # 2 plies = one system move
    h = float(h_fixed) if h_fixed else h_own
    xs = np.arange(lo, hi + step, step, dtype=float)
    dens = np.zeros_like(xs)
    for t, w in jumps:
        dens += w * (np.exp(-0.5 * ((xs - t) / h) ** 2)
                     + np.exp(-0.5 * ((xs + t - 2 * lo) / h) ** 2)) / (h * math.sqrt(2 * math.pi))
    return {"grid": [[float(x), float(y)] for x, y in zip(xs, dens)],
            "h": float(h), "h_own": float(h_own), "h_shared": bool(h_fixed), "area": float(np.trapezoid(dens, xs)) if hasattr(np, "trapezoid")
            else float(np.trapz(dens, xs)),
            "n_events": km["n_events"], "n": km["n"]}


def fb_hazard(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Life-table (actuarial) discrete hazard of the FIRST blunder.

    h(window) = P(first blunder falls in this window | the game was still being
    played and had not blundered when the window opened). Games censored inside
    the window are credited half a window of exposure (the standard actuarial
    correction), so the denominator is n_start - c/2.
    """
    windows =  FB_HAZARD_WINDOWS
    out = []
    for lo, hi in windows:
        n_start = 0
        d = 0
        c = 0
        for r in runs:
            last = r.get("last_sys_ply")
            if last is None or last < lo:
                continue
            fb = r.get("first_blunder_ply")
            if fb is not None and fb < lo:
                continue
            n_start += 1
            if fb is not None and lo <= fb <= hi:
                d += 1
            elif fb is None and lo <= last <= hi:
                c += 1
        n_eff = n_start - c / 2.0
        rate = (d / n_eff) if n_eff > 0 else None
        out.append({"window": f"{lo}–{hi}", "lo": lo, "hi": hi,
                    "at_risk": n_start, "events": d, "censored": c,
                    "n_eff": n_eff, "hazard": rate})
    return out


def loss_exceedance(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Move-level complementary CDF: P(Win% points shed >= x) per system move.

    The x axis is Win% points, so it is bounded at 100 and every point on it
    means the same thing regardless of when in the game the move was played —
    unlike the centipawn version, whose tail was almost entirely large cp swings
    inside positions that were already won or lost. The blunder rate quoted
    elsewhere is literally this curve read at x = BLUNDER_DWP.
    """
    grid = LOSS_CCDF_GRID
    vals = [v for r in runs for v in r.get("_losses_dwp", [])]
    n = len(vals)
    if not n:
        return {"n": 0, "points": [], "zero_share": None, "median": None, "p90": None}
    svals = sorted(vals)
    pts = []
    for x in grid:
        k = sum(1 for v in svals if v >= x)
        pts.append({"dwp": x, "k": k, "p": k / n})

    def pct(q):
        return float(svals[min(n - 1, int(q * n))])
    return {"n": n, "points": pts,
            "zero_share": sum(1 for v in svals if v <= 0.0) / n,
            "median": pct(0.50), "p75": pct(0.75), "p90": pct(0.90),
            "p99": pct(0.99),
            "blunder_share": sum(1 for v in svals if v >= BLUNDER_DWP) / n}


def collapse_cascade(runs: List[Dict[str, Any]], max_ply: int = 200) -> Dict[str, Any]:
    """Cumulative incidence of three milestones over ONE shared denominator.

    All three curves are shares of the SAME set of exposed games, so they are
    directly comparable and each plateaus at the share of games that ever reach
    that milestone (first blunder ~ all of them; point of no return only the
    games that end up lost; game over ~ 100%):

      first blunder  — first system move losing >= 150 cp
      point of no return — first ply from which the evaluation never again
                       came back above -150 cp
      game over      — the game's last ply

    Also returns the per-game lag from first blunder to point of no return.
    """
    exposed = [r for r in runs if r.get("last_sys_ply") is not None]
    n = len(exposed)
    out: Dict[str, Any] = {"n": n, "series": {}, "lag": None, "lag_zero_share": None,
                           "lag_n": 0, "by_mode": []}
    if not n:
        return out
    fields = {"first_blunder": "first_blunder_ply", "ponr": "ponr_ply", "end": "last_ply"}
    for name, key in fields.items():
        vals = sorted(int(r[key]) for r in exposed if r.get(key) is not None)
        pts = [{"ply": 0, "F": 0.0, "k": 0}]
        for i, t in enumerate(vals, start=1):
            if pts and pts[-1]["ply"] == t:
                pts[-1] = {"ply": t, "F": i / n, "k": i}
            else:
                pts.append({"ply": t, "F": i / n, "k": i})
        # the denominator is fixed (all exposed games), so the curve genuinely stays
        # flat after the last event — carry it to the axis end rather than stopping
        # mid-chart, which would read as missing data
        if pts and pts[-1]["ply"] < max_ply:
            pts.append({"ply": max_ply, "F": pts[-1]["F"], "k": pts[-1]["k"]})
        out["series"][name] = {"points": pts, "reached": len(vals),
                               "share": len(vals) / n,
                               "median": (float(vals[len(vals) // 2]) if vals else None)}
    lags = [int(r["ponr_ply"]) - int(r["first_blunder_ply"]) for r in exposed
            if r.get("ponr_ply") is not None and r.get("first_blunder_ply") is not None]
    if lags:
        out["lag"] = dist_stats([float(x) for x in lags])
        out["lag_zero_share"] = sum(1 for x in lags if x <= 0) / len(lags)
        out["lag_n"] = len(lags)
    for mm in MODE_ORDER:
        rs = [r for r in exposed if r["mode"] == mm]
        lg = [int(r["ponr_ply"]) - int(r["first_blunder_ply"]) for r in rs
              if r.get("ponr_ply") is not None and r.get("first_blunder_ply") is not None]
        fb = sorted(r["first_blunder_ply"] for r in rs if r.get("first_blunder_ply") is not None)
        pn = sorted(r["ponr_ply"] for r in rs if r.get("ponr_ply") is not None)
        out["by_mode"].append({
            "mode": mm, "n": len(rs),
            "median_first_blunder": (fb[len(fb) // 2] if fb else None),
            "n_ponr": len(pn),
            "median_ponr": (pn[len(pn) // 2] if pn else None),
            "median_lag": (sorted(lg)[len(lg) // 2] if lg else None),
            "lag_zero": (sum(1 for x in lg if x <= 0) / len(lg)) if lg else None,
            "lag_n": len(lg)})
    return out



# =============================================================================
# Error triage — decides which "errored" runs carry information vs. which never
# got a fair shot. Rules operate on the FULL error message text because the
# stored error_type is unreliable (runner._classify_error maps every
# RuntimeError to "invalid_move", including harness ImportErrors).
# Rule order matters: first match wins.
# =============================================================================

# (triage_code, include?, human label, substrings)
ERROR_TRIAGE_RULES: List[Tuple[str, bool, str, Tuple[str, ...]]] = [
    ("interrupted", False, "run interrupted by the operator (Ctrl-C)",
     ("KeyboardInterrupt",)),
    ("infra_environment", False, "host environment failure (engine binary/files/permissions)",
     ("PermissionError", "FileNotFoundError", "Stockfish binary", "stockfish-ubu",
      "STOCKFISH_PATH")),
    ("infra_harness_bug", False, "harness code bug (missing/broken dependency)",
     ("ImportError", "ModuleNotFoundError", "AttributeError", "NameError")),
    ("infra_api", False, "API/platform failure (quota, server, network)",
     ("ResourceExhausted", "ServerError", "InternalServerError", "ServiceUnavailable",
      "DeadlineExceeded", "ConnectionError", "ConnectTimeout", "429", "503",
      "google.api_core", "grpc")),
    ("consensus_failure", True, "crew could not reach consensus within the cycle budget",
     ("ConsensusFailureError",)),
    ("illegal_move_failure", True, "model exhausted its retry budget producing illegal moves",
     ("not in legal actions", "not in legal moves")),
    ("context_overflow", True, "prompt exceeded the model's context window (capability limit)",
     ("ContextWindowExceededError", "context window", "context_overflow")),
]

REAL_WINNERS = ("system", "draw")     # plus any engine-name string = loss


def triage_error(error_text: str) -> Tuple[str, bool, str]:
    """Return (triage_code, informative?, label) for an error message."""
    for code, keep, label, pats in ERROR_TRIAGE_RULES:
        if any(p in error_text for p in pats):
            return code, keep, label
    return "unknown_error", False, "unrecognised error (excluded pending manual review)"


# =============================================================================
# Selection pipeline
# =============================================================================

def _difficulty_key(e: Exp) -> Optional[str]:
    """Return the hardness name iff (hardness, elo, depth) matches a current
    definition exactly; None otherwise."""
    expect = FOCUS_DIFFICULTIES.get(e.hardness)
    if expect and (e.elo, e.play_depth) == expect:
        return e.hardness
    return None

def _winner_is_real(winner: Any) -> bool:
    return isinstance(winner, str) and winner not in ("incomplete", "", "(none)") and winner is not None

def _game_outcome(winner: Any) -> str:
    if winner == "system":
        return "win"
    if winner == "draw":
        return "draw"
    return "loss"

def select_runs(all_exps: List[Exp]) -> Tuple[List[Dict[str, Any]], List[Exp]]:
    """Walk the full universe and decide, run by run, what enters the focused set.

    Returns (selection_records, included_exps). Every universe run gets exactly
    one record: {id, decision, stage, category, reason, evidence, facts...}.
    """
    records: List[Dict[str, Any]] = []
    included: List[Exp] = []

    for e in all_exps:
        d = e.doc
        rec: Dict[str, Any] = {
            "id": e.id,
            "model": e.model, "hardness": e.hardness, "elo": e.elo,
            "play_depth": e.play_depth, "mode": e.mode,
            "session": e.session, "schema": e.schema, "host": e.hostname,
            "status": e.status, "stored_outcome": e.outcome,
            "winner": d.get("winner"), "result": e.result,
            "n_move_docs": len(e.moves),
            "prototype_id": d.get("prototype_id"),
            "error_type_stored": e.error_type or None,
            "error_first_line": (e.error or "").split("\n")[0][:200] or None,
        }

        def exclude(stage: str, reason: str, evidence: str = "") -> None:
            rec.update(decision="excluded", stage=stage, reason=reason, evidence=evidence)
            records.append(rec)

        # -- stage 1: scope --------------------------------------------------
        # FOCUS_PROTOTYPE is None for the baseline (keep runs with no
        # prototype_id) and a prototype name otherwise (keep only that one), so
        # the two sweeps never mix even if they share a cache directory.
        if (d.get("prototype_id") or None) != FOCUS_PROTOTYPE:
            exclude("scope",
                    ("prototype run (only the main baseline system is in scope)"
                     if FOCUS_PROTOTYPE is None
                     else f"not part of the {FOCUS_PROTOTYPE!r} prototype sweep"),
                    f"prototype_id={d.get('prototype_id')}")
            continue
        if e.model != FOCUS_MODEL:
            exclude("scope", f"model is not {FOCUS_MODEL}",f"model={e.model}")
            continue
        if e.mode not in FOCUS_MODES:
            exclude("scope", "mode outside the three main systems", f"mode={e.mode}")
            continue
        diff = _difficulty_key(e)
        if diff is None:
            exclude("scope",
                    "difficulty uses an old/other definition (only low=1320/d1, "
                    "medium=1400/d2, high=1500/d3 are in scope)",
                    f"hardness={e.hardness} elo={e.elo} depth={e.play_depth}")
            continue

        # -- stage 2: artifacts ---------------------------------------------
        if e.is_empty:
            exclude("artifact", "empty pre-created document: the run was registered at "
                    "sweep start but never executed (no summary, no moves)",
                    f"status={e.status}")
            continue
        if e.is_unfinalized:
            exclude("artifact", "unfinalized run: move documents exist but the run never "
                    "wrote its summary (process killed mid-game or still in flight at "
                    "pull time) — no trustworthy outcome",
                    f"status={e.status} n_move_docs={len(e.moves)}")
            continue

        # -- stage 3: error triage -------------------------------------------
        if e.error:
            code, keep, label = triage_error(f"{e.error_type} {e.error} {e.error_tb}")
            if not keep:
                exclude("error_triage",
                        f"{label} — the system never got a fair shot at the game",
                        rec["error_first_line"] or code)
                continue
            # Informative failure. If the game nevertheless reached a terminal
            # state (checkmate/draw before the error was raised), score the game
            # and keep the error as an annotation, per the analysis rules.
            if _winner_is_real(d.get("winner")):
                rec.update(
                    decision="included", stage="error_triage",
                    category=_game_outcome(d.get("winner")),
                    reason=f"game reached a terminal state; run was additionally marked "
                           f"errored ({label}) — scored by its game result, error kept "
                           f"as an annotation",
                    evidence=rec["error_first_line"] or "")
            else:
                rec.update(
                    decision="included", stage="error_triage", category=code,
                    reason=f"informative failure: {label}",
                    evidence=rec["error_first_line"] or "")
            records.append(rec)
            included.append(e)
            continue

        # -- stage 4: normal completions --------------------------------------
        if not e.has_summary:
            exclude("artifact", "no summary and not caught by earlier stages (unexpected)", "")
            continue
        if e.outcome == "incomplete":
            rec.update(decision="included", stage="completion", category="ply_cap_incomplete",
                       reason="game hit the 200-ply cap without a decisive result "
                              "(informative: neither side converted)",
                       evidence=f"n_move_docs={len(e.moves)}")
        elif e.outcome in ("model_win", "model_loss", "draw"):
            rec.update(decision="included", stage="completion",
                       category={"model_win": "win", "model_loss": "loss",
                                 "draw": "draw"}[e.outcome],
                       reason="game completed normally",
                       evidence=f"result={e.result} winner={d.get('winner')}")
        else:
            exclude("completion", f"unrecognised stored outcome {e.outcome!r} "
                    "(excluded pending manual review)", "")
            continue
        if e.status == "completed_with_write_errors" or e.write_failures:
            rec["storage_note"] = (
                "some move documents were lost to Firestore's 1 MiB doc limit "
                "(gcc-1.0 schema); summary aggregates were recomputed in memory "
                "before writing and remain trustworthy; per-move analyses use the "
                f"{len(e.moves)} stored moves and disclose the gaps")
        records.append(rec)
        included.append(e)

    return records, included



def cap_cells(records: List[Dict[str, Any]],
              included: List[Exp]) -> Tuple[List[Exp], Dict[str, Any]]:
    """Trim every (difficulty x mode) cell back to CELL_CAP_N runs.

    Runs are ordered by created_at and the CELL_CAP_KEEP end of each cell is
    kept; the surplus is flipped to decision="excluded" with stage="cell_cap" so
    the selection report still accounts for every run in the universe. Cells
    that never reached CELL_CAP_N are left as they are.
    """
    rec_by_id = {r["id"]: r for r in records}
    cells: Dict[Tuple[str, str], List[Exp]] = defaultdict(list)
    for e in included:
        cells[(e.hardness, e.mode)].append(e)

    kept: List[Exp] = []
    summary: Dict[str, Any] = {"cap_n": CELL_CAP_N, "keep": CELL_CAP_KEEP,
                               "trimmed": 0, "cells": []}
    for (diff, mode), exps in sorted(cells.items()):
        # created_at is "YYYY-MM-DD_HH-MM-SS-TZ"; every run in the focused set
        # carries the same zone, so the string order is the time order. The id
        # breaks ties deterministically.
        ordered = sorted(exps, key=lambda e: (str(e.created), e.id))
        keep = (ordered[-CELL_CAP_N:] if CELL_CAP_KEEP == "latest"
                else ordered[:CELL_CAP_N])
        keep_ids = {e.id for e in keep}
        dropped = [e for e in ordered if e.id not in keep_ids]
        for e in dropped:
            rec = rec_by_id[e.id]
            rec.pop("category", None)
            rec.update(
                decision="excluded", stage="cell_cap",
                reason=f"surplus run: this {diff}/{mode} cell held {len(ordered)} "
                       f"accepted games and the design is {CELL_CAP_N}; the "
                       f"{CELL_CAP_KEEP} {CELL_CAP_N} by start time were kept so "
                       f"every cell weighs the same and stays on one harness version",
                evidence=f"created={e.created} schema={e.schema} "
                         f"cell_size={len(ordered)}")
        summary["trimmed"] += len(dropped)
        summary["cells"].append({
            "diff": diff, "mode": mode, "accepted": len(ordered),
            "kept": len(keep), "trimmed": len(dropped),
            "kept_from": str(keep[0].created) if keep else None,
            "kept_to": str(keep[-1].created) if keep else None,
            "schemas_kept": sorted({e.schema for e in keep}),
        })
        kept.extend(keep)

    kept.sort(key=lambda e: str(e.created))
    return kept, summary


# =============================================================================
# Small helpers
# =============================================================================

def hr(ch: str = "=") -> str:
    return ch * W


def mean_or_none(vals: List[float]) -> Optional[float]:
    return statistics.mean(vals) if vals else None


# =============================================================================
# Environment / credential sanity checks (all read-only)
# =============================================================================

def read_env_project() -> str:
    """Read GCLOUD_PROJECT from CrewForge/.env without touching os.environ."""
    from dotenv import dotenv_values
    if not ENV_FILE.exists():
        sys.exit(f"ERROR: env file not found: {ENV_FILE}")
    values = dotenv_values(ENV_FILE)
    project = (values.get("GCLOUD_PROJECT") or "").strip()
    if not project:
        sys.exit(f"ERROR: GCLOUD_PROJECT is not set in {ENV_FILE}")
    return project


def _gcloud(*args: str) -> Optional[str]:
    """Run a read-only gcloud query; None if gcloud is unavailable/errors."""
    try:
        out = subprocess.run(["gcloud", *args], capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except Exception:
        return None


def sanity_check_gcloud(project: str) -> None:
    """Verify the terminal's gcloud matches the project in CrewForge/.env.

    Hard-fails on an active-config project mismatch (the check the dashboard is
    required to make); warns on ADC quota-project drift, which is the other
    common cause of 403s against this Firestore DB.
    """
    cfg_project = _gcloud("config", "get-value", "project")
    account = _gcloud("config", "get-value", "account") or "(unknown)"

    if cfg_project is None:
        sys.exit("ERROR: could not run `gcloud config get-value project` — is gcloud installed "
                 "and logged in? (This dashboard requires an authorized gcloud + ADC.)")
    if cfg_project != project:
        sys.exit(
            "ERROR: gcloud project mismatch.\n"
            f"  gcloud active config project : {cfg_project!r}\n"
            f"  GCLOUD_PROJECT in {ENV_FILE.name}   : {project!r}\n"
            "  The terminal's gcloud is logged into a different project than this "
            "experiment suite uses.\n"
            f"  Fix with:  gcloud config set project {project}\n"
            "  (and make sure ADC is for the right account: "
            "gcloud auth application-default login)"
        )

    print(f"  gcloud sanity check : OK (config project matches .env; account={account})")

    # ADC quota-project drift is a warning, not a failure: the pull will still
    # surface a clear 403 if the ADC identity itself lacks access.
    adc_path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    if adc_path.exists():
        try:
            quota = json.loads(adc_path.read_text()).get("quota_project_id")
        except Exception:
            quota = None
        if quota and quota != project:
            print(f"  WARNING: ADC quota project is {quota!r}, not {project!r}. "
                  "If the pull 403s, run:")
            print(f"           gcloud auth application-default set-quota-project {project}")


# =============================================================================
# Cache handling
# =============================================================================

def cache_manifest() -> Optional[Dict[str, Any]]:
    mpath = LOCAL_CACHE_DIR / MANIFEST_NAME
    if not mpath.exists():
        return None
    try:
        m = json.loads(mpath.read_text())
    except Exception:
        return None
    if not m.get("complete"):
        return None
    return m


def cache_age_hours(manifest: Dict[str, Any]) -> Optional[float]:
    try:
        pulled = datetime.fromisoformat(manifest["pulled_at_utc"])
        return (datetime.now(timezone.utc) - pulled).total_seconds() / 3600.0
    except Exception:
        return None


def clear_cache() -> None:
    if LOCAL_CACHE_DIR.exists():
        shutil.rmtree(LOCAL_CACHE_DIR)
    (LOCAL_CACHE_DIR / "experiments").mkdir(parents=True)


# =============================================================================
# Firestore pull (READ-ONLY: .stream() / document reads only)
# =============================================================================

def _stream_with_retry(col_ref, what: str) -> List[Any]:
    """Stream a collection fully, retrying transient failures with backoff."""
    delay = PULL_BASE_DELAY_S
    for attempt in range(1, PULL_MAX_RETRIES + 1):
        try:
            return list(col_ref.stream(timeout=STREAM_TIMEOUT_S))
        except Exception as e:
            if attempt == PULL_MAX_RETRIES:
                raise RuntimeError(f"failed to stream {what} after "
                                   f"{PULL_MAX_RETRIES} attempts: {e}") from e
            print(f"    retry {attempt}/{PULL_MAX_RETRIES} for {what}: "
                  f"{type(e).__name__}: {e} — waiting {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    return []


def pull_from_firestore(project: str) -> Dict[str, Any]:
    """Full fresh pull of the experiment collection into LOCAL_CACHE_DIR.

    One JSON file per experiment (written as soon as it is fetched), then a
    manifest marking the pull complete. Chunked deliberation calls
    (moves/{seq}/calls/{chunk}) are pulled too, stored under the same
    `__subcollection_calls` key the historical JSON export used.
    """
    from google.cloud import firestore

    db = firestore.Client(project=project, database=FIRESTORE_DATABASE)
    col = db.collection(FIRESTORE_COLLECTION)

    print(f"\n  Pulling {FIRESTORE_COLLECTION} from Firestore db={FIRESTORE_DATABASE!r} "
          "(read-only; this can take a while — reliability over speed) ...")
    t0 = time.time()

    exp_snaps = _stream_with_retry(col, f"collection {FIRESTORE_COLLECTION}")
    print(f"  Found {len(exp_snaps)} experiment documents.")
    if not exp_snaps:
        sys.exit("ERROR: no experiment documents found — refusing to build an empty cache.")

    exp_dir = LOCAL_CACHE_DIR / "experiments"
    n_moves_total = 0
    n_chunk_docs_total = 0

    for i, snap in enumerate(sorted(exp_snaps, key=lambda s: s.id), 1):
        exp_id = snap.id
        exp_doc = snap.to_dict() or {}

        move_snaps = _stream_with_retry(snap.reference.collection("moves"),
                                        f"{exp_id}/moves")
        moves: Dict[str, Any] = {}
        n_chunks = 0
        for ms in sorted(move_snaps, key=lambda s: s.id):
            mdoc = ms.to_dict() or {}
            if mdoc.get("calls_storage") == "chunked":
                chunk_snaps = _stream_with_retry(ms.reference.collection("calls"),
                                                 f"{exp_id}/moves/{ms.id}/calls")
                mdoc["__subcollection_calls"] = {
                    cs.id: (cs.to_dict() or {})
                    for cs in sorted(chunk_snaps, key=lambda s: s.id)
                }
                n_chunks += len(chunk_snaps)
            moves[ms.id] = mdoc

        payload = {"experiment_id": exp_id, "experiment": exp_doc, "moves": moves}
        fpath = exp_dir / f"{exp_id}.json"
        tmp = fpath.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, default=str)
        tmp.replace(fpath)

        n_moves_total += len(moves)
        n_chunk_docs_total += n_chunks
        chunk_note = f", {n_chunks} calls-chunk docs" if n_chunks else ""
        print(f"    [{i:>3}/{len(exp_snaps)}] {exp_id}: {len(moves)} moves{chunk_note}")

    manifest = {
        "complete": True,
        "pulled_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project": project,
        "database": FIRESTORE_DATABASE,
        "collection": FIRESTORE_COLLECTION,
        "experiment_count": len(exp_snaps),
        "move_doc_count": n_moves_total,
        "calls_chunk_doc_count": n_chunk_docs_total,
        "pull_duration_seconds": round(time.time() - t0, 1),
    }
    (LOCAL_CACHE_DIR / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    print(f"  Pull complete in {manifest['pull_duration_seconds']:.0f}s "
          f"({len(exp_snaps)} experiments, {n_moves_total} move docs, "
          f"{n_chunk_docs_total} calls-chunk docs).")
    return manifest


def ensure_data(project: str, stale_ok: bool = False) -> Tuple[Dict[str, Any], str]:
    """Return (manifest, source_note), reusing the cache if fresh enough.

    With stale_ok (--cached), any complete cache is accepted regardless of age
    or project — useful offline; the report labels the data's age either way.
    """
    m = cache_manifest()
    if m is not None:
        age = cache_age_hours(m)
        if stale_ok:
            age_note = f"age {age:.1f}h" if age is not None else "age unknown"
            return m, f"cached pull from {m['pulled_at_utc']} ({age_note}; --cached)"
        if age is not None and age <= CACHE_MAX_AGE_HOURS and m.get("project") == project:
            return m, f"cached pull from {m['pulled_at_utc']} (age {age:.1f}h)"
        why = "older than 24h" if (age is None or age > CACHE_MAX_AGE_HOURS) else "different project"
        print(f"  Cache present but {why} — clearing and re-pulling.")
    elif stale_ok:
        sys.exit("ERROR: --cached given but no complete cache exists in "
                 f"{LOCAL_CACHE_DIR} — run once without --cached first.")
    else:
        print("  No usable cache — doing a fresh full pull.")
    clear_cache()
    m = pull_from_firestore(project)
    return m, f"fresh pull at {m['pulled_at_utc']}"


# =============================================================================
# Load + derive per-experiment views
# =============================================================================

class Exp:
    """A loaded experiment with derived, deterministic analysis fields."""

    def __init__(self, payload: Dict[str, Any]):
        self.id: str = payload["experiment_id"]
        self.doc: Dict[str, Any] = payload["experiment"] or {}
        raw_moves = payload.get("moves", {}) or {}
        self.moves: List[Dict[str, Any]] = sorted(
            raw_moves.values(), key=lambda m: m.get("seq", 0))

        d = self.doc
        self.status: str = d.get("status", "(missing)")
        self.schema: str = d.get("schema_version", "(missing)")
        self.session: str = d.get("session_id", "(missing)")
        self.hostname: str = (d.get("environment") or {}).get("hostname", "?")
        self.created: str = d.get("created_at_est", d.get("created_at_utc", "?"))
        self.hardness: str = d.get("hardness", "?")
        self.mode: str = d.get("mode", "?")
        self.model: str = d.get("model", "?")
        self.elo = d.get("stockfish_elo")
        self.play_depth = d.get("stockfish_play_depth")
        self.eval_depth = d.get("stockfish_eval_depth")

        # A doc that never got its finalize() update has no 'outcome' key.
        self.has_summary: bool = "outcome" in d
        self.outcome: str = d.get("outcome", "(none)")
        self.result: str = d.get("result", "")
        self.error: str = d.get("error") or ""
        self.error_type: str = d.get("error_type") or ""
        self.error_tb: str = d.get("error_traceback") or ""
        self.duration_s: float = float(d.get("duration_seconds") or 0.0)
        self.di: Dict[str, Any] = d.get("data_integrity") or {}

        # Empty pre-created run: never finalized AND produced no move docs.
        self.is_empty: bool = (not self.has_summary) and not self.moves
        # Crashed mid-run without finalize (has moves but no summary) — rare.
        self.is_unfinalized: bool = (not self.has_summary) and bool(self.moves)

        # --- move-level derivations (system moves = the LLM side) ---
        self.sys_moves = [m for m in self.moves if m.get("is_system_move")]
        self.cp_losses: List[int] = []          # |delta| for harmful system moves
        self.quality = Counter()
        self.sys_evals: List[int] = []          # eval after each system move, in order
        qual_mismatch = 0
        for m in self.sys_moves:
            delta = m.get("centipawn_eval_delta")
            if isinstance(delta, int) and delta < 0:
                self.cp_losses.append(-delta)
            q = m.get("quality_category", "unknown")
            self.quality[q] += 1
            if isinstance(delta, int) and _quality_category(delta) != q:
                qual_mismatch += 1
            ev = m.get("centipawn_eval")
            if isinstance(ev, int):
                self.sys_evals.append(ev)
        self.quality_mismatches = qual_mismatch

        # Recomputed-from-visible-moves aggregates (may differ from the stored
        # summary when Firestore lost move writes — that gap is itself reported).
        self.re_prompt = sum(int(m.get("prompt_tokens", 0) or 0) for m in self.sys_moves)
        self.re_completion = sum(int(m.get("completion_tokens", 0) or 0) for m in self.sys_moves)
        self.re_thoughts = sum(int(m.get("thoughts_tokens", 0) or 0) for m in self.sys_moves)
        self.re_cycles = sum(int(m.get("deliberation_cycles", 0) or 0) for m in self.sys_moves)
        self.re_calls = sum(int(m.get("call_count", 0) or 0) for m in self.sys_moves)
        self.sys_latency_ms = sum(int(m.get("latency_ms", 0) or 0) for m in self.sys_moves)

        self.acpl: Optional[float] = mean_or_none([float(x) for x in self.cp_losses])

        # Missing-move detection: gaps in the seq sequence 1..max(seq).
        seqs = sorted(m.get("seq", 0) for m in self.moves)
        self.seq_gaps: List[int] = []
        if seqs:
            self.seq_gaps = sorted(set(range(1, seqs[-1] + 1)) - set(seqs))
        summary_total = self.doc.get("total_moves")
        self.moves_missing_vs_summary: Optional[int] = (
            (summary_total - len(self.moves))
            if isinstance(summary_total, int) and self.has_summary else None)

        # Firestore write failures recorded by the orchestrator + 1MB matching.
        self.write_failures: List[str] = list(
            self.di.get("firestore_write_failure_details") or [])
        blob = " ".join(self.write_failures) + " " + self.error + " " + self.error_tb
        self.hit_1mb: bool = any(p in blob for p in FS_1MB_PATTERNS)
        self.chunked_moves: int = int(self.di.get("chunked_moves") or 0)

    @property
    def difficulty(self) -> Tuple[str, Any, Any]:
        return (self.hardness, self.elo, self.play_depth)

    @property
    def difficulty_label(self) -> str:
        return f"{self.hardness} [elo {self.elo}, depth {self.play_depth}]"

    @property
    def triad(self) -> Tuple[str, str, str]:
        return (self.difficulty_label, self.mode, self.model)


def _quality_category(centipawn_delta: Optional[int]) -> str:
    """Mirror of run_gcc_experiments._quality_category (used only to cross-check
    the stored quality_category field, never to replace it)."""
    if centipawn_delta is None:
        return "unknown"
    if centipawn_delta >= 0:
        return "excellent"
    loss = abs(centipawn_delta)
    if loss < 10:
        return "excellent"
    if loss < 30:
        return "good"
    if loss < 80:
        return "inaccuracy"
    if loss < 150:
        return "mistake"
    return "blunder"


def load_experiments() -> List[Exp]:
    exp_dir = LOCAL_CACHE_DIR / "experiments"
    exps = []
    for fp in sorted(exp_dir.glob("*.json")):
        with open(fp, encoding="utf-8") as f:
            exps.append(Exp(json.load(f)))
    return exps


# =============================================================================
# Error classification (deterministic substring rules — shown in the report)
# =============================================================================

ERROR_BUCKET_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("firestore-1mb-doc-limit", FS_1MB_PATTERNS),
    ("environment (stockfish/files/perms)", ("PermissionError", "FileNotFoundError",
                                             "stockfish", "Stockfish binary")),
    ("interrupted by user", ("KeyboardInterrupt",)),
    ("model output: invalid move", ("not in legal actions", "invalid_move")),
    ("model output: malformed JSON", ("JSONDecodeError",)),
    ("crew consensus/tooling failure", ("ConsensusFailureError", "consensus",
                                        "Tool talk()", "Tool consensus()",
                                        "AgentCreationError")),
    ("API/server error", ("ServerError", "ResourceExhausted", "InternalServerError",
                          "ServiceUnavailable", "DeadlineExceeded", "429", "503")),
]


def classify_error(e: Exp) -> str:
    blob = f"{e.error_type} {e.error} {e.error_tb}"
    for bucket, pats in ERROR_BUCKET_RULES:
        if any(p in blob for p in pats):
            return bucket
    return "other"


# =============================================================================
# HTML report generation
# =============================================================================
#
# The heavy lifting (aggregation, charts, filtering, deep-dives) happens in the
# browser: gcc_dashboard_template.html is a self-contained static page whose
# `__GCC_DATA__` token gets replaced with the JSON payload built below. The
# payload carries one record per experiment with the derived fields plus a
# slimmed move list — per-move stats only, never the LLM call payloads, so the
# generated file stays a few MB even though the raw pull can be huge.

# Doc keys surfaced structurally in the report; anything else lands in `extra`
# (truncated), mirroring the old terminal deep-dive's "other raw fields".
_SHOWN_TOP_DOC_KEYS = {
    "status", "schema_version", "session_id", "environment", "created_at_est",
    "created_at_utc", "hardness", "mode", "model", "stockfish_elo",
    "stockfish_play_depth", "stockfish_eval_depth", "outcome", "result", "winner",
    "final_fen", "error", "error_type", "error_traceback", "duration_seconds",
    "data_integrity", "ended_at_est", "ended_at_utc", "total_moves", "system_moves",
    "opponent_moves", "total_plies", "total_prompt_tokens", "total_completion_tokens",
    "total_thoughts_tokens", "total_tokens", "system_time_ms", "opponent_time_ms",
    "avg_centipawn_loss", "quality_counts", "total_deliberation_cycles",
    "avg_cycles_per_move", "call_errors", "summary_checksum",
}
_EMBED_DI_KEYS = {
    "live_aggregate_matches_recompute", "live_prompt_tokens", "live_completion_tokens",
    "moves_written_locally", "firestore_write_failures",
    "chunked_moves", "max_chunks_in_move", "truncated_call_moves",
    "chunk_verify_failures",
}

# Column order of the per-move arrays; must match `MOVE` in the template.
_MOVE_COLS = ("seq", "ply", "player", "sys", "uci", "san", "cp", "delta", "q",
              "tokens", "calls", "cycles", "lat", "chunk")

DIFF_ORDER = ["low", "medium", "high"]
MODE_ORDER = ["single", "multi", "hardcoded_multi"]

def _slim_moves(e: Exp) -> List[List[Any]]:
    out = []
    for m in e.moves:
        out.append([
            m.get("seq"), m.get("ply"), m.get("player", ""),
            1 if m.get("is_system_move") else 0,
            m.get("uci", ""), m.get("san", ""),
            m.get("centipawn_eval"), m.get("centipawn_eval_delta"),
            m.get("quality_category", ""),
            m.get("total_tokens", 0), m.get("call_count", 0),
            m.get("deliberation_cycles", 0), m.get("latency_ms", 0),
            1 if m.get("calls_storage") == "chunked" else 0,
        ])
    return out


def _extra_fields(e: Exp) -> Dict[str, str]:
    """Raw doc fields not surfaced structurally, truncated for embedding.

    For empty pre-created runs everything is 'extra' — the deep-dive shows the
    whole (tiny) doc, exactly like the old terminal report did.
    """
    keys = e.doc.keys() if e.is_empty else (set(e.doc) - _SHOWN_TOP_DOC_KEYS)
    out = {}
    for k in sorted(keys):
        v = e.doc[k]
        s = json.dumps(v, default=str) if isinstance(v, (dict, list)) else str(v)
        out[k] = s[:300] + " …(truncated)" if len(s) > 300 else s
    return out


def exp_record(e: Exp) -> Dict[str, Any]:
    err = None
    if e.error or e.error_type or e.error_tb:
        tb = e.error_tb
        err = {
            "type": e.error_type, "msg": e.error,
            "tb": tb[:TB_MAX_CHARS], "tb_truncated": len(tb) > TB_MAX_CHARS,
            "bucket": classify_error(e),
        }
    d = e.doc
    return {
        "id": e.id, "status": e.status, "schema": e.schema, "session": e.session,
        "host": e.hostname, "created": str(e.created),
        "ended": str(d.get("ended_at_est", d.get("ended_at_utc", ""))) or None,
        "hard": e.hardness, "elo": e.elo, "pdepth": e.play_depth, "edepth": e.eval_depth,
        "mode": e.mode, "model": e.model,
        "outcome": e.outcome, "result": e.result, "winner": d.get("winner"),
        "fen": d.get("final_fen"), "dur": e.duration_s or None,
        "empty": e.is_empty, "unfin": e.is_unfinalized, "hasSummary": e.has_summary,
        "err": err,
        "nSys": len(e.sys_moves), "acpl": e.acpl,
        "worst": max(e.cp_losses) if e.cp_losses else None,
        "losses": e.cp_losses, "quality": dict(e.quality), "qmis": e.quality_mismatches,
        "evals": e.sys_evals,
        "pt": e.re_prompt, "ct": e.re_completion, "tt": e.re_thoughts,
        "cycles": e.re_cycles, "calls": e.re_calls, "latms": e.sys_latency_ms,
        "sumPT": d.get("total_prompt_tokens"), "sumCT": d.get("total_completion_tokens"),
        "sumACPL": d.get("avg_centipawn_loss"),
        "gaps": e.seq_gaps, "missVsSum": e.moves_missing_vs_summary,
        "wfails": e.write_failures, "hit1mb": e.hit_1mb, "chunked": e.chunked_moves,
        "di": {k: e.di[k] for k in sorted(_EMBED_DI_KEYS & set(e.di))},
        "extra": _extra_fields(e),
        "moves": _slim_moves(e),
    }


# =============================================================================
# Focused cache writer
# =============================================================================

def write_focused_cache(records: List[Dict[str, Any]], included: List[Exp],
                        pull_manifest: Dict[str, Any]) -> None:
    """(Re)build gcc_focused_data/ from scratch: verbatim payload copies of the
    accepted runs plus the full selection report. Deterministic and cheap, so it
    is rebuilt on every invocation from the full-pull cache."""
    saved_cache = None
    if BEST_MOVES_CACHE.exists():          # expensive stockfish results survive rebuilds
        saved_cache = BEST_MOVES_CACHE.read_bytes()
    if FOCUSED_DIR.exists():
        shutil.rmtree(FOCUSED_DIR)
    FOCUSED_RUNS_DIR.mkdir(parents=True)
    FOCUSED_META_DIR.mkdir(parents=True)
    FOCUSED_ANALYSIS_DIR.mkdir(parents=True)
    if saved_cache is not None:
        BEST_MOVES_CACHE.write_bytes(saved_cache)

    src_dir = LOCAL_CACHE_DIR / "experiments"
    for e in included:
        shutil.copyfile(src_dir / f"{e.id}.json", FOCUSED_RUNS_DIR / f"{e.id}.json")

    by_decision = Counter(r["decision"] for r in records)
    by_category = Counter(r.get("category") for r in records if r["decision"] == "included")
    report = {
        "filter_version": FILTER_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_pull": pull_manifest,
        "focus_scope": {
            "model": FOCUS_MODEL, "modes": list(FOCUS_MODES),
            "difficulties": {k: {"elo": v[0], "play_depth": v[1]}
                             for k, v in FOCUS_DIFFICULTIES.items()},
        },
        "universe_runs": len(records),
        "decisions": dict(by_decision),
        "included_by_category": dict(by_category),
        "records": records,
    }
    (FOCUSED_META_DIR / "selection_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str))
    (FOCUSED_META_DIR / "_manifest.json").write_text(json.dumps({
        "filter_version": FILTER_VERSION,
        "built_at_utc": report["built_at_utc"],
        "included_runs": len(included),
        "universe_runs": len(records),
        "source_pulled_at_utc": pull_manifest.get("pulled_at_utc"),
    }, indent=2))
    print(f"  Focused cache  : {len(included)} runs accepted / {len(records)} in universe "
          f"-> {FOCUSED_DIR}")

def _load_best_cache() -> Dict[str, str]:
    if BEST_MOVES_CACHE.exists():
        try:
            return json.loads(BEST_MOVES_CACHE.read_text())
        except Exception:
            return {}
    return {}


def _save_best_cache(cache: Dict[str, str]) -> None:
    # The cache is shared across sweeps and so lives outside FOCUSED_ANALYSIS_DIR,
    # which write_focused_cache creates. Derive the directory from the file so it
    # stays correct wherever BEST_MOVES_CACHE points.
    BEST_MOVES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = BEST_MOVES_CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache))
    tmp.replace(BEST_MOVES_CACHE)


def _stockfish_path() -> Optional[str]:
    """STOCKFISH_PATH from CrewForge/.env, if the binary exists."""
    from dotenv import dotenv_values
    values = dotenv_values(ENV_FILE)
    p = (values.get("STOCKFISH_PATH") or "").strip().strip("'\"")
    return p if p and Path(p).exists() else None


def compute_best_moves(fens: List[str]) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Best move (UCI) for every FEN via local Stockfish at the paper's settings.
    Cached on disk; resumable; returns (fen->best_uci, provenance)."""

    cache = _load_best_cache()
    todo = [f for f in dict.fromkeys(fens) if f not in cache]
    prov: Dict[str, Any] = {"depth": PAPER_SF_DEPTH, "threads": PAPER_SF_THREADS,
                            "hash_mb": PAPER_SF_HASH_MB, "multipv": 1,
                            "engine": None, "computed_now": len(todo),
                            "from_cache": len(dict.fromkeys(fens)) - len(todo)}
    sf = _stockfish_path()
    if todo and not sf:
        prov["error"] = "stockfish binary not found — best-move rate unavailable"
        return cache, prov
    if todo:
        print(f"  Best-move analysis: {len(todo)} positions to evaluate at depth "
              f"{PAPER_SF_DEPTH} (cached: {prov['from_cache']}) ...")
        engine = chess.engine.SimpleEngine.popen_uci(sf) # type: ignore -- this cannot be None due to the if todo and not sf
        try:
            prov["engine"] = engine.id.get("name")
            engine.configure({"Threads": PAPER_SF_THREADS, "Hash": PAPER_SF_HASH_MB})
            t0 = time.time()
            for i, fen in enumerate(todo, 1):
                board = chess.Board(fen)
                # game=object() forces python-chess to send `ucinewgame` before every
                # position. Without it, it sends `ucinewgame` exactly once (its `game`
                # argument defaults to None and only a CHANGE triggers a new game), so a
                # single transposition table, history and killer tables accumulate across
                # all ~5k unrelated positions. That carried state changes move ordering,
                # which changes late-move reductions and TT cutoffs, which changes the
                # depth-20 PV: the "best move" then depends on how many positions were
                # analysed before this one and in what order, not on the position alone.
                # Clearing it per position costs the warm-hash speedup and buys a result
                # that is a function of the FEN only, which is what the metric claims.
                info = engine.analyse(board, chess.engine.Limit(depth=PAPER_SF_DEPTH),
                                      game=object())
                pv = info.get("pv") or []
                cache[fen] = pv[0].uci() if pv else ""
                if i % 200 == 0 or i == len(todo):
                    rate = i / (time.time() - t0)
                    print(f"    [{i}/{len(todo)}] {rate:.1f} pos/s "
                          f"(~{(len(todo) - i) / max(rate, 0.1) / 60:.1f} min left)")
                    _save_best_cache(cache)
        finally:
            engine.quit()
        _save_best_cache(cache)
    else:
        sf2 = _stockfish_path()
        if sf2:
            try:
                engine = chess.engine.SimpleEngine.popen_uci(sf2)
                prov["engine"] = engine.id.get("name")
                engine.quit()
            except Exception:
                pass
    return cache, prov


def replay_verify(e: Exp) -> Dict[str, Any]:
    """Replay every stored move with python-chess and cross-check the stored data.

    Returns verification results + per-seq piece counts (before each move) +
    a PGN movetext built from the verified SANs + termination detail.
    """
    import chess

    v: Dict[str, Any] = {
        "moves_checked": 0, "fen_mismatches": 0, "san_mismatches": 0,
        "illegal_uci": 0, "chain_breaks": 0, "seq_gaps": list(e.seq_gaps),
        "pieces_before": {},          # seq -> piece count before the move
        "termination": None, "terminal_consistent": None,
    }
    sans: List[Tuple[int, str, bool]] = []   # (ply, san, is_system)

    prev_fen_after: Optional[str] = None
    prev_seq: Optional[int] = None
    for m in e.moves:
        seq, uci, fen_b, fen_a = m.get("seq"), m.get("uci"), m.get("fen_before"), m.get("fen_after")
        if not (uci and fen_b and fen_a):
            continue
        v["moves_checked"] += 1
        try:
            board = chess.Board(fen_b)
        except ValueError:
            v["fen_mismatches"] += 1
            continue
        v["pieces_before"][seq] = len(board.piece_map())
        try:
            mv = chess.Move.from_uci(uci)
            if mv not in board.legal_moves:
                v["illegal_uci"] += 1
                continue
            san = board.san(mv)
            if san != m.get("san"):
                v["san_mismatches"] += 1
            board.push(mv)
            if board.fen() != fen_a:
                v["fen_mismatches"] += 1
        except ValueError:
            v["illegal_uci"] += 1
            continue
        if prev_fen_after is not None and prev_seq is not None and seq == prev_seq + 1:
            if fen_b != prev_fen_after:
                v["chain_breaks"] += 1
        prev_fen_after, prev_seq = fen_a, seq
        ply = m.get("ply")
        san = m.get("san")
        is_system = m.get("is_system_move")

        if isinstance(ply, int) and isinstance(san, str) and isinstance(is_system, bool):
            sans.append((ply, san, is_system))
        else:
            print(f"ERROR: failed to extract ply, san, or is_system_move from {e.id}. This will affect results")

    # PGN movetext (system = White always; ply odd = White's move).
    parts: List[str] = []
    for ply, san, _ in sans:
        if ply is None or not san:
            continue
        if ply % 2 == 1:
            parts.append(f"{(ply + 1) // 2}. {san}")
        else:
            parts.append(san)
    v["pgn"] = " ".join(parts)

    # Termination detail from the final position (and full replay when gap-free).
    final_fen = e.doc.get("final_fen") or (e.moves[-1].get("fen_after") if e.moves else None)
    if final_fen:
        try:
            fb = chess.Board(final_fen)
            if fb.is_checkmate():
                term = "checkmate"
                winner_by_board = "white" if fb.turn == chess.BLACK else "black"
                stored = e.doc.get("winner")
                v["terminal_consistent"] = (
                    (winner_by_board == "white" and stored == "system")
                    or (winner_by_board == "black" and stored not in ("system", "draw",
                                                                      "incomplete", None, "")))
            elif fb.is_stalemate():
                term = "stalemate"
            elif fb.is_insufficient_material():
                term = "insufficient material"
            elif fb.halfmove_clock >= 150:
                term = "75-move rule"
            else:
                term = None   # repetition or ply-cap — resolved below
            v["termination"] = term
        except ValueError:
            v["termination"] = None
    return v

def analyze_run(e: Exp, rec: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the full derived record for one accepted run."""
    d = e.doc
    category = rec["category"]
    ver = replay_verify(e)

    # ---- termination label ----
    term = ver.get("termination")
    if category == "ply_cap_incomplete" and term is None:
        term = "200-ply cap reached"
    elif category == "consensus_failure":
        term = "no move: crew never reached consensus"
    elif category == "illegal_move_failure":
        term = "aborted: illegal-move retry budget exhausted"
    elif term is None and category in DECIDED_CATEGORIES:
        # game_over without mate/stalemate/material/75-move => repetition
        term = "fivefold repetition" if category == "draw" else "(unresolved terminal type)"
    ver["termination"] = term

    # ---- system-move series ----
    # Two parallel loss series are kept per move:
    #   losses_std / losses_harm   centipawns (legacy ACPL definitions, §5/§7)
    #   losses_dwp                 Win% points shed — the scale every quality,
    #                              blunder and timing statistic now reads.
    sys_moves = e.sys_moves
    losses_std: List[float] = []       # min(max(0,-delta), clamp) over ALL system moves
    losses_harm: List[float] = []      # |delta| over delta<0 moves (orchestrator def)
    losses_dwp: List[float] = []       # Win% points shed, over ALL system moves
    phase_data = {p: {"n": 0, "losses_std": [], "losses_dwp": [],
                      "blunders": 0, "mistakes": 0}
                  for p in ("opening", "middlegame", "endgame")}
    quality_wp = Counter()             # Win%-based move-quality mix
    dwp_by_seq: Dict[int, Optional[float]] = {}
    wp_by_seq: Dict[int, Optional[float]] = {}
    blunder_plies: List[int] = []
    first_blunder_ply: Optional[int] = None
    worst_loss, worst_ply, worst_san = None, None, None      # cp (legacy)
    worst_dwp, worst_dwp_ply, worst_dwp_san = None, None, None
    qmis = 0
    for m in sys_moves:
        delta = m.get("centipawn_eval_delta")
        cp_after = m.get("centipawn_eval")
        ply = m.get("ply")
        seq = m.get("seq")
        if ply is None or seq is None or cp_after is None:
            print(f"ERROR: sequence, ply, or centipawn after move could not be retrieved from {e.id}. This will invalidate results.")
            continue


        # Win% view of this move. cp_before is reconstructed as after - delta,
        # which is exactly how the orchestrator formed delta in the first place.
        wp_after = lichess_win_pct(cp_after)
        dwp = (delta_win_pct(cp_after - delta, cp_after)
               if isinstance(cp_after, int) and isinstance(delta, int) else None)
        dwp_by_seq[seq] = dwp
        wp_by_seq[seq] = wp_after

        if dwp is not None:
            losses_dwp.append(dwp)
            j = win_pct_judgment(dwp)
            quality_wp[j] += 1
            if dwp >= BLUNDER_DWP:
                blunder_plies.append(ply)
                if first_blunder_ply is None:
                    first_blunder_ply = ply
            if worst_dwp is None or dwp > worst_dwp:
                worst_dwp, worst_dwp_ply, worst_dwp_san = dwp, ply, m.get("san")

        if isinstance(delta, int):
            loss = max(0, -delta)
            capped = min(loss, 1000)
            losses_std.append(float(capped))
            if delta < 0:
                losses_harm.append(float(loss))
            if _quality_category(delta) != m.get("quality_category"):
                qmis += 1
            if worst_loss is None or loss > worst_loss:
                worst_loss, worst_ply, worst_san = loss, ply, m.get("san")
            ph = _phase_of(ply, ver["pieces_before"].get(seq))
            pd = phase_data[ph]
            pd["n"] += 1
            pd["losses_std"].append(float(capped))
            if dwp is not None:
                pd["losses_dwp"].append(dwp)
                if dwp >= BLUNDER_DWP:
                    pd["blunders"] += 1
                elif dwp >= MISTAKE_DWP:
                    pd["mistakes"] += 1

    acpl_std = (sum(losses_std) / len(losses_std)) if losses_std else None
    acpl_harm = (sum(losses_harm) / len(losses_harm)) if losses_harm else None
    stored_acpl = d.get("avg_centipawn_loss")

    # ---- eval trajectory (all plies, White POV) ----
    # Only mate scores are collapsed (to +/-MATE_EQUIV_CP); real evaluations are
    # carried at full value, so a legitimately winning -1400 cp is no longer
    # indistinguishable from a forced mate the way the old +/-1000 clamp made it.
    # Each point carries both cp and its Win%, so charts can use either.
    evals: List[Tuple[int, int, float]] = []   # (ply, cp, win%)
    for m in e.moves:
        cp, ply = m.get("centipawn_eval"), m.get("ply")
        if isinstance(cp, int) and isinstance(ply, int):
            cpc = clamp_mate_cp(cp)
            lcpc = lichess_win_pct(cpc)
            evals.append((ply, cpc, lcpc))
    evals.sort()
    # Point of no return: first ply from which the LLM's winning chances stay
    # below PONR_WIN_PCT to the end of the game.
    ponr = None
    if evals:
        run_ok = [wp >= PONR_WIN_PCT for _, _, wp in evals]
        last_ok = max((i for i, ok in enumerate(run_ok) if ok), default=-1)
        if last_ok < len(evals) - 1:
            ponr = evals[last_ok + 1][0]
    final_eval = evals[-1][1] if evals else None
    final_win_pct = evals[-1][2] if evals else None
    # "Ahead"/"behind" bands are Win% too: +/-50 cp at balance is ~54%/46%, so the
    # 55/45 band below is the same idea expressed on the comparable scale.
    share_ahead = (sum(1 for _, _, wp in evals if wp > 55) / len(evals)) if evals else None
    share_behind = (sum(1 for _, _, wp in evals if wp < 45) / len(evals)) if evals else None
    lead_changes = 0
    prev_sign = 0
    for _, _, wp in evals:
        sign = 1 if wp > 55 else (-1 if wp < 45 else 0)
        if sign != 0 and prev_sign != 0 and sign != prev_sign:
            lead_changes += 1
        if sign != 0:
            prev_sign = sign

    # ---- resources ----
    n_sys = len(sys_moves)
    pt, ct, tt = e.re_prompt, e.re_completion, e.re_thoughts
    total_tokens = pt + ct
    lat_ms = e.sys_latency_ms
    cycles, calls = e.re_cycles, e.re_calls
    call_errors = sum(1 for m in sys_moves
                      for c in (m.get("calls") or []) if c.get("error"))
    # per-move token series for pooled distributions
    tok_per_move = [int(m.get("total_tokens", 0) or 0) for m in sys_moves]
    cyc_per_move = [int(m.get("deliberation_cycles", 0) or 0) for m in sys_moves]
    lat_per_move = [int(m.get("latency_ms", 0) or 0) for m in sys_moves]

    dtotal_tokens = d.get("total_tokens")

    # ---- integrity summary ----
    summary_total = d.get("total_moves")
    integrity = {
        "moves_stored": len(e.moves),
        "moves_per_summary": summary_total,
        "moves_missing": (summary_total - len(e.moves))
                         if isinstance(summary_total, int) else None,
        "seq_gaps": ver["seq_gaps"],
        "fen_mismatches": ver["fen_mismatches"],
        "san_mismatches": ver["san_mismatches"],
        "illegal_uci": ver["illegal_uci"],
        "chain_breaks": ver["chain_breaks"],
        "quality_mismatches": qmis,
        "acpl_recompute_matches_stored": (
            None if (acpl_harm is None or not isinstance(stored_acpl, (int, float)))
            else abs(acpl_harm - float(stored_acpl)) < 0.5 or bool(ver["seq_gaps"])),
        "token_recompute_delta": (
            (dtotal_tokens - total_tokens) if isinstance(dtotal_tokens, int) else None),
        "write_failures": len(e.write_failures),
        "chunked_moves": e.chunked_moves,
    }

    # ---- slim per-move rows for the deep-dive ----
    # Columns 13-15 are the Win% view of the move (only meaningful for system
    # moves); compute_paper_metrics appends the best-move flag at 16.
    move_rows = []
    for m in e.moves:
        seq = m.get("seq")
        ply = m.get("ply")
        cp_eval = m.get("centipawn_eval")
        if seq is None or ply is None or cp_eval is None: 
            print(f"ERROR: sequence, ply, or centipawn eval could not be retrieved from {e.id}. This will invalidate results.")
            continue
        is_sys = bool(m.get("is_system_move"))
        dwp = dwp_by_seq.get(seq) if is_sys else None
        move_rows.append([
            seq, ply, 1 if is_sys else 0,
            m.get("san", ""), m.get("uci", ""),
            cp_eval, m.get("centipawn_eval_delta"),
            m.get("quality_category", ""),
            _phase_of(ply, ver["pieces_before"].get(seq)) if is_sys else "",
            m.get("total_tokens", 0), m.get("call_count", 0),
            m.get("deliberation_cycles", 0), m.get("latency_ms", 0),
            dwp,                                          # 13: Win% points shed
            lichess_win_pct(clamp_mate_cp(cp_eval)),   # 14: Win% after
            (win_pct_judgment(dwp) or "") if is_sys else "",           # 15: judgment
        ])

    return {
        # identity / provenance
        "id": e.id, "session": e.session, "schema": e.schema, "host": e.hostname,
        "created": str(e.created), "duration_s": e.duration_s or None,
        "diff": e.hardness, "elo": e.elo, "play_depth": e.play_depth, "mode": e.mode,
        # outcome
        "category": category, "score": SCORE_MAP.get(category),
        "result": e.result or None, "winner": d.get("winner"),
        "termination": term,
        "error_note": rec.get("error_first_line"),
        "error_annotated": bool(e.error),
        "storage_note": rec.get("storage_note"),
        # counts
        "n_plies": summary_total if isinstance(summary_total, int) else len(e.moves),
        "n_sys": n_sys,
        # exposure window for the time-to-event analytics: the last ply the game
        # reached, and the last ply on which the SYSTEM actually moved (its risk set)
        "last_ply": (e.moves[-1].get("ply") if e.moves else None),
        "last_sys_ply": (sys_moves[-1].get("ply") if sys_moves else None),
        # strength — Win% scale is primary; the cp/ACPL fields are legacy
        "acpl_std": acpl_std, "acpl_harm": acpl_harm, "acpl_stored": stored_acpl,
        "median_loss": (_percentiles(losses_std) or {}).get("median") if losses_std else None,
        "quality": dict(e.quality),               # orchestrator's cp categories
        "quality_wp": dict(quality_wp),           # Win% judgment categories
        "avg_dwp": (sum(losses_dwp) / len(losses_dwp)) if losses_dwp else None,
        "median_dwp": (_percentiles(losses_dwp) or {}).get("median") if losses_dwp else None,
        # blunder = a move shedding >= BLUNDER_DWP Win% points
        "blunders": len(blunder_plies), "blunder_plies": blunder_plies,
        "blunder_rate": (len(blunder_plies) / n_sys) if n_sys else None,
        "worst_loss": worst_loss, "worst_ply": worst_ply, "worst_san": worst_san,
        "worst_dwp": worst_dwp, "worst_dwp_ply": worst_dwp_ply,
        "worst_dwp_san": worst_dwp_san,
        "first_blunder_ply": first_blunder_ply, "ponr_ply": ponr,
        "final_eval": final_eval, "final_win_pct": final_win_pct,
        "share_ahead": share_ahead, "share_behind": share_behind,
        "lead_changes": lead_changes,
        "evals": [[p, cp, round(wp, 2)] for p, cp, wp in evals],
        "phase": {p: {"n": pd["n"],
                      "acpl_std": (sum(pd["losses_std"]) / pd["n"]) if pd["n"] else None,
                      "avg_dwp": (sum(pd["losses_dwp"]) / len(pd["losses_dwp"])
                                  if pd["losses_dwp"] else None),
                      "blunders": pd["blunders"], "mistakes": pd["mistakes"]}
                  for p, pd in phase_data.items()},
        # resources
        "pt": pt, "ct": ct, "tt": tt, "total_tokens": total_tokens,
        "tokens_per_sys_move": (total_tokens / n_sys) if n_sys else None,
        "thinking_share": (tt / ct) if ct else None,
        "lat_ms": lat_ms, "lat_per_sys_move_s": (lat_ms / n_sys / 1000) if n_sys else None,
        "cycles": cycles, "cycles_per_sys_move": (cycles / n_sys) if n_sys else None,
        "calls": calls, "calls_per_sys_move": (calls / n_sys) if n_sys else None,
        "call_errors": call_errors,
        "_tok_per_move": tok_per_move, "_cyc_per_move": cyc_per_move,
        "_lat_per_move": lat_per_move, "_losses_std": losses_std,
        "_losses_dwp": losses_dwp,
        # integrity + game record
        "integrity": integrity,
        "pgn": ver["pgn"],
        "moves": move_rows,
    }

def analyze_all(included: List[Exp], records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze every accepted run and build all aggregate structures."""
    import numpy as np
    global RNG

    rec_by_id = {r["id"]: r for r in records}
    runs = [analyze_run(e, rec_by_id[e.id]) for e in
            sorted(included, key=lambda e: str(e.created))]
    print(f"  Analyzed       : {len(runs)} runs "
          f"({sum(r['n_sys'] for r in runs)} system moves replay-verified)")

    def grp(diff=None, mode=None):  
        return [r for r in runs
                if (diff is None or r["diff"] == diff)
                and (mode is None or r["mode"] == mode)]

    groups = {"overall": summarize_group(runs)}
    for dd in DIFF_ORDER:
        groups[f"diff:{dd}"] = summarize_group(grp(diff=dd))
    for mm in MODE_ORDER:
        groups[f"mode:{mm}"] = summarize_group(grp(mode=mm))
    for dd in DIFF_ORDER:
        for mm in MODE_ORDER:
            groups[f"cell:{dd}:{mm}"] = summarize_group(grp(diff=dd, mode=mm))

    trajectories = {}
    for mm in MODE_ORDER:
        trajectories[f"mode:{mm}"] = eval_trajectory_bands(grp(mode=mm))
    for dd in DIFF_ORDER:
        trajectories[f"diff:{dd}"] = eval_trajectory_bands(grp(diff=dd))

    blunder_time = {f"mode:{mm}": blunder_timing(grp(mode=mm)) for mm in MODE_ORDER}

    # ---- first-blunder time-to-event block (section 3 PDF/CDF charts) ----
    fb_km = {"overall": km_event(runs)}
    fb_dens = {"overall": fb_density(runs)}
    # one pooled bandwidth for every per-group density curve, so differences in
    # peak height are differences in the data and not in the smoothing
    h_shared = fb_dens["overall"]["h"]
    fb_haz = {"overall": fb_hazard(runs)}
    fb_loss = {"overall": loss_exceedance(runs)}
    for mm in MODE_ORDER:
        g = grp(mode=mm)
        fb_km[f"mode:{mm}"] = km_event(g)
        fb_dens[f"mode:{mm}"] = fb_density(g, h_fixed=h_shared)
        fb_haz[f"mode:{mm}"] = fb_hazard(g)
        fb_loss[f"mode:{mm}"] = loss_exceedance(g)
    for dd in DIFF_ORDER:
        g = grp(diff=dd)
        fb_km[f"diff:{dd}"] = km_event(g)
        fb_dens[f"diff:{dd}"] = fb_density(g, h_fixed=h_shared)
    fb_logrank = []
    for ma, mb in [("single", "multi"), ("single", "hardcoded_multi"),
                   ("multi", "hardcoded_multi")]:
        fb_logrank.append({"kind": "mode", "a": ma, "b": mb,
                           **logrank_test(grp(mode=ma), grp(mode=mb))})
    for da, db in [("low", "medium"), ("low", "high"), ("medium", "high")]:
        fb_logrank.append({"kind": "difficulty", "a": da, "b": db,
                           **logrank_test(grp(diff=da), grp(diff=db))})
    first_blunder = {
        "km": fb_km, "density": fb_dens, "hazard": fb_haz,
        "loss_exceedance": fb_loss, "logrank": fb_logrank,
        "cascade": collapse_cascade(runs),
        "hazard_windows": [list(w) for w in FB_HAZARD_WINDOWS],
        "density_bandwidth": h_shared,
        "blunder_threshold_dwp": BLUNDER_DWP,
        "ponr_threshold_wp": PONR_WIN_PCT,
        "judgment_cutoffs_dwp": {"inaccuracy": INACCURACY_DWP,
                                 "mistake": MISTAKE_DWP,
                                 "blunder": BLUNDER_DWP},
    }
    # ---- point-of-no-return time-to-event block -------------------------
    # Same estimator, same censoring rule, different event: when the position
    # became unrecoverable rather than when the system first erred. Built here
    # rather than in the figure code so both curves come from one definition.
    pr_km = {"overall": km_event(runs, "ponr_ply")}
    for mm in MODE_ORDER:
        pr_km[f"mode:{mm}"] = km_event(grp(mode=mm), "ponr_ply")
    for dd in DIFF_ORDER:
        pr_km[f"diff:{dd}"] = km_event(grp(diff=dd), "ponr_ply")
    pr_logrank = []
    for ma, mb in [("single", "multi"), ("single", "hardcoded_multi"),
                   ("multi", "hardcoded_multi")]:
        pr_logrank.append({"kind": "mode", "a": ma, "b": mb,
                           **logrank_test(grp(mode=ma), grp(mode=mb), "ponr_ply")})
    for da, db in [("low", "medium"), ("low", "high"), ("medium", "high")]:
        pr_logrank.append({"kind": "difficulty", "a": da, "b": db,
                           **logrank_test(grp(diff=da), grp(diff=db), "ponr_ply")})
    ponr = {"km": pr_km, "logrank": pr_logrank, "threshold_wp": PONR_WIN_PCT}

    delib_quality = {mm: deliberation_by_quality(grp(mode=mm))
                     for mm in ("multi", "hardcoded_multi")}
    delib_bucket = {mm: deliberation_by_bucket(grp(mode=mm))
                    for mm in ("multi", "hardcoded_multi")}

    return {
        "runs": runs, "groups": groups,
        "trajectories": trajectories, "blunder_timing": blunder_time,
        "first_blunder": first_blunder,
        "ponr": ponr,
        "deliberation_by_quality": delib_quality,
        "deliberation_by_bucket": delib_bucket,
    }

PAPER_SCORE = {"win": 1.0, "draw": 0.5, "ply_cap_incomplete": 0.5,
               "loss": 0.0, "consensus_failure": 0.0, "illegal_move_failure": 0.0}
PAPER_JUDGMENTS = ("blunder", "mistake", "inaccuracy", "ok")


def compute_paper_metrics(included: List[Exp],
                          runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The full paper-replication computation. Mutates each analyzed run with
    per-run paper metrics and returns the aggregate `paper` payload block."""
    runs_by_id = {r["id"]: r for r in runs}

    # ---- gather (fen_before, uci) for every system move --------------------
    sys_positions: Dict[str, List[Tuple[int, str, str]]] = {}   # id -> [(seq, fen, uci)]
    all_fens: List[str] = []
    for e in included:
        lst = []
        for m in e.sys_moves:
            fen, uci = m.get("fen_before"), m.get("uci")
            if fen and uci:
                lst.append((m.get("seq"), fen, uci))
                all_fens.append(fen)
        sys_positions[e.id] = lst

    best_map, best_prov = compute_best_moves(all_fens)

    # ---- per-run, per-move paper metrics ----------------------------------
    for e in included:
        r = runs_by_id[e.id]
        judg = Counter()
        win_pcts: List[float] = []
        deltas: List[float] = []
        best_n = best_hits = 0
        per_move: Dict[int, Optional[int]] = {}
        for m in e.sys_moves:
            cp_after = m.get("centipawn_eval")
            delta_cp = m.get("centipawn_eval_delta")
            seq = m.get("seq")
            if cp_after is not None and seq is not None:
                wp_after = lichess_win_pct(cp_after)
            else:
                print(f"ERROR: failed to retreive cp delta or seq for {e.id}. This will affect results.")
                continue
            dwp = (delta_win_pct(cp_after - delta_cp, cp_after)
                   if isinstance(cp_after, int) and isinstance(delta_cp, int) else None)
            j = win_pct_judgment(dwp)
            if j:
                judg[j] += 1
            if wp_after is not None:
                win_pcts.append(wp_after)
            if dwp is not None:
                deltas.append(dwp)
            fen, uci = m.get("fen_before"), m.get("uci")
            is_best = None
            if fen and uci and best_map.get(fen):
                best_n += 1
                is_best = (best_map[fen] == uci)
                if is_best:
                    best_hits += 1
            per_move[seq] = 1 if is_best else (0 if is_best is not None else None)
        n_j = sum(judg.values())
        r["paper"] = {
            "score": PAPER_SCORE.get(r["category"]),
            "avg_win_pct": (sum(win_pcts) / len(win_pcts)) if win_pcts else None,
            "final_win_pct": win_pcts[-1] if win_pcts else None,
            "judgments": dict(judg), "n_judged": n_j,
            "blunder_rate": (judg["blunder"] / n_j) if n_j else None,
            "mistake_rate": (judg["mistake"] / n_j) if n_j else None,
            "inaccuracy_rate": (judg["inaccuracy"] / n_j) if n_j else None,
            "best_rate": (best_hits / best_n) if best_n else None,
            "best_n": best_n, "best_hits": best_hits,
            "avg_delta_win_pct": (sum(deltas) / len(deltas)) if deltas else None,
        }
        # extend the deep-dive move rows: [16]=is_best. The Win% judgment that
        # used to be appended here is now column 15, written by analyze_run —
        # both sections read the same numbers rather than deriving them twice.
        for row in r["moves"]:
            row.append(per_move.get(row[0]))

    # ---- group aggregates --------------------------------------------------
    def agg(sel: List[Dict[str, Any]]) -> Dict[str, Any]:
        games = [r for r in sel if r["paper"]["score"] is not None]
        chess_games = [r for r in sel if r["category"] in ("win", "loss", "draw")]
        wins = sum(1 for r in games if r["category"] == "win")
        opp_wins = sum(1 for r in games if PAPER_SCORE.get(r["category"]) == 0.0)
        wl_paper = (0.5 * (wins - opp_wins) / len(games) + 0.5) if games else None
        c_wins = sum(1 for r in chess_games if r["category"] == "win")
        c_losses = sum(1 for r in chess_games if r["category"] == "loss")
        wl_chess = (0.5 * (c_wins - c_losses) / len(chess_games) + 0.5) if chess_games else None
        jc = Counter()
        n_j = 0
        best_hits = best_n = 0
        for r in sel:
            jc.update(r["paper"]["judgments"])
            n_j += r["paper"]["n_judged"]
            best_hits += r["paper"]["best_hits"]
            best_n += r["paper"]["best_n"]
        wp = [r["paper"]["avg_win_pct"] for r in sel if r["paper"]["avg_win_pct"] is not None]
        # loss-reason breakdown in the paper's table shape
        reasons = {"instruction": 0, "draw_rules": 0, "draw_cap": 0,
                   "mate_by_llm": 0, "mate_by_engine": 0, "other_loss": 0}
        for r in games:
            c, t = r["category"], (r["termination"] or "")
            if c in ("consensus_failure", "illegal_move_failure"):
                reasons["instruction"] += 1
            elif c == "draw":
                reasons["draw_rules"] += 1
            elif c == "ply_cap_incomplete":
                reasons["draw_cap"] += 1
            elif c == "win":
                reasons["mate_by_llm"] += 1
            elif c == "loss":
                reasons["mate_by_engine" if t == "checkmate" else "other_loss"] += 1
        moves_avg = (sum(r["n_plies"] / 2 for r in games) / len(games)) if games else None
        return {
            "n_paper_games": len(games), "n_chess_games": len(chess_games),
            "wins": wins, "opp_wins": opp_wins,
            "winloss_paper": wl_paper, "winloss_chess": wl_chess,
            "judgments": dict(jc), "n_judged": n_j,
            "blunder_rate": (jc["blunder"] / n_j) if n_j else None,
            "mistake_rate": (jc["mistake"] / n_j) if n_j else None,
            "inaccuracy_rate": (jc["inaccuracy"] / n_j) if n_j else None,
            "best_rate": (best_hits / best_n) if best_n else None,
            "best_n": best_n,
            "avg_win_pct": dist_stats(wp),
            "loss_reasons": reasons,
            "avg_moves": moves_avg,
        }

    per_mode: Dict[str, Any] = {}
    per_cell: Dict[str, Any] = {}
    for mm in MODE_ORDER:
        sel = [r for r in runs if r["mode"] == mm]
        block = agg(sel)
        per_mode[mm] = block
        for dd in DIFF_ORDER:
            per_cell[f"{dd}:{mm}"] = agg([r for r in sel if r["diff"] == dd])

    # ---- Win% trajectory per mode (mean over running games at sampled plies) --
    trajectories: Dict[str, List[Dict[str, Any]]] = {}
    for mm in MODE_ORDER:
        sel = [r for r in runs if r["mode"] == mm]
        out = []
        for ply in range(2, 201, 2):
            vals = []
            for r in sel:
                if not r["evals"] or r["evals"][-1][0] < ply:
                    continue
                bestv = None
                for p, cp, wp in r["evals"]:
                    if p <= ply:
                        bestv = wp
                    else:
                        break
                if bestv is not None:
                    vals.append(bestv)
            if len(vals) >= 3:
                out.append({"ply": ply, "n": len(vals),
                            "mean": sum(vals) / len(vals)})
        trajectories[mm] = out

    return {
        "per_mode": per_mode,
        "per_cell": per_cell,
        "trajectories": trajectories,
    }



# =============================================================================
# Main
# =============================================================================

print(hr("="))
print("CREWFORGE METRICS  (read-only)")
print(hr("="))
args = sys.argv[1:]

USAGE = """usage: gcc_dashboard.py [options]

  --collection NAME   Firestore collection to analyse
                      (default: ChessExperiments)
  --prototype NAME    keep only runs whose prototype_id is NAME; omit to keep
                      the baseline runs, which carry no prototype_id
  --refresh           discard the cache and re-pull
  --cached            use the existing cache, skip the gcloud check

  The prototype sweep lives in its own collection and gets its own cache and
  focused-set directories, so it can never overwrite the baseline's:

    python3 gcc_dashboard.py --collection ChessExperiments_p4_refutation \\
                             --prototype p4_refutation
"""

if "--help" in args or "-h" in args:
    print(USAGE)
    sys.exit(0)


def _opt(flag: str, default: Optional[str] = None) -> Optional[str]:
    """Value of `--flag VALUE`, or default. Exits if the value is missing."""
    if flag not in args:
        return default
    i = args.index(flag)
    if i + 1 >= len(args) or args[i + 1].startswith("--"):
        sys.exit(f"ERROR: {flag} needs a value.\n\n{USAGE}")
    return args[i + 1]


_known = {"--refresh", "--cached", "--collection", "--prototype", "--help", "-h"}
_values = {args[i + 1] for i, a in enumerate(args)
           if a in ("--collection", "--prototype") and i + 1 < len(args)}
for a in args:
    if a.startswith("-") and a not in _known:
        sys.exit(f"ERROR: unknown option {a}\n\n{USAGE}")
    if not a.startswith("-") and a not in _values:
        sys.exit(f"ERROR: unexpected argument {a!r}\n\n{USAGE}")

configure_sweep(_opt("--collection", DEFAULT_COLLECTION), _opt("--prototype", )) # type: ignore (DEFAULT COLLECTION cannot be None)

print(f"  Collection     : {FIRESTORE_COLLECTION}"
      + (f"   prototype={FOCUS_PROTOTYPE}" if FOCUS_PROTOTYPE else "   (baseline)"))
print(f"  Cache          : {LOCAL_CACHE_DIR.name}  ->  {FOCUSED_DIR.name}")

force = "--refresh" in args
cached = "--cached" in args
if force and cached:
    sys.exit("ERROR: --refresh and --cached are mutually exclusive.")

project = read_env_project()
if cached:
    print("  --cached given: using the existing cache; skipping the gcloud sanity check.")
else:
    sanity_check_gcloud(project)
    if force and LOCAL_CACHE_DIR.exists():
        print("  --refresh given: discarding cache.")
        shutil.rmtree(LOCAL_CACHE_DIR)

manifest, source_note = ensure_data(project, stale_ok=cached)

all_exps = load_experiments()
n_empty = sum(1 for e in all_exps if e.is_empty)
n_unfin = sum(1 for e in all_exps if e.is_unfinalized)
print(f"  (Universe) Loaded {len(all_exps)} experiments "
        f"({len(all_exps) - n_empty - n_unfin} analyzed, {n_empty} empty, {n_unfin} unfinalized).")

records, included = select_runs(all_exps)
included, cap_summary = cap_cells(records, included)
if cap_summary["trimmed"]:
    print(f"  Cell cap       : trimmed {cap_summary['trimmed']} surplus run(s) to "
            f"n={CELL_CAP_N} per cell (keeping the {CELL_CAP_KEEP} by start time)")
    for c in cap_summary["cells"]:
        if c["trimmed"]:
            print(f"      {c['diff']:6} {c['mode']:16} {c['accepted']:>2} -> "
                    f"{c['kept']:>2}  ({', '.join(c['schemas_kept'])})")
write_focused_cache(records, included, manifest)

for cat, n in sorted(Counter(r.get("category") for r in records if r["decision"] == "included").items()):
    print(f"      included {cat:22} {n}")
excl = Counter((r["stage"], r["reason"][:60]) for r in records if r["decision"] == "excluded")
for (stage, reason), n in sorted(excl.items()):
    print(f"      excluded [{stage}] {reason:62} {n}")

unknown = [r for r in records if r.get("category") == "unknown_error" or "unrecognised" in (r.get("reason") or "")]
if unknown:
    print("  WARNING: runs excluded with UNRECOGNISED errors/outcomes — review these:")
    for r in unknown:
        print(f"      {r['id']}: {r.get('evidence') or r.get('error_first_line')}")

analyzed = analyze_all(included, records)
analyzed["paper"] = compute_paper_metrics(included, analyzed["runs"])

print(hr("="))

