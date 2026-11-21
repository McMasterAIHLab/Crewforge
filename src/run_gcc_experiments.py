#!/usr/bin/env python3
"""
run_gcc_experiments.py — Vertex-only
=====================================================================================

Firestore layout
----------------
    {FIRESTORE_COLLECTION}/{experiment_id}                 -> config + environment + summary
    {FIRESTORE_COLLECTION}/{experiment_id}/moves/{seq}     -> per-move detail: chess move
                                                              details, centipawn scores,
                                                              timing, token aggregates, and
                                                              the deliberation chat (see below).
    {FIRESTORE_COLLECTION}/{experiment_id}/moves/{seq}/calls/{chunk}
                                                           -> (chunked moves only) byte-bounded
                                                              slices of the deliberation chat.

Deliberation-chat storage (1 MiB doc-limit safety)
--------------------------------------------------
Firestore caps a single document at 1,048,576 bytes. In the multi-agent modes a
single move can accumulate dozens of deliberation calls (each with a large
`user_prompt`), pushing the move doc past that limit — which made `create()`
fail and silently dropped the move from Firestore (it survived only in the local
backup). To prevent this, every move's `calls` are sized:

  * If the whole move doc (including inline `calls`) is at or under
    MOVE_INLINE_MAX_BYTES, it is stored exactly as before with
    `calls_storage = "inline"` (true for `single` mode and most multi moves).
  * Otherwise the `calls` are packed into byte-bounded chunks
    (CALLS_CHUNK_BUDGET_BYTES each, far below 1 MiB) written to a
    `moves/{seq}/calls/{chunk}` sub-subcollection. The move doc then carries
    `calls_storage = "chunked"`, an empty `calls` list, and a `calls_manifest`
    describing every chunk (id, call range, per-chunk checksum) plus a
    `calls_checksum` over the full ordered call list for verified reassembly.

The local backup mirror always keeps the full `calls` inline regardless of how
Firestore stored them, so it remains the authoritative, lossless copy.

Experiments are generated as TRIADS = (hardness x mode x model), each run
EXPERIMENTS_PER_TRIAD times. With 3 hardness levels, 3 modes, and 3 models that
is 27 experiments per pass at EXPERIMENTS_PER_TRIAD = 1.

"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Import path setup — make sibling src/ modules AND the parent-dir
# cloud_llm_provider importable regardless of the current working directory.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent          # .../CrewForge/src
_PARENT_DIR = _THIS_DIR.parent                        # .../CrewForge
for _p in (str(_THIS_DIR), str(_PARENT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv

# =============================================================================
# CONFIGURATION — set everything here before running. There is no CLI.
# =============================================================================

# Communication prototype to run, or None for the unmodified baseline system.
# When set, multi-agent modes are built from prototypes/<id> instead of
# MultiLLMSystem, results go to their own Firestore collection and backup
# directory, and every experiment document records the id. The single-LLM mode
# is never affected by a prototype — it has no inter-agent communication — but
# it is still worth running alongside as an in-session control.
# Registered ids: see prototypes.list_ids().
PROTOTYPE_ID: Optional[str] = None

# How many times each triad (hardness x mode x model) is run.
EXPERIMENTS_PER_TRIAD = 1

# Hardness levels: each maps to a Stockfish ELO and search/play depth.
HARDNESS_LEVELS: Dict[str, Dict[str, int]] = {
    "low":    {"stockfish_elo": 1320, "stockfish_play_depth": 1},
    # "medium": {"stockfish_elo": 1400, "stockfish_play_depth": 2},
    # "high":   {"stockfish_elo": 1500, "stockfish_play_depth": 3},
}

# The three play modes.
#   single          -> one LLM plays directly (SingleLLMSystem)
#   multi           -> multi-agent crew, dynamically designed by the meta-agent
#   hardcoded_multi -> multi-agent crew loaded from a fixed preset (same for all)
MODES: List[str] = ["single"]

# The three Gemini models under test (full Vertex model IDs).
MODELS: List[str] = [
    # "gemini-3.1-pro-preview",
    "gemini-3.5-flash",
    # "gemini-3.1-flash-lite",
]

# Gemini sampling / reasoning settings (apply to every model).
TEMPERATURE = 1.0
THINKING_LEVEL = "low"          # minimum reasoning level (cannot be fully disabled)

# Game / generation limits.
MAX_PLIES = 200                 # hard cap on plies per game (~100 model moves)
MAX_OUTPUT_TOKENS = 16384       # per-call output cap (includes thinking tokens)
MAX_CONTEXT_TOKENS = 1_048_576  # context window used for prompt-size guards
MOVE_HISTORY_LIMIT = 1          # recent-white-moves history hint (provenance only)
STOCKFISH_EVAL_DEPTH = 10       # depth used for centipawn evaluation of positions
MOVE_DELAY_SECONDS = 0.0        # delay between turns (serial vertex needs none)
MAX_SYSTEM_RETRIES = 3          # per-move retry budget inside the system

# Symmetric invalid-move handling for both systems.
SYMMETRIC_INVALID_MOVE = False

# Hardcoded crew preset (used by the hardcoded_multi mode for ALL experiments).
HARDCODED_CREW_PATH = str(_THIS_DIR / "hardcoded_crew_v1.json")
HARDCODED_CREW_NAME = "hardcoded_crew_v1"

# Firestore. A prototype writes to its own collection: the reporting scripts
# group by (hardness, mode, model), so prototype runs sharing a collection with
# the baseline would be pooled into the same cells and silently corrupt every
# existing aggregate.
FIRESTORE_COLLECTION = ("ChessExperiments" if PROTOTYPE_ID is None
                        else f"ChessExperiments_{PROTOTYPE_ID}")
FIRESTORE_DATABASE: Optional[str] = "experiment-data"  # named DB in the project; None = (default)

# Local durable backup mirror (gitignored). One JSON + one JSONL per experiment.
LOCAL_BACKUP_DIR = _PARENT_DIR / ("gcc_runs_backup" if PROTOTYPE_ID is None
                                  else f"gcc_runs_backup_{PROTOTYPE_ID}")

# Firestore write resilience.
FS_WRITE_MAX_RETRIES = 5
FS_WRITE_BASE_DELAY = 1.0

# Firestore per-document size management (hard limit is 1,048,576 bytes).
# All thresholds are measured against a UTF-8 JSON proxy of the payload and kept
# well under the true limit to absorb Firestore's own (slightly larger) internal
# accounting and leave generous headroom.
#   MOVE_INLINE_MAX_BYTES    -> move docs at/under this keep `calls` inline.
#   CALLS_CHUNK_BUDGET_BYTES -> target max size for each externalized calls chunk.
#   FS_DOC_SAFE_BYTES        -> absolute ceiling for any single chunk doc; a lone
#                               call larger than this is truncated (with a marker)
#                               so it can never exceed the 1 MiB limit. The local
#                               backup retains the untruncated call.
MOVE_INLINE_MAX_BYTES = 200_000
CALLS_CHUNK_BUDGET_BYTES = 200_000
FS_DOC_SAFE_BYTES = 900_000
# Read chunked moves back after writing and verify the reassembled calls match
# the checksum. Adds a few reads per chunked move (a small minority of moves).
VERIFY_CHUNKED_WRITES = True

SCHEMA_VERSION = "gcc-2.0"
SEED = 42
ENV_FILE = _PARENT_DIR / ".env"

# =============================================================================
# END CONFIGURATION
# =============================================================================


# ---------------------------------------------------------------------------
# Time / id helpers
# ---------------------------------------------------------------------------

_EST = ZoneInfo("America/New_York")


def _now_est() -> datetime:
    return datetime.now(_EST)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _est_stamp(dt: Optional[datetime] = None) -> str:
    """Human-readable Eastern-time stamp, e.g. '2026-06-17_14-30-05-EDT'."""
    dt = dt or _now_est()
    return dt.strftime("%Y-%m-%d_%H-%M-%S-%Z")


def _sanitize_id(text: str) -> str:
    """Firestore document ids may not contain '/'. Keep everything else readable."""
    return text.replace("/", "_")


def _make_experiment_id(triad_tag: str) -> str:
    code = uuid.uuid4().hex[:6]
    return _sanitize_id(f"{_est_stamp()}_{triad_tag}_{code}")


# ---------------------------------------------------------------------------
# Chess move-quality classification (mirrors the legacy audit module exactly)
# ---------------------------------------------------------------------------

def _quality_category(centipawn_delta: int) -> str:
    if centipawn_delta is None:
        return "unknown"
    if centipawn_delta >= 0:
        return "excellent"
    loss = abs(centipawn_delta)
    if loss < 10:
        return "excellent"
    elif loss < 30:
        return "good"
    elif loss < 80:
        return "inaccuracy"
    elif loss < 150:
        return "mistake"
    else:
        return "blunder"


# ---------------------------------------------------------------------------
# Deliberation-chat (`calls`) sizing helpers — keep every Firestore doc under
# the 1 MiB limit by packing calls into byte-bounded chunks when a move is big.
# ---------------------------------------------------------------------------

def _json_bytes(obj: Any) -> int:
    """Size of `obj` as compact UTF-8 JSON — our proxy for the Firestore doc size."""
    return len(json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"))


def _calls_checksum(calls: List[Dict[str, Any]]) -> str:
    """Stable checksum over an ordered list of call records (for verified reassembly)."""
    return hashlib.sha256(
        json.dumps(calls, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def _truncate_oversized_call(call: Dict[str, Any], limit: int) -> Dict[str, Any]:
    """Defensive net for a single call larger than `limit` bytes (never observed
    in practice — biggest seen ~0.37 MB). Returns a copy whose largest string
    field(s) are head-truncated until it fits, annotated so the loss is explicit.
    The untruncated original is still kept in the local backup."""
    import copy
    c = copy.deepcopy(call)
    while _json_bytes(c) > limit:
        str_fields = [(k, v) for k, v in c.items()
                      if isinstance(v, str) and len(v) > 64 and not k.startswith("_")]
        if not str_fields:
            break
        k, v = max(str_fields, key=lambda kv: len(kv[1].encode("utf-8")))
        keep = max(64, len(v) // 2)
        if keep >= len(v):
            break
        c[k] = v[:keep]
        marks = c.setdefault("_truncated_fields", {})
        marks[k] = {"original_chars": len(v), "kept_chars": keep}
    if "_truncated_fields" in c:
        c["_truncated"] = True
    return c


def _pack_calls_into_chunks(calls: List[Dict[str, Any]], budget: int,
                            hard_limit: int) -> List[Dict[str, Any]]:
    """Greedily pack consecutive calls into chunks whose JSON stays at/under
    `budget`. Order is preserved. Returns chunk descriptors:
        {index, call_start, call_end, calls, bytes}
    A single call exceeding `hard_limit` is truncated (see above) and placed in
    its own chunk so no chunk doc can ever exceed the Firestore limit."""
    chunks: List[Dict[str, Any]] = []
    cur: List[Dict[str, Any]] = []
    cur_bytes = 2  # the enclosing "[]"
    start = 0
    for i, call in enumerate(calls):
        c = call
        csz = _json_bytes(c)
        if csz > hard_limit:
            c = _truncate_oversized_call(c, hard_limit)
            csz = _json_bytes(c)
        # Flush the current chunk if appending this call would exceed the budget.
        if cur and cur_bytes + csz + 1 > budget:
            chunks.append({"call_start": start, "call_end": i - 1,
                           "calls": cur, "bytes": cur_bytes})
            cur, cur_bytes, start = [], 2, i
        cur.append(c)
        cur_bytes += csz + 1
    if cur:
        chunks.append({"call_start": start, "call_end": len(calls) - 1,
                       "calls": cur, "bytes": cur_bytes})
    for idx, ch in enumerate(chunks):
        ch["index"] = idx
    return chunks


# ---------------------------------------------------------------------------
# Read-side helpers — reassemble a move's full `calls` regardless of whether it
# was stored inline or chunked. Use these in analysis/reporting code instead of
# reading `move["calls"]` directly (which is empty for chunked moves).
# ---------------------------------------------------------------------------

def fetch_move_calls(move_ref) -> List[Dict[str, Any]]:
    """Return the full, ordered `calls` for a live Firestore move document ref,
    transparently reassembling chunked storage and verifying the checksum."""
    snap = move_ref.get()
    move = snap.to_dict() or {}
    if move.get("calls_storage") != "chunked":
        return move.get("calls", []) or []
    chunk_docs = sorted(
        (d.to_dict() for d in move_ref.collection("calls").stream()),
        key=lambda d: d.get("chunk_index", 0))
    calls: List[Dict[str, Any]] = []
    for cd in chunk_docs:
        calls.extend(cd.get("calls", []) or [])
    _verify_reassembled(move, calls)
    return calls


def reassemble_exported_move(move_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the full, ordered `calls` from a move dict produced by the JSON
    export (chunk docs live under the `__subcollection_calls` key)."""
    if move_dict.get("calls_storage") != "chunked":
        return move_dict.get("calls", []) or []
    chunk_map = move_dict.get("__subcollection_calls", {}) or {}
    chunk_docs = sorted(chunk_map.values(), key=lambda d: d.get("chunk_index", 0))
    calls: List[Dict[str, Any]] = []
    for cd in chunk_docs:
        calls.extend(cd.get("calls", []) or [])
    _verify_reassembled(move_dict, calls)
    return calls


def _verify_reassembled(move: Dict[str, Any], calls: List[Dict[str, Any]]) -> None:
    manifest = move.get("calls_manifest", {}) or {}
    expected = manifest.get("stored_calls_checksum") or move.get("calls_checksum")
    if expected and _calls_checksum(calls) != expected:
        raise ValueError(
            f"reassembled calls checksum mismatch for move seq={move.get('seq')} "
            f"(expected {expected}); chunks may be missing — consult the local backup")


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

@dataclass
class ExpConfig:
    experiment_id: str
    session_id: str
    triad_index: int            # 1-based index within the full sweep
    repetition: int             # which EXPERIMENTS_PER_TRIAD repeat this is (1-based)

    hardness: str               # "low" | "medium" | "high"
    mode: str                   # "single" | "multi" | "hardcoded_multi"
    model: str                  # full Gemini model id
    # Communication prototype id, or None for the unmodified baseline system.
    # Part of the experiment-defining set, so it is covered by config_hash().
    prototype_id: Optional[str]

    system_type: str            # "single" | "multi_agent"
    stockfish_elo: int
    stockfish_play_depth: int
    stockfish_eval_depth: int

    preset_crew_path: Optional[str]
    preset_crew_name: Optional[str]

    temperature: float
    thinking_level: str
    max_plies: int
    max_output_tokens: int
    max_context_tokens: int
    move_history_limit: int
    seed: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        # Hash only the experiment-defining fields (not the unique id / index).
        defining = {
            k: v for k, v in self.to_dict().items()
            if k not in ("experiment_id", "session_id", "triad_index", "repetition")
        }
        return hashlib.sha256(
            json.dumps(defining, sort_keys=True).encode()
        ).hexdigest()[:16]


def build_configs(session_id: str) -> List[ExpConfig]:
    """Generate the full ordered list of experiment configs for the sweep.

    Ordering: hardness -> mode -> model -> repetition, matching the project's
    triad enumeration, so repeats of the same triad run consecutively.
    """
    configs: List[ExpConfig] = []
    idx = 0
    for hardness, hcfg in HARDNESS_LEVELS.items():
        for mode in MODES:
            for model in MODELS:
                for rep in range(1, EXPERIMENTS_PER_TRIAD + 1):
                    idx += 1
                    if mode == "single":
                        system_type = "single"
                        preset_path = None
                        preset_name = None
                    elif mode == "multi":
                        system_type = "multi_agent"
                        preset_path = None
                        preset_name = None
                    elif mode == "hardcoded_multi":
                        system_type = "multi_agent"
                        preset_path = HARDCODED_CREW_PATH
                        preset_name = HARDCODED_CREW_NAME
                    else:
                        raise ValueError(f"Unknown mode: {mode}")

                    triad_tag = f"{hardness}_{mode}_{model}"
                    if EXPERIMENTS_PER_TRIAD > 1:
                        triad_tag = f"{triad_tag}_r{rep}"
                    if PROTOTYPE_ID is not None:
                        # so an experiment id identifies its system on sight
                        triad_tag = f"{triad_tag}_{PROTOTYPE_ID}"

                    configs.append(ExpConfig(
                        experiment_id=_make_experiment_id(triad_tag),
                        session_id=session_id,
                        triad_index=idx,
                        repetition=rep,
                        hardness=hardness,
                        mode=mode,
                        model=model,
                        prototype_id=PROTOTYPE_ID,
                        system_type=system_type,
                        stockfish_elo=hcfg["stockfish_elo"],
                        stockfish_play_depth=hcfg["stockfish_play_depth"],
                        stockfish_eval_depth=STOCKFISH_EVAL_DEPTH,
                        preset_crew_path=preset_path,
                        preset_crew_name=preset_name,
                        temperature=TEMPERATURE,
                        thinking_level=THINKING_LEVEL,
                        max_plies=MAX_PLIES,
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                        max_context_tokens=MAX_CONTEXT_TOKENS,
                        move_history_limit=MOVE_HISTORY_LIMIT,
                        seed=SEED,
                    ))
    return configs


# ---------------------------------------------------------------------------
# Credentials / environment setup
# ---------------------------------------------------------------------------

def setup_credentials() -> Dict[str, str]:
    """Load CrewForge/.env explicitly and bridge GCLOUD_* -> GOOGLE_CLOUD_*.

    The Vertex SDK (and cloud_llm_provider) read GOOGLE_CLOUD_PROJECT /
    GOOGLE_CLOUD_LOCATION. We force these from the GCLOUD_* values in the
    project's own .env, OVERRIDING anything already present in the environment,
    so a stray .env from a different project can never leak the wrong project id.
    """
    if not ENV_FILE.exists():
        raise SystemExit(f"Expected env file not found: {ENV_FILE}")

    load_dotenv(ENV_FILE)

    project = os.environ.get("GCLOUD_PROJECT")
    location = os.environ.get("GCLOUD_LOCATION") or "global"
    if not project:
        raise SystemExit("GCLOUD_PROJECT is not set in .env — cannot proceed.")

    # Explicit override (do not trust pre-existing GOOGLE_CLOUD_* values).
    os.environ["GOOGLE_CLOUD_PROJECT"] = project
    os.environ["GOOGLE_CLOUD_LOCATION"] = location

    stockfish_path = os.environ.get("STOCKFISH_PATH", "")
    if not stockfish_path or not os.path.exists(stockfish_path):
        raise SystemExit(
            f"Stockfish binary not found (STOCKFISH_PATH='{stockfish_path}'). "
            "Set STOCKFISH_PATH in .env to a valid engine binary."
        )

    return {"project": project, "location": location, "stockfish_path": stockfish_path}


def get_environment_info(creds: Dict[str, str]) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "platform": platform.system(),
        "platform_release": platform.release(),
        "architecture": platform.machine(),
        "python_version": sys.version,
        "hostname": platform.node(),
        "provider": "vertex",
        "gcloud_project": creds["project"],
        "gcloud_location": creds["location"],
        "captured_at_utc": _utc_iso(),
    }


# ---------------------------------------------------------------------------
# Firestore client + write-with-retry
# ---------------------------------------------------------------------------

def make_firestore_client(project: str):
    from google.cloud import firestore
    if FIRESTORE_DATABASE:
        return firestore.Client(project=project, database=FIRESTORE_DATABASE)
    return firestore.Client(project=project)


def _fs_create_with_retry(doc_ref, data: Dict[str, Any], what: str):
    """Create a Firestore document (never overwrites). Retries transient errors.

    Returns (ok: bool, error: Optional[str]). AlreadyExists is treated as a hard
    failure (we refuse to overwrite) and is NOT retried.
    """
    from google.api_core import exceptions as gexc
    delay = FS_WRITE_BASE_DELAY
    for attempt in range(1, FS_WRITE_MAX_RETRIES + 1):
        try:
            doc_ref.create(data)
            return True, None
        except gexc.AlreadyExists as e:
            return False, f"AlreadyExists (refusing to overwrite {what}): {e}"
        except Exception as e:  # transient / network
            if attempt == FS_WRITE_MAX_RETRIES:
                return False, f"{type(e).__name__}: {e}"
            print(f"    [FIRESTORE] write failed for {what} "
                  f"(attempt {attempt}/{FS_WRITE_MAX_RETRIES}): {e} — retrying in {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    return False, "exhausted retries"


def _fs_update_with_retry(doc_ref, data: Dict[str, Any], what: str):
    delay = FS_WRITE_BASE_DELAY
    for attempt in range(1, FS_WRITE_MAX_RETRIES + 1):
        try:
            doc_ref.update(data)
            return True, None
        except Exception as e:
            if attempt == FS_WRITE_MAX_RETRIES:
                return False, f"{type(e).__name__}: {e}"
            print(f"    [FIRESTORE] update failed for {what} "
                  f"(attempt {attempt}/{FS_WRITE_MAX_RETRIES}): {e} — retrying in {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    return False, "exhausted retries"


# ---------------------------------------------------------------------------
# Experiment writer — owns the experiment doc, its moves subcollection, the
# local backup mirror, and the integrity-checked summary.
# ---------------------------------------------------------------------------

class ExperimentWriter:
    def __init__(self, db, cfg: ExpConfig, env_info: Dict[str, Any]):
        self.db = db
        self.cfg = cfg
        self.env_info = env_info

        self.is_multi = (cfg.system_type == "multi_agent")
        self.created_utc = _utc_iso()
        self.created_est = _est_stamp()

        # In-memory copies for backup + integrity recompute.
        self.move_docs: List[Dict[str, Any]] = []
        self.seq = 0

        # Live-streamed aggregates (cross-checked against recompute at finalize).
        self._live_prompt_tokens = 0
        self._live_completion_tokens = 0

        # Failure tracking — surfaced honestly in the final status.
        self.firestore_write_failures: List[str] = []

        # Chunked-storage stats — surfaced in the summary's data_integrity block.
        self.chunked_move_count = 0
        self.max_chunks_in_move = 0
        self.truncated_call_moves = 0
        self.chunk_verify_failures: List[str] = []

        # Resolve the experiment document, guaranteeing no overwrite.
        self.exp_ref = self._reserve_experiment_doc()

        # Local backup files.
        LOCAL_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        self.backup_json = LOCAL_BACKUP_DIR / f"{self.cfg.experiment_id}.json"
        self.backup_jsonl = LOCAL_BACKUP_DIR / f"{self.cfg.experiment_id}.moves.jsonl"
        # Truncate any stale jsonl for this (unique) id, then write a header line.
        with open(self.backup_jsonl, "w", encoding="utf-8") as f:
            f.write(json.dumps({"_record": "experiment_header",
                                "experiment_doc": self._initial_doc()},
                               ensure_ascii=False, default=str) + "\n")

    # -- experiment doc creation -------------------------------------------

    def _initial_doc(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.cfg.experiment_id,
            "session_id": self.cfg.session_id,
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "created_at_est": self.created_est,
            "created_at_utc": self.created_utc,
            "hardness": self.cfg.hardness,
            "mode": self.cfg.mode,
            "model": self.cfg.model,
            # None for the baseline system; a registered id for a prototype run
            "prototype_id": self.cfg.prototype_id,
            "system_type": self.cfg.system_type,
            "stockfish_elo": self.cfg.stockfish_elo,
            "stockfish_play_depth": self.cfg.stockfish_play_depth,
            "stockfish_eval_depth": self.cfg.stockfish_eval_depth,
            "config": self.cfg.to_dict(),
            "config_hash": self.cfg.config_hash(),
            "environment": self.env_info,
        }

    def _reserve_experiment_doc(self):
        from google.api_core import exceptions as gexc
        for _ in range(5):
            ref = self.db.collection(FIRESTORE_COLLECTION).document(self.cfg.experiment_id)
            ok, err = _fs_create_with_retry(ref, self._initial_doc(),
                                            what=f"experiment {self.cfg.experiment_id}")
            if ok:
                return ref
            if err and err.startswith("AlreadyExists"):
                # Astronomically unlikely with unique ids — mint a fresh one.
                old = self.cfg.experiment_id
                self.cfg.experiment_id = _make_experiment_id(
                    f"{self.cfg.hardness}_{self.cfg.mode}_{self.cfg.model}")
                print(f"    [GUARD] experiment id {old} already existed; "
                      f"regenerated as {self.cfg.experiment_id}")
                continue
            # Transient failure that exhausted retries: we cannot safely proceed
            # without an experiment doc — raise so the run is not silently lost.
            raise RuntimeError(f"Could not create experiment document: {err}")
        raise RuntimeError("Could not reserve a unique experiment document id.")

    # -- per-move handling --------------------------------------------------

    def add_move(self, move_data: Dict[str, Any]) -> None:
        """move_callback target. Builds the move doc, mirrors it locally, then
        writes it to Firestore. Local write happens first so data survives even
        if the Firestore write fails."""
        self.seq += 1
        doc = self._build_move_doc(move_data, self.seq)

        if doc["is_system_move"]:
            self._live_prompt_tokens += doc["prompt_tokens"]
            self._live_completion_tokens += doc["completion_tokens"]

        self.move_docs.append(doc)

        # 1) Durable local append (never lose data).
        try:
            with open(self.backup_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps({"_record": "move", "move": doc},
                                   ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            print(f"    [BACKUP] WARNING: failed to append move {self.seq} locally: {e}")

        # 2) Firestore move document(s) — inline if small, else chunked calls.
        self._persist_move_to_firestore(doc)

        self._print_move_line(doc)

    def _persist_move_to_firestore(self, doc: Dict[str, Any]) -> None:
        """Write a move to Firestore so it can never exceed the 1 MiB doc limit.

        Small moves are written as a single doc with inline `calls` (unchanged
        legacy shape). Large moves have their `calls` externalized into a
        byte-bounded `moves/{seq}/calls/{chunk}` sub-subcollection; the move doc
        then carries a manifest + checksum for verified reassembly. Chunks are
        written before the move doc, so the move doc is the commit point that
        only ever references chunks that were attempted. Any failure is recorded
        and the full move always remains in the local backup."""
        seq = doc["seq"]
        move_ref = self.exp_ref.collection("moves").document(f"{seq:04d}")
        calls = doc.get("calls", []) or []
        full_checksum = _calls_checksum(calls)

        # --- inline path: the whole move comfortably fits in one document ---
        if _json_bytes(doc) <= MOVE_INLINE_MAX_BYTES:
            fs_doc = dict(doc)
            fs_doc["calls_storage"] = "inline"
            fs_doc["calls_checksum"] = full_checksum
            ok, err = _fs_create_with_retry(move_ref, fs_doc, what=f"move {seq:04d}")
            if not ok:
                self.firestore_write_failures.append(f"move {seq:04d}: {err}")
                print(f"    [FIRESTORE] FAILED to write move {seq:04d}: {err} "
                      f"(retained in local backup)")
            return

        # --- chunked path: externalize the deliberation calls ---
        chunks = _pack_calls_into_chunks(calls, CALLS_CHUNK_BUDGET_BYTES, FS_DOC_SAFE_BYTES)
        calls_col = move_ref.collection("calls")
        manifest_chunks: List[Dict[str, Any]] = []
        stored_calls: List[Dict[str, Any]] = []   # the (possibly truncated) calls as written
        failed_chunks: List[str] = []
        any_truncated = False

        for ch in chunks:
            chunk_id = f"{ch['index']:04d}"
            chunk_checksum = _calls_checksum(ch["calls"])
            chunk_doc = {
                "seq": seq,
                "chunk_index": ch["index"],
                "call_start": ch["call_start"],
                "call_end": ch["call_end"],
                "num_calls": len(ch["calls"]),
                "calls_checksum": chunk_checksum,
                "calls": ch["calls"],
            }
            ok, err = _fs_create_with_retry(
                calls_col.document(chunk_id), chunk_doc,
                what=f"move {seq:04d} calls chunk {chunk_id}")
            if not ok:
                failed_chunks.append(chunk_id)
                self.firestore_write_failures.append(
                    f"move {seq:04d} calls chunk {chunk_id}: {err}")
                print(f"    [FIRESTORE] FAILED to write move {seq:04d} calls chunk "
                      f"{chunk_id}: {err} (retained in local backup)")
            any_truncated = any_truncated or any(c.get("_truncated") for c in ch["calls"])
            stored_calls.extend(ch["calls"])
            manifest_chunks.append({
                "chunk_id": chunk_id,
                "chunk_index": ch["index"],
                "call_start": ch["call_start"],
                "call_end": ch["call_end"],
                "num_calls": len(ch["calls"]),
                "calls_checksum": chunk_checksum,
                "written": ok,
            })

        stored_checksum = _calls_checksum(stored_calls)
        fs_doc = {k: v for k, v in doc.items() if k != "calls"}
        fs_doc["calls"] = []
        fs_doc["calls_storage"] = "chunked"
        fs_doc["calls_checksum"] = full_checksum
        fs_doc["calls_manifest"] = {
            "num_chunks": len(chunks),
            "num_calls": len(calls),
            "chunk_budget_bytes": CALLS_CHUNK_BUDGET_BYTES,
            "calls_checksum": full_checksum,            # over the original calls
            "stored_calls_checksum": stored_checksum,   # over the calls as written
            "calls_truncated": any_truncated,
            "failed_chunks": failed_chunks,
            "chunks": manifest_chunks,
        }
        ok, err = _fs_create_with_retry(move_ref, fs_doc, what=f"move {seq:04d}")
        if not ok:
            self.firestore_write_failures.append(f"move {seq:04d}: {err}")
            print(f"    [FIRESTORE] FAILED to write move {seq:04d}: {err} "
                  f"(retained in local backup)")

        # --- stats + optional read-back verification ---
        self.chunked_move_count += 1
        self.max_chunks_in_move = max(self.max_chunks_in_move, len(chunks))
        if any_truncated:
            self.truncated_call_moves += 1
        if VERIFY_CHUNKED_WRITES and ok and not failed_chunks:
            self._verify_chunked_move(move_ref, seq, stored_checksum)

    def _verify_chunked_move(self, move_ref, seq: int, expected_checksum: str) -> None:
        """Read the calls chunks back, reassemble in order, and confirm the
        checksum matches what we stored. A mismatch is recorded honestly."""
        try:
            chunk_docs = sorted(
                (d.to_dict() for d in move_ref.collection("calls").stream()),
                key=lambda d: d.get("chunk_index", 0))
            reassembled: List[Dict[str, Any]] = []
            for cd in chunk_docs:
                reassembled.extend(cd.get("calls", []) or [])
            got = _calls_checksum(reassembled)
            if got != expected_checksum:
                msg = (f"move {seq:04d}: chunk read-back checksum mismatch "
                       f"(expected {expected_checksum}, got {got})")
                self.chunk_verify_failures.append(msg)
                print(f"    [FIRESTORE] WARNING {msg}")
        except Exception as e:
            msg = f"move {seq:04d}: chunk read-back verification error: {type(e).__name__}: {e}"
            self.chunk_verify_failures.append(msg)
            print(f"    [FIRESTORE] WARNING {msg}")

    def _build_move_doc(self, move_data: Dict[str, Any], seq: int) -> Dict[str, Any]:
        env_state = move_data.get("env_state", {}) or {}
        smr = move_data.get("system_move_record", {}) or {}
        player = move_data.get("player", "")
        is_system = player in ("single_llm", "multi_llm")
        cp_delta = env_state.get("centipawn_eval_delta", 0)
        calls = smr.get("calls", []) or []

        thoughts_tokens = sum(int(c.get("thoughts_tokens", 0) or 0) for c in calls)
        deliberation_cycles = (
            len({c.get("cycle", 0) for c in calls}) if (self.is_multi and calls) else 0
        )

        return {
            "seq": seq,
            "ply": env_state.get("ply"),
            "turn": move_data.get("turn"),
            "player": player,
            "is_system_move": is_system,
            "action": move_data.get("action", ""),
            # chess move details
            "uci": env_state.get("uci", ""),
            "san": env_state.get("san", ""),
            "fen_before": env_state.get("fen_before", ""),
            "fen_after": env_state.get("fen_after", ""),
            "centipawn_eval": env_state.get("centipawn_eval"),
            "centipawn_eval_delta": cp_delta,
            "quality_category": _quality_category(cp_delta) if is_system else "engine",
            # timing
            "latency_ms": move_data.get("latency_ms", 0),
            "timestamp": move_data.get("timestamp", ""),
            # token usage (aggregated over all calls in this move)
            "prompt_tokens": int(smr.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(smr.get("completion_tokens", 0) or 0),
            "thoughts_tokens": thoughts_tokens,
            "total_tokens": int(smr.get("total_tokens", 0) or 0),
            "total_latency_ms": int(smr.get("total_latency_ms", 0) or 0),
            "call_count": int(smr.get("call_count", len(calls)) or 0),
            "deliberation_cycles": deliberation_cycles,
            # FULL deliberation chat content for this move: every call's prompts,
            # response, thinking, tokens, and (for MAS) caller/cycle/tool data.
            "calls": calls,
        }

    def _print_move_line(self, doc: Dict[str, Any]) -> None:
        if doc["is_system_move"]:
            extra = f" calls={doc['call_count']}"
            if self.is_multi:
                extra += f" cycles={doc['deliberation_cycles']}"
            print(f"    ply {str(doc['ply']):>3} {doc['player']:<11} {doc['uci']:<6} "
                  f"cp={str(doc['centipawn_eval']):>6} Δ={str(doc['centipawn_eval_delta']):>6} "
                  f"toks={doc['total_tokens']:>6}{extra}  [{doc['quality_category']}]")
        else:
            print(f"    ply {str(doc['ply']):>3} {doc['player']:<11} {doc['uci']:<6} "
                  f"cp={str(doc['centipawn_eval']):>6} (opponent)")

    # -- finalize -----------------------------------------------------------

    def finalize(self, run_result, error: Optional[str] = None,
                 error_traceback: Optional[str] = None) -> Dict[str, Any]:
        """Recompute the summary from the stored move docs, cross-check, write
        the experiment summary to Firestore, and write the consolidated local
        backup JSON. Returns the summary dict."""
        ended_utc = _utc_iso()

        result_dict = (run_result.result if run_result else {}) or {}
        duration = getattr(run_result, "duration_seconds", 0.0) if run_result else 0.0
        run_error = error if error is not None else (getattr(run_result, "error", None) if run_result else None)
        run_error_type = getattr(run_result, "error_type", None) if run_result else None

        # --- recompute aggregates from the authoritative move docs ---
        system_moves = [m for m in self.move_docs if m["is_system_move"]]
        opponent_moves = [m for m in self.move_docs if not m["is_system_move"]]

        recomputed_prompt = sum(m["prompt_tokens"] for m in system_moves)
        recomputed_completion = sum(m["completion_tokens"] for m in system_moves)
        recomputed_thoughts = sum(m["thoughts_tokens"] for m in system_moves)
        recomputed_total = recomputed_prompt + recomputed_completion
        total_cycles = sum(m["deliberation_cycles"] for m in system_moves)

        quality_counts = {"excellent": 0, "good": 0, "inaccuracy": 0,
                          "mistake": 0, "blunder": 0}
        cp_losses: List[int] = []
        for m in system_moves:
            q = m["quality_category"]
            if q in quality_counts:
                quality_counts[q] += 1
            d = m["centipawn_eval_delta"]
            if isinstance(d, int) and d < 0:
                cp_losses.append(abs(d))
        avg_cp_loss = (sum(cp_losses) / len(cp_losses)) if cp_losses else 0.0

        system_time_ms = sum(m["latency_ms"] for m in system_moves)
        opponent_time_ms = sum(m["latency_ms"] for m in opponent_moves)

        # --- outcome ---
        result_str = result_dict.get("result", "incomplete")
        winner = result_dict.get("winner", "incomplete")
        final_fen = result_dict.get("final_fen", "")
        outcome = self._derive_outcome(winner, result_str, run_error)

        # --- integrity / honesty checks ---
        live_matches_recompute = (
            self._live_prompt_tokens == recomputed_prompt
            and self._live_completion_tokens == recomputed_completion
        )
        call_errors = sum(
            1 for m in self.move_docs for c in m["calls"] if c.get("error")
        )

        if run_error:
            status = "errored"
        elif self.firestore_write_failures or self.chunk_verify_failures:
            status = "completed_with_write_errors"
        else:
            status = "completed"

        summary = {
            "status": status,
            "outcome": outcome,
            "result": result_str,
            "winner": winner,
            "final_fen": final_fen,
            "ended_at_utc": ended_utc,
            "ended_at_est": _est_stamp(),
            "duration_seconds": duration,
            # move counts
            "total_moves": len(self.move_docs),
            "system_moves": len(system_moves),
            "opponent_moves": len(opponent_moves),
            "total_plies": len(self.move_docs),
            # tokens (recomputed from stored move docs)
            "total_prompt_tokens": recomputed_prompt,
            "total_completion_tokens": recomputed_completion,
            "total_thoughts_tokens": recomputed_thoughts,
            "total_tokens": recomputed_total,
            # timing
            "system_time_ms": system_time_ms,
            "opponent_time_ms": opponent_time_ms,
            # chess quality
            "avg_centipawn_loss": round(avg_cp_loss, 2),
            "quality_counts": quality_counts,
            # multi-agent deliberation
            "total_deliberation_cycles": total_cycles,
            "avg_cycles_per_move": round(total_cycles / len(system_moves), 3) if system_moves else 0.0,
            # errors
            "error": run_error,
            "error_type": run_error_type,
            "error_traceback": error_traceback,
            "call_errors": call_errors,
            # data integrity
            "data_integrity": {
                "live_aggregate_matches_recompute": live_matches_recompute,
                "live_prompt_tokens": self._live_prompt_tokens,
                "live_completion_tokens": self._live_completion_tokens,
                "moves_written_locally": len(self.move_docs),
                "firestore_write_failures": len(self.firestore_write_failures),
                "firestore_write_failure_details": self.firestore_write_failures,
                # calls-chunking (1 MiB doc-limit safety)
                "chunked_moves": self.chunked_move_count,
                "max_chunks_in_move": self.max_chunks_in_move,
                "truncated_call_moves": self.truncated_call_moves,
                "chunk_verify_failures": len(self.chunk_verify_failures),
                "chunk_verify_failure_details": self.chunk_verify_failures,
            },
        }
        summary["summary_checksum"] = self._summary_checksum(summary)

        # --- write summary to Firestore (update; doc already exists) ---
        ok, err = _fs_update_with_retry(self.exp_ref, summary,
                                        what=f"summary {self.cfg.experiment_id}")
        if not ok:
            self.firestore_write_failures.append(f"summary: {err}")
            summary["status"] = "completed_with_write_errors" if not run_error else "errored"
            summary["data_integrity"]["firestore_write_failures"] = len(self.firestore_write_failures)
            summary["data_integrity"]["firestore_write_failure_details"] = self.firestore_write_failures
            print(f"    [FIRESTORE] FAILED to write summary: {err} (retained in local backup)")

        # --- consolidated local backup ---
        self._write_local_backup(summary)

        return summary

    @staticmethod
    def _derive_outcome(winner: str, result_str: str, run_error: Optional[str]) -> str:
        if run_error:
            return "error"
        if winner == "system":
            return "model_win"
        if winner == "draw":
            return "draw"
        if winner in ("incomplete", "", None):
            return "incomplete"
        # any other winner string is the engine's name
        return "model_loss"

    @staticmethod
    def _summary_checksum(summary: Dict[str, Any]) -> str:
        core = {k: v for k, v in summary.items() if k != "summary_checksum"}
        return hashlib.sha256(
            json.dumps(core, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

    def _write_local_backup(self, summary: Dict[str, Any]) -> None:
        payload = {
            "experiment_doc": self._initial_doc(),
            "summary": summary,
            "moves": self.move_docs,
        }
        try:
            tmp = self.backup_json.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
            tmp.replace(self.backup_json)
        except Exception as e:
            print(f"    [BACKUP] WARNING: failed to write consolidated backup: {e}")


# ---------------------------------------------------------------------------
# Single experiment execution
# ---------------------------------------------------------------------------

def _build_system(cfg: ExpConfig, env, chat_fn):
    from agent_config import LLMConfig
    from single_system import SingleLLMSystem
    from multi_system import MultiLLMSystem

    env_intro = env.get_env_intro_prompt()

    if cfg.system_type == "single":
        llm_config = LLMConfig(
            model_name=cfg.model,
            temperature=cfg.temperature,
            max_context_tokens=cfg.max_context_tokens,
            max_token=cfg.max_output_tokens,
            think=False,        # thinking level is applied by the chat_fn wrapper
            seed=cfg.seed,
        )
        return SingleLLMSystem(
            llm_config=llm_config,
            env_intro_prompt=env_intro,
            max_retries=MAX_SYSTEM_RETRIES,
            chat_fn=chat_fn,
            symmetric_invalid_move=SYMMETRIC_INVALID_MOVE,
        )

    meta_cfg = LLMConfig(
        model_name=cfg.model,
        temperature=cfg.temperature,
        max_context_tokens=cfg.max_context_tokens,
        max_token=cfg.max_output_tokens,
        think=False,
        seed=cfg.seed,
    )
    crew_cfg = LLMConfig(
        model_name=cfg.model,            # same model for meta and crew
        temperature=cfg.temperature,
        max_context_tokens=cfg.max_context_tokens,
        max_token=cfg.max_output_tokens,
        think=False,
        seed=cfg.seed,
    )

    preset_path = cfg.preset_crew_path
    preset_name = cfg.preset_crew_name

    # getattr, not cfg.prototype_id: this function is shared with the puzzle
    # harness and must keep working for any config that predates the field.
    proto_id = getattr(cfg, "prototype_id", None)
    if proto_id is None:
        system_cls = MultiLLMSystem
        extra: Dict[str, Any] = {}
    else:
        import prototypes
        spec = prototypes.get(proto_id)
        system_cls = spec.multi_system_cls
        extra = {"unit_spec": prototypes.resolve_unit_spec(spec, env)}
        # A prototype may ship its own hardcoded-crew preset, written for its
        # own channel. It replaces the harness preset only for the modes that
        # use one, so `multi` still designs its crew from scratch.
        if preset_path is not None and spec.preset_crew_path is not None:
            preset_path = spec.preset_crew_path
            preset_name = spec.preset_crew_name
            print(f"USING PROTOTYPE CREW {preset_name} FOUND AT: {preset_path}.")

    return system_cls(
        meta_llm_config=meta_cfg,
        crew_llm_config=crew_cfg,
        env_intro_prompt=env_intro,
        every_other=env.has_opponent,
        max_retries=MAX_SYSTEM_RETRIES,
        chat_fn=chat_fn,
        preset_crew_path=Path(preset_path) if preset_path else None,
        preset_crew_name=preset_name,
        symmetric_invalid_move=SYMMETRIC_INVALID_MOVE,
        **extra,
    )


def run_one_experiment(db, cfg: ExpConfig, stockfish_path: str,
                       env_info: Dict[str, Any], total: int) -> Dict[str, Any]:
    from chess_engine_module import StockfishEngine
    from chess_env import ChessEnv
    from runner import run_game
    from cloud_llm_provider import create_cloud_chat_fn

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT {cfg.triad_index}/{total}   [{cfg.session_id}]")
    print(f"  id      : {cfg.experiment_id}")
    print(f"  triad   : hardness={cfg.hardness}  mode={cfg.mode}  model={cfg.model}"
          + (f"  prototype={cfg.prototype_id}" if cfg.prototype_id else ""))
    print(f"  engine  : Stockfish ELO={cfg.stockfish_elo} play_depth={cfg.stockfish_play_depth}")
    print(f"  thinking: {cfg.thinking_level}   temperature: {cfg.temperature}")
    print(f"{'=' * 70}")

    writer = ExperimentWriter(db, cfg, env_info)

    try:
        engine = StockfishEngine(stockfish_path, cfg.stockfish_play_depth, cfg.stockfish_eval_depth)
        engine.engine.configure({
            "UCI_LimitStrength": True,
            "Skill Level": 0,
            "UCI_Elo": cfg.stockfish_elo,
        })
        env = ChessEnv(chess_engine=engine, max_plies=cfg.max_plies)

        chat_fn = create_cloud_chat_fn(provider="vertex", thinking_level=cfg.thinking_level)
        system = _build_system(cfg, env, chat_fn)

        run_result = run_game(
            env=env,
            system=system,
            config=cfg.to_dict(),
            move_callback=writer.add_move,
            verbose=False,                # we print our own concise move lines
            move_delay=MOVE_DELAY_SECONDS,
        )
        summary = writer.finalize(run_result=run_result, error=run_result.error)

    except KeyboardInterrupt:
        summary = writer.finalize(run_result=None,
                                  error="KeyboardInterrupt: user interrupted")
        _print_experiment_summary(cfg, summary)
        raise
    except Exception as e:
        tb = traceback.format_exc()
        summary = writer.finalize(run_result=None,
                                  error=f"{type(e).__name__}: {e}",
                                  error_traceback=tb)

    _print_experiment_summary(cfg, summary)
    return {"experiment_id": cfg.experiment_id, "cfg": cfg, "summary": summary}


def _print_experiment_summary(cfg: ExpConfig, summary: Dict[str, Any]) -> None:
    di = summary.get("data_integrity", {})
    print(f"\n  -- result: {summary['outcome']} ({summary['result']}, winner={summary['winner']})")
    print(f"     status: {summary['status']}   moves: {summary['total_moves']} "
          f"(model: {summary['system_moves']})")
    print(f"     tokens: prompt={summary['total_prompt_tokens']:,} "
          f"completion={summary['total_completion_tokens']:,} "
          f"(thinking={summary['total_thoughts_tokens']:,}) "
          f"total={summary['total_tokens']:,}")
    if cfg.system_type == "multi_agent":
        print(f"     deliberation cycles: {summary['total_deliberation_cycles']} "
              f"(avg/move {summary['avg_cycles_per_move']})")
    print(f"     avg centipawn loss: {summary['avg_centipawn_loss']}   "
          f"call errors: {summary['call_errors']}")
    if summary.get("error"):
        print(f"     ERROR: {summary['error']}")
    if di.get("chunked_moves"):
        print(f"     chunked moves: {di['chunked_moves']} "
              f"(max {di['max_chunks_in_move']} chunks/move; "
              f"calls stored in moves/<seq>/calls/<chunk>)")
    if di.get("truncated_call_moves"):
        print(f"     WARNING: {di['truncated_call_moves']} move(s) had an oversized "
              f"single call truncated for Firestore (full copy in local backup).")
    if di.get("firestore_write_failures"):
        print(f"     FIRESTORE WRITE FAILURES: {di['firestore_write_failures']} "
              f"(data preserved in local backup)")
    if di.get("chunk_verify_failures"):
        print(f"     CHUNK VERIFY FAILURES: {di['chunk_verify_failures']} "
              f"(see chunk_verify_failure_details)")
    if not di.get("live_aggregate_matches_recompute", True):
        print("     WARNING: live token aggregate did not match recompute — see data_integrity.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    creds = setup_credentials()
    env_info = get_environment_info(creds)

    session_id = _sanitize_id(f"{_est_stamp()}_{uuid.uuid4().hex[:6]}")
    configs = build_configs(session_id)

    print("=" * 70)
    print("CrewForge GCC Chess Experiments (Vertex + Firestore)")
    print("=" * 70)
    print(f"  GCloud project   : {creds['project']}")
    print(f"  GCloud location  : {creds['location']}")
    print(f"  Firestore coll.  : {FIRESTORE_COLLECTION}"
          f"{' (db=' + FIRESTORE_DATABASE + ')' if FIRESTORE_DATABASE else ' (default db)'}")
    print(f"  Local backup dir : {LOCAL_BACKUP_DIR}")
    print(f"  Session id       : {session_id}")
    print(f"  System           : {'BASELINE' if PROTOTYPE_ID is None else 'PROTOTYPE ' + PROTOTYPE_ID}")
    print(f"  Models           : {', '.join(MODELS)}")
    print(f"  Hardness levels  : {', '.join(HARDNESS_LEVELS)}")
    print(f"  Modes            : {', '.join(MODES)}")
    print(f"  Experiments/triad: {EXPERIMENTS_PER_TRIAD}")
    print(f"  Total experiments: {len(configs)}")
    print(f"  Thinking level   : {THINKING_LEVEL}   Temperature: {TEMPERATURE}   "
          f"Max plies: {MAX_PLIES}")
    print("=" * 70)

    db = make_firestore_client(creds["project"])

    results: List[Dict[str, Any]] = []
    interrupted = False
    for cfg in configs:
        try:
            results.append(run_one_experiment(db, cfg, creds["stockfish_path"],
                                               env_info, len(configs)))
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Stopping sweep after current experiment.")
            interrupted = True
            break

    _print_session_summary(session_id, results, interrupted, len(configs))


def _print_session_summary(session_id: str, results: List[Dict[str, Any]],
                           interrupted: bool, total: int) -> None:
    print("\n" + "=" * 70)
    print(f"SESSION COMPLETE: {session_id}")
    print("=" * 70)
    print(f"  Ran {len(results)}/{total} experiments"
          + ("  (INTERRUPTED)" if interrupted else ""))
    n_ok = sum(1 for r in results if r["summary"]["status"] == "completed")
    n_writeerr = sum(1 for r in results if r["summary"]["status"] == "completed_with_write_errors")
    n_err = sum(1 for r in results if r["summary"]["status"] == "errored")
    print(f"  completed: {n_ok}   completed_with_write_errors: {n_writeerr}   errored: {n_err}")
    total_tokens = sum(r["summary"]["total_tokens"] for r in results)
    print(f"  total tokens across session: {total_tokens:,}")
    print(f"  Firestore: {FIRESTORE_COLLECTION}/<experiment_id>/moves/<seq>"
          f"  (large moves: .../moves/<seq>/calls/<chunk>)")
    print(f"  Local backups: {LOCAL_BACKUP_DIR}")
    if n_writeerr or n_err:
        print("  NOTE: some experiments had errors or write failures — review their "
              "status/data_integrity fields in Firestore and the local backups.")
    print("=" * 70)


if __name__ == "__main__":
    main()
