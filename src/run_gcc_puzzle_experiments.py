
from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make sibling src/ modules importable regardless of the current working
# directory, so this module can be imported (not just executed) from anywhere.
# run_gcc_experiments performs the same setup, but relying on that would make
# this module's own import of it order-dependent.
_THIS_DIR = Path(__file__).resolve().parent          # .../CrewForge/src
_PARENT_DIR = _THIS_DIR.parent                        # .../CrewForge
for _p in (str(_THIS_DIR), str(_PARENT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- reused BY REFERENCE from the in-flight module (no edits to that file) ---
import run_gcc_experiments as gcc
from run_gcc_experiments import (
    _utc_iso,
    _est_stamp,
    _sanitize_id,
    _make_experiment_id,
    _json_bytes,
    _calls_checksum,
    _pack_calls_into_chunks,
    _fs_create_with_retry,
    _fs_update_with_retry,
    setup_credentials,
    get_environment_info,
    make_firestore_client,
    _build_system,
    SCHEMA_VERSION as BASE_SCHEMA_VERSION,
    LOCAL_BACKUP_DIR as BASE_BACKUP_DIR,
    MOVE_INLINE_MAX_BYTES,
    CALLS_CHUNK_BUDGET_BYTES,
    FS_DOC_SAFE_BYTES,
    VERIFY_CHUNKED_WRITES,
    MOVE_DELAY_SECONDS,
)

from runner import run_game
from system import System


# ---------------------------------------------------------------------------
# Puzzle experiment configuration
# ---------------------------------------------------------------------------

# Schema tag for puzzle documents. Deliberately NOT the chess harness'
# SCHEMA_VERSION: these documents have a different field set, and a consumer that
# branches on schema_version must be able to tell them apart. The chess value is
# still recorded as base_schema_version for lineage.
PUZZLE_SCHEMA_VERSION = "gcc-puzzle-1.0"

# Communication prototype to run, or None for the unmodified baseline system.
# Independent of the chess harness' own PROTOTYPE_ID: the two campaigns are
# configured separately and neither should ever be flipped as a side effect of
# the other. Registered ids: see prototypes.list_ids().
PROTOTYPE_ID: Optional[str] = "p4_refutation"

# Firestore collection for puzzle experiments (separate from ChessExperiments).
# A prototype writes to its own collection: gcc_puzzle_report.py groups by
# (mode, model), so sharing a collection with the baseline would pool prototype
# runs into the same cells and corrupt every existing aggregate.
PUZZLE_FIRESTORE_COLLECTION = ("ChessPuzzleExperiments" if PROTOTYPE_ID is None
                               else f"ChessPuzzleExperiments_{PROTOTYPE_ID}")

# Local durable backup mirror, kept in its own subdirectory so puzzle runs do not
# intermingle with an in-flight chess campaign's backups.
LOCAL_BACKUP_DIR = BASE_BACKUP_DIR / ("puzzles" if PROTOTYPE_ID is None
                                      else f"puzzles_{PROTOTYPE_ID}")

# The puzzle sweep axes. Difficulty is NOT an axis — it is a property of each
# puzzle (its Lichess rating), carried in the data and bucketed post-hoc.
MODES: List[str] = ["multi"]#, "multi", "hardcoded_multi"]
MODELS: List[str] = [
    "gemini-3.5-flash",
]

# Repetitions of each (mode, model, shard) cell. The paper's protocol is greedy
# pass@1 — a single deterministic sample — so 1 repetition is the matching
# setting. Raise this only if TEMPERATURE is also raised above 0.
EXPERIMENTS_PER_REP = 1

# Puzzle dataset selection (loaded live inside ChessPuzzleEnv).
PUZZLE_DATASET_REPO = "UofTCSSLab/C1-data"
PUZZLE_DATASET_CONFIG = "test"
PUZZLE_DATASET_SPLIT = "test"
PUZZLE_TASK_LIMIT = 1         # total puzzles to cover; -1 = the whole split
PUZZLE_K = 1                     # sweeps of the whole set (dormant; keep at 1)

# --- test mode -------------------------------------------------------------
# A small, difficulty-stratified set of puzzle ids used to smoke-test MAS
# changes without running a full shard sweep. When TEST_MODE is on, TEST_SET
# replaces the sharded window entirely and the run is tagged "test_" so it
# cannot be confused with a reportable sweep.
TEST_MODE: bool = True
TEST_SET: List[str] = [
    # 8 hard
    "1n2Vk", "WWtDw", "u6zzA", "rLfdK", "ZssUU", "Z89dh", "fnzeX", "0DvFA",
    # 2 easy, as a floor check
    "sSXWs", "0OIiM",
]


# Known size of the C1 test split. Used ONLY to plan shard boundaries before the
# dataset is loaded; the environment validates the window it actually receives.
PUZZLE_DATASET_TOTAL = 900

# Puzzles per experiment. One experiment is one unit of restartable work: at
# measured multi-agent rates a full 900-puzzle cell runs ~11 h with no
# checkpoint, so a crash or a run_game timeout forfeits the whole cell. Shards
# bound that loss, and can be re-run or distributed independently.
# Set to None to emit one experiment per (mode, model) covering everything.
PUZZLE_SHARD_SIZE: Optional[int] = 60

# --- sampling / generation (declared locally on purpose; see module docstring) ---
# C1 reports greedy pass@1. 
# 
# Set to 1.0 as per Gemini-recommendations. Temperature 0 matches the paper though; anything higher
# makes the numbers non-comparable to the published baselines.
TEMPERATURE = 1.0
THINKING_LEVEL = "low"           # minimum reasoning level (cannot be fully disabled)
MAX_OUTPUT_TOKENS = 16384        # per-call output cap (includes thinking tokens)
MAX_CONTEXT_TOKENS = 1_048_576   # context window used for prompt-size guards
# Recorded for provenance only: the Vertex adapter maps just temperature and
# num_predict into GenerateContentConfig, so this seed does NOT reach the API.
# Reproducibility comes from TEMPERATURE = 0, not from the seed.
SEED = 42
SEED_APPLIED_BY_PROVIDER = False

# Hardcoded crew preset (mirrors the chess harness' hardcoded_multi mode).
HARDCODED_CREW_PATH = str(_THIS_DIR / "hardcoded_crew_v1.json")
HARDCODED_CREW_NAME = "hardcoded_crew_v1"

# Conversation history the crew sees, in puzzles. Each puzzle is an independent
# position, so the answer is "this puzzle only" — see _isolate_crew_memory().
PUZZLE_MOVE_HISTORY_LIMIT = 1


@dataclass
class PuzzleExpConfig:
    """Config for a single puzzle experiment (one shard of the puzzle set).

    Deliberately free of the chess-game fields (hardness, stockfish_elo,
    play/eval depth, max_plies): none apply when there is no engine opponent and
    difficulty is a data property.
    """
    experiment_id: str
    session_id: str
    sweep_index: int            # 1-based index within the full sweep
    # which EXPERIMENTS_PER_REP repeat this is (1-based)
    repetition: int

    mode: str                   # "single" | "multi" | "hardcoded_multi"
    model: str                  # full Gemini model id
    system_type: str            # "single" | "multi_agent"
    # Communication prototype id, or None for the unmodified baseline system.
    # Part of the experiment-defining set, so it is covered by config_hash().
    prototype_id: Optional[str]

    preset_crew_path: Optional[str]
    preset_crew_name: Optional[str]

    dataset_repo: str
    dataset_config: str
    dataset_split: str
    task_offset: int            # first dataset row this experiment covers
    task_limit: int             # rows covered from task_offset (-1 = to the end)
    puzzle_ids: Optional[List[str]]  # explicit selection; None = use the window
    shard_index: int            # 0-based shard number
    num_shards: int             # total shards making up one (mode, model) cell
    k: int

    temperature: float
    thinking_level: str
    max_output_tokens: int
    max_context_tokens: int
    seed: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        defining = {
            k: v for k, v in self.to_dict().items()
            if k not in ("experiment_id", "session_id", "sweep_index", "repetition")
        }
        return hashlib.sha256(
            json.dumps(defining, sort_keys=True).encode()
        ).hexdigest()[:16]


def _plan_shards() -> List[Tuple[int, int]]:
    """Work out the (task_offset, task_limit) window for each shard.

    Returns:
        One (offset, limit) pair per shard, covering the first PUZZLE_TASK_LIMIT
        rows of the split (or all PUZZLE_DATASET_TOTAL of them when it is -1).
        A single (0, PUZZLE_TASK_LIMIT) window when sharding is disabled.

    Raises:
        ValueError: If PUZZLE_SHARD_SIZE is set to a non-positive value.
    """
    if PUZZLE_SHARD_SIZE is None:
        return [(0, PUZZLE_TASK_LIMIT)]
    if PUZZLE_SHARD_SIZE <= 0:
        raise ValueError(
            f"PUZZLE_SHARD_SIZE must be a positive count or None, "
            f"got {PUZZLE_SHARD_SIZE}")

    total = (PUZZLE_DATASET_TOTAL if PUZZLE_TASK_LIMIT == (-1)
             else min(PUZZLE_TASK_LIMIT, PUZZLE_DATASET_TOTAL))
    return [(off, min(PUZZLE_SHARD_SIZE, total - off))
            for off in range(0, total, PUZZLE_SHARD_SIZE)]


def build_puzzle_configs(session_id: str) -> List[PuzzleExpConfig]:
    """Generate the ordered list of puzzle experiment configs.

    Ordering is SHARD-major (shard -> mode -> model -> repetition). If a session
    is interrupted, that leaves every mode covering the same prefix of the
    benchmark, which stays directly comparable; mode-major ordering would instead
    leave one mode complete and another untouched.

    Args:
        session_id: The session identifier shared by all configs in this sweep.

    Returns:
        The ordered list of PuzzleExpConfig.
    """
    if TEST_MODE:
        if not TEST_SET or not all(pid.strip() for pid in TEST_SET):
            raise ValueError(
                "TEST_MODE is on but TEST_SET is empty or contains blank ids")
        # ChessPuzzleEnv rejects puzzle_ids combined with a window, so the
        # window fields stay at their no-op values for the single test shard.
        shards: List[Tuple[int, int]] = [(0, -1)]
    else:
        shards = _plan_shards()

    configs: List[PuzzleExpConfig] = []
    idx = 0
    for shard_index, (task_offset, task_limit) in enumerate(shards):
        for mode in MODES:
            for model in MODELS:
                for rep in range(1, EXPERIMENTS_PER_REP + 1):
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

                    configs.append(PuzzleExpConfig(
                        experiment_id=_make_experiment_id(
                            _experiment_tag(mode, model, shard_index,
                                            len(shards), rep)),
                        session_id=session_id,
                        sweep_index=idx,
                        repetition=rep,
                        mode=mode,
                        model=model,
                        system_type=system_type,
                        prototype_id=PROTOTYPE_ID,
                        preset_crew_path=preset_path,
                        preset_crew_name=preset_name,
                        dataset_repo=PUZZLE_DATASET_REPO,
                        dataset_config=PUZZLE_DATASET_CONFIG,
                        dataset_split=PUZZLE_DATASET_SPLIT,
                        task_offset=task_offset,
                        task_limit=task_limit,
                        puzzle_ids=list(TEST_SET) if TEST_MODE else None,
                        shard_index=shard_index,
                        num_shards=len(shards),
                        k=PUZZLE_K,
                        temperature=TEMPERATURE,
                        thinking_level=THINKING_LEVEL,
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                        max_context_tokens=MAX_CONTEXT_TOKENS,
                        seed=SEED,
                    ))
    return configs


def _experiment_tag(mode: str, model: str, shard_index: int,
                    num_shards: int, rep: int) -> str:
    """Build the human-readable portion of an experiment id."""
    tag = f"test_puzzle_{mode}_{model}" if TEST_MODE else f"puzzle_{mode}_{model}"
    if num_shards > 1:
        tag = f"{tag}_s{shard_index}"
    if EXPERIMENTS_PER_REP > 1:
        tag = f"{tag}_r{rep}"
    if PROTOTYPE_ID is not None:
        # so an experiment id identifies its system on sight
        tag = f"{tag}_{PROTOTYPE_ID}"
    return tag


# ---------------------------------------------------------------------------
# Firestore payload safety
# ---------------------------------------------------------------------------

def _fs_safe_keys(obj: Any) -> Any:
    """Recursively coerce every mapping key to a string.

    Firestore maps are map<string, Value> and the client's encoder raises
    TypeError on any other key type, which _fs_update_with_retry can only report
    as a failed write after burning its whole retry budget. A metric dict keyed
    by an int (a pass@k, a rating bucket) is an easy and near-invisible way to
    lose an entire summary, so every payload is normalized on the way out.
    Values are untouched.

    Args:
        obj: Any JSON-like structure destined for Firestore.

    Returns:
        The same structure with all mapping keys as str.
    """
    if isinstance(obj, dict):
        return {str(k): _fs_safe_keys(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_fs_safe_keys(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Per-puzzle failure isolation
# ---------------------------------------------------------------------------

class _ResilientSystem(System):
    """Wraps a System so one unanswerable puzzle cannot end the whole sweep.

    SingleLLMSystem raises RuntimeError once its retry budget is spent (repeated
    JSON-parse failures or illegal moves), and MultiLLMSystem raises
    ConsensusFailureError when the crew never converges. runner.run_game() wraps
    its entire loop in a single try/except, so either exception ends the run. In
    a chess game that costs one game; in a batch benchmark it discards every
    remaining puzzle — potentially hours of compute — over a single item.

    This wrapper turns such a failure into what it actually is scientifically: a
    puzzle the system failed to answer, scored incorrect. It returns "", which is
    never a legal UCI move, so the environment records it as wrong and moves on.
    The underlying system's own call records (including the failed attempts) are
    untouched and still reach the move document, and every failure is recorded
    here for the summary.

    It is also the hook point for per-puzzle crew memory isolation, since it is
    the one place that sees every puzzle boundary.
    """

    def __init__(self, inner: System, memory: Any = None):
        """
        Args:
            inner: The system to delegate to.
            memory: Optional multi-agent Memory to prune between puzzles.
        """
        self._inner = inner
        self._memory = memory
        self.failures: List[str] = []

    def next_move(self, env_state_prompt) -> str:
        self._begin_puzzle(env_state_prompt)
        try:
            return self._inner.next_move(env_state_prompt)
        except (KeyboardInterrupt, SystemExit):
            # Operator intent and interpreter shutdown must still propagate.
            raise
        except Exception as e:
            msg = (f"move_id {env_state_prompt.move_id}: "
                   f"{type(e).__name__}: {e}")
            self.failures.append(msg)
            print(f"    [RESILIENT] {msg}")
            print("                -> scoring this puzzle incorrect, continuing sweep")
            return ""

    def _begin_puzzle(self, env_state_prompt) -> None:
        """Drop stored deliberation from earlier puzzles.

        Purely a memory-footprint optimization: _isolate_crew_memory() has
        already narrowed the history window so that entries from other puzzles
        cannot appear in any prompt. Without pruning, the memory tree still grows
        for the length of the run and every view_* call scans all of it, which
        turns into a quadratic cost over hundreds of puzzles.

        Failures here are swallowed — a pruning problem must never be able to
        take down a run that would otherwise succeed.
        """
        if self._memory is None:
            return
        try:
            root = self._memory._get_tree().getroot()
            if root is None:
                return
            keep = env_state_prompt.move_id
            for section in list(root):
                for entry in list(section):
                    raw = entry.get("move_id")
                    if raw is None:
                        continue
                    try:
                        mid = int(raw)
                    except (TypeError, ValueError):
                        continue
                    # move_id -1 marks pre-game/setup messages, which the memory
                    # always considers in scope; leave those alone.
                    if mid >= 0 and mid != keep:
                        section.remove(entry)
        except Exception as e:
            print(f"    [RESILIENT] WARNING: could not prune crew memory: "
                  f"{type(e).__name__}: {e}")

    # -- pass-through --------------------------------------------------------

    def call_ollama(self, user_prompt, system_prompt, llm_config, format=None):
        return self._inner.call_ollama(user_prompt, system_prompt,
                                       llm_config, format)

    def get_move_record(self):
        return self._inner.get_move_record()

    def get_name(self) -> str:
        return self._inner.get_name()

    def close(self) -> None:
        self._inner.close()


def _isolate_crew_memory(system: Any) -> bool:
    """Restrict the multi-agent crew to the puzzle currently being solved.

    The MAS memory keys every message by move_id and shows each agent the last
    MOVE_HISTORY_LIMIT (4) move ids of conversation — sensible in a chess game,
    where recent moves are the same game. In a puzzle batch, move ids are
    successive independent positions, so the crew deliberating puzzle n would
    receive the full deliberation about puzzles n-1, n-2 and n-3: three unrelated
    tactical positions presented as recent context. That contaminates the
    multi-agent conditions, makes results depend on dataset ordering, and adds a
    confound against the single condition, which is stateless by construction.

    Narrowing the window to 1 leaves each agent seeing only the current puzzle.
    This is done from the outside, via the documented limit, rather than by
    editing multi_system/memory_module, which the in-flight chess campaign uses.

    Args:
        system: The system built by _build_system (single systems have no crew).

    Returns:
        True if a crew memory was found and narrowed.
    """
    memory = getattr(system, "memory", None)
    if memory is None:
        return False
    system.MOVE_HISTORY_LIMIT = PUZZLE_MOVE_HISTORY_LIMIT
    memory.MOVE_HISTORY_LIMIT = PUZZLE_MOVE_HISTORY_LIMIT
    return True


def describe_crew(system: Any) -> Optional[Dict[str, Any]]:
    """Capture which agents the crew actually ended up with.

    Worth recording because sharding changes what a "cell" is for the `multi`
    mode: the meta-agent designs a fresh crew per MultiLLMSystem, so each shard
    of a multi cell is deliberated by a differently-designed team. That is
    arguably a feature — it turns one arbitrary crew into a small distribution of
    them — but only if the variation is visible. (`hardcoded_multi` is unaffected:
    its crew comes from a fixed preset.)

    Args:
        system: The system built by _build_system.

    Returns:
        A serializable description of the crew, or None for a single system.
    """
    memory = getattr(system, "memory", None)
    registry = getattr(memory, "registry", None) if memory else None
    if registry is None:
        return None
    try:
        agents = registry.all_agents()
    except Exception:
        return None
    return {
        "num_agents": len(agents),
        "agents": [
            {
                "name": getattr(a, "name", ""),
                "agent_id": getattr(a, "id", ""),
                "responsibility": getattr(a, "responsibility", ""),
                "provides": getattr(a, "provides", ""),
                "talks_to": list(getattr(a, "talksto_strs", []) or []),
            }
            for a in agents
        ],
    }


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

def setup_puzzle_credentials() -> Dict[str, str]:
    """setup_credentials() without its Stockfish requirement.

    The chess harness refuses to start without a valid engine binary, which is
    correct there and meaningless here: the puzzle environment never constructs
    an engine. Rather than fork the .env loading and GCLOUD_* -> GOOGLE_CLOUD_*
    override logic (the whole point of importing it), the guard is satisfied with
    a path that certainly exists and the value is dropped from the result.

    Returns:
        The credential dict, without the unused stockfish_path key.
    """
    original = os.environ.get("STOCKFISH_PATH")
    installed_sentinel = False
    if not original or not os.path.exists(original):
        # load_dotenv() does not override variables already present, so this also
        # keeps a stale .env value from re-failing the guard.
        os.environ["STOCKFISH_PATH"] = sys.executable
        installed_sentinel = True
    try:
        creds = setup_credentials()
    finally:
        if installed_sentinel:
            if original is None:
                os.environ.pop("STOCKFISH_PATH", None)
            else:
                os.environ["STOCKFISH_PATH"] = original
    creds.pop("stockfish_path", None)
    return creds


# ---------------------------------------------------------------------------
# Puzzle experiment writer
# ---------------------------------------------------------------------------

class PuzzlesExperimentWriter:
    """Firestore + local-backup writer for a single puzzle experiment.

    Standalone (does not subclass ExperimentWriter, whose __init__ requires the
    chess ExpConfig). It owns the puzzle-shaped experiment/move/summary docs and
    delegates all Firestore-durability work (chunking, retry, checksums) to the
    free functions imported from run_gcc_experiments, so there is a single copy
    of that hardened code.
    """

    def __init__(self, db, cfg: PuzzleExpConfig, env_info: Dict[str, Any]):
        self.db = db
        self.cfg = cfg
        self.env_info = env_info

        self.is_multi = (cfg.system_type == "multi_agent")
        self.created_utc = _utc_iso()
        self.created_est = _est_stamp()

        self.move_docs: List[Dict[str, Any]] = []
        self.seq = 0

        self._live_prompt_tokens = 0
        self._live_completion_tokens = 0

        # Dataset facts, supplied by set_dataset_info() as soon as the
        # environment has loaded. Held here so that the true denominator is known
        # even for a run that is interrupted before produce a RunResult.
        self.total_puzzles: Optional[int] = None
        self.total_k: Optional[int] = None
        self.dataset_name: str = ""
        self.dataset_hash: str = ""
        # Composition of the crew that actually ran, recorded because the
        # meta-designed crew differs from shard to shard (see describe_crew).
        self.crew: Optional[Dict[str, Any]] = None

        self.firestore_write_failures: List[str] = []
        self.chunked_move_count = 0
        self.max_chunks_in_move = 0
        self.truncated_call_moves = 0
        self.chunk_verify_failures: List[str] = []

        self.exp_ref = self._reserve_experiment_doc()

        LOCAL_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        self.backup_json = LOCAL_BACKUP_DIR / f"{self.cfg.experiment_id}.json"
        self.backup_jsonl = LOCAL_BACKUP_DIR / \
            f"{self.cfg.experiment_id}.moves.jsonl"
        with open(self.backup_jsonl, "w", encoding="utf-8") as f:
            f.write(json.dumps({"_record": "experiment_header",
                                "experiment_doc": self._initial_doc()},
                               ensure_ascii=False, default=str) + "\n")

    def set_dataset_info(self, total_puzzles: int, total_k: int,
                         dataset_name: str, dataset_hash: str) -> None:
        """Record what the environment actually loaded.

        Called immediately after the environment is constructed so that the
        summary can always tell a complete sweep from a partial one. Deriving the
        denominator from the RunResult instead would leave it unknown exactly
        when it matters most — an interrupted run, whose RunResult never exists.

        Args:
            total_puzzles: Puzzles in this experiment's window (per sweep).
            total_k: Number of sweeps the environment will perform.
            dataset_name: "<repo>:<config>:<split>" provenance string.
            dataset_hash: Hash of the loaded window.
        """
        self.total_puzzles = total_puzzles
        self.total_k = total_k
        self.dataset_name = dataset_name
        self.dataset_hash = dataset_hash

    def set_crew(self, crew: Optional[Dict[str, Any]]) -> None:
        """Record the crew composition for this experiment (None for single)."""
        self.crew = crew

    # -- experiment doc creation -------------------------------------------

    def _initial_doc(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.cfg.experiment_id,
            "session_id": self.cfg.session_id,
            "schema_version": PUZZLE_SCHEMA_VERSION,
            "base_schema_version": BASE_SCHEMA_VERSION,
            "environment_kind": "chess_puzzle",
            "status": "running",
            "created_at_est": self.created_est,
            "created_at_utc": self.created_utc,
            "mode": self.cfg.mode,
            "model": self.cfg.model,
            # None for the baseline system; a registered id for a prototype run
            "prototype_id": self.cfg.prototype_id,
            "system_type": self.cfg.system_type,
            "dataset_repo": self.cfg.dataset_repo,
            "dataset_config": self.cfg.dataset_config,
            "dataset_split": self.cfg.dataset_split,
            "shard_index": self.cfg.shard_index,
            "num_shards": self.cfg.num_shards,
            "task_offset": self.cfg.task_offset,
            "task_limit": self.cfg.task_limit,
            # Explicit sampling provenance: what was requested, and what the
            # provider actually honours.
            "sampling": {
                "temperature": self.cfg.temperature,
                "decoding": "greedy" if self.cfg.temperature == 0 else "sampled",
                "thinking_level": self.cfg.thinking_level,
                "max_output_tokens": self.cfg.max_output_tokens,
                "seed": self.cfg.seed,
                "seed_applied_by_provider": SEED_APPLIED_BY_PROVIDER,
            },
            "config": self.cfg.to_dict(),
            "config_hash": self.cfg.config_hash(),
            "environment": self.env_info,
        }

    def _reserve_experiment_doc(self):
        for _ in range(5):
            ref = self.db.collection(PUZZLE_FIRESTORE_COLLECTION).document(
                self.cfg.experiment_id)
            ok, err = _fs_create_with_retry(ref, _fs_safe_keys(self._initial_doc()),
                                            what=f"experiment {self.cfg.experiment_id}")
            if ok:
                return ref
            if err and err.startswith("AlreadyExists"):
                old = self.cfg.experiment_id
                self.cfg.experiment_id = _make_experiment_id(
                    _experiment_tag(self.cfg.mode, self.cfg.model,
                                    self.cfg.shard_index, self.cfg.num_shards,
                                    self.cfg.repetition))
                print(f"    [GUARD] experiment id {old} already existed; "
                      f"regenerated as {self.cfg.experiment_id}")
                continue
            raise RuntimeError(f"Could not create experiment document: {err}")
        raise RuntimeError(
            "Could not reserve a unique experiment document id.")

    # -- per-move handling --------------------------------------------------

    def add_move(self, move_data: Dict[str, Any]) -> None:
        """move_callback target. Builds the puzzle move doc, mirrors it locally,
        then persists to Firestore. Local write happens first so data survives a
        Firestore failure."""
        self.seq += 1
        doc = self._build_move_doc(move_data, self.seq)

        if doc["is_system_move"]:
            self._live_prompt_tokens += doc["prompt_tokens"]
            self._live_completion_tokens += doc["completion_tokens"]

        self.move_docs.append(doc)

        try:
            with open(self.backup_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps({"_record": "move", "move": doc},
                                   ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            print(f"    [BACKUP] WARNING: failed to append move "
                  f"{self.seq} locally: {e}")

        self._persist_move_to_firestore(doc)
        self._print_move_line(doc)

    def _persist_move_to_firestore(self, doc: Dict[str, Any]) -> None:
        """Write a move to Firestore, chunking oversized `calls` via the imported
        chunking helpers so no document exceeds the 1 MiB limit. Identical
        durability strategy to the chess writer; only the doc payload differs."""
        seq = doc["seq"]
        move_ref = self.exp_ref.collection("moves").document(f"{seq:04d}")
        calls = doc.get("calls", []) or []
        full_checksum = _calls_checksum(calls)

        # inline path
        if _json_bytes(doc) <= MOVE_INLINE_MAX_BYTES:
            fs_doc = dict(doc)
            fs_doc["calls_storage"] = "inline"
            fs_doc["calls_checksum"] = full_checksum
            ok, err = _fs_create_with_retry(
                move_ref, _fs_safe_keys(fs_doc), what=f"move {seq:04d}")
            if not ok:
                self.firestore_write_failures.append(f"move {seq:04d}: {err}")
                print(f"    [FIRESTORE] FAILED to write move {seq:04d}: {err} "
                      f"(retained in local backup)")
            return

        # chunked path
        chunks = _pack_calls_into_chunks(
            calls, CALLS_CHUNK_BUDGET_BYTES, FS_DOC_SAFE_BYTES)
        calls_col = move_ref.collection("calls")
        manifest_chunks: List[Dict[str, Any]] = []
        stored_calls: List[Dict[str, Any]] = []
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
                calls_col.document(chunk_id), _fs_safe_keys(chunk_doc),
                what=f"move {seq:04d} calls chunk {chunk_id}")
            if not ok:
                failed_chunks.append(chunk_id)
                self.firestore_write_failures.append(
                    f"move {seq:04d} calls chunk {chunk_id}: {err}")
                print(f"    [FIRESTORE] FAILED to write move {seq:04d} calls chunk "
                      f"{chunk_id}: {err} (retained in local backup)")
            any_truncated = any_truncated or any(
                c.get("_truncated") for c in ch["calls"])
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
            "calls_checksum": full_checksum,
            "stored_calls_checksum": stored_checksum,
            "calls_truncated": any_truncated,
            "failed_chunks": failed_chunks,
            "chunks": manifest_chunks,
        }
        ok, err = _fs_create_with_retry(
            move_ref, _fs_safe_keys(fs_doc), what=f"move {seq:04d}")
        if not ok:
            self.firestore_write_failures.append(f"move {seq:04d}: {err}")
            print(f"    [FIRESTORE] FAILED to write move {seq:04d}: {err} "
                  f"(retained in local backup)")

        self.chunked_move_count += 1
        self.max_chunks_in_move = max(self.max_chunks_in_move, len(chunks))
        if any_truncated:
            self.truncated_call_moves += 1
        if VERIFY_CHUNKED_WRITES and ok and not failed_chunks:
            self._verify_chunked_move(move_ref, seq, stored_checksum)

    def _verify_chunked_move(self, move_ref, seq: int,
                             expected_checksum: str) -> None:
        """Read the calls chunks back, reassemble in order, confirm the checksum."""
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
            msg = (f"move {seq:04d}: chunk read-back verification error: "
                   f"{type(e).__name__}: {e}")
            self.chunk_verify_failures.append(msg)
            print(f"    [FIRESTORE] WARNING {msg}")

    def _build_move_doc(
            self, move_data: Dict[str, Any], seq: int) -> Dict[str, Any]:
        """Build one PUZZLE move doc. Each 'move' is one puzzle attempt: the
        per-puzzle correctness replaces the chess centipawn/quality fields, and
        the puzzle's own rating/themes ride along so difficulty and theme
        breakdowns can be computed from the collected data alone."""
        env_state = move_data.get("env_state", {}) or {}
        smr = move_data.get("system_move_record", {}) or {}
        player = move_data.get("player", "")
        is_system = player in ("single_llm", "multi_llm")
        calls = smr.get("calls", []) or []
        action = move_data.get("action", "") or ""

        thoughts_tokens = sum(int(c.get("thoughts_tokens", 0) or 0)
                              for c in calls)
        deliberation_cycles = (
            len({c.get("cycle", 0) for c in calls}) if (
                self.is_multi and calls) else 0
        )

        return {
            "seq": seq,
            "turn": move_data.get("turn"),
            "player": player,
            "is_system_move": is_system,
            "action": action,
            # puzzle attempt details (replaces chess
            # uci/san/fen_before/centipawn)
            "puzzle_id": env_state.get("current_puzzle_id", ""),
            "fen": env_state.get("fen", ""),
            "predicted_move": env_state.get("predicted_move", ""),
            "predicted_move_raw": env_state.get("predicted_move_raw", ""),
            "ground_truth": env_state.get("ground_truth", ""),
            "correct": bool(env_state.get("correct", False)),
            "current_k": env_state.get("current_k"),
            # An empty action on a system move means _ResilientSystem caught a
            # failure for this puzzle: no legal UCI move is ever empty.
            "system_failed": bool(is_system and not action.strip()),
            # puzzle metadata, for post-hoc difficulty / theme slicing
            "rating": env_state.get("rating"),
            "themes": env_state.get("themes", []) or [],
            "primary_theme": env_state.get("primary_theme", ""),
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
            "calls": calls,
        }

    def _print_move_line(self, doc: Dict[str, Any]) -> None:
        if not doc["is_system_move"]:
            return
        if doc["system_failed"]:
            mark = "ERR"
        elif doc["correct"]:
            mark = "OK "
        else:
            mark = "  x"
        rating = doc["rating"]
        extra = f" calls={doc['call_count']}"
        if self.is_multi:
            extra += f" cycles={doc['deliberation_cycles']}"
        print(f"    [{mark}] {str(doc['seq']):>4} {str(doc['puzzle_id']):<12} "
              f"pred={str(doc['predicted_move']):<6} "
              f"gt={str(doc['ground_truth']):<6} "
              f"r={str(rating if rating is not None else '-'):>5} "
              f"toks={doc['total_tokens']:>6}{extra}")

    # -- finalize -----------------------------------------------------------

    def finalize(self, run_result, error: Optional[str] = None,
                 error_traceback: Optional[str] = None,
                 system_failures: Optional[List[str]] = None) -> Dict[str, Any]:
        """Recompute aggregates from stored move docs, build the PUZZLE summary
        (accuracy / pass_at_k, no winner or centipawn quality), write it to
        Firestore, and write the local backup. Returns the summary dict."""
        ended_utc = _utc_iso()
        system_failures = list(system_failures or [])

        result_dict = (run_result.result if run_result else {}) or {}
        duration = getattr(
            run_result,
            "duration_seconds",
            0.0) if run_result else 0.0
        run_error = error if error is not None else (
            getattr(run_result, "error", None) if run_result else None)
        run_error_type = getattr(
            run_result,
            "error_type",
            None) if run_result else None

        system_moves = [m for m in self.move_docs if m["is_system_move"]]

        recomputed_prompt = sum(m["prompt_tokens"] for m in system_moves)
        recomputed_completion = sum(m["completion_tokens"]
                                    for m in system_moves)
        recomputed_thoughts = sum(m["thoughts_tokens"] for m in system_moves)
        recomputed_total = recomputed_prompt + recomputed_completion
        total_cycles = sum(m["deliberation_cycles"] for m in system_moves)

        # --- puzzle accuracy, recomputed from the authoritative move docs ---
        # Two denominators, named rather than conflated. `attempted` measures how
        # far the run got; `expected_attempts` is the benchmark's true size. A
        # partial run that happens to have solved everything it reached scores
        # 1.0 on the first and correctly far less on the second, so only the
        # second is ever a benchmark number — and only when is_complete_sweep.
        attempted = len(system_moves)
        num_correct_recompute = sum(1 for m in system_moves if m["correct"])

        total_puzzles = (self.total_puzzles if self.total_puzzles is not None
                         else int(result_dict.get("total_puzzles", 0) or 0))
        total_k = self.total_k if self.total_k is not None else self.cfg.k
        expected_attempts = total_puzzles * total_k
        is_complete_sweep = bool(
            expected_attempts) and attempted == expected_attempts

        accuracy_over_attempted = (
            num_correct_recompute / attempted) if attempted else 0.0
        accuracy_over_expected = (
            num_correct_recompute / expected_attempts) if expected_attempts else 0.0
        # The headline number, when the denominator is known.
        accuracy = (accuracy_over_expected if expected_attempts
                    else accuracy_over_attempted)

        # First-sweep-only recount, for comparison against the environment's own
        # num_correct (which is a pass@1 / first-attempt count). Comparing that
        # against a recount over ALL sweeps would flag every k>1 run as corrupt.
        first_sweep = [m for m in system_moves if (m.get("current_k") or 0) == 0]
        num_correct_first_sweep = sum(1 for m in first_sweep if m["correct"])

        system_time_ms = sum(m["latency_ms"] for m in system_moves)
        failed_puzzles = sum(1 for m in system_moves if m["system_failed"])

        # aggregate reported by the env (PuzzleResult)
        pass_at_k = _fs_safe_keys(result_dict.get("pass_at_k", {}) or {})
        env_num_correct = result_dict.get("num_correct", num_correct_first_sweep)
        dataset_hash = result_dict.get("dataset_hash", "") or self.dataset_hash
        dataset_name = result_dict.get("dataset_name", "") or self.dataset_name

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
        elif not is_complete_sweep:
            # No exception, but the sweep did not cover its window — the accuracy
            # is not a benchmark number and must not read like one.
            status = "incomplete"
        else:
            status = "completed"

        summary = {
            "status": status,
            "environment_kind": "chess_puzzle",
            # puzzle outcome (accuracy replaces winner/outcome)
            "accuracy": round(accuracy, 4),
            "accuracy_over_attempted": round(accuracy_over_attempted, 4),
            "accuracy_over_expected": round(accuracy_over_expected, 4),
            "is_complete_sweep": is_complete_sweep,
            "pass_at_k": pass_at_k,
            "num_correct": env_num_correct,
            "num_correct_recompute": num_correct_recompute,
            "num_correct_first_sweep": num_correct_first_sweep,
            "total_puzzles": total_puzzles,
            "total_k": total_k,
            "expected_attempts": expected_attempts,
            "attempted": attempted,
            "shard_index": self.cfg.shard_index,
            "num_shards": self.cfg.num_shards,
            "task_offset": self.cfg.task_offset,
            "task_limit": self.cfg.task_limit,
            "dataset_name": dataset_name,
            "dataset_hash": dataset_hash,
            # the crew that actually deliberated (None for single-LLM runs);
            # written here rather than at reservation time because it does not
            # exist until the meta-agent has designed it
            "crew": self.crew,
            "ended_at_utc": ended_utc,
            "ended_at_est": _est_stamp(),
            "duration_seconds": duration,
            # move counts (each move == one puzzle attempt)
            "total_moves": len(self.move_docs),
            "system_moves": len(system_moves),
            # tokens (recomputed from stored move docs)
            "total_prompt_tokens": recomputed_prompt,
            "total_completion_tokens": recomputed_completion,
            "total_thoughts_tokens": recomputed_thoughts,
            "total_tokens": recomputed_total,
            # timing
            "system_time_ms": system_time_ms,
            # multi-agent deliberation
            "total_deliberation_cycles": total_cycles,
            "avg_cycles_per_move": round(total_cycles / len(system_moves), 3) if system_moves else 0.0,
            # errors
            "error": run_error,
            "error_type": run_error_type,
            "error_traceback": error_traceback,
            "call_errors": call_errors,
            # puzzles the system could not answer at all (isolated, not fatal)
            "failed_puzzles": failed_puzzles,
            "system_failure_details": system_failures[:50],
            # data integrity
            "data_integrity": {
                "live_aggregate_matches_recompute": live_matches_recompute,
                # Only meaningful on a complete sweep; None means "not comparable"
                # rather than "mismatch".
                "env_num_correct_matches_recompute": (
                    (env_num_correct == num_correct_first_sweep)
                    if is_complete_sweep else None),
                "live_prompt_tokens": self._live_prompt_tokens,
                "live_completion_tokens": self._live_completion_tokens,
                "moves_written_locally": len(self.move_docs),
                "firestore_write_failures": len(self.firestore_write_failures),
                "firestore_write_failure_details": self.firestore_write_failures,
                "chunked_moves": self.chunked_move_count,
                "max_chunks_in_move": self.max_chunks_in_move,
                "truncated_call_moves": self.truncated_call_moves,
                "chunk_verify_failures": len(self.chunk_verify_failures),
                "chunk_verify_failure_details": self.chunk_verify_failures,
            },
        }
        summary["summary_checksum"] = self._summary_checksum(summary)

        ok, err = _fs_update_with_retry(self.exp_ref, _fs_safe_keys(summary),
                                        what=f"summary {self.cfg.experiment_id}")
        if not ok:
            self.firestore_write_failures.append(f"summary: {err}")
            summary["status"] = "completed_with_write_errors" if not run_error else "errored"
            summary["data_integrity"]["firestore_write_failures"] = len(
                self.firestore_write_failures)
            summary["data_integrity"]["firestore_write_failure_details"] = self.firestore_write_failures
            print(
                f"    [FIRESTORE] FAILED to write summary: {err} (retained in local backup)")

        self._write_local_backup(summary)
        return summary

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
                json.dump(
                    payload,
                    f,
                    ensure_ascii=False,
                    indent=2,
                    default=str)
            tmp.replace(self.backup_json)
        except Exception as e:
            print(
                f"    [BACKUP] WARNING: failed to write consolidated backup: {e}")


# ---------------------------------------------------------------------------
# Single puzzle experiment execution
# ---------------------------------------------------------------------------

def run_one_puzzle_experiment(db, cfg: PuzzleExpConfig,
                              env_info: Dict[str, Any], total: int) -> Dict[str, Any]:
    """Run one puzzle experiment: one shard of the puzzle set for (mode, model).

    Args:
        db: Firestore client.
        cfg: The puzzle experiment config.
        env_info: Environment provenance dict.
        total: Total number of experiments in the sweep (for progress printing).

    Returns:
        A dict with experiment_id, cfg, and the finalized summary.
    """
    from chess_puzzle_env import ChessPuzzleEnv
    from cloud_llm_provider import create_cloud_chat_fn

    window = (f"rows {cfg.task_offset}.."
              f"{cfg.task_offset + cfg.task_limit - 1}" if cfg.task_limit != (-1)
              else f"rows {cfg.task_offset}..end")

    print(f"\n{'=' * 70}")
    print(f"PUZZLE EXPERIMENT {cfg.sweep_index}/{total}   [{cfg.session_id}]")
    print(f"  id      : {cfg.experiment_id}")
    print(f"  cell    : mode={cfg.mode}  model={cfg.model}"
          + (f"  prototype={cfg.prototype_id}" if cfg.prototype_id else ""))
    print(f"  shard   : {cfg.shard_index + 1}/{cfg.num_shards}  ({window})")
    print(f"  dataset : {cfg.dataset_repo}:{cfg.dataset_config}:{cfg.dataset_split}"
          f"  (k={cfg.k})")
    print(f"  sampling: temperature={cfg.temperature} "
          f"({'greedy' if cfg.temperature == 0 else 'sampled'})  "
          f"thinking={cfg.thinking_level}")
    print(f"{'=' * 70}")

    writer = PuzzlesExperimentWriter(db, cfg, env_info)
    system: Optional[_ResilientSystem] = None

    try:
        env = ChessPuzzleEnv(
            dataset_repo=cfg.dataset_repo,
            config=cfg.dataset_config,
            split=cfg.dataset_split,
            task_limit=cfg.task_limit,
            task_offset=cfg.task_offset,
            puzzle_ids=cfg.puzzle_ids,
            k=cfg.k,
        )
        writer.set_dataset_info(total_puzzles=len(env.puzzle_ids),
                                total_k=env.k,
                                dataset_name=env.dataset_name,
                                dataset_hash=env.dataset_hash)
        print(f"  loaded  : {len(env.puzzle_ids)} puzzles "
              f"(hash {env.dataset_hash})")

        chat_fn = create_cloud_chat_fn(
            provider="vertex",
            thinking_level=cfg.thinking_level)
        inner = _build_system(cfg, env, chat_fn)
        crew = describe_crew(inner)
        writer.set_crew(crew)
        if _isolate_crew_memory(inner):
            names = ", ".join(a["name"] for a in (crew or {}).get("agents", []))
            print(f"  crew    : {(crew or {}).get('num_agents', 0)} agents "
                  f"[{names}]")
            print(f"            memory window narrowed to the current puzzle "
                  f"(history limit {PUZZLE_MOVE_HISTORY_LIMIT})")
        system = _ResilientSystem(inner, memory=getattr(inner, "memory", None))

        run_result = run_game(
            env=env,
            system=system,
            config=cfg.to_dict(),
            move_callback=writer.add_move,
            verbose=False,
            move_delay=MOVE_DELAY_SECONDS,
        )
        summary = writer.finalize(
            run_result=run_result,
            error=run_result.error,
            system_failures=system.failures)

    except KeyboardInterrupt:
        summary = writer.finalize(
            run_result=None,
            error="KeyboardInterrupt: user interrupted",
            system_failures=system.failures if system else None)
        _print_puzzle_summary(cfg, summary)
        raise
    except Exception as e:
        tb = traceback.format_exc()
        summary = writer.finalize(
            run_result=None,
            error=f"{type(e).__name__}: {e}",
            error_traceback=tb,
            system_failures=system.failures if system else None)

    _print_puzzle_summary(cfg, summary)
    return {"experiment_id": cfg.experiment_id, "cfg": cfg, "summary": summary}


def _print_puzzle_summary(cfg: PuzzleExpConfig,
                          summary: Dict[str, Any]) -> None:
    di = summary.get("data_integrity", {})
    print(f"\n  -- result: {summary['status']}   "
          f"accuracy={summary['accuracy']:.4f} "
          f"({summary['num_correct_recompute']} correct / "
          f"{summary['attempted']} attempted / "
          f"{summary['expected_attempts']} expected)")
    print(f"     pass@k: {summary.get('pass_at_k')}")
    print(f"     tokens: {summary['total_tokens']:,}   "
          f"cycles: {summary['total_deliberation_cycles']}   "
          f"duration: {summary['duration_seconds']}s")
    if not summary["is_complete_sweep"]:
        print(f"     INCOMPLETE SWEEP — 'accuracy' is not a benchmark number; "
              f"accuracy_over_attempted="
              f"{summary['accuracy_over_attempted']:.4f}")
    if summary.get("failed_puzzles"):
        print(f"     {summary['failed_puzzles']} puzzle(s) the system could not "
              f"answer (scored incorrect; see system_failure_details)")
    if summary.get("error"):
        print(f"     ERROR: {summary['error']}")
    if di.get("chunked_moves"):
        print(f"     chunked moves: {di['chunked_moves']} "
              f"(max {di['max_chunks_in_move']} chunks/move)")
    if di.get("truncated_call_moves"):
        print(f"     WARNING: {di['truncated_call_moves']} move(s) had an oversized "
              f"single call truncated for Firestore (full copy in local backup).")
    if di.get("firestore_write_failures"):
        print(f"     FIRESTORE WRITE FAILURES: {di['firestore_write_failures']} "
              f"(data preserved in local backup)")
    if di.get("chunk_verify_failures"):
        print(f"     CHUNK VERIFY FAILURES: {di['chunk_verify_failures']}")
    if not di.get("live_aggregate_matches_recompute", True):
        print("     WARNING: live token aggregate did not match recompute — "
              "see data_integrity.")


def _print_session_summary(session_id: str, results: List[Dict[str, Any]],
                           interrupted: bool, total: int) -> None:
    print("\n" + "=" * 70)
    print(f"SESSION COMPLETE: {session_id}")
    print("=" * 70)
    print(f"  Ran {len(results)}/{total} puzzle experiments"
          + ("  (INTERRUPTED)" if interrupted else ""))

    counts: Dict[str, int] = {}
    for r in results:
        st = r["summary"]["status"]
        counts[st] = counts.get(st, 0) + 1
    print("  " + ("   ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
                  if counts else "(no experiments completed)"))

    print("\n  per-shard:")
    for r in results:
        s, c = r["summary"], r["cfg"]
        flag = "" if s["is_complete_sweep"] else "  [INCOMPLETE]"
        print(f"    {c.mode:<16} {c.model:<20} shard {c.shard_index + 1}/{c.num_shards} "
              f"acc={s['accuracy']:.4f} "
              f"({s['num_correct_recompute']}/{s['expected_attempts']}){flag}")

    # Cells are only comparable when every shard of them completed, so the
    # aggregate is reported alongside its own completeness.
    print("\n  per-cell (shards combined):")
    cells: Dict[tuple, Dict[str, Any]] = {}
    for r in results:
        s, c = r["summary"], r["cfg"]
        cell = cells.setdefault((c.mode, c.model),
                                {"correct": 0, "expected": 0, "attempted": 0,
                                 "shards": 0, "complete": True,
                                 "failed": 0, "tokens": 0})
        cell["correct"] += s["num_correct_recompute"]
        cell["expected"] += s["expected_attempts"]
        cell["attempted"] += s["attempted"]
        cell["failed"] += s.get("failed_puzzles", 0)
        cell["tokens"] += s["total_tokens"]
        cell["shards"] += 1
        cell["complete"] = cell["complete"] and s["is_complete_sweep"]
    for (mode, model), cell in sorted(cells.items()):
        acc = cell["correct"] / cell["expected"] if cell["expected"] else 0.0
        flag = "" if cell["complete"] else "  [PARTIAL — not a benchmark number]"
        print(f"    {mode:<16} {model:<20} acc={acc:.4f} "
              f"({cell['correct']}/{cell['expected']})  "
              f"shards={cell['shards']}  unanswered={cell['failed']}  "
              f"tokens={cell['tokens']:,}{flag}")

    print(f"\n  Firestore: {PUZZLE_FIRESTORE_COLLECTION}/<experiment_id>/moves/<seq>"
          f"  (large moves: .../moves/<seq>/calls/<chunk>)")
    print(f"  Local backups: {LOCAL_BACKUP_DIR}")
    if any(not r["summary"]["is_complete_sweep"] for r in results):
        print("  NOTE: some shards are incomplete — re-run those shards before "
              "reporting cell-level accuracy.")
    print("=" * 70)


def main() -> None:
    creds = setup_puzzle_credentials()
    env_info = get_environment_info(creds)

    session_id = _sanitize_id(f"{_est_stamp()}_{uuid.uuid4().hex[:6]}")
    configs = build_puzzle_configs(session_id)
    shards = _plan_shards()
    covered = sum(lim for _, lim in shards) if shards[0][1] != (-1) else None

    print("=" * 70)
    print("CrewForge GCC Chess PUZZLE Experiments (Vertex + Firestore)")
    print("=" * 70)
    print(f"  GCloud project   : {creds['project']}")
    print(f"  GCloud location  : {creds['location']}")
    print(f"  Firestore coll.  : {PUZZLE_FIRESTORE_COLLECTION}"
          f"{' (db=' + gcc.FIRESTORE_DATABASE + ')' if gcc.FIRESTORE_DATABASE else ' (default db)'}")
    print(f"  Local backup dir : {LOCAL_BACKUP_DIR}")
    print(f"  Schema version   : {PUZZLE_SCHEMA_VERSION} "
          f"(base {BASE_SCHEMA_VERSION})")
    print(f"  Session id       : {session_id}")
    print(f"  System           : {'BASELINE' if PROTOTYPE_ID is None else 'PROTOTYPE ' + PROTOTYPE_ID}")
    print(f"  Models           : {', '.join(MODELS)}")
    print(f"  Modes            : {', '.join(MODES)}")
    print(f"  Dataset          : {PUZZLE_DATASET_REPO}:{PUZZLE_DATASET_CONFIG}"
          f":{PUZZLE_DATASET_SPLIT}  (k={PUZZLE_K})")
    print(f"  Shards           : {len(shards)} x {PUZZLE_SHARD_SIZE or 'all'} puzzles"
          + (f"  ({covered} puzzles per cell)" if covered else ""))
    print(f"  Reps per cell    : {EXPERIMENTS_PER_REP}")
    print(f"  Total experiments: {len(configs)}")
    print(f"  Sampling         : temperature={TEMPERATURE} "
          f"({'greedy' if TEMPERATURE == 0 else 'sampled'})  "
          f"thinking={THINKING_LEVEL}  max_output={MAX_OUTPUT_TOKENS}")
    print("=" * 70)

    db = make_firestore_client(creds["project"])

    results: List[Dict[str, Any]] = []
    interrupted = False
    for cfg in configs:
        try:
            results.append(
                run_one_puzzle_experiment(
                    db, cfg, env_info, len(configs)))
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Stopping sweep after current experiment.")
            interrupted = True
            break

    _print_session_summary(session_id, results, interrupted, len(configs))


if __name__ == "__main__":
    main()