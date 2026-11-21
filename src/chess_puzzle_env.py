"""Chess puzzle environment.

A self-iterating implementation of the Environment ABC for chess puzzle solving,
modeled on EvalPlusEnv. The environment owns its dataset end to end: it loads the
C1 benchmark live from Hugging Face in __init__ (as EvalPlusEnv calls
get_mbpp_plus), holds the batch of puzzles, and walks through them one per turn
within a single run_game() call. There is no opponent and no chess engine is
instantiated or used.

Scoring mirrors the C1 paper's protocol: each puzzle is a single tactical
position with one ground-truth move, and correctness is exact string match of
the submitted UCI move against that ground truth (pass@1). The k parameter
re-sweeps the whole set k times; it defaults to 1 (each puzzle once) and is a
repetition count, not the 1/10/100 pass@k estimator used by EvalPlus.

The C1 "test" config is already pre-moved: extra_info.fen is the position the
solver is asked about and reward_model.ground_truth is the answer, so no setup
move is applied here. That assumption is NOT taken on trust — __init__ verifies
that every ground-truth move is legal in its own FEN (see _validate_puzzles).
A config that has not been pre-moved fails loudly at load rather than producing
a clean, error-free 0% after many hours of compute.

rating, themes and primary_theme are carried verbatim (raw, unbucketed) all the
way into the per-move state, so downstream analysis can reproduce the C1
per-theme (20 themes x 25) and per-difficulty (4 tiers x 100) breakdowns from
the collected data alone, without re-downloading the dataset.

The batch can be restricted to a contiguous window with task_offset/task_limit,
which is how the runner shards a 900-puzzle sweep into independently restartable
experiments.
"""

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from datasets import load_dataset

from environment import Environment, EnvState, EnvStatePrompt, EnvIntroPrompt, EnvResult
from chess_env import fen_to_board_description
import chess


def convert_numpy_types(obj: Any) -> Any:
    """
    TAKEN DIRECTLY FROM https://github.com/CSSLab/C1, utils.py

    Recursively coerce numpy scalar/array types to native Python types.

    Copied from the C1 paper's utils.py. Hugging Face / parquet rows return numpy
    types (e.g. np.int64 ratings, numpy arrays for the themes list) that do not
    serialize cleanly to JSON or Firestore; this coerces them to native types.

    Args:
        obj: Any value, possibly containing numpy types nested in dicts/lists.

    Returns:
        The same structure with numpy scalars/arrays converted to Python types.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


def _normalize_themes(raw: Any) -> List[str]:
    """Coerce the dataset's themes field into a plain list of strings.

    The upstream representation is not guaranteed: Lichess ships themes as a
    single space-separated string, while the parquet/HF round-trip can surface
    them as a numpy array or a Python list. All three are normalized here so that
    downstream code (and Firestore, which rejects numpy types) sees one shape.

    Args:
        raw: The themes value as read from extra_info, in any of those forms.

    Returns:
        A list of non-empty theme strings; empty if the field was absent.
    """
    raw = convert_numpy_types(raw)
    if raw is None:
        return []
    if isinstance(raw, str):
        return [t for t in raw.split() if t]
    if isinstance(raw, (list, tuple)):
        return [str(t).strip() for t in raw if str(t).strip()]
    return [str(raw)]


def _normalize_rating(raw: Any) -> Optional[int]:
    """Coerce the dataset's rating field to an int, or None if unusable.

    Args:
        raw: The rating value as read from extra_info (int, np.int64, str, None).

    Returns:
        The rating as a Python int, or None when absent or non-numeric. Rating is
        analysis metadata rather than a load-bearing field, so an unparseable
        value degrades to None instead of failing the load.
    """
    raw = convert_numpy_types(raw)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _row_to_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a puzzle record from one UofTCSSLab/C1-data "test" row.

    Reaches into the nested reward_model / extra_info structs the way the paper's
    full_eval.py does, coercing numpy types on the way out. The position is taken
    as-is (already pre-moved in this config; verified by _validate_puzzles).

    Args:
        row: One row of the loaded dataset (a dict with reward_model / extra_info).

    Returns:
        A record dict with keys: puzzle_id, fen, ground_truth, rating, themes,
        primary_theme.

    Raises:
        ValueError: If the row is missing reward_model/extra_info, or if any
            load-bearing field (puzzle_id, fen, ground_truth) is absent or empty.
            These identify a puzzle and make it solvable, so a blank is fatal and
            is never silently substituted. rating, themes and primary_theme are
            analysis metadata and may be absent (None / []).
    """
    if not isinstance(row.get("reward_model"), dict) or not isinstance(
            row.get("extra_info"), dict):
        raise ValueError(
            f"row missing reward_model/extra_info structs; got keys {sorted(row.keys())}")

    reward_model = row["reward_model"]
    extra_info = row["extra_info"]

    puzzle_id = str(
        convert_numpy_types(
            extra_info.get("puzzle_id")) or "").strip()
    fen = str(extra_info.get("fen") or "").strip()
    ground_truth = str(reward_model.get("ground_truth") or "").strip().lower()

    for field, value in (("puzzle_id", puzzle_id),
                         ("fen", fen), ("ground_truth", ground_truth)):
        if not value:
            raise ValueError(
                f"puzzle row is missing required field '{field}' "
                f"(puzzle_id={puzzle_id!r}, extra_info keys {sorted(extra_info.keys())})")

    return {
        "puzzle_id": puzzle_id,
        "fen": fen,
        "ground_truth": ground_truth,
        "rating": _normalize_rating(extra_info.get("rating")),
        "themes": _normalize_themes(extra_info.get("themes")),
        # present in the C1 "test" split only; "" elsewhere
        "primary_theme": str(convert_numpy_types(
            extra_info.get("primary_theme")) or "").strip(),
    }


@dataclass
class PuzzleState(EnvState):
    """State for the current/just-completed puzzle in the batch sweep."""
    num_puzzles_completed: int   # puzzles finished so far (in the current sweep)
    total_puzzles: int           # total puzzles per sweep
    current_puzzle_id: str       # id of the current puzzle
    current_k: int               # current sweep index (0-based)
    total_k: int                 # total sweeps
    fen: str                     # the (pre-moved) puzzle position
    # submitted move, NORMALIZED exactly as it was scored (stripped + lowercased);
    # "" for pre-submission states and for puzzles the system failed to answer
    predicted_move: str
    # the submitted move exactly as the system emitted it, before normalization.
    # Kept so `predicted_move == ground_truth` always reproduces `correct`, while
    # the raw string remains available for audit.
    predicted_move_raw: str
    ground_truth: str            # the correct UCI move
    correct: bool                # whether the submitted move matched ground truth
    # --- puzzle metadata, carried verbatim for downstream slicing ---
    rating: Optional[int]        # Lichess puzzle rating (the difficulty axis)
    themes: List[str]            # tactical theme tags
    primary_theme: str           # the single headline theme ("" if not provided)


@dataclass
class PuzzleResult(EnvResult):
    """Aggregate result over the whole puzzle batch, for metrics and analysis."""
    # Keys are STRINGS ("pass@1", "pass@<k>"), not ints. Firestore maps are
    # map<string, Value> and its encoder raises TypeError on a non-string key, so
    # an int-keyed dict here silently fails every summary write downstream. This
    # also matches EvalPlusEnv's existing "pass@1" naming.
    pass_at_k: Dict[str, float]
    dataset_name: str                        # "<repo>:<config>:<split>" provenance
    dataset_hash: str                        # hash of the puzzle set, for reproducibility
    total_puzzles: int                       # puzzles per sweep
    total_k: int                             # number of sweeps
    # first-attempt correct count (pass@1)
    num_correct: int
    per_puzzle_results: List[Dict[str, Any]]  # detailed per-puzzle records


class ChessPuzzleEnv(Environment):
    """Implementation of Environment for solving a batch of chess puzzles.

    Loads the C1 puzzle benchmark live from Hugging Face and iterates it one
    puzzle per turn. It is active until every puzzle has been attempted (k times,
    for k sweeps), after which it becomes inactive. It has no opponent, so
    opponent_move() is a no-op. See the Environment class for documentation on
    the methods within.
    """

    def __init__(
        self,
        dataset_repo: str = "UofTCSSLab/C1-data",
        config: str = "test",
        split: str = "test",
        task_limit: int = (-1),
        task_offset: int = 0,
        puzzle_ids: Optional[Sequence[str]] = None,
        k: int = 1,
        validate: bool = True,
    ) -> None:
        """Initialize the puzzle environment by loading the dataset live.

        Args:
            dataset_repo: Hugging Face dataset repo id to load.
            config: Dataset config/name (the C1 benchmark is the "test" config).
            split: Dataset split to read.
            task_limit: Optionally cap the number of puzzles used, counted from
                task_offset. -1 means "to the end of the split".
            task_offset: Index of the first dataset row to include (0-based). With
                task_limit this selects a contiguous window, which is how a large
                benchmark is sharded into independently restartable experiments.
            k: Number of times to sweep the whole set (repetition count, not the
                pass@k estimator). Defaults to 1 (each puzzle attempted once).
            validate: Run the load-time integrity checks (see _validate_puzzles).
                Only disable for tests with deliberately synthetic positions.

        Raises:
            ValueError: If k < 1, if task_offset is negative or lands past the end
                of the split, or if the loaded window fails validation.
        """
        super().__init__()
        self.has_opponent = False
        self.opponent_name = ""

        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if task_offset < 0:
            raise ValueError(f"task_offset must be >= 0, got {task_offset}")
        if task_limit == 0:
            raise ValueError(
                "task_limit must be -1 (no cap) or a positive count, got 0")
        if puzzle_ids is not None:
            if task_offset or task_limit != (-1):
                raise ValueError(
                    "puzzle_ids selects rows explicitly and cannot be combined "
                    "with task_offset/task_limit")
            if not puzzle_ids:
                raise ValueError("puzzle_ids must be non-empty when provided")
            duplicate_requests = [pid for pid in set(puzzle_ids)
                                  if list(puzzle_ids).count(pid) > 1]
            if duplicate_requests:
                raise ValueError(
                    f"puzzle_ids contains repeated id(s): "
                    f"{sorted(duplicate_requests)}")
        self.k = k
        self.current_k = 0

        # [k_idx][puzzle_id] -> submitted move
        self.samples: Dict[int, Dict[str, str]] = {
            i: {} for i in range(self.k)}

        self.dataset_name = f"{dataset_repo}:{config}:{split}"

        # Live retrieval, mirroring EvalPlusEnv's
        # get_mbpp_plus/get_human_eval_plus.
        dataset = load_dataset(dataset_repo, config, split=split)

        stop = None if task_limit == (-1) else task_offset + task_limit
        wanted = set(puzzle_ids) if puzzle_ids is not None else None

        self.puzzles: Dict[str, Dict[str, Any]] = {}
        self.puzzle_ids: List[str] = []
        rows_seen = 0
        duplicates: List[str] = []
        for idx, row in enumerate(dataset):
            rows_seen = idx + 1
            if wanted is None:
                if idx < task_offset:
                    continue
                if stop is not None and idx >= stop:
                    break

            record = _row_to_record(dict(row))
            puzzle_id = record["puzzle_id"]

            if wanted is not None and puzzle_id not in wanted:
                continue

            if puzzle_id in self.puzzles:
                duplicates.append(f"row {idx}: {puzzle_id}")
                continue

            self.puzzles[puzzle_id] = {
                "fen": record["fen"],
                "ground_truth": record["ground_truth"],
                "rating": record["rating"],
                "themes": record["themes"],
                "primary_theme": record["primary_theme"],
            }
            self.puzzle_ids.append(puzzle_id)

        if wanted is not None:
            missing = [pid for pid in puzzle_ids if pid not in self.puzzles]
            if missing:
                raise ValueError(
                    f"{len(missing)} requested puzzle_id(s) not found in "
                    f"{self.dataset_name}: {missing[:5]}"
                    f"{' ...' if len(missing) > 5 else ''}")
            # Preserve the caller's ordering rather than dataset order.
            self.puzzle_ids = list(puzzle_ids)
        # An offset past the end of the split yields an empty environment, which
        # would otherwise "complete" instantly with a meaningless 0% — a badly
        # configured shard must fail loudly instead.
        if task_offset and not self.puzzle_ids:
            raise ValueError(
                f"task_offset={task_offset} selected no rows from "
                f"{self.dataset_name} (split has {rows_seen} rows) — check the "
                f"shard configuration")

        self.puzzle_idx = 0
        self.task_limit = task_limit
        self.task_offset = task_offset
        self.dataset_rows_available = rows_seen
        self.dataset_hash = self._compute_dataset_hash()

        if validate:
            self._validate_puzzles(duplicates)
        elif duplicates:
            print(f"    [PUZZLE ENV] WARNING: {len(duplicates)} duplicate "
                  f"puzzle_id(s) dropped (validation disabled)")

    def _validate_puzzles(self, duplicates: List[str]) -> None:
        """Verify the loaded window is internally consistent, or fail loudly.

        Three things are checked, and every violation across the whole batch is
        collected before raising so the error message is diagnostic rather than
        just reporting whichever row happened to be first:

          * every FEN parses as a chess position;
          * every ground-truth move is LEGAL in its own FEN. This is the check
            that verifies the "already pre-moved" assumption this environment is
            built on. If the configured split still needs C1's make_pre_move()
            applied, the ground truth is generally not legal in the stored
            position — and because the systems are constrained to legal moves,
            the model could never emit it. Without this check that manifests as a
            clean, error-free 0% after many hours of compute;
          * no duplicate puzzle ids (collected by the caller during the load).

        Args:
            duplicates: Duplicate-id descriptions collected during the load.

        Raises:
            ValueError: If any check fails, with counts, the first few offending
                puzzles, and a hint at the likely cause.
        """
        bad_fen: List[str] = []
        illegal_gt: List[str] = []

        for pid in self.puzzle_ids:
            entry = self.puzzles[pid]
            try:
                board = chess.Board(entry["fen"])
            except ValueError as e:
                bad_fen.append(f"{pid}: {entry['fen']!r} ({e})")
                continue
            if entry["ground_truth"] not in {m.uci() for m in board.legal_moves}:
                illegal_gt.append(
                    f"{pid}: gt={entry['ground_truth']!r} fen={entry['fen']!r}")

        problems: List[str] = []
        if bad_fen:
            problems.append(
                f"{len(bad_fen)} unparseable FEN(s); first: "
                + "; ".join(bad_fen[:3]))
        if illegal_gt:
            total = len(self.puzzle_ids)
            problems.append(
                f"{len(illegal_gt)}/{total} ground-truth move(s) are NOT legal in "
                f"their own FEN; first: " + "; ".join(illegal_gt[:3]))
        if duplicates:
            problems.append(
                f"{len(duplicates)} duplicate puzzle_id(s); first: "
                + "; ".join(duplicates[:3]))

        if not problems:
            return

        hint = ""
        # A broad failure points at the config, not at individual bad rows.
        if illegal_gt and len(illegal_gt) > 0.2 * max(len(self.puzzle_ids), 1):
            hint = (
                "\n  HINT: most ground truths being illegal means the positions are "
                "probably NOT pre-moved — this environment expects a split where "
                "extra_info.fen is already the position the solver is asked about "
                "(C1's make_pre_move() applied). Check dataset_repo/config/split.")

        raise ValueError(
            f"puzzle dataset {self.dataset_name} failed validation "
            f"(window offset={self.task_offset}, limit={self.task_limit}):\n  "
            + "\n  ".join(problems) + hint)

    def _compute_dataset_hash(self) -> str:
        """Return a stable hash of the puzzle set (ids + answers) for provenance.

        Covers only the window this environment loaded, so two shards of the same
        split hash differently — which is what makes the hash useful for
        confirming that a set of shards were assembled from the same source and
        cover disjoint content. The window bounds themselves are recorded
        separately as task_offset/task_limit.

        Returns:
            A short hex digest over the ordered (puzzle_id, ground_truth) pairs.
        """
        payload = ";".join(
            f"{pid}:{self.puzzles[pid]['ground_truth']}" for pid in self.puzzle_ids)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _current_state(self, predicted_move: str = "",
                       correct: bool = False) -> EnvState:
        """Build a PuzzleState for the puzzle at the current cursor.

        Args:
            predicted_move: The submitted move, if any (empty for pre-submission).
            correct: Whether the submitted move matched ground truth.

        Returns:
            A PuzzleState reflecting the current cursor position.
        """
        # the len(self.puzzle_ids) -1 prevents a OOB array error if opponent_move()
        # is called after all puzzles have been completed.
        idx = min(self.puzzle_idx, len(self.puzzle_ids) - 1)
        pid = self.puzzle_ids[idx]
        entry = self.puzzles[pid]
        return PuzzleState(
            num_puzzles_completed=self.puzzle_idx,
            total_puzzles=len(self.puzzle_ids),
            current_puzzle_id=pid,
            current_k=self.current_k,
            total_k=self.k,
            fen=entry["fen"],
            predicted_move=predicted_move,
            predicted_move_raw=predicted_move,
            ground_truth=entry["ground_truth"],
            correct=correct,
            rating=entry["rating"],
            themes=entry["themes"],
            primary_theme=entry["primary_theme"],
        )

    def is_active(self) -> bool:
        return self.puzzle_idx < len(
            self.puzzle_ids) and self.current_k != self.k

    def give_move(self, action: str) -> EnvState:
        pid = self.puzzle_ids[self.puzzle_idx]
        entry = self.puzzles[pid]

        predicted = action.strip().lower()
        self.samples[self.current_k][pid] = predicted
        correct = predicted == entry["ground_truth"]

        # Build state BEFORE advancing so it reflects the puzzle just
        # completed. predicted_move carries the NORMALIZED move — the exact
        # string `correct` was computed from — so that anyone recomputing
        # accuracy from the stored records gets the same answer. The raw string
        # is preserved alongside it.
        state = PuzzleState(
            num_puzzles_completed=self.puzzle_idx + 1,
            total_puzzles=len(self.puzzle_ids),
            current_puzzle_id=pid,
            current_k=self.current_k,
            total_k=self.k,
            fen=entry["fen"],
            predicted_move=predicted,
            predicted_move_raw=action,
            ground_truth=entry["ground_truth"],
            correct=correct,
            rating=entry["rating"],
            themes=entry["themes"],
            primary_theme=entry["primary_theme"],
        )

        self.puzzle_idx += 1  # advance to the next puzzle

        if self.puzzle_idx >= len(self.puzzle_ids):  # exhausted this sweep
            if self.current_k < self.k:              # more sweeps remain
                self.current_k += 1
                self.puzzle_idx = 0

        return state

    def opponent_move(self) -> EnvState:
        # No opponent: return the current state without advancing the cursor.
        return self._current_state()

    def get_env_state_prompt(self) -> EnvStatePrompt:
        pid = self.puzzle_ids[self.puzzle_idx]
        fen = self.puzzles[pid]["fen"]
        board = chess.Board(fen)

        position_description = fen_to_board_description(fen)

        legal_moves_list = [m.uci() for m in board.legal_moves]
        legal_moves_str = "You MUST choose from these valid moves: " + \
            ", ".join(legal_moves_list) + "\n\n"

        combined_str = (
            f"Board Position (FEN): {fen}\n\n"
            "FEN explanation: This single string represents the entire board state.\n"
            "- Lowercase letters = Black pieces (r=rook, n=knight, b=bishop, q=queen, k=king, p=pawn)\n"
            "- Uppercase letters = White pieces (R=Rook, N=Knight, B=Bishop, Q=Queen, K=King, P=Pawn)\n"
            "- Numbers = consecutive empty squares\n"
            "- The 'w' or 'b' after the board indicates whose turn it is\n\n"
            f"{position_description}\n\n"
            "This is a chess puzzle: there is a single best move for the side to play.\n"
            "Find that move.\n\n"
            f"{legal_moves_str}"
        )

        return EnvStatePrompt(prompt=combined_str, move_id=self._global_move_id(),
                              metadata=None, legal_actions=legal_moves_list)

    def _global_move_id(self) -> int:
        """A move id that increases monotonically across the whole run.

        Consumers of EnvStatePrompt.move_id (notably the multi-agent memory, which
        keys every stored message by it and filters conversation history on it)
        assume move ids never repeat. self.puzzle_idx alone resets to 0 at the
        start of every k-sweep, which would make sweep 2's puzzle 0 collide with
        sweep 1's and pull stale deliberation into the prompt. Offsetting by the
        sweep index keeps ids unique for the lifetime of the environment.

        Returns:
            current_k * puzzles_per_sweep + puzzle_idx.
        """
        return self.current_k * len(self.puzzle_ids) + self.puzzle_idx

    def get_env_intro_prompt(self) -> EnvIntroPrompt:
        # NOTE the deliberate leading space: multi_system's crew-creation system
        # prompt interpolates this as "...designated to{domain_prompt}" with no
        # separator, which would otherwise render as "designated tosolve". That
        # shared prompt builder is used by the in-flight chess campaign and is not
        # edited here; the space is contributed from this side instead.
        intro = (
            " solve a chess puzzle by finding the single best move for the side to "
            "play. Each puzzle is a tactical position with one objectively best move.\n\n"
        )
        rules = (
            "CRITICAL - UCI NOTATION:\n"
            "ALL chess moves MUST be written in UCI notation (not algebraic notation).\n"
            "UCI format: starting_square + ending_square (e.g., e2e4, g1f3, e1g1)\n"
            "WRONG: 1.e4, Nf3, O-O, Qxd5\n"
            "CORRECT: e2e4, g1f3, e1g1, d1d5\n"
            "This applies to ALL move discussions \n"
        )

        consensus_tool = {
            'type': 'function',
            'function': {
                'name': 'consensus',
                'description': 'Declare that the team has reached consensus on the puzzle move',
                'parameters': {
                    'type': 'object',
                    'required': ['move', 'reasoning'],
                    'properties': {
                        'move': {'type': 'string', 'description': 'The move that the team has agreed upon. Should be in UCI notation.'},
                        'reasoning': {'type': 'string', 'description': 'Brief explanation of why this move was chosen'},
                    },
                },
            },
        }

        consensus_move_field = "move"

        reflection = (f"- Is there anything important about the position that hasn't been discussed yet?\n"
                      f"- Do you have concerns or ideas that weren't covered in your recent conversations?\n"
                      f"- Based on your role, is there analysis you should share proactively?\n\n"
                      f"If you have something new to contribute, use your tools. If not, that's fine too.\n\n")


        reply = (
            f"Remember, if talking about or proposing moves, use UCI notation")

        return EnvIntroPrompt(domain_prompt=intro, notation_rules=rules,
                              consensus_schema=consensus_tool,
                              consensus_move_field=consensus_move_field,
                              reflection=reflection, reply=reply)

    def get_env_name(self) -> str:
        return "ChessPuzzle"

    def get_result(self) -> EnvResult:
        total = len(self.puzzle_ids)
        if total == 0:
            return PuzzleResult(
                pass_at_k={"pass@1": 0.0}, dataset_name=self.dataset_name,
                dataset_hash=self.dataset_hash, total_puzzles=0, total_k=self.k,
                num_correct=0, per_puzzle_results=[])

        solved_first = 0
        solved_any = 0
        per_puzzle_results: List[Dict[str, Any]] = []

        for pid in self.puzzle_ids:
            entry = self.puzzles[pid]
            ground_truth = entry["ground_truth"]
            attempts = [self.samples[kx][pid]
                        for kx in range(self.k) if pid in self.samples[kx]]

            first_correct = bool(attempts) and attempts[0] == ground_truth
            any_correct = any(a == ground_truth for a in attempts)

            solved_first += int(first_correct)
            solved_any += int(any_correct)

            per_puzzle_results.append({
                "puzzle_id": pid,
                "fen": entry["fen"],
                "ground_truth": ground_truth,
                "attempts": attempts,
                "first_correct": first_correct,
                "any_correct": any_correct,
                "rating": entry.get("rating"),
                "themes": entry.get("themes"),
                "primary_theme": entry.get("primary_theme", ""),
            })

        # String keys — see the note on PuzzleResult.pass_at_k.
        pass_at_k: Dict[str, float] = {"pass@1": solved_first / total}
        if self.k > 1:
            pass_at_k[f"pass@{self.k}"] = solved_any / total

        return PuzzleResult(
            pass_at_k=pass_at_k, dataset_name=self.dataset_name,
            dataset_hash=self.dataset_hash, total_puzzles=total, total_k=self.k,
            num_correct=solved_first, per_puzzle_results=per_puzzle_results)

    def close(self) -> None:
        pass