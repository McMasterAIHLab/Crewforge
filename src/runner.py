"""
Unified Runner — Environment-Agnostic Game Loop

Replaces run_single_llm_game() and run_multi_agent_game() with a single
run_game() that interacts exclusively with the Environment and System ABCs.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from environment import Environment
from system import System
from recording import RunnerMoveRecord, RunResult

# Maximum seconds for a single game. 
GAME_TIMEOUT_SECONDS = 86400


class GameTimeoutError(Exception):
    """Raised when a game exceeds the time for a single game."""
    pass


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _generate_game_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _classify_error(exc: Exception) -> str:
    """Return a short, machine-readable error_type tag for dashboard filtering.

    Categories:
        timeout          – game or call exceeded time limit
        consensus_failure – MAS could not reach consensus
        invalid_move     – system produced illegal / unparseable moves
        agent_creation   – MAS failed to create crew
        context_overflow – prompt exceeded context window
        api_error        – Ollama / network level failure
        unknown          – anything else
    """
    name = type(exc).__name__

    if name == "GameTimeoutError" or "timeout" in str(exc).lower():
        return "timeout"
    if name == "ConsensusFailureError":
        return "consensus_failure"
    if name == "RuntimeError" and "not in legal" in str(exc).lower():
        return "invalid_move"
    if name == "RuntimeError":
        return "invalid_move"
    if name == "AgentCreationError":
        return "agent_creation"
    if name == "ContextWindowExceededError":
        return "context_overflow"
    if "connection" in name.lower() or "ollama" in str(exc).lower():
        return "api_error"
    return "unknown"


def run_game(
    env: Environment,
    system: System,
    config: Dict[str, Any],
    move_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    verbose: bool = True,
    move_delay: float = 0.5,
    game_timeout: float = GAME_TIMEOUT_SECONDS,
) -> RunResult:
    """Run a complete game using the given environment and system.

    This is the unified, environment-agnostic game loop. It drives the
    interaction between a System (single-LLM or multi-agent) and an
    Environment (chess, EvalPlus, or any future domain) using their ABCs.

    Args:
        env: An initialized Environment instance.
        system: An initialized System instance (already has its intro prompt).
        config: Configuration dict stored in the RunResult for provenance.
        move_callback: Optional callback invoked after every move with the
            RunnerMoveRecord.to_dict().  Used for live data saving.
        verbose: Print progress to stdout.
        move_delay: Seconds to wait between turns (0 to disable).
        game_timeout: Maximum wall-clock seconds for the entire game.

    Returns:
        A RunResult with all move records and the final environment result.
    """
    game_id = _generate_game_id()
    start_time = _utc_iso()
    start_ts = time.time()

    moves: List[RunnerMoveRecord] = []
    error_msg: Optional[str] = None
    error_type: Optional[str] = None
    turn = 0

    try:
        if verbose:
            print(f"[{game_id}] Starting game: {system.get_name()} in {env.get_env_name()}")
            print("-" * 60)

        while env.is_active():
            # --- game-level timeout check ---
            elapsed = time.time() - start_ts
            if elapsed > game_timeout:
                raise GameTimeoutError(
                    f"Game exceeded {game_timeout}s wall-clock limit "
                    f"(elapsed {elapsed:.0f}s, turn {turn})"
                )

            # --- system's turn ---
            env_state_prompt = env.get_env_state_prompt()

            t0 = time.time()
            action = system.next_move(env_state_prompt)
            system_latency_ms = int((time.time() - t0) * 1000)

            system_move_record = system.get_move_record()
            env_state = env.give_move(action)

            ts_now = _utc_iso()
            record = RunnerMoveRecord(
                turn=turn,
                player=system.get_name(),
                action=action,
                env_state=env_state,
                system_move_record=system_move_record,
                timestamp=ts_now,
                latency_ms=system_latency_ms,
            )
            moves.append(record)

            if verbose:
                print(f"Turn {turn}: {system.get_name()} -> {action}  ({system_latency_ms}ms)")

            if move_callback is not None:
                try:
                    move_callback(record.to_dict())
                except Exception as cb_err:
                    print(f"[WARNING] move_callback failed: {cb_err}")

            # --- opponent's turn (if applicable) ---
            if env.has_opponent and env.is_active():
                t0 = time.time()
                opponent_state = env.opponent_move()
                opponent_latency_ms = int((time.time() - t0) * 1000)

                opp_record = RunnerMoveRecord(
                    turn=turn,
                    player=env.opponent_name,
                    action="",
                    env_state=opponent_state,
                    system_move_record=None,
                    timestamp=_utc_iso(),
                    latency_ms=opponent_latency_ms,
                )
                moves.append(opp_record)

                if verbose:
                    print(f"Turn {turn}: {env.opponent_name} responded  ({opponent_latency_ms}ms)")

                if move_callback is not None:
                    try:
                        move_callback(opp_record.to_dict())
                    except Exception as cb_err:
                        print(f"[WARNING] move_callback failed: {cb_err}")

            turn += 1

            if move_delay > 0:
                time.sleep(move_delay)

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Game finished after {turn} turns.")
            print(f"{'=' * 60}")

    except Exception as e:
        error_type = _classify_error(e)
        error_msg = f"{type(e).__name__}: {e}"
        if verbose:
            import traceback
            traceback.print_exc()
            print(f"ERROR [{error_type}]: {error_msg}")

    finally:
        try:
            system.close()
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass

    end_time = _utc_iso()
    duration = round(time.time() - start_ts, 2)

    # Retrieve final result from environment (safe even after error)
    try:
        env_result = env.get_result().to_dict()
    except Exception:
        env_result = {}

    return RunResult(
        game_id=game_id,
        system_name=system.get_name(),
        environment_name=env.get_env_name(),
        start_time=start_time,
        end_time=end_time,
        duration_seconds=duration,
        result=env_result,
        moves=moves,
        total_turns=turn,
        config=config,
        error=error_msg,
        error_type=error_type,
    )
