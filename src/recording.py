from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from environment import EnvState, EnvResult
from system import SystemMoveRecord


@dataclass
class RunnerMoveRecord:
    """A single move record produced by the runner.

    For system moves: action is the string returned by system.next_move(),
    system_move_record contains the LLM call data.
    For opponent moves: action is "" (the action is internal to the environment),
    system_move_record is None.
    """
    turn: int
    player: str
    action: str
    env_state: EnvState
    system_move_record: Optional[SystemMoveRecord]
    timestamp: str
    latency_ms: int

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "player": self.player,
            "action": self.action,
            "env_state": self.env_state.to_dict(),
            "system_move_record": self.system_move_record.to_dict() if self.system_move_record else {},
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
        }


@dataclass
class RunResult:
    """Complete result of a single run_game() invocation."""
    game_id: str
    system_name: str
    environment_name: str
    start_time: str
    end_time: str
    duration_seconds: float
    result: Dict[str, Any]  # env.get_result().to_dict()
    moves: List[RunnerMoveRecord]
    total_turns: int
    config: dict
    error: Optional[str] = None
    error_type: Optional[str] = None  # machine-readable category: timeout, consensus_failure, invalid_move, etc.

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "system_name": self.system_name,
            "environment_name": self.environment_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "result": self.result,
            "moves": [m.to_dict() for m in self.moves],
            "total_turns": self.total_turns,
            "config": self.config,
            "error": self.error,
            "error_type": self.error_type,
        }
