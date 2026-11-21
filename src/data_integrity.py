"""
Data Integrity Module
Provides checksums, validation, and integrity verification for experiment data.
"""

from __future__ import annotations
import hashlib
import json
import secrets
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone


def compute_sha256(data: Any) -> str:
    """Compute SHA256 hash of any JSON-serializable data."""
    if isinstance(data, str):
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    else:
        # Use canonical JSON representation
        canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def compute_move_checksum(move_data: Dict[str, Any]) -> str:
    """
    Compute integrity checksum for a single move record.
    
    Includes: ply, uci, san, fen_before, fen_after, color, timestamp, deliberation data.
    """
    checksum_data = {
        "ply": move_data.get("ply"),
        "uci": move_data.get("uci"),
        "san": move_data.get("san"),
        "fen_before": move_data.get("fen_before"),
        "fen_after": move_data.get("fen_after"),
        "color": move_data.get("color"),
        "time_ms": move_data.get("time_ms"),
        "deliberation_cycles": move_data.get("deliberation_cycles"),
        "raw_llm_output": move_data.get("raw_llm_output"),
        "move_source": move_data.get("move_source", "llm"),
    }
    return compute_sha256(checksum_data)


def compute_game_checksum(game_data: Dict[str, Any]) -> str:
    """
    Compute integrity checksum for an entire game.
    
    Includes all moves with their checksums, result, and metadata.
    """
    checksum_data = {
        "moves_count": len(game_data.get("moves", [])),
        "move_checksums": [m.get("move_checksum") for m in game_data.get("moves", [])],
        "result": game_data.get("result"),
        "termination": game_data.get("termination"),
        "total_time_ms": game_data.get("total_time_ms"),
    }
    return compute_sha256(checksum_data)


def verify_fen_chain(moves: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    Verify FEN continuity across all moves in a game.
    
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    if not moves:
        return True, []
    
    for i in range(1, len(moves)):
        prev_fen_after = moves[i - 1].get("fen_after")
        curr_fen_before = moves[i].get("fen_before")
        
        if prev_fen_after != curr_fen_before:
            errors.append(
                f"FEN discontinuity at ply {moves[i].get('ply')}: "
                f"previous fen_after != current fen_before"
            )
    
    return len(errors) == 0, errors


def compute_config_hash(config_data: Dict[str, Any], min_entropy_bits: int = 64) -> str:
    full_hash = compute_sha256(config_data)
    # Use 20 characters = 80 bits, giving collision probability < 10^-8 at 10k experiments
    # Birthday paradox: p(collision) ≈ n² / (2 * 2^k) where k = bits, n = experiments
    # For k=80, n=10000: p ≈ 10^8 / (2 * 2^80) ≈ 4 × 10^-17
    min_chars = (min_entropy_bits + 3) // 4 + 4  # Extra margin
    return full_hash[:max(20, min_chars)]


@dataclass
class IntegrityRecord:
    """Complete integrity record for a single move."""
    ply: int
    move_checksum: str
    fen_before: str
    fen_after: str
    timestamp: str
    previous_checksum: Optional[str] = None
    chain_valid: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GameIntegrityTracker:
    """Tracks integrity records for an entire game."""
    def __init__(self, game_id: str):
        self.game_id = game_id
        self.records: List[IntegrityRecord] = []
        self.initial_fen: str = ""
        self._chain_salt = secrets.token_hex(8)
    
    def set_initial_fen(self, fen: str):
        """Set the starting FEN for the game."""
        self.initial_fen = fen
    
    def add_move_record(self, move_data: Dict[str, Any]) -> str:
        """
        Add a move and compute its chained checksum.
        
        Returns the computed checksum.
        """
        ply = move_data.get("ply", len(self.records))
        
        # Get previous checksum for chaining
        prev_checksum = None
        if self.records:
            prev_checksum = self.records[-1].move_checksum
        
        # Compute checksum including chain
        checksum_input = {
            **move_data,
            "previous_checksum": prev_checksum,
            "chain_salt": self._chain_salt,
        }
        move_checksum = compute_sha256(checksum_input)[:32]  # 128 bits for move
        
        # Verify FEN continuity
        chain_valid = True
        if self.records:
            expected_fen = self.records[-1].fen_after
            actual_fen = move_data.get("fen_before", "")
            chain_valid = (expected_fen == actual_fen)
        elif self.initial_fen:
            chain_valid = (move_data.get("fen_before", "") == self.initial_fen)
        
        record = IntegrityRecord(
            ply=ply,
            move_checksum=move_checksum,
            fen_before=move_data.get("fen_before", ""),
            fen_after=move_data.get("fen_after", ""),
            timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            previous_checksum=prev_checksum,
            chain_valid=chain_valid,
        )
        self.records.append(record)
        
        return move_checksum
    
    def verify_chain(self) -> Tuple[bool, List[str]]:
        """Verify the entire chain integrity."""
        errors = []
        
        for i, record in enumerate(self.records):
            if not record.chain_valid:
                errors.append(f"FEN discontinuity at ply {record.ply}")
            
            if i > 0:
                if record.previous_checksum != self.records[i - 1].move_checksum:
                    errors.append(f"Checksum chain broken at ply {record.ply}")
        
        return len(errors) == 0, errors
    
    def get_game_checksum(self) -> str:
        """Compute final game checksum."""
        if not self.records:
            return compute_sha256({"game_id": self.game_id, "empty": True})
        
        return compute_sha256({
            "game_id": self.game_id,
            "move_count": len(self.records),
            "final_checksum": self.records[-1].move_checksum,
            "chain_salt": self._chain_salt,
            "all_checksums": [r.move_checksum for r in self.records],
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Export integrity data."""
        return {
            "game_id": self.game_id,
            "initial_fen": self.initial_fen,
            "move_count": len(self.records),
            "game_checksum": self.get_game_checksum(),
            "chain_verified": self.verify_chain()[0],
            "records": [r.to_dict() for r in self.records],
        }
