from dataclasses import dataclass
from typing import Any, List

from chess_engine_module import ChessEngine
from environment import Environment, EnvState, EnvStatePrompt, EnvIntroPrompt, EnvResult
import chess

# Piece name mappings
PIECE_NAMES = {
    'P': 'White Pawn',
    'N': 'White Knight',
    'B': 'White Bishop',
    'R': 'White Rook',
    'Q': 'White Queen',
    'K': 'White King',
    'p': 'Black Pawn',
    'n': 'Black Knight',
    'b': 'Black Bishop',
    'r': 'Black Rook',
    'q': 'Black Queen',
    'k': 'Black King',
}

# File (column) labels a-h
FILES = 'abcdefgh'
# Rank (row) labels 1-8 (FEN starts from rank 8)
RANKS = '87654321'


def fen_to_board_description(fen: str) -> str:
    """
    Convert a FEN string to a clear, human-readable board state description.

    This provides a deterministic, unambiguous description of where each piece
    is located on the board, making it easier for LLMs to understand the position.

    Args:
        fen: A FEN (Forsyth-Edwards Notation) string representing the board state.
             Example: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    Returns:
        A multi-line string describing the board state in plain English.

    Example output:
        "=====BOARD STATE DESCRIPTION=====
        White Pieces:
          White Rook at a1
          White Knight at b1
          ...
        Black Pieces:
          Black Rook at a8
          Black Knight at b8
          ..."
    """
    # Split FEN into components
    parts = fen.split()
    board_fen = parts[0]  # The board position part
    active_color = parts[1] if len(parts) > 1 else 'w'
    castling = parts[2] if len(parts) > 2 else '-'
    en_passant = parts[3] if len(parts) > 3 else '-'
    halfmove = parts[4] if len(parts) > 4 else '0'
    fullmove = parts[5] if len(parts) > 5 else '1'

    # Parse the board position
    white_pieces = []
    black_pieces = []

    ranks = board_fen.split('/')

    for rank_idx, rank_str in enumerate(ranks):
        file_idx = 0
        for char in rank_str:
            if char.isdigit():
                # Empty squares
                file_idx += int(char)
            else:
                # Piece
                square = FILES[file_idx] + RANKS[rank_idx]
                piece_name = PIECE_NAMES.get(char, f"Unknown({char})")

                if char.isupper():
                    white_pieces.append(f"  {piece_name} at {square}")
                else:
                    black_pieces.append(f"  {piece_name} at {square}")

                file_idx += 1

    # Build the description
    lines = ["=====BOARD STATE DESCRIPTION====="]

    # Turn indicator
    turn = "White" if active_color == 'w' else "Black"
    lines.append(f"Turn: {turn} to move")
    lines.append("")

    # White pieces
    lines.append("White Pieces:")
    if white_pieces:
        lines.extend(sorted(white_pieces))
    else:
        lines.append("  (none)")
    lines.append("")

    # Black pieces
    lines.append("Black Pieces:")
    if black_pieces:
        lines.extend(sorted(black_pieces))
    else:
        lines.append("  (none)")
    lines.append("")

    # Castling rights
    if castling != '-':
        castling_desc = []
        if 'K' in castling:
            castling_desc.append("White can castle kingside")
        if 'Q' in castling:
            castling_desc.append("White can castle queenside")
        if 'k' in castling:
            castling_desc.append("Black can castle kingside")
        if 'q' in castling:
            castling_desc.append("Black can castle queenside")
        lines.append("Castling Rights: " + "; ".join(castling_desc))
    else:
        lines.append("Castling Rights: None")

    # En passant
    if en_passant != '-':
        lines.append(f"En Passant Target Square: {en_passant}")

    # Move counts
    lines.append(f"Move Number: {fullmove} (Halfmove clock: {halfmove})")

    return "\n".join(lines)


def get_board_description_from_board(board: chess.Board) -> str:
    """
    Get a human-readable board description from a chess.Board object.

    Args:
        board: A python-chess Board object.

    Returns:
        A multi-line string describing the board state in plain English.
    """
    return fen_to_board_description(board.fen())


@dataclass
class ChessMove():
    """Stores chess moves for the purposes of the move history in the promp"""
    side: str
    uci: str
    move_id: int


@dataclass
class ChessState(EnvState):
    player: str
    san: str
    uci: str
    fen_before: str
    fen_after: str
    centipawn_eval: int
    centipawn_eval_delta: int
    ply: int


@dataclass
class ChessResult(EnvResult):
    # store the winner as a string (e.g. "system" or "<engine name>")
    winner: str
    result: str  # "1-0", "0-1", "1/2-1/2", "incomplete"
    final_fen: str


class ChessEnv(Environment):
    """Implementation of Environment for playing Chess against a chess engine. See the Environment class
    for documentation on the methods within
    """

    def __init__(self, chess_engine: ChessEngine, max_plies: int):
        super().__init__()
        self.has_opponent = True
        self.opponent_name = chess_engine.get_name()

        self.chess_engine = chess_engine

        self.max_plies = max_plies

        self.board = chess.Board()
        self.ply = 0

        self.move_history: List[ChessMove] = []
        self.prev_centipawn: int = 0

    def is_active(self) -> bool:
        return not self.board.is_game_over() and self.ply < self.max_plies

    def _apply_move(self, move: chess.Move, chess_state_player: str, chess_move_side: str) -> EnvState:
        """Apply the given move by the given player to the board, and returns the resulting state.

        Args:
            move: the chess.Move object representing the move to apply to the board.
            chess_state_player: a string to store in the ChessState representing who made the move (i.e. "system").
            chess_move_side: a string to store in the move history representing who made the move (white or black)

        Returns:
            the ChessState (EnvState implementation) representing the state of the chess board after the move.
        """
        # capture the move and the stats before the move
        san = self.board.san(move)
        fen_before = self.board.fen()
        centipawns_before = self.prev_centipawn

        self.board.push(move)  # affect the board with the given move
        self.ply += 1

        # capture stats after the move
        fen_after = self.board.fen()
        centipawns_after = self.chess_engine.evaluate_board(self.board)
        self.prev_centipawn = centipawns_after

        # construct state based off of making the move
        state = ChessState(player=chess_state_player, san=san, uci=move.uci(), fen_before=fen_before,
                           fen_after=fen_after, centipawn_eval=centipawns_after,
                           centipawn_eval_delta=centipawns_after - centipawns_before, ply=self.ply)

        # add the move to the tracked move history (for prompt purposes)
        self.move_history.append(
            ChessMove(side=chess_move_side, uci=move.uci(), move_id=self.ply))

        return state

    def give_move(self, action: str) -> EnvState:
        nm_move = chess.Move.from_uci(action)

        return self._apply_move(nm_move, "system", "white")

    def opponent_move(self) -> EnvState:
        engine_move = self.chess_engine.next_move(self.board)
        return self._apply_move(engine_move, self.chess_engine.get_name(), "black")

    def get_env_state_prompt(self) -> EnvStatePrompt:
        # Generate human-readable board description from FEN
        board_description = get_board_description_from_board(self.board)

        legal_moves_list = [m.uci() for m in self.board.legal_moves]
        legal_moves_str = "You MUST choose from these valid moves: " + ", ".join(legal_moves_list) + "\n\n"

        # format move history
        total_moves = len(self.move_history)
        lines = [
            f"MOVE HISTORY (Current Game - {total_moves} moves so far):",
        ]

        if total_moves == 0:
            lines.append("No moves yet - this is the opening position.")
        else:
            for entry in self.move_history:
                move_id = entry.move_id
                move_uci = entry.uci
                side = entry.side
                # reasoning = entry.get("reasoning", "") # doesn't exist here, the env cannot know the reasoning

                if side == "white":
                    lines.append(f"  Move {move_id}: {move_uci} (Your move)")
                else:
                    lines.append(f"  Move {move_id}: {move_uci} ({self.chess_engine.get_name()}'s move)")

        history_text = "\n".join(lines)

        combined_str = (
            f"Board Position (FEN): {self.board.fen()}\n\n"
            "FEN explanation: This single string represents the entire board state.\n"
            "- Lowercase letters = Black pieces (r=rook, n=knight, b=bishop, q=queen, k=king, p=pawn)\n"
            "- Uppercase letters = White pieces (R=Rook, N=Knight, B=Bishop, Q=Queen, K=King, P=Pawn)\n"
            "- Numbers = consecutive empty squares\n"
            "- The 'w' or 'b' after the board indicates whose turn it is\n\n"
            f"{board_description}\n\n"
            f"{history_text}\n\n"
            f"{legal_moves_str}"
        )

        prompt = EnvStatePrompt(prompt=combined_str, move_id=self.ply,
                                metadata=None, legal_actions=legal_moves_list)

        return prompt

    def get_env_name(self) -> str:
        return "Chess"

    def close(self) -> None:
        self.chess_engine.close()

    def get_env_intro_prompt(self) -> EnvIntroPrompt:
        engine_name = self.chess_engine.get_name()
        intro = (
            f"win a chess game against {engine_name} (a chess engine). "
            f"Your team plays WHITE, {engine_name} plays BLACK.\n\n"
        )
        rules = (
            "CRITICAL - UCI NOTATION:\n"
            "ALL chess moves MUST be written in UCI notation (not algebraic notation).\n"
            "UCI format: starting_square + ending_square (e.g., e2e4, g1f3, e1g1)\n"
            "WRONG: 1.e4, Nf3, O-O, Qxd5\n"
            "CORRECT: e2e4, g1f3, e1g1, d1d5\n"
            # - in direct messages, common space, and especially when expressing agreement on a move.
            "This applies to ALL move discussions \n"
        )

        consensus_tool = {
            'type': 'function',
            'function': {
                'name': 'consensus',
                'description': 'Declare that the team has reached consensus on a move',
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
            f"Remember, if talking about or proposing positions, use UCI notation")

        return EnvIntroPrompt(domain_prompt=intro, notation_rules=rules, consensus_schema=consensus_tool, consensus_move_field=consensus_move_field, reflection=reflection, reply=reply)

    def get_result(self) -> EnvResult:
        result_str = self.board.result() if self.board.is_game_over() else "incomplete"
        winner_map = {
            "1-0": "system",
            "0-1": self.chess_engine.get_name(),
            "1/2-1/2": "draw",
        }
        winner = winner_map.get(result_str, "incomplete")
        return ChessResult(winner=winner, result=result_str, final_fen=self.board.fen())
