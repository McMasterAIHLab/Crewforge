"""
Chess Engine Module

This module provides an abstract base class for chess engines and 
concrete implementations of engines using these classes.

The engines may be interacted with in two ways: through the next_move()
function, which returns the engine's next move, and evaluate_board()
function, which returns the engine's evaluation of the current board state
(from white's/LLM perspective)

Usage Example:
    engine = StockfishEngine("/path/to/stockfish", default_depth=10)
    move = engine.next_move(board)
    eval_score = engine.evaluate_board(board) 
    engine.close()
"""

from abc import ABC, abstractmethod
import chess
import chess.engine


class ChessEngine(ABC):
    """
    Abstract base class for chess engines.
    
    This class defines the interface that all chess engine implementations
    must follow. The engine is initialized and ready to use immediately
    after construction.
    """
    
    def __init__(self, engine_path: str, default_play_depth: int, default_eval_depth: int):
        """
        Initialize the chess engine and set the engine_path and depth.
        
        Args:
            engine_path: Path to the engine executable
            default_play_depth: Default search depth in plies for making moves
            default_eval_depth: Default search depth in plies for evaluating the board
            
        Raises:
            FileNotFoundError: If the engine executable is not found
            RuntimeError: If engine initialization fails
        """
        self.engine_path = engine_path
        self.default_play_depth = default_play_depth
        self.default_eval_depth = default_eval_depth

    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the engine's name

        Returns:
            The engine's name as a str
        """
        pass
    

    @abstractmethod
    def next_move(self, board: chess.Board, depth=None) -> chess.Move:
        """
        Get the engine's move for the current position.
        
        Args:
            board: Current chess board state
            depth: Search depth in plies (uses default_play_depth if None)
            
        Returns:
            Chess.Move object (can be converted to UCI notation (e.g., 'e2e4', 'e7e8q')
            by calling .uci() on output)
        """
        pass
    

    @abstractmethod
    def evaluate_board(self, board: chess.Board, depth=None) -> int:
        """
        Evaluate the current position from white's/LLM perspective.
        
        Args:
            board: Current chess board state
            depth: Search depth in plies for evaluating (uses default_eval_depth if None)
            
        Returns:
            Evaluation score in centipawns from white's/LLM perspective.
            Positive values indicate white advantage, negative values
            indicate black advantage. Mate is scored as +/-30000.
        """
        pass
    

    @abstractmethod
    def close(self):
        """
        Close the engine connection and release resources.
        
        This method should gracefully shut down the engine and clean up
        any resources. It should be safe to call multiple times.
        """
        pass


class StockfishEngine(ChessEngine):
    """
    Stockfish chess engine implementation.
    
    This class provides a concrete implementation of the ChessEngine interface
    for the Stockfish chess engine.
    
    The engine is initialized and ready to use immediately after construction.
    """
    

    def __init__(self, engine_path: str, default_play_depth: int, default_eval_depth: int):
        """
        Initialize Stockfish engine and establish connection via UCI protocol.
        
        Args:
            engine_path: Path to the Stockfish executable
            default_play_depth: Default search depth in plies for making moves
            default_eval_depth: Default search depth in plies for evaluating the board
        
        Raises:
            FileNotFoundError: If Stockfish executable is not found at engine_path
        """
        super().__init__(engine_path, default_play_depth, default_eval_depth)
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)

            # nerf stockfish as much as possible
            self.engine.configure({
                "UCI_LimitStrength": True,  # Enable strength limiting
                "Skill Level": 0,           # Lowest skill (plays badly)
                "UCI_Elo": 1320,            # Beginner level, lowest that stockfish allows
            })            
        except FileNotFoundError:
            raise FileNotFoundError(f"Stockfish not found at \"{self.engine_path}\". Please verify the path to the Stockfish executable.")
    

    def get_name(self):
        return "Stockfish"

    
    def next_move(self, board: chess.Board, depth=None) -> chess.Move:
        """
        Get Stockfish's recommended move for the current position.
        
        Args:
            board: Current chess board state
            depth: Search depth in plies (uses default_play_depth if None)
            
        Returns:
            Chess.Move object (can be converted to UCI notation (e.g., 'e2e4', 'e7e8q')
            by calling .uci() on output)
        """
        limit = chess.engine.Limit(depth=depth if depth is not None else self.default_play_depth)
        return self.engine.play(board, limit).move
    

    def evaluate_board(self, board: chess.Board, depth=None) -> int:
        """
        Evaluate the current board using Stockfish's analysis from white's/LLM perspective
        
        Args:
            board: Current chess board state
            depth: Search depth in plies for evaluating (uses default_eval_depth if None)
                        
        Returns:
            Evaluation score in centipawns from white's/LLM perspective.
            - Positive values: white has an advantage
            - Negative values: black has an advantage
            - Checkmate is scored as +/-30000 centipawns
            - Example: +100 = white is up 1 pawn (100 centipawns = 1 pawn)
        """
        info = self.engine.analyse(board, chess.engine.Limit(depth=depth if depth is not None else self.default_eval_depth))
        score = info["score"].pov(chess.WHITE)  # NOTE: may want to use a symbolic constant to represent
                                                # the LLM here, instead of assuming the rest of the modules
                                                # will use chess.WHITE as the LLM player
        return int(score.score(mate_score=30000))
    
    
    def close(self):
        """
        Close the Stockfish engine connection.
        
        It is safe to call multiple times.
        """
        if self.engine is not None:
            self.engine.quit()
            self.engine = None