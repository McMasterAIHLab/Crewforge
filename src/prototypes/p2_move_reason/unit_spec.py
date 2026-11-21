
from __future__ import annotations

from prototypes.base import UnitSpec

# Identical to p1: the move half of a message is still exactly one UCI move.
# Normalization lowercases before matching, so the promotion piece is lowercase
# only. Castling has no special form in UCI: it is the king's two-square move
# (e1g1), which this pattern already covers.
_UCI_PATTERN = r"[a-h][1-8][a-h][1-8][qrbn]?"

ABSTAIN = "--"

#: Hard cap on the reason field, in whitespace-separated words. A reason over
#: this length is rejected and corrected, exactly as a malformed move is. The
#: cap is the whole point of the prototype: it restores a reason channel
#: without restoring open-ended deliberation.
MAX_REASON_WORDS = 25

_FORMAT_RULES = (
    "Every message you send has two parts, and both are required.\n"
    "  move: exactly one move in UCI notation and nothing else - the from-square, "
    "then the to-square, and one extra letter only for a pawn promotion. "
    "Examples: e2e4, g1f3, e7e8q. Not algebraic notation such as Nf3 or O-O.\n"
    f"  reason: why that move, in at most {MAX_REASON_WORDS} words, judged by your "
    "own specialism. One or two sentences.\n\n"
    f"If you have no move to put forward, send exactly {ABSTAIN} as the move, with "
    "a reason saying what stopped you. The team reads that as you making no "
    "proposal.\n\n"
    "A message whose move is not exactly one move, or whose reason is missing or "
    "over the word limit, is rejected and you will be asked for it again."
)

_CORRECTIVE = (
    "=== FORMAT CORRECTION (YOUR PREVIOUS MESSAGE WAS REJECTED) ===\n"
    "{bad}\n\n"
    "A message has two parts, both required:\n"
    "  move:   exactly one chess move in UCI notation and nothing else - the\n"
    "          from-square, then the to-square, and one extra letter only for a\n"
    "          pawn promotion (q, r, b, or n). Examples: {examples}\n"
    f"  reason: why that move, in at most {MAX_REASON_WORDS} words.\n\n"
    "If you have no move to put forward, the move is exactly: {abstain}\n\n"
    "Do not put into the move field:\n"
    "  words, explanation, or punctuation\n"
    "  markdown, code fences, or backticks\n"
    "  JSON, braces, quotes, or field names\n"
    "  algebraic notation such as Nf3, O-O, or Qxd5\n"
    "  more than one move\n\n"
    "Send the move and its reason now.\n"
    "=== END FORMAT CORRECTION ==="
)

MOVE_REASON = UnitSpec(
    name="move_with_reason",
    pattern=_UCI_PATTERN,
    examples=("e2e4", "g1f3", "e7e8q"),
    description="one chess move in UCI notation",
    format_rules=_FORMAT_RULES,
    corrective_template=_CORRECTIVE,
    abstain_token=ABSTAIN,
    check_legal=False,
)
