

from __future__ import annotations

from prototypes.base import UnitSpec

# Normalization lowercases before matching, so the promotion piece is lowercase
# only. Castling has no special form in UCI: it is the king's two-square move
# (e1g1), which this pattern already covers.
_UCI_PATTERN = r"[a-h][1-8][a-h][1-8][qrbn]?"

ABSTAIN = "--"

_FORMAT_RULES = (
    "The team communicates only in moves. Every message you send is exactly one "
    "move in UCI notation and nothing else: the from-square, then the to-square, "
    "and one extra letter only for a pawn promotion. Examples: e2e4, g1f3, e7e8q. "
    "No words, no punctuation, no explanation, no markdown, no JSON. A message "
    "that is not exactly one move is rejected and you will be asked for it again.\n\n"
    f"If you have no move to put forward, send exactly {ABSTAIN}. The team reads "
    "that as you making no proposal."
)

_CORRECTIVE = (
    "=== FORMAT CORRECTION (YOUR PREVIOUS MESSAGE WAS REJECTED) ===\n"
    "Your message was: {bad}\n"
    "That is not a valid move.\n\n"
    "A message is exactly one chess move in UCI notation and nothing else:\n"
    "the from-square, then the to-square, and one extra letter only for a\n"
    "pawn promotion (q, r, b, or n). Examples: {examples}\n\n"
    "If you have no move to put forward, the message is exactly: {abstain}\n\n"
    "Send only the move, or {abstain}. Do not add:\n"
    "  words, explanation, or punctuation\n"
    "  markdown, code fences, or backticks\n"
    "  JSON, braces, quotes, or field names\n"
    "  algebraic notation such as Nf3, O-O, or Qxd5\n"
    "  more than one move\n\n"
    "Send the move now.\n"
    "=== END FORMAT CORRECTION ==="
)

UCI_MOVE = UnitSpec(
    name="uci_move",
    pattern=_UCI_PATTERN,
    examples=("e2e4", "g1f3", "e7e8q"),
    description="one chess move in UCI notation",
    format_rules=_FORMAT_RULES,
    corrective_template=_CORRECTIVE,
    abstain_token=ABSTAIN,
    check_legal=False,
)
