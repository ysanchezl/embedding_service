import math
import re
from typing import Callable

# Counts tokens for a batch of strings. Injected so chunking stays tokenizer-agnostic.
TokenCounter = Callable[[list[str]], list[int]]

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?…])\s+")
_WORD = re.compile(r"\s+")


def chunk_text(
    text: str,
    count_tokens: TokenCounter,
    max_tokens: int,
    overlap_tokens: int = 0,
) -> list[str]:
    """Split text into chunks of at most max_tokens, cutting on natural boundaries.

    Prefers paragraph, then sentence, then word boundaries. Consecutive chunks
    share up to overlap_tokens worth of trailing units.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")

    text = text.strip()
    if not text:
        return []

    units = _split(text, count_tokens, max_tokens, [_PARAGRAPH, _SENTENCE, _WORD])
    return _pack(units, count_tokens, max_tokens, overlap_tokens)


def _split(text: str, count_tokens: TokenCounter, max_tokens: int, separators: list) -> list[str]:
    """Recursively break text into units that each fit within max_tokens."""
    if count_tokens([text])[0] <= max_tokens:
        return [text]

    if not separators:
        return _hard_split(text, count_tokens, max_tokens)

    parts = [p for p in separators[0].split(text) if p.strip()]
    if len(parts) <= 1:
        return _split(text, count_tokens, max_tokens, separators[1:])

    units: list[str] = []
    for part in parts:
        units.extend(_split(part, count_tokens, max_tokens, separators[1:]))
    return units


def _hard_split(text: str, count_tokens: TokenCounter, max_tokens: int) -> list[str]:
    """Last resort for a single run of characters that is too big (no spaces)."""
    size = max(1, len(text) // math.ceil(count_tokens([text])[0] / max_tokens))
    pieces = [text[i : i + size] for i in range(0, len(text), size)]

    # Piece sizes are an estimate; keep halving any piece that still does not fit.
    result: list[str] = []
    for piece in pieces:
        if len(piece) > 1 and count_tokens([piece])[0] > max_tokens:
            half = len(piece) // 2
            result.extend(_hard_split(piece[:half], count_tokens, max_tokens))
            result.extend(_hard_split(piece[half:], count_tokens, max_tokens))
        else:
            result.append(piece)
    return result


def _pack(units: list[str], count_tokens: TokenCounter, max_tokens: int, overlap_tokens: int) -> list[str]:
    sizes = count_tokens(units)
    # Overlap must leave room for at least one new unit.
    overlap_tokens = max(0, min(overlap_tokens, max_tokens - 1))

    chunks: list[str] = []
    current: list[int] = []  # indexes into units
    current_size = 0

    for index, size in enumerate(sizes):
        if current and current_size + size > max_tokens:
            chunks.append(_join(units, current))
            current = _overlap_tail(current, sizes, overlap_tokens, max_tokens - size)
            current_size = sum(sizes[i] for i in current)

        current.append(index)
        current_size += size

    if current:
        chunks.append(_join(units, current))

    return chunks


def _overlap_tail(current: list[int], sizes: list[int], overlap_tokens: int, room: int) -> list[int]:
    """Trailing units of the finished chunk to repeat at the start of the next one."""
    limit = min(overlap_tokens, room)
    tail: list[int] = []
    total = 0
    for i in reversed(current):
        if total + sizes[i] > limit:
            break
        tail.append(i)
        total += sizes[i]
    return tail[::-1]


def _join(units: list[str], indexes: list[int]) -> str:
    return " ".join(units[i] for i in indexes)
