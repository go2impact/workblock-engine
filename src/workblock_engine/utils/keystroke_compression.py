"""Utility functions for compressing keystroke sequences."""
from __future__ import annotations

import re
from typing import List

PLAIN_TEXT_REPEAT_THRESHOLD = 4

# Pattern captures angle-bracket tokens and any text between them.
_SEGMENT_PATTERN = re.compile(r'(<[^>]+>|[^<]+)')


def _split_segments(text: str) -> List[str]:
    """Split keystroke text into keystroke tokens and plain text segments."""
    return _SEGMENT_PATTERN.findall(text)


def _compress_plain_segment(segment: str) -> str:
    """Compress runs of identical characters in plain-text segments."""
    pieces: List[str] = []
    i = 0
    length = len(segment)

    while i < length:
        char = segment[i]
        run_length = 1

        while i + run_length < length and segment[i + run_length] == char:
            run_length += 1

        if char.isspace():
            pieces.append(char * run_length)
        elif run_length > PLAIN_TEXT_REPEAT_THRESHOLD:
            pieces.append(f"{char}{{{run_length}}}")
        else:
            pieces.append(char * run_length)

        i += run_length

    return ''.join(pieces)


def compress_keystrokes(text: str) -> str:
    """Compress repeated keystroke sequences to reduce token count.

    Consecutive identical keystroke tokens are collapsed (``<Cmd>{3}``), and
    repeated multi-token sequences are grouped (``(<Ctrl><Ctrl + c>){4}``).
    Plain text segments pass through unchanged.
    """
    if not text:
        return text

    segments = _split_segments(text)
    if not segments:
        return text

    result: List[str] = []
    i = 0
    length = len(segments)

    while i < length:
        current = segments[i]

        if current.startswith('<'):
            best_len = 1
            best_count = 1
            max_group_len = length - i

            for group_len in range(1, max_group_len + 1):
                group = segments[i:i + group_len]
                if not all(part.startswith('<') for part in group):
                    break

                count = 1
                while i + (count * group_len) < length and segments[i + (count * group_len): i + ((count + 1) * group_len)] == group:
                    count += 1

                if count > 1 and group_len * count > best_len * best_count:
                    best_len = group_len
                    best_count = count

            if best_count > 1:
                group_text = ''.join(segments[i:i + best_len])
                if best_len == 1:
                    result.append(f"{group_text}{{{best_count}}}")
                else:
                    result.append(f"({group_text}){{{best_count}}}")
                i += best_len * best_count
                continue

        if current.startswith('<'):
            result.append(current)
        else:
            result.append(_compress_plain_segment(current))
        i += 1

    return ''.join(result)

__all__ = ["compress_keystrokes"]
