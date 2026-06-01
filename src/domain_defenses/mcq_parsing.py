from __future__ import annotations

import re


LETTER_TO_NUMBER = {"A": "1", "B": "2", "C": "3", "D": "4"}

# High-confidence explicit markers — safe for any text length
_EXPLICIT_PATTERNS = [
    # "Answer: A", "answer is 3", "The correct answer is B"
    re.compile(r"\b(?:correct\s+)?answer\s*(?:is\s*|:\s*|=\s*)([ABCD1-4])\b", re.IGNORECASE),
    # "I choose 4", "select option A", "pick B"
    re.compile(r"\b(?:choose|select|pick)\s+(?:option\s+)?([ABCD1-4])\b", re.IGNORECASE),
    # "option D"
    re.compile(r"\boption\s+([ABCD1-4])\b", re.IGNORECASE),
    # "(A)" or "(3)"
    re.compile(r"\(([ABCD1-4])\)", re.IGNORECASE),
]

# Answer at start of text followed by a strong separator (not just a space)
# Matches "3. Because...", "B) Option text...", "A: explanation..."
_START_PATTERN = re.compile(r"^\s*([ABCD1-4])[.):]\s", re.IGNORECASE)

# Free-search patterns — used only for short outputs to avoid false positives
# from echoed question text (e.g. "A patient presents with..." → A ≠ answer)
_FREE_LETTER = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
_FREE_NUMBER = re.compile(r"\b([1-4])\b")

_SHORT_THRESHOLD = 30  # chars; beyond this, free search yields too many false positives


def _normalize(raw: str) -> str:
    upper = raw.upper()
    return LETTER_TO_NUMBER.get(upper, upper)


_THINK_CLOSED = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)


def parse_mcq_answer(text: str) -> str | None:
    """Return one of {'1','2','3','4'} from model output, or None on parse failure."""
    if not text:
        return None

    # Strip closed <think>...</think> blocks (reasoning models).
    # Unclosed <think> means generation was cut off before the answer — fail fast.
    if _THINK_OPEN.search(text):
        if not re.search(r"</think>", text, re.IGNORECASE):
            return None
        text = _THINK_CLOSED.sub("", text)

    stripped = text.strip()
    upper = stripped.upper()

    # 1. Exact match
    if upper in {"1", "2", "3", "4"}:
        return upper
    if upper in LETTER_TO_NUMBER:
        return LETTER_TO_NUMBER[upper]

    # 2. Near-exact: answer with trailing punctuation ("3.", "B)", "A.")
    core = stripped.rstrip(".):,!? ")
    if core.upper() in {"1", "2", "3", "4"}:
        return core.upper()
    if core.upper() in LETTER_TO_NUMBER:
        return LETTER_TO_NUMBER[core.upper()]

    # 3. Explicit high-confidence markers (safe for any text length)
    for pattern in _EXPLICIT_PATTERNS:
        m = pattern.search(stripped)
        if m:
            return _normalize(m.group(1))

    # 4. Answer at start of text with strong separator
    m = _START_PATTERN.match(stripped)
    if m:
        return _normalize(m.group(1))

    # 5. Free fallback — only for short outputs
    #    Require a single unambiguous occurrence to avoid false positives.
    #    Letters must not be sentence-initial (position > 0) to avoid "A patient…" → 1.
    if len(stripped) <= _SHORT_THRESHOLD:
        number_matches = _FREE_NUMBER.findall(stripped)
        if len(number_matches) == 1:
            return number_matches[0]
        letter_matches = list(_FREE_LETTER.finditer(upper))
        if len(letter_matches) == 1 and letter_matches[0].start() > 0:
            return LETTER_TO_NUMBER[letter_matches[0].group(1).upper()]

    return None
