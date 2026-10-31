"""Tokenisation helpers shared by BM25, evaluation and the extractive fallback."""
import re
from typing import List

# Keeps financial codes intact: "8-k", "1.05", "ifrs", "s2", "97.5", "ebitda"
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", re.IGNORECASE)

STOPWORDS = frozenset(
    """a an and are as at be but by for from has have how if in into is it its of on or that the their
    then there these this to was were what when where which who why will with under over per can do does""".split()
)


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Lower-case, punctuation-stripped tokens. 'EBITDA.' -> 'ebitda', 'Form 8-K' -> ['form', '8-k']."""
    tokens = _TOKEN_RE.findall((text or "").lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def content_terms(text: str, min_len: int = 3) -> set:
    return {t for t in tokenize(text) if len(t) >= min_len}
