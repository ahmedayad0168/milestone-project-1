"""Conversation memory + follow-up question rewriting (the 'chat' feature).

A follow-up like "and what about Scope 3?" is useless to a retriever on its own. Before retrieval we
rewrite it into a standalone question using the recent chat history, so hybrid search + reranking
work exactly as they do for a first question.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.config import settings
from src.generation.llm import BaseLLM
from src.generation.prompts import REWRITE_SYSTEM_PROMPT
from src.logging_setup import get_logger

log = get_logger(__name__)
History = List[Dict[str, str]]  # [{"role": "user"|"assistant", "content": str, "standalone"?: str}]

_FOLLOWUP_CUES = re.compile(
    r"\b(it|its|they|them|their|this|that|those|these|the same|former|latter|above|previous|instead|also|too|else)\b"
    r"|^\s*(and|but|so|then|what about|how about|why|how come|what if|compare|explain|more|elaborate|can you|could you)\b",
    re.IGNORECASE,
)


def trim_history(history: Optional[History], max_turns: int = settings.MEMORY_TURNS) -> History:
    """Keep the last ``max_turns`` user/assistant exchanges."""
    history = [h for h in (history or []) if h.get("role") in ("user", "assistant") and h.get("content")]
    return history[-2 * max_turns:]


def needs_rewrite(question: str, history: Optional[History]) -> bool:
    """Cheap heuristic that avoids an LLM call for self-contained questions."""
    if not trim_history(history):
        return False
    words = question.split()
    return len(words) <= 14 and bool(_FOLLOWUP_CUES.search(question))


def _heuristic_rewrite(question: str, history: History) -> str:
    """No-LLM fallback: anchor the follow-up to the previous standalone question."""
    for turn in reversed(history):
        if turn["role"] == "user":
            previous = turn.get("standalone") or turn["content"]
            return f"{question.strip()} (regarding: {previous.strip()})"
    return question


def _clean_rewrite(text: str, original: str) -> str:
    lines = (text or "").strip().splitlines()
    line = lines[0].strip() if lines else ""
    line = re.sub(r"^(rewritten|standalone)( question)?\s*:\s*", "", line, flags=re.IGNORECASE).strip().strip("\"'`").strip()
    return line if 3 <= len(line) <= 400 else ""


def rewrite_followup(question: str, history: Optional[History], llm: Optional[BaseLLM]) -> str:
    """Return a standalone version of ``question`` (or the question itself if no rewrite is needed)."""
    history = trim_history(history)
    if not needs_rewrite(question, history):
        return question
    if llm is None or not llm.is_generative:
        return _heuristic_rewrite(question, history)

    transcript = "\n".join(
        f"{'User' if h['role'] == 'user' else 'Assistant'}: {(h.get('standalone') or h['content'])[:350] if h['role'] == 'user' else h['content'][:350]}"
        for h in history[-4:]
    )
    messages = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Chat history:\n{transcript}\n\nFollow-up question: {question}\n\nStandalone question:"},
    ]
    try:
        rewritten = _clean_rewrite(llm.chat(messages, max_tokens=120, temperature=0.0), question)
        return rewritten or _heuristic_rewrite(question, history)
    except Exception as e:  # never let rewriting break the answer
        log.warning("Follow-up rewrite failed (%s); using heuristic.", e)
        return _heuristic_rewrite(question, history)
