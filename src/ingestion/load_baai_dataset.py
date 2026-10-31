"""Optional: build a large finance corpus from BAAI/IndustryInstruction_Finance-Economics.

Needs ``datasets`` (uncomment it in requirements.txt). The deployed app does NOT need this.
Outputs go to data/raw (raw dump, git-ignored), data/processed (git-ignored) and data/golden_eval_set.json.

    python -m src.ingestion.load_baai_dataset --num-samples 5000
    python -m src.ingestion.run_ingestion --include-processed
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Tuple

import numpy as np

from src.config import settings

STOP = {"this", "that", "with", "from", "have", "which", "their", "they", "been", "were", "about", "there",
        "these", "would", "could", "should", "more", "also", "such"}


def extract_instruction_response(convs, fallback_instruction: str = "") -> Tuple[str, str]:
    """Robust extractor for conversations stored as ndarray, list, dict-list or stringified list."""
    instruction, response = "", ""
    if isinstance(convs, str):
        try:
            convs = ast.literal_eval(convs)
        except Exception:
            convs = None
    if isinstance(convs, (list, tuple, np.ndarray)):
        for item in convs:
            if not isinstance(item, dict):
                continue
            sender, val = item.get("from", ""), str(item.get("value", ""))
            if sender in ("human", "user") and not instruction:
                instruction = val
            elif sender in ("gpt", "assistant") and not response:
                response = val
    return (instruction or str(fallback_instruction or "")).strip(), response.strip()


def load_baai_finance_dataset(num_samples: int = 75000, max_processed: int = 15000):
    from datasets import load_dataset  # lazy: only needed for this script
    import pandas as pd

    raw_csv = settings.DATA_RAW_DIR / f"baai_finance_economics_raw_{num_samples // 1000}k.csv"
    processed_json = settings.DATA_PROCESSED_DIR / "financial_corpus_processed.json"
    golden_path = settings.BASE_DIR / "data" / "golden_eval_set.json"
    settings.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    settings.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Downloading {num_samples} samples from the Hugging Face Hub...")
    ds = load_dataset("BAAI/IndustryInstruction_Finance-Economics")["train"].select(range(num_samples))
    ds.to_pandas().to_csv(raw_csv, index=False)

    print("[2/3] Parsing conversations...")
    records, candidates = [], []
    for idx, item in enumerate(ds):
        instruction, response = extract_instruction_response(item.get("conversations"), item.get("instruction", ""))
        if not (instruction or response):
            continue
        records.append({
            "id": f"baai_fin_{idx:06d}",
            "title": instruction[:120] or f"Finance topic {idx + 1}",
            "content": f"Question: {instruction}\n\nAnswer:\n{response}" if instruction else response,
            "lang": str(item.get("lang", "en")),
            "origin_source": raw_csv.name,
        })
        if len(candidates) < 25 and len(instruction) > 40 and len(response) > 80 and str(item.get("lang", "")) == "en":
            candidates.append((instruction, response))
    processed_json.write_text(json.dumps(records[:max_processed], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"      {len(records)} records parsed; {min(len(records), max_processed)} written to {processed_json.name}")

    print("[3/3] Building golden evaluation set...")
    golden = []
    for instruction, response in candidates:
        words = re.findall(r"\b[A-Za-z]{4,}\b", response)
        keywords = list(dict.fromkeys(w for w in words if w.lower() not in STOP))[:5]
        if len(keywords) >= 2:
            golden.append({"query": instruction, "ground_truth_answer": response, "expected_keywords": keywords,
                           "expected_source": processed_json.name})
    golden_path.write_text(json.dumps(golden, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"      {len(golden)} golden items -> {golden_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=75000)
    ap.add_argument("--max-processed", type=int, default=15000)
    a = ap.parse_args()
    load_baai_finance_dataset(a.num_samples, a.max_processed)
