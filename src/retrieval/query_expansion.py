"""Pre-retrieval NER + domain query expansion."""
from __future__ import annotations

import re
from typing import Dict, List

from src.logging_setup import get_logger

log = get_logger(__name__)

_REGEX_PATTERNS = [
    (r"\b(IFRS\s+S[12]|IFRS\s+1[56]|ASC\s+\d{3}|CSRD|ESRS|Basel\s+(?:III|IV)|FRTB|Item\s+106|Item\s+1\.05|Form\s+8-K)\b", "FIN_STANDARD"),
    (r"\b(Scope\s+[123]|CET1|RWA|EBITDA|Expected\s+Shortfall|Output\s+Floor|DSO|DPO|DIO)\b", "FIN_METRIC"),
    (r"\b(Double\s+Materiality|Limited\s+Assurance|TCFD|ISSB)\b", "FIN_GOV"),
    (r"(?:[$€£]\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|[KMBkmb]\b))?|\b\d[\d,]*(?:\.\d+)?\s?(?:USD|EUR|GBP)\b)", "MONEY"),
    (r"\b\d+(?:\.\d+)?\s?%", "PERCENT"),
]


class QueryExpander:
    FINCON_EXPANSION_DICT: Dict[str, List[str]] = {
        "ifrs s1": ["general sustainability requirements", "issb", "governance", "strategy", "risk management", "metrics and targets"],
        "ifrs s2": ["climate-related disclosures", "scope 1", "scope 2", "scope 3 emissions", "financed emissions", "tcfd"],
        "ifrs 16": ["lease liability", "right-of-use asset", "ebitda", "depreciation", "interest expense", "operating lease"],
        "asc 606": ["revenue from contracts with customers", "performance obligations", "transaction price", "five-step model"],
        "csrd": ["corporate sustainability reporting directive", "esrs", "double materiality", "limited assurance", "esef"],
        "esrs": ["european sustainability reporting standards", "impact materiality", "financial materiality"],
        "basel iii": ["basel iii endgame", "frtb", "expected shortfall", "output floor", "rwa", "cet1"],
        "frtb": ["fundamental review of the trading book", "market risk", "internal models approach", "standardised approach", "expected shortfall"],
        "sec cyber": ["item 1.05", "form 8-k", "four business days", "material cybersecurity incident", "item 106"],
        "form 8-k": ["item 1.05", "four business days", "materiality determination"],
        "8-k": ["item 1.05", "four business days", "material cybersecurity incident"],
        "scope 3": ["indirect value chain emissions", "category 15", "upstream", "downstream"],
        "ebitda": ["earnings before interest taxes depreciation amortisation", "alternative performance measure", "non-gaap"],
        "variance": ["price variance", "volume variance", "mix variance", "flexible budget", "favourable", "unfavourable"],
        "working capital": ["dso", "dpo", "dio", "cash conversion cycle"],
    }

    def __init__(self):
        self.nlp = self._init_spacy()

    def _init_spacy(self):
        try:
            import spacy

            try:
                nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
            except Exception:
                nlp = spacy.blank("en")
            if "entity_ruler" not in nlp.pipe_names:
                ruler = nlp.add_pipe("entity_ruler", before="ner") if "ner" in nlp.pipe_names else nlp.add_pipe("entity_ruler")
                ruler.add_patterns([
                    {"label": "FIN_STANDARD", "pattern": [{"LOWER": a}, {"LOWER": b}]}
                    for a, b in [("ifrs", "s1"), ("ifrs", "s2"), ("ifrs", "16"), ("ifrs", "15"), ("asc", "606"), ("asc", "842"),
                                 ("basel", "iii"), ("basel", "iv"), ("item", "106"), ("item", "1.05"), ("form", "8-k")]
                ] + [{"label": "FIN_STANDARD", "pattern": [{"LOWER": w}]} for w in ("csrd", "esrs", "frtb")]
                  + [{"label": "FIN_METRIC", "pattern": [{"LOWER": "scope"}, {"LOWER": n}]} for n in ("1", "2", "3")]
                  + [{"label": "FIN_METRIC", "pattern": [{"LOWER": w}]} for w in ("cet1", "rwa", "ebitda")]
                  + [{"label": "FIN_METRIC", "pattern": [{"LOWER": "expected"}, {"LOWER": "shortfall"}]},
                     {"label": "FIN_METRIC", "pattern": [{"LOWER": "output"}, {"LOWER": "floor"}]},
                     {"label": "FIN_GOV", "pattern": [{"LOWER": "double"}, {"LOWER": "materiality"}]},
                     {"label": "FIN_GOV", "pattern": [{"LOWER": "limited"}, {"LOWER": "assurance"}]},
                     {"label": "FIN_GOV", "pattern": [{"LOWER": "tcfd"}]},
                     {"label": "FIN_GOV", "pattern": [{"LOWER": "issb"}]}]
                )
            return nlp
        except Exception as e:
            log.info("spaCy unavailable (%s); using regex-only entity extraction.", e)
            return None

    def extract_entities(self, text: str) -> List[Dict[str, object]]:
        """spaCy entities merged with regex entities (money/percent/standards), de-duplicated by span."""
        found: List[Dict[str, object]] = []
        if self.nlp is not None:
            try:
                found += [{"text": e.text, "label": e.label_, "start": e.start_char, "end": e.end_char} for e in self.nlp(text).ents]
            except Exception:
                pass
        for pattern, label in _REGEX_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                found.append({"text": m.group(0), "label": label, "start": m.start(), "end": m.end()})
        found.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))
        merged: List[Dict[str, object]] = []
        for e in found:
            if merged and e["start"] < merged[-1]["end"]:  # overlaps an earlier (longer/first) span
                continue
            merged.append(e)
        return merged

    def expand_query(self, query: str) -> str:
        """Append domain synonyms for recognised terms (helps both BM25 and the embedder)."""
        lower = query.lower()
        extra: List[str] = []
        for term, synonyms in self.FINCON_EXPANSION_DICT.items():
            if term in lower:
                for s in synonyms:
                    if s not in lower and s not in extra:
                        extra.append(s)
        return f"{query} {' '.join(extra[:12])}" if extra else query
