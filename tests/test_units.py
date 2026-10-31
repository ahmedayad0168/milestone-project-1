from src.generation.chain import extract_citations, extractive_answer
from src.ingestion.hierarchical import HierarchicalChunker, sanitize_metadata
from src.ingestion.parser import Document
from src.retrieval.hybrid_search import rrf_fuse
from src.retrieval.query_expansion import QueryExpander
from src.text_utils import tokenize


def test_tokenize_keeps_financial_codes_and_strips_punctuation():
    assert tokenize("EBITDA.") == ["ebitda"]
    assert "8-k" in tokenize("Form 8-K, Item 1.05!")
    assert "1.05" in tokenize("Item 1.05")
    assert "the" not in tokenize("the impact of IFRS 16")


def test_sanitize_metadata_for_chroma():
    clean = sanitize_metadata({"a": 1, "b": None, "c": [1, 2], "d": {"x": 1}, "e": float("nan"), "f": "ok"})
    assert clean["a"] == 1 and "b" not in clean and "e" not in clean
    assert clean["c"] == "1, 2" and isinstance(clean["d"], str) and clean["f"] == "ok"


def test_chunk_ids_are_deterministic_and_parent_linked():
    docs = [Document("Sentence one. " * 200, {"source": "a.json", "record_id": "r1"})]
    c1, p1 = HierarchicalChunker().process_documents(docs)
    c2, p2 = HierarchicalChunker().process_documents(docs)
    assert [c["id"] for c in c1] == [c["id"] for c in c2] and set(p1) == set(p2)
    assert all(c["parent_id"] in p1 for c in c1)
    assert len({c["id"] for c in c1}) == len(c1)


def test_rrf_fusion_ranks_overlap_first_and_does_not_mutate():
    dense = [{"id": "a", "text": "a"}, {"id": "b", "text": "b"}, {"id": "c", "text": "c"}]
    sparse = [{"id": "b", "text": "b"}, {"id": "d", "text": "d"}]
    out = rrf_fuse({"dense": dense, "sparse": sparse}, k=60, top_k=4)
    assert out[0]["id"] == "b" and out[0]["dense_rank"] == 2 and out[0]["sparse_rank"] == 1
    assert "dense_rank" not in dense[1] and "rrf_score" not in dense[1]


def test_entities_and_expansion():
    qe = QueryExpander()
    labels = {(e["text"], e["label"]) for e in qe.extract_entities("IFRS 16 raises EBITDA by 5% or $2 million")}
    assert ("IFRS 16", "FIN_STANDARD") in labels and ("5%", "PERCENT") in labels and ("$2 million", "MONEY") in labels
    assert "lease liability" in qe.expand_query("impact of IFRS 16")
    assert qe.expand_query("hello world") == "hello world"


def test_citation_extraction_ignores_invalid_numbers():
    parents = [{"text": "x", "metadata": {"source": "s1", "parent_id": "p1"}}, {"text": "y", "metadata": {"source": "s2", "parent_id": "p2"}}]
    cites = extract_citations("Claim [2] and again [2], bogus [7], first [1].", parents)
    assert [c["index"] for c in cites] == [1, 2] and cites[1]["source"] == "s2"


def test_extractive_answer_cites_and_abstains():
    parents = [{"text": "The output floor is 72.5% of standardised RWA. Unrelated filler text here today.", "metadata": {}}]
    assert "[1]" in extractive_answer("output floor percentage", parents)
    assert "don't have enough information" in extractive_answer("zebra quantum", parents)
