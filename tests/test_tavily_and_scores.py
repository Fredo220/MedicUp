import asyncio

from openoe.core.models import Document
from openoe.pipeline import web_context
from openoe.pipeline.answer import _build_context, _format_markdown
from openoe.pipeline.merge import merge_documents
from openoe.sources.tavily import TavilySource, clean_title, extract_ids
from openoe.utils.score import assign_percent, to_percent


def _doc(id_, route, title, doi=None, score=1.0, **meta):
    md = {"doi": doi, **meta} if doi else dict(meta)
    return Document(id=id_, source=route, route=route, doc_type="paper", title=title, content="", metadata=md, score=score)


def test_to_percent_bounds():
    assert to_percent(0.5, 1.0) == 50
    assert to_percent(2.0, 1.0) == 100
    assert to_percent(0.001, 1.0) == 1
    assert to_percent(None, 1.0) == 1
    assert to_percent(0.3, 0) == 1


def test_assign_percent_relative_to_best():
    docs = [_doc("a", "r", "A", score=0.04), _doc("b", "r", "B", score=0.01)]
    assign_percent(docs)
    assert [d.score_pct for d in docs] == [100, 25]


def test_extract_ids():
    ids = extract_ids("https://pubmed.ncbi.nlm.nih.gov/42358277/", "Journal · DOI: 10.1002/pro6.70058 · PMCID: PMC13292151.")
    assert ids == {"doi": "https://doi.org/10.1002/pro6.70058", "pmid": "42358277", "pmcid": "PMC13292151"}
    assert extract_ids("https://onlinelibrary.wiley.com/doi/full/10.1002/abc.123")["doi"] == "https://doi.org/10.1002/abc.123"
    assert extract_ids("https://example.com/news") == {}


def test_merge_dedups_doi_variants_and_tracks_found_via():
    oa = _doc("W1", "openalex_kw", "CAR-T in solid tumors", doi="https://doi.org/10.1/ABC", score=30)
    tv = _doc("https://pmc/x", "tavily_web", "CAR-T in solid tumours: a review", doi="10.1/abc", score=0.9,
              url="https://pmc/x", pmcid="PMC1")
    only_tv = _doc("https://nature/y", "tavily_web", "Something else", score=0.5, url="https://nature/y")
    merged = merge_documents([oa, tv, only_tv], cfg={})
    top = merged[0]
    assert len(merged) == 2
    assert top.id == "W1"
    assert top.metadata["found_via"] == ["openalex_kw", "tavily_web"]
    assert top.metadata["url"] == "https://pmc/x" and top.metadata["pmcid"] == "PMC1"
    assert top.metadata["doi"] == "https://doi.org/10.1/ABC"


def test_tavily_source_builds_paper_docs(monkeypatch):
    seen = {}

    async def fake(query, **kw):
        seen.update(query=query, **kw)
        return {"results": [
            {"url": "https://pubmed.ncbi.nlm.nih.gov/123456/", "title": "  Paper  A ", "content": "DOI: 10.5555/xyz.", "score": 0.8},
            {"url": "", "title": "no url", "content": "", "score": 0.1},
        ]}

    monkeypatch.setattr("openoe.sources.tavily.tavily_search", fake)
    docs = asyncio.run(TavilySource().search(
        query="a AND b", keywords=["natural question"], limit=5, route_name="tavily_web",
        config={"api_key": "k", "include_domains": ["pubmed.ncbi.nlm.nih.gov"]},
    ))
    assert seen["query"] == "natural question" and seen["include_domains"] == ["pubmed.ncbi.nlm.nih.gov"]
    assert len(docs) == 1
    d = docs[0]
    assert d.title == "Paper A" and d.score == 0.8 and d.source == "tavily"
    assert d.metadata["doi"] == "https://doi.org/10.5555/xyz" and d.metadata["pmid"] == "123456"


def test_question_context_and_supporting_sources():
    async def fake(query, **kw):
        if query == "q":
            return {"answer": "bg", "results": [{"url": "https://who.int/a", "title": "WHO note", "content": "x", "score": 0.73}]}
        return {"results": [
            {"url": "https://doi.org/10.1000/abc", "title": "Same paper mirror", "content": "", "score": 0.99},
            {"url": "https://news.com/b", "title": "CAR-T in solid tumors", "content": "", "score": 0.9},
            {"url": "https://trials.gov/c", "title": "Trial NCT1", "content": "trial", "score": 0.6},
        ]}

    cfg = {"enabled": True, "api_key": "k", "supporting": {"top_k": 1, "max_results": 3}}
    ctx = asyncio.run(web_context.question_context("q", cfg, search=fake))
    assert ctx["answer"] == "bg"
    assert ctx["sources"][0]["score_pct"] == 73 and ctx["sources"][0]["domain"] == "who.int"

    papers = [
        {"doc_type": "paper", "title": "CAR-T in solid tumors", "doi": "https://doi.org/10.1000/ABC"},
        {"doc_type": "paper", "title": "Second"},
    ]
    n = asyncio.run(web_context.add_supporting_sources(papers, cfg, search=fake))
    assert n == 1
    assert [s["url"] for s in papers[0]["supporting_sources"]] == ["https://news.com/b", "https://trials.gov/c"]  # same-title commentary kept, DOI mirror dropped
    assert "supporting_sources" not in papers[1]


def test_web_context_failure_is_swallowed():
    async def boom(query, **kw):
        raise RuntimeError("down")

    cfg = {"enabled": True, "api_key": "k"}
    assert asyncio.run(web_context.question_context("q", cfg, search=boom)) == {}
    papers = [{"doc_type": "paper", "title": "T"}]
    assert asyncio.run(web_context.add_supporting_sources(papers, cfg, search=boom)) == 0
    assert asyncio.run(web_context.question_context("q", {"enabled": False}, search=boom)) == {}


def test_answer_context_includes_score_pct_and_web_sources():
    p = _doc("W1", "openalex_kw", "Paper", doi="https://doi.org/10.1/a")
    p.score_pct = 87
    web = [{"title": "Web page", "url": "https://x.org/p", "domain": "x.org", "snippet": "s", "score_pct": 64}]
    ctx, sources = _build_context([p], [], web)
    assert [c["type"] for c in ctx] == ["paper", "web"] and ctx[1]["id"] == 2
    assert sources[0]["score_pct"] == 87 and sources[1]["doc_type"] == "web"
    md = _format_markdown("q", "a [^1][^2]", sources)
    assert "Score 87%" in md and "[x.org](https://x.org/p)" in md and "Score 64%" in md


def test_clean_title():
    assert clean_title("Frontiers | CAR-T cells in solid tumors: engineering") == "CAR-T cells in solid tumors: engineering"
    assert clean_title("Mechanisms of resistance - PubMed") == "Mechanisms of resistance"
    assert clean_title("Plain title") == "Plain title"
