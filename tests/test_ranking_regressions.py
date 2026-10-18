"""Offline regressions for reviewed ranking/evidence failures (no API credentials)."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from openoe.discovery.followup import answer_followup
from openoe.discovery.models import Paper
from openoe.discovery.rank import _seed_link_count, rank_candidates
from openoe.llm.client import LLMError
from openoe.llm.prompts import PromptManager
from openoe.medicap import discover as legacy


class StubLLM:
    def __init__(self, responses=None, *, failure=None, embedding_failure=None):
        self.responses = responses or {}
        self.failure = failure
        self.embedding_failure = embedding_failure
        self.calls = []

    async def embed(self, texts, model):
        if self.embedding_failure:
            raise self.embedding_failure
        return [[1.0, 0.0] for _ in texts]

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.failure:
            raise self.failure
        response = self.responses.get(kwargs["step"], "[]")
        if callable(response):
            response = response(kwargs)
        return SimpleNamespace(content=response)


def candidate(wid, title, *, retracted=False):
    return {"paper": Paper(openalex_id=wid, title=title, is_retracted=retracted),
            "signals": [{"type": "search"}]}


def rank(candidates, llm, cfg=None):
    return asyncio.run(rank_candidates("query", [], candidates, llm, PromptManager(), {
        "embedding_model": "embed", "small_model": "small", "strong_model": "strong",
        **(cfg or {}),
    }))


def judgment(relevance=8):
    return json.dumps({"relevance": relevance, "why": "The title addresses the research question.",
                       "quote": "", "related_seeds": [], "study_design": "unclear"})


def test_rank_gates_explicit_irrelevance_and_uses_configured_judge_model():
    llm = StubLLM({"discovery_judge": lambda kw: judgment(0 if "Title: irrelevant" in kw["prompt"] else 8)})
    results, n, errors, timings, assessments = rank(
        [candidate("W1", "irrelevant"), candidate("W2", "relevant")], llm,
    )
    assert [p["openalex_id"] for p in results] == ["W2"]
    assert results[0]["assessment_status"] == "assessed"
    assert all(c["model"] == "strong" for c in llm.calls)
    assert {a["openalex_id"]: a["assessment_status"] for a in assessments} == {
        "W1": "excluded", "W2": "assessed",
    }


def test_explicit_judge_model_takes_precedence():
    llm = StubLLM({"discovery_judge": judgment()})
    rank([candidate("W1", "relevant")], llm, {"judge_model": "explicit"})
    assert llm.calls[0]["model"] == "explicit"


def test_all_judge_outages_have_unassessed_metadata_and_no_ordinary_recommendations():
    llm = StubLLM(failure=RuntimeError("temporary test outage"))
    results, n, errors, timings, assessments = rank([candidate("W1", "candidate")], llm)
    assert results == [] and errors
    assert assessments[0]["assessment_status"] == "unassessed"
    assert "relevance" not in assessments[0]


def test_embedding_failure_stops_the_ranking_cascade():
    llm = StubLLM(embedding_failure=RuntimeError("embedding test outage"))
    results, n, errors, timings, assessments = rank([candidate("W1", "candidate")], llm)
    assert results == [] and n == 0 and errors
    assert llm.calls == []
    assert assessments[0]["assessment_status"] == "unassessed"


def test_systemic_provider_failure_propagates_without_a_zero_relevance_fallback():
    error = LLMError("model_unavailable", "Choose an available model.")
    llm = StubLLM(failure=error)
    with pytest.raises(LLMError) as raised:
        rank([candidate("W1", "candidate")], llm)
    assert raised.value is error


def test_individual_invalid_response_is_unassessed_without_aborting_other_candidates():
    error = LLMError("invalid_response", "Malformed response.", systemic=False)
    llm = StubLLM(failure=error)
    results, n, errors, timings, assessments = rank([candidate("W1", "candidate")], llm)
    assert results == [] and "invalid_response" in errors[0]
    assert assessments[0]["assessment_status"] == "unassessed"


def test_false_supporting_quote_is_not_presented_as_an_explained_recommendation():
    response = json.dumps({"relevance": 9, "why": "Supports the question.",
                           "quote": "Not in the abstract.", "related_seeds": []})
    llm = StubLLM({"discovery_judge": response})
    results, n, errors, timings, assessments = rank([candidate("W1", "candidate")], llm)
    assert results == [] and errors
    assert assessments[0]["assessment_status"] == "unassessed"


def test_related_recommendation_is_not_counted_as_a_citation_link():
    assert _seed_link_count([{"type": "related", "seeds": [0]}]) == 0
    assert _seed_link_count([{"type": "cites_seeds", "seeds": [0]}]) == 1


def test_retracted_papers_do_not_enter_the_ranked_evidence_cohort():
    llm = StubLLM({"discovery_judge": judgment()})
    results, n, errors, timings, assessments = rank([candidate("W1", "retracted", retracted=True)], llm)
    assert results == [] and n == 0 and llm.calls == []
    assert assessments[0]["assessment_status"] == "excluded"


def work(wid, title, refs=()):
    return {"id": f"https://openalex.org/{wid}", "display_name": title,
            "referenced_works": [f"https://openalex.org/{r}" for r in refs]}


class StubOA:
    limited = False

    def __init__(self, works):
        self.works = works

    async def batch(self, ids):
        return [self.works[i] for i in ids if i in self.works]

    async def citing(self, seed, n):
        return []

    async def search(self, question, n):
        return []


def test_empty_retrieval_does_not_publish_leaked_loo_recall(monkeypatch):
    monkeypatch.setattr(legacy, "_client", lambda *a: StubOA({
        "W1": work("W1", "alphaseed"), "W2": work("W2", "betaseed"),
    }))
    result = asyncio.run(legacy.discover("", ["W1", "W2"], {}))
    assert result["stats"]["returned"] == 0
    assert result["loo"] == []
    assert result["loo_status"]["status"] == "unavailable"
    assert result["loo_status"]["reason"]


def test_legacy_judge_budget_cannot_promote_an_unjudged_candidate(monkeypatch):
    ids = [f"W{i}" for i in range(100, 131)]
    works = {"W1": work("W1", "alpha beta", ids), **{i: work(i, "alpha beta") for i in ids}}
    monkeypatch.setattr(legacy, "_client", lambda *a: StubOA(works))
    llm = StubLLM({"medicap_judge": "0"})
    cfg = {"llm": {"default_provider": "mock", "providers": {
        "mock": {"base_url": "mock://unused", "api_key": "not-a-real-key"},
    }}}
    result = asyncio.run(legacy.discover("", ["W1"], cfg, llm=llm))
    assert result["papers"] == []
    states = {a["id"]: a["assessment_status"] for a in result["assessments"]}
    assert states["W130"] == "unassessed"
    assert sum(v == "excluded" for v in states.values()) == 30


def test_followup_filters_retracted_and_rejected_without_renumbering_survivors():
    llm = StubLLM({"discovery_followup": "The trial addresses this question [^3].",
                   "discovery_suggestions": "[]"})
    papers = [
        {"title": "Retracted source", "openalex_id": "W1", "is_retracted": True},
        {"title": "Rejected source", "openalex_id": "W2", "decision": "rejected"},
        {"title": "Eligible source", "openalex_id": "W3", "abstract": "A trial result."},
    ]
    result = asyncio.run(answer_followup(llm, PromptManager(), {"strong_model": "strong"},
                                        "question", [], papers))
    prompt = llm.calls[0]["prompt"]
    assert "Retracted source" not in prompt and "Rejected source" not in prompt
    assert '"n": 3' in prompt
    assert [c["n"] for c in result["citations"]] == [3]


def test_no_eligible_followup_evidence_makes_no_model_call():
    llm = StubLLM()
    result = asyncio.run(answer_followup(llm, PromptManager(), {"strong_model": "strong"},
                                        "question", [], [{"title": "Bad", "is_retracted": True}]))
    assert llm.calls == [] and result["citations"] == []
    assert result["assessment_status"] == "no_eligible_evidence"


def test_followup_accepts_a_footnote_immediately_after_sentence_punctuation():
    llm = StubLLM({"discovery_followup": "A cited sentence. [^1]", "discovery_suggestions": "[]"})
    result = asyncio.run(answer_followup(llm, PromptManager(), {"strong_model": "strong"},
                                        "question", [], [{"title": "Source", "abstract": "A study."}]))
    assert result["answer"] == "A cited sentence [^1]."
    assert [c["n"] for c in result["citations"]] == [1]


@pytest.mark.parametrize("response", [
    "Unsupported claim [^999].",
    "Unsupported claim [1].",
    "Unsupported claim with no citation.",
])
def test_followup_removes_unsupported_passages_and_unknown_citations(response):
    llm = StubLLM({"discovery_followup": response, "discovery_suggestions": "[]"})
    result = asyncio.run(answer_followup(llm, PromptManager(), {"strong_model": "strong"},
                                        "question", [], [{"title": "Source", "abstract": "A study."}]))
    assert "Unsupported claim" not in result["answer"]
    assert "[^999]" not in result["answer"]
    assert result["citations"] == []
    assert result["assessment_status"] == "insufficient_cited_evidence"


def test_followup_keeps_valid_passages_and_discards_unknown_citation_passage():
    llm = StubLLM({"discovery_followup": "Supported claim [^1]. Unsupported claim [^999].",
                   "discovery_suggestions": "[]"})
    result = asyncio.run(answer_followup(llm, PromptManager(), {"strong_model": "strong"},
                                        "question", [], [{"title": "Source", "abstract": "A study."}]))
    assert result["answer"] == "Supported claim [^1]."
    assert [c["n"] for c in result["citations"]] == [1]
