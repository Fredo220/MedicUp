import asyncio

import httpx

import openoe.llm.client as llm_client_module
from openoe.llm.client import LLMClient
from openoe.discovery.rank import (
    _BARE_CITATION_RE,
    calculate_final_score,
    calculate_pre_score,
    parse_judge_json,
    verify_quote,
)
from openoe.discovery.seeds import parse_seed
from openoe.discovery.followup import _BARE_CITATION_RE as _FOLLOWUP_BARE_CITATION_RE
from openoe.discovery.openalex import _describe_error
from openoe.pipeline.answer import _BARE_NUMERIC_CITATION_RE


def test_parse_seed_doi_and_doi_url():
    assert parse_seed("10.1056/NEJMoa1003466") == ("doi", "10.1056/NEJMoa1003466")
    assert parse_seed("https://doi.org/10.1056/NEJMoa1003466.") == (
        "doi",
        "10.1056/NEJMoa1003466",
    )


def test_parse_seed_pmid_and_pubmed_url():
    assert parse_seed("12345678") == ("pmid", "12345678")
    assert parse_seed("https://pubmed.ncbi.nlm.nih.gov/12345678/") == (
        "pmid",
        "12345678",
    )


def test_parse_seed_openalex_and_title():
    assert parse_seed("W123456789") == ("openalex", "W123456789")
    assert parse_seed("https://openalex.org/W123456789") == ("openalex", "W123456789")
    assert parse_seed("A study title") == ("title", "A study title")


def test_quote_verification_normalizes_case_whitespace_and_quotes():
    assert verify_quote('"A useful result was observed."', "The study states: a useful\nresult was observed.")
    assert not verify_quote("", "Some abstract text.")
    assert not verify_quote("A result was observed.", "No such sentence.")


def test_pre_score_and_final_ordering_for_toy_candidates():
    high_pre = calculate_pre_score(
        similarity=1.0,
        min_similarity=0.0,
        max_similarity=1.0,
        seed_links=2,
        n_seeds=2,
        n_signal_types=2,
        has_s2_recommendation=True,
    )
    low_pre = calculate_pre_score(
        similarity=0.0,
        min_similarity=0.0,
        max_similarity=1.0,
        seed_links=0,
        n_seeds=2,
        n_signal_types=1,
        has_s2_recommendation=False,
    )
    assert high_pre > low_pre
    assert calculate_final_score(2, high_pre) > calculate_final_score(7, low_pre)


def test_bare_numeric_citation_cleanup_keeps_footnotes():
    text = "[13] [1, 2] [^3]"
    assert _BARE_NUMERIC_CITATION_RE.sub("", text).split() == ["[^3]"]
    assert _BARE_CITATION_RE.sub("", text).split() == ["[^3]"]
    assert _FOLLOWUP_BARE_CITATION_RE.sub("", text).split() == ["[^3]"]


def test_judge_json_parser_ignores_surrounding_prose():
    parsed = parse_judge_json(
        'Here is the assessment:\n{"relevance": 8, "why": "Relevant.", '
        '"quote": "An abstract sentence.", "related_seeds": ["S1"]}\nDone.'
    )
    assert parsed["relevance"] == 8
    assert parsed["related_seeds"] == ["S1"]


def test_openalex_error_description_does_not_expose_request_query():
    response = httpx.Response(
        429,
        request=httpx.Request("GET", "https://api.openalex.org/works?api_key=private"),
    )
    error = httpx.HTTPStatusError("rate limited", request=response.request, response=response)
    assert _describe_error(error) == "HTTP 429"
    assert "private" not in _describe_error(error)


def test_embedding_batches_preserve_order_with_concurrent_http_posts(monkeypatch):
    active_posts = 0
    max_active_posts = 0

    class StubResponse:
        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            pass

        def json(self):
            return {"data": self.data}

    class StubAsyncClient:
        def __init__(self, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def post(self, url, *, headers, json):
            nonlocal active_posts, max_active_posts
            active_posts += 1
            max_active_posts = max(max_active_posts, active_posts)
            if json["input"][0] == "text-0":
                await asyncio.sleep(0.04)
            else:
                await asyncio.sleep(0.005)
            data = [
                {
                    "index": index,
                    "embedding": [float(text.split("-")[1])],
                }
                for index, text in reversed(list(enumerate(json["input"])))
            ]
            active_posts -= 1
            return StubResponse(data)

    monkeypatch.setattr(llm_client_module.httpx, "AsyncClient", StubAsyncClient)
    llm = LLMClient({
        "default_provider": "test",
        "providers": {"test": {"base_url": "http://stub/v1", "api_key": "test-key"}},
        "retry": {"max_retries": 1, "backoff": []},
    })
    embeddings = asyncio.run(llm.embed([f"text-{index}" for index in range(129)], "stub-model"))

    assert embeddings == [[float(index)] for index in range(129)]
    assert 2 <= max_active_posts <= 4


def test_study_type_labels_only_what_title_or_abstract_states():
    from openoe.discovery.models import study_type
    assert study_type("SGLT2 inhibitors in heart failure: a meta-analysis of five trials", "randomized") == "Meta-analysis"
    assert study_type("2023 Focused Update of the 2021 ESC Guidelines for heart failure") == "Guideline"
    assert study_type("Empagliflozin in Heart Failure", "In this double-blind trial, we randomly assigned... placebo-controlled") == "RCT"
    assert study_type("Dapagliflozin for heart failure according to body mass index: the DELIVER trial", "randomized") == "Trial sub-analysis"
    assert study_type("Rationale and design of the DELIVER trial", "randomized") == "Trial design"
    assert study_type("A retrospective cohort of DOAC users", "") == "Observational"
    assert study_type("EMPEROR-Preserved: a promise fulfilled", "", "editorial") == "Editorial"
    assert study_type("Heart failure therapy", "", "review") == "Review"
    assert study_type("Heart failure therapy", "") is None
    assert study_type("Rationale for and Design of the EMPEROR-Preserved Trial", "randomized") == "Trial design"
    assert study_type("Dapagliflozin and empagliflozin in HFpEF: pooled data from DELIVER and EMPEROR-Preserved", "randomized") == "Pooled analysis"


def test_pick_title_match_prefers_primary_report_and_rejects_nonsense():
    from openoe.discovery.models import Paper
    from openoe.discovery.seeds import pick_title_match
    primary = Paper(openalex_id="W1", title="Empagliflozin in Heart Failure with a Preserved Ejection Fraction",
                    abstract="EMPEROR-Preserved trial ...", cited_by_count=3000)
    editorial = Paper(openalex_id="W2", title="EMPEROR-Preserved: A promise fulfilled", cited_by_count=12)
    assert pick_title_match("EMPEROR-Preserved", [editorial, primary]).openalex_id == "W1"
    assert pick_title_match("asdf qwer", [editorial, primary]) is None


def test_plain_seed_labels_rewrites_internal_labels():
    from openoe.discovery.rank import plain_seed_labels
    assert plain_seed_labels("Same trial as S1 and the seed paper S2.") == "Same trial as your paper 1 and your paper 2."
    assert plain_seed_labels("SGLT2 inhibitors") == "SGLT2 inhibitors"
    assert plain_seed_labels("Extends the seed papers to HFrEF.") == "Extends your papers to HFrEF."


def test_openalex_source_raises_when_database_unavailable(monkeypatch):
    import asyncio
    import httpx
    import pytest
    from openoe.sources import openalex as src

    class Down:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, *a, **k): raise httpx.HTTPError("429 Too Many Requests")

    async def no_sleep(_): return None
    monkeypatch.setattr(src.httpx, "AsyncClient", Down)
    monkeypatch.setattr(src.asyncio, "sleep", no_sleep)
    with pytest.raises(RuntimeError, match="OpenAlex unavailable"):
        asyncio.run(src.OpenAlexSource().search(query="sglt2 hfpef", keywords=["sglt2 heart failure"]))
