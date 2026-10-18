import asyncio

import httpx

import openoe.llm.client as llm_client_module
from openoe.llm.client import LLMClient
from openoe.discovery.rank import (
    _BARE_CITATION_RE,
    calculate_final_score,
    calculate_pre_score,
    parse_judge_json,
    reconcile_study_design,
    score_to_pct,
    verify_quote,
)
from openoe.discovery.seeds import parse_seed
from openoe.discovery.followup import _BARE_CITATION_RE as _FOLLOWUP_BARE_CITATION_RE
import pytest
from openoe.discovery.models import DiscoverRequest, paper_from_work
from openoe.discovery.openalex import SELECT_FIELDS, _describe_error
from openoe.discovery.pubmed import parse_pubmed_abstracts, parse_pubmed_articles
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
    for seed in ("PMID 20525992", "PMID: 20525992", "pmid:20525992", "PMID20525992"):
        assert parse_seed(seed) == ("pmid", "20525992")


def test_parse_seed_openalex_and_title():
    assert parse_seed("W123456789") == ("openalex", "W123456789")
    assert parse_seed("openalex:W123") == ("openalex", "W123")
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


def test_pubmed_abstract_parser_joins_labelled_parts_and_inline_tags():
    xml = """\
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>12345</PMID>
          <Article>
            <Abstract>
              <AbstractText Label="BACKGROUND">We observed <i>melanoma</i> response.</AbstractText>
              <AbstractText Label="CONCLUSION">Survival improved.</AbstractText>
            </Abstract>
          </Article>
        </MedlineCitation>
      </PubmedArticle>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>67890</PMID>
          <Article><ArticleTitle>No abstract available</ArticleTitle></Article>
        </MedlineCitation>
      </PubmedArticle>
    </PubmedArticleSet>
    """

    assert parse_pubmed_abstracts(xml) == {
        "12345": "BACKGROUND: We observed melanoma response. CONCLUSION: Survival improved."
    }


def test_judge_study_design_is_normalized_and_defaults_to_unclear():
    assert parse_judge_json('{"study_design":"rct"}')["study_design"] == "RCT"
    assert parse_judge_json('{"study_design":"banana"}')["study_design"] == "unclear"
    assert parse_judge_json('{"relevance":5}')["study_design"] == "unclear"


def test_reconcile_study_design_corrects_randomized_trial_labels():
    assert reconcile_study_design(
        "non-randomized trial",
        "LBA47 … Primary analysis of the randomized phase II NIBIT-ML1 study",
        "",
    ) == "RCT"
    assert reconcile_study_design(
        "non-randomized trial", "A non-randomized phase II study", "",
    ) == "non-randomized trial"
    assert reconcile_study_design(
        "cohort", "A randomized cohort study", "",
    ) == "cohort"


def test_score_pct_is_rounded_and_clamped():
    assert score_to_pct(0.0) == 1
    assert score_to_pct(0.834) == 83
    assert score_to_pct(1.2) == 100


def test_paper_from_work_maps_retraction_and_primary_topic():
    paper = paper_from_work({
        "id": "https://openalex.org/W123",
        "title": "A study",
        "is_retracted": True,
        "primary_topic": {
            "display_name": "Melanoma",
            "field": {"display_name": "Medicine"},
        },
    })

    assert paper is not None
    assert paper.is_retracted is True
    assert paper.topic == "Melanoma"
    assert paper.field == "Medicine"
    assert "is_retracted" in SELECT_FIELDS
    assert "primary_topic" in SELECT_FIELDS


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


def test_discover_request_accepts_question_only_or_seeds_only():
    assert DiscoverRequest(question="  melanoma immunotherapy  ").seeds == []
    assert DiscoverRequest(seeds=["10.1056/NEJMoa1003466", " "]).seeds == ["10.1056/NEJMoa1003466"]
    with pytest.raises(ValueError):
        DiscoverRequest(question="  ", seeds=["", "  "])


def test_pubmed_articles_become_papers_with_doi_year_and_links():
    xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID><Article>
      <Journal><JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue><Title>Lancet</Title></Journal>
      <ArticleTitle>Adjuvant <i>pembrolizumab</i> in melanoma.</ArticleTitle>
      <ELocationID EIdType="doi">10.1000/abc</ELocationID>
      <Abstract><AbstractText Label="RESULTS">RFS improved.</AbstractText></Abstract>
      <AuthorList><Author><LastName>Eggermont</LastName><Initials>A</Initials></Author></AuthorList>
      <PublicationTypeList><PublicationType>Randomized Controlled Trial</PublicationType></PublicationTypeList>
    </Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""
    papers = parse_pubmed_articles(xml)
    assert len(papers) == 1
    paper = papers[0]
    assert paper.openalex_id == "pmid:123" and paper.pmid == "123" and paper.doi == "10.1000/abc"
    assert paper.title == "Adjuvant pembrolizumab in melanoma." and paper.year == 2021 and paper.venue == "Lancet"
    assert paper.abstract == "RESULTS: RFS improved." and paper.authors == ["Eggermont A"]
    assert paper.links["pubmed"] == "https://pubmed.ncbi.nlm.nih.gov/123/" and paper.links["doi"] == "https://doi.org/10.1000/abc"
