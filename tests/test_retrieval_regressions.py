"""Provider-free regression coverage for seed identity and candidate fusion."""
import asyncio
from types import SimpleNamespace

import httpx
import pytest

from openoe.discovery.candidates import gather_candidates
from openoe.discovery.models import Paper
from openoe.discovery.seeds import resolve_seeds
from openoe.llm.client import LLMError


_SIGNALS = ("cited_by_seeds", "cites_seeds", "related", "s2_recommendation", "search", "web")


def _config(*enabled):
    return {
        "small_model": "stub",
        "signals": {key: {"enabled": key in enabled} for key in _SIGNALS},
    }


def _work(wid, title="A paper", **fields):
    return {"id": f"https://openalex.org/{wid}", "title": title, **fields}


class _Client:
    rate_limited = False

    def __init__(self):
        self.errors = []


class _LLM:
    async def chat(self, **kwargs):
        return SimpleNamespace(content='["query"]')


class _Prompts:
    def render(self, *args, **kwargs):
        return "prompt"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    async def forbidden(*args, **kwargs):
        raise AssertionError("Regression tests must not use live providers")

    async def no_pubmed_match(*args, **kwargs):
        return None

    monkeypatch.setattr(httpx.AsyncClient, "request", forbidden)
    monkeypatch.setattr("openoe.discovery.seeds.lookup_pubmed", no_pubmed_match)


def test_title_seed_does_not_accept_unrelated_search_hit():
    class Client(_Client):
        async def search(self, query, n):
            return [_work("W99", "An unrelated publication")]

    papers, unresolved = asyncio.run(resolve_seeds(["Immunotherapy in melanoma"], Client()))
    assert papers == []
    assert unresolved == ["Immunotherapy in melanoma"]


def test_title_seed_selects_matching_title_beyond_first_result():
    class Client(_Client):
        async def search(self, query, n):
            return [
                _work("W98", "A related but different publication"),
                _work("W99", "Immunotherapy in Melanoma: a trial."),
            ]

    papers, unresolved = asyncio.run(resolve_seeds(["immunotherapy in melanoma — a trial"], Client()))
    assert [paper.openalex_id for paper in papers] == ["W99"]
    assert unresolved == []


def test_ambiguous_identical_titles_remain_unresolved():
    class Client(_Client):
        async def search(self, query, n):
            return [
                _work("W98", "Editorial", doi="10.1234/one"),
                _work("W99", "Editorial", doi="10.1234/two"),
            ]

    papers, unresolved = asyncio.run(resolve_seeds(["Editorial"], Client()))
    assert papers == []
    assert unresolved == ["Editorial"]


def test_pubmed_title_fallback_cannot_substitute_an_unrelated_title(monkeypatch):
    class Client(_Client):
        async def search(self, query, n):
            return []

    async def unrelated(*args, **kwargs):
        return Paper(openalex_id="pmid:99", pmid="99", title="An unrelated publication")

    monkeypatch.setattr("openoe.discovery.seeds.lookup_pubmed", unrelated)
    papers, unresolved = asyncio.run(resolve_seeds(["Immunotherapy in melanoma"], Client()))
    assert papers == []
    assert unresolved == ["Immunotherapy in melanoma"]


def test_newly_kept_seed_after_ten_originals_is_resolved():
    class Client(_Client):
        async def get_work(self, key):
            return _work(key, f"Seed {key}")

    supplied = [f"W{i}" for i in range(1, 12)]
    papers, unresolved = asyncio.run(resolve_seeds(supplied, Client()))
    assert [paper.openalex_id for paper in papers] == supplied
    assert unresolved == []


def test_global_budget_keeps_multiseed_candidate_regardless_of_response_order():
    class Client(_Client):
        def __init__(self, references_slow):
            super().__init__()
            self.references_slow = references_slow

        async def get_works(self, ids):
            await asyncio.sleep(0.001 if self.references_slow else 0)
            return [_work("W999", "Cited by every seed")]

        async def citing(self, wid):
            await asyncio.sleep(0 if self.references_slow else 0.001)
            seed_index = int(wid[1:]) - 1
            return [_work(f"W{100 + seed_index * 25 + j}") for j in range(25)]

    async def run():
        seeds = [Paper(openalex_id=f"W{i}", title=f"Seed {i}", referenced_works=["W999"]) for i in range(1, 11)]
        results = []
        for slow in (True, False):
            candidates, _, _ = await gather_candidates(
                "question", seeds, [], Client(slow), _LLM(), _Prompts(),
                _config("cited_by_seeds", "cites_seeds"),
            )
            results.append([candidate["paper"].openalex_id for candidate in candidates])
        return results

    slow, fast = asyncio.run(run())
    assert "W999" in slow
    assert len(slow) == 200
    assert slow == fast


@pytest.mark.parametrize("disliked", ["pmid:123", "123", "https://pubmed.ncbi.nlm.nih.gov/123/"])
def test_rejected_pmid_is_excluded_under_its_openalex_identity(disliked):
    class Client(_Client):
        async def search(self, query, n):
            return [_work("W99", ids={"pmid": "https://pubmed.ncbi.nlm.nih.gov/123/"})]

    candidates, _, _ = asyncio.run(gather_candidates(
        "question", [], [disliked], Client(), _LLM(), _Prompts(), _config("search")
    ))
    assert candidates == []


def test_pubmed_and_openalex_metadata_fuse_independently_of_response_order(monkeypatch):
    class Client(_Client):
        def __init__(self, openalex_slow):
            super().__init__()
            self.openalex_slow = openalex_slow

        async def get_works(self, ids):
            await asyncio.sleep(0.001 if self.openalex_slow else 0)
            return [_work(
                "W99", "Same paper", doi="https://doi.org/10.1234/full",
                ids={"pmid": "https://pubmed.ncbi.nlm.nih.gov/123/"},
                cited_by_count=300, open_access={"is_oa": True}, is_retracted=True,
            )]

        async def search(self, query, n):
            await asyncio.sleep(0 if self.openalex_slow else 0.001)
            return []

    async def pubmed(*args, **kwargs):
        return [Paper(openalex_id="pmid:123", pmid="123", title="Same paper", abstract="Full PubMed abstract")]

    monkeypatch.setattr("openoe.discovery.candidates.search_pubmed", pubmed)

    async def run():
        results = []
        for slow in (True, False):
            candidates, _, _ = await gather_candidates(
                "question", [Paper(openalex_id="W1", title="Seed", referenced_works=["W99"])], [],
                Client(slow), _LLM(), _Prompts(), _config("cited_by_seeds", "search"),
            )
            assert len(candidates) == 1
            results.append(candidates[0])
        return results

    slow, fast = asyncio.run(run())
    assert slow["paper"].to_api_dict() == fast["paper"].to_api_dict()
    assert slow["paper"].openalex_id == "W99"
    assert slow["paper"].doi == "10.1234/full"
    assert slow["paper"].abstract == "Full PubMed abstract"
    assert slow["paper"].cited_by_count == 300
    assert slow["paper"].is_oa is True
    assert slow["paper"].is_retracted is True
    assert slow["signals"] == fast["signals"]


def test_transitive_aliases_merge_before_seed_and_disliked_exclusion(monkeypatch):
    class Client(_Client):
        async def get_works(self, ids):
            return [_work("W99", "Same paper", doi="10.1234/full")]

        async def search(self, query, n):
            return [
                _work("W100", "Same paper", ids={"pmid": "https://pubmed.ncbi.nlm.nih.gov/123/"}),
                _work("W101", "Same paper", doi="10.1234/full", ids={"pmid": "https://pubmed.ncbi.nlm.nih.gov/123/"}),
            ]

    candidates, _, _ = asyncio.run(gather_candidates(
        "question", [Paper(openalex_id="W1", title="Seed", referenced_works=["W99"])], ["pmid:123"],
        Client(), _LLM(), _Prompts(), _config("cited_by_seeds", "search"),
    ))
    assert candidates == []


def test_tavily_uses_configured_key_and_keeps_successful_scholarly_branch(monkeypatch):
    class Client(_Client):
        async def by_dois(self, dois):
            return [_work("W99", "Discovered paper", doi="10.1234/found")]

    async def post(self, url, *, headers, json):
        assert headers["Authorization"] == "Bearer local-test-key"
        request = httpx.Request("POST", url)
        if "include_domains" not in json:
            return httpx.Response(503, request=request)
        return httpx.Response(200, request=request, json={"results": [
            {"url": "https://doi.org/10.1234/found", "title": "Discovered paper", "content": "A study"},
        ]})

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    cfg = {**_config("web"), "tavily_api_key": "local-test-key"}
    candidates, context, errors = asyncio.run(gather_candidates(
        "question", [], [], Client(), _LLM(), _Prompts(), cfg
    ))
    assert [candidate["paper"].openalex_id for candidate in candidates] == ["W99"]
    assert context == []
    assert len(errors) == 1 and "Tavily" in errors[0]


def test_missing_tavily_key_is_reported_once(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    candidates, context, errors = asyncio.run(gather_candidates(
        "question", [], [], _Client(), _LLM(), _Prompts(), _config("web")
    ))
    assert candidates == [] and context == []
    assert len(errors) == 1
    assert "Tavily" in errors[0] and "not configured" in errors[0]


def test_systemic_query_failure_propagates_and_cancels_other_retrieval():
    async def run():
        started = asyncio.Event()
        cancelled = asyncio.Event()

        class Client(_Client):
            async def get_works(self, ids):
                started.set()
                try:
                    await asyncio.sleep(0.03)
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
                return []

            async def search(self, query, n):
                return [_work("W99")]

        class LLM(_LLM):
            async def chat(self, **kwargs):
                await started.wait()
                raise LLMError("authentication", "Provider rejected the configured key.", status_code=401)

        with pytest.raises(LLMError, match="configured key"):
            await gather_candidates(
                "question", [Paper(openalex_id="W1", title="Seed", referenced_works=["W99"])], [],
                Client(), LLM(), _Prompts(), _config("cited_by_seeds", "search"),
            )
        assert cancelled.is_set()

    asyncio.run(run())


def test_non_systemic_query_response_failure_can_fall_back_to_literal_search():
    class Client(_Client):
        async def search(self, query, n):
            return [_work("W99", "Relevant candidate")]

    class LLM(_LLM):
        async def chat(self, **kwargs):
            raise LLMError("invalid_response", "Invalid query response.", systemic=False)

    candidates, _, errors = asyncio.run(gather_candidates(
        "question", [], [], Client(), LLM(), _Prompts(), _config("search"),
    ))
    assert [candidate["paper"].openalex_id for candidate in candidates] == ["W99"]
    assert candidates[0]["signals"] == [{"type": "search", "query": "question"}]
    assert len(errors) == 1 and "query generation" in errors[0]


def test_pubmed_only_seed_reports_unavailable_graph_without_invalid_openalex_lookup():
    class Client(_Client):
        def __init__(self):
            super().__init__()
            self.citing_ids = []

        async def citing(self, wid):
            self.citing_ids.append(wid)
            return [_work("W99")]

    client = Client()
    candidates, _, errors = asyncio.run(gather_candidates(
        "question", [
            Paper(openalex_id="pmid:123", pmid="123", title="PubMed-only seed"),
            Paper(openalex_id="W2", title="OpenAlex seed"),
        ], [], client, _LLM(), _Prompts(), _config("cites_seeds"),
    ))
    assert client.citing_ids == ["W2"]
    assert candidates[0]["signals"] == [{"type": "cites_seeds", "seeds": [1]}]
    assert len(errors) == 1
    assert "Citation graph unavailable" in errors[0]
