import asyncio

from openoe.medicap import discover as M
from openoe.medicap.openalex import abstract, to_paper


def _work(wid, title, refs=(), year=2020, doi=None, retracted=False):
    return {"id": f"https://openalex.org/{wid}", "display_name": title, "publication_year": year,
            "doi": f"https://doi.org/{doi}" if doi else None, "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/123"},
            "type": "article", "cited_by_count": 5, "is_retracted": retracted,
            "referenced_works": [f"https://openalex.org/{r}" for r in refs],
            "authorships": [{"author": {"display_name": "Ada M Lovelace"}}],
            "primary_location": {"source": {"display_name": "J Test", "type": "journal"}},
            "abstract_inverted_index": {"Hello": [0], "world.": [1]}, "primary_topic": {"display_name": "CAR-T"}}


def test_to_paper_and_abstract():
    p = to_paper(_work("W1", "A title", refs=["W2"], doi="10.1234/X", retracted=True))
    assert p["id"] == "W1" and p["doi"] == "10.1234/X" and p["pmid"] == "123"
    assert p["au"] == "Lovelace AM" and p["flag"] == "retracted" and p["refs"] == ["W2"]
    assert abstract({"b": [1], "a": [0]}) == "a b"


def test_combine_ignores_missing_judge():
    assert M.combine(1.0, 0.0, None) == 40 / 75
    assert M.combine(1.0, 1.0, 1.0) == 1.0


class FakeOA:
    def __init__(self, works, hits):
        self.works, self.hits = works, hits

    async def work(self, ref):
        return self.works.get(ref)

    async def search(self, q, n=25, filt=""):
        return self.hits


def test_resolve_line_kinds():
    oa = FakeOA({"doi:10.1234/abc": _work("W1", "Known paper"), "pmid:12345678": _work("W2", "Other")},
                [_work("W3", "CAR T cells in solid tumors"), _work("W4", "CAR T cells in solid tumours review"),
                 _work("W5", "Unrelated topic entirely")])
    run = lambda raw: asyncio.run(M.resolve_line(oa, raw))  # noqa: E731
    assert run("https://doi.org/10.1234/abc")["p"]["id"] == "W1"
    assert run("PMID: 12345678")["p"]["id"] == "W2"
    assert run("10.9999/missing")["st"] == "miss"
    assert run("CAR T cells in solid tumors")["st"] == "ok"           # exact title
    amb = run("CAR T solid tumors cells review overview")
    assert amb["st"] in ("amb", "ok") and "W5" not in str(amb)


def test_graph_and_loo_rank_hidden_seed():
    # seeds S1=W1, S2=W2; W2 cites W1 and both cite W9, so hiding either still finds it via the graph.
    pool = {k: to_paper(w) for k, w in {
        "W1": _work("W1", "alpha beta", refs=["W9"]), "W2": _work("W2", "alpha gamma", refs=["W1", "W9"]),
        "W9": _work("W9", "alpha delta")}.items()}
    seeds = ["W1", "W2"]
    links = {"W1": {"cited_by": {1}}, "W9": {"cited_by": {0, 1}}}
    vecs = M._tfidf({k: p["t"] for k, p in pool.items()})
    sem = lambda use: {k: 1.0 for k in vecs}  # noqa: E731
    rows = M._loo(pool, seeds, links, ["W9"], [], sem, {})
    assert rows[0]["seed"] == 1 and rows[0]["graph"] is not None
    assert rows[0]["keyword"] is None and rows[0]["web"] is None


def test_figure_matching_only_accepts_the_papers_own_images():
    from openoe.medicap.figures import figures, match_keys, pick_image
    keys = match_keys("10.1038/s41571-023-00832-4", "PMC4800826")
    imgs = ["https://media.springernature.com/x/art%3A10.1038%2Fs41571-023-00832-4/Fig1.png",
            "https://cdn.ncbi.nlm.nih.gov/pmc/blobs/b8ce/4800826/f4.jpg"]
    assert pick_image(["https://other.org/unrelated.png", imgs[0]], keys)["url"] == imgs[0]
    assert pick_image([imgs[1]], keys)["url"] == imgs[1]
    assert pick_image(["https://other.org/unrelated.png"], keys) is None
    assert pick_image(["http://insecure.org/s41571-023-00832-4.png"], keys) is None

    async def fake_search(q, **kw):
        assert kw["include_images"]
        return {"images": [{"url": "https://x.org/" + q.replace(" ", "_") + ".png"}]}
    cfg = {"stages": {"web_context": {"api_key": "k"}}}
    res = asyncio.run(figures([{"id": "Wfig1", "t": "no doi"}], cfg, search=fake_search))
    assert res == {"Wfig1": None}   # no identifiers → nothing provable → no image


def test_image_proxy_only_serves_selected_urls():
    from openoe.medicap.figures import fetch_image
    assert asyncio.run(fetch_image("http://169.254.169.254/latest/meta-data")) is None
