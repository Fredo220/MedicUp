"""Medicap discovery: resolve seeds, expand via citation graph + keyword + Tavily, score, evaluate.

Score components (each 0..1, best candidate in the run = 1):
  g  graph     — direct citation links to seeds + shared references with seeds (bibliographic coupling)
  s  semantic  — TF-IDF cosine of title+abstract vs the seed/question centroid
  l  judge     — LLM relevance 0–100 (only when an LLM key is configured, else ``None``)
The UI combines them with user weights and shows everything as 1–100 %.
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set

import httpx

from openoe.medicap.openalex import OpenAlex, short_id, to_paper
from openoe.pipeline.web_context import add_supporting_sources, question_context
from openoe.sources.tavily import clean_title, extract_ids, tavily_search

log = logging.getLogger(__name__)

RX_DOI = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)?(10\.\d{4,9}/\S+)$", re.I)
RX_PMID = re.compile(r"^(?:pmid:?\s*)?(\d{6,9})$", re.I)
RX_WID = re.compile(r"^(?:https?://openalex\.org/)?(W\d+)$", re.I)
_STOP = set("""a an and the of in on for to with after from by at is are was were be been this that these those
as or not but into than then via its it their our we using use used based between during within among over under
study studies results result patients patient analysis data new role effect effects associated association""".split())
DEFAULT_W = {"g": 40, "s": 35, "l": 25}


def toks(s: str) -> List[str]:
    return [w for w in re.split(r"[^a-z0-9]+", (s or "").lower()) if len(w) > 2 and w not in _STOP]


def dice(a: str, b: str) -> float:
    A, B = set(toks(a)), set(toks(b))
    return 2 * len(A & B) / (len(A) + len(B)) if A or B else 0.0


def _brief(p: Dict[str, Any]) -> Dict[str, Any]:
    return {k: p[k] for k in ("id", "t", "v", "y", "doi", "pmid", "au")}


# ── Seed resolution ──────────────────────────────────────────────────────────

async def resolve_line(oa: OpenAlex, raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    if m := RX_DOI.match(raw):
        w = await oa.work(f"doi:{m.group(1)}")
        return {"kind": "doi", "st": "ok", "p": _brief(to_paper(w))} if w else \
            {"kind": "doi", "st": "miss", "msg": "OpenAlex has no record for this DOI"}
    if m := RX_PMID.match(raw):
        w = await oa.work(f"pmid:{m.group(1)}")
        return {"kind": "pmid", "st": "ok", "p": _brief(to_paper(w))} if w else \
            {"kind": "pmid", "st": "miss", "msg": "OpenAlex has no record for this PMID"}
    if m := RX_WID.match(raw):
        w = await oa.work(m.group(1).upper())
        return {"kind": "openalex", "st": "ok", "p": _brief(to_paper(w))} if w else \
            {"kind": "openalex", "st": "miss", "msg": "OpenAlex has no record for this ID"}
    hits = [to_paper(w) for w in await oa.search(raw, 5)]
    norm = " ".join(toks(raw))
    for p in hits:
        if " ".join(toks(p["t"])) == norm:
            return {"kind": "title", "st": "ok", "p": _brief(p)}
    cands = sorted(({"p": _brief(p), "s": round(dice(raw, p["t"]), 3)} for p in hits),
                   key=lambda x: -x["s"])
    cands = [c for c in cands if c["s"] >= 0.5][:3]
    if len(cands) == 1 or (len(cands) > 1 and cands[0]["s"] >= 0.9 and cands[1]["s"] < 0.75):
        return {"kind": "title", "st": "ok", "p": cands[0]["p"], "fuzzy": cands[0]["s"]}
    if cands:
        return {"kind": "title", "st": "amb", "cands": cands}
    return {"kind": "title", "st": "miss", "msg": "no close title match in OpenAlex"}


async def resolve(lines: List[str], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    async with httpx.AsyncClient(timeout=30.0) as c:
        oa = _client(c, cfg)
        res = list(await asyncio.gather(*[resolve_line(oa, ln) for ln in lines]))
        if oa.limited:
            for r in res:
                if r["st"] == "miss":
                    r["msg"] = LIMIT_MSG
        return res


def _client(c: httpx.AsyncClient, cfg: Dict[str, Any]) -> OpenAlex:
    route = (cfg.get("retrieval", {}).get("routes", {}).get("openalex_kw", {}) or {}).get("config", {})
    return OpenAlex(c, route.get("base_url", "https://api.openalex.org"), route.get("email") or None,
                    api_key=route.get("api_key") or None)


class OpenAlexLimited(RuntimeError):
    """OpenAlex refused requests because the daily rate limit/credits are used up."""


LIMIT_MSG = ("OpenAlex rate limit reached for this server (daily credits used up). "
             "Set OPENALEX_API_KEY / OPENALEX_EMAIL in .env or retry later.")


# ── Scoring helpers ──────────────────────────────────────────────────────────

def _tfidf(docs: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    tf = {k: Counter(toks(v)) for k, v in docs.items()}
    df = Counter(w for c in tf.values() for w in c)
    n = len(docs)
    vecs = {}
    for k, c in tf.items():
        v = {w: (cnt / max(1, sum(c.values()))) * (math.log((n + 1) / (df[w] + 1)) + 1) for w, cnt in c.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs[k] = {w: x / norm for w, x in v.items()}
    return vecs


def _centroid(vecs: List[Dict[str, float]]) -> Dict[str, float]:
    out: Counter = Counter()
    for v in vecs:
        out.update(v)
    norm = math.sqrt(sum(x * x for x in out.values())) or 1.0
    return {w: x / norm for w, x in out.items()}


def _cos(a: Dict[str, float], b: Dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(w, 0.0) for w, x in a.items())


def _graph(pool: Dict[str, Dict], seeds: List[str], links: Dict[str, Dict[str, Set[int]]],
           use: Set[int]) -> Dict[str, Dict[str, Any]]:
    """Graph evidence per candidate, only counting seeds whose index is in ``use``."""
    seed_refs = {i: set(pool[s]["refs"]) for i, s in enumerate(seeds) if i in use and s in pool}
    out = {}
    for pid, p in pool.items():
        direct = {i for i in links.get(pid, {}).get("cites", set()) | links.get(pid, {}).get("cited_by", set()) if i in use}
        refs = set(p["refs"])
        shared = {i: len(refs & r) for i, r in seed_refs.items() if seeds[i] != pid and refs & r}
        out[pid] = {"direct": direct, "shared": shared,
                    "raw": len(direct) + 0.25 * sum(min(n, 4) for n in shared.values())}
    return out


def _norm(vals: Dict[str, float]) -> Dict[str, float]:
    top = max(vals.values(), default=0.0)
    return {k: (v / top if top > 0 else 0.0) for k, v in vals.items()}


def combine(g: float, s: float, l: Optional[float], w: Dict[str, float] = DEFAULT_W) -> float:
    parts = [(w["g"], g), (w["s"], s)] + ([(w["l"], l)] if l is not None else [])
    tot = sum(x for x, _ in parts) or 1.0
    return sum(x * v for x, v in parts) / tot


def _rank(order: List[str], pid: str) -> Optional[int]:
    return order.index(pid) + 1 if pid in order else None


# ── LLM judge ────────────────────────────────────────────────────────────────

def llm_available(cfg: Dict[str, Any]) -> bool:
    prov = cfg.get("llm", {}).get("providers", {}).get(cfg.get("llm", {}).get("default_provider", "openrouter"), {})
    return bool(prov.get("api_key") and prov.get("base_url"))


async def _judge(llm, cfg, question: str, seed_titles: List[str], papers: List[Dict]) -> Dict[str, float]:
    jc = cfg.get("stages", {}).get("filter", {}).get("paper_relevance", {})
    sem = asyncio.Semaphore(int(jc.get("concurrency", 10)))
    seeds_txt = "\n".join(f"- {t}" for t in seed_titles)

    async def _one(p: Dict) -> Optional[float]:
        prompt = (f"Research question: {question}\nSeed papers the researcher trusts:\n{seeds_txt}\n\n"
                  f"Candidate paper: {p['t']}\nAbstract: {p['abstract'][:1500]}\n\n"
                  "How relevant is the candidate to the question and seeds? Reply with only an integer 0-100.")
        async with sem:
            try:
                r = await llm.chat(step="medicap_judge", prompt=prompt, model=jc.get("model", "openai/gpt-oss-20b"),
                                   provider=jc.get("provider"), max_tokens=1024, reasoning_effort="low")
                m = re.search(r"\d{1,3}", r.content)
                return min(100, int(m.group())) / 100 if m else None
            except Exception as e:
                log.warning("Judge failed for %s: %s", p["id"], e)
                return None

    res = await asyncio.gather(*[_one(p) for p in papers])
    return {p["id"]: v for p, v in zip(papers, res) if v is not None}


# ── Discovery ────────────────────────────────────────────────────────────────

async def _tavily_hits(oa: OpenAlex, question: str, cfg: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
    """Tavily scholarly search on the question; map hits onto OpenAlex works via DOI/PMID when possible."""
    route = cfg.get("retrieval", {}).get("routes", {}).get("tavily_web", {}) or {}
    rc = route.get("config", {}) or {}
    if not route.get("enabled") or not rc.get("api_key"):
        return []
    try:
        data = await tavily_search(question, api_key=rc["api_key"], max_results=n,
                                   include_domains=rc.get("include_domains"), exclude_domains=rc.get("exclude_domains"))
    except Exception as e:
        log.warning("Tavily discovery failed: %s", e)
        return []

    async def _map(item: Dict[str, Any]) -> Dict[str, Any]:
        url, content = item.get("url", ""), item.get("content", "")
        ids = extract_ids(url, content)
        w = None
        if ids.get("doi"):
            w = await oa.work("doi:" + re.sub(r"^https?://doi\.org/", "", ids["doi"]))
        if w is None and ids.get("pmid"):
            w = await oa.work("pmid:" + ids["pmid"])
        return {"url": url, "title": clean_title(item.get("title", "")), "snippet": content[:400],
                "tavily_score": item.get("score"), "work": w}

    return list(await asyncio.gather(*[_map(i) for i in data.get("results", []) if i.get("url")]))


async def discover(question: str, seed_ids: List[str], cfg: Dict[str, Any], llm=None,
                   limit: int = 40) -> Dict[str, Any]:
    seed_ids = list(dict.fromkeys(short_id(s) for s in seed_ids if s))
    web_cfg = cfg.get("stages", {}).get("web_context", {}) or {}
    async with httpx.AsyncClient(timeout=30.0) as c:
        oa = _client(c, cfg)
        seed_works, keyword, web, web_ctx = await asyncio.gather(
            oa.batch(seed_ids),
            oa.search(question, 25) if question else asyncio.sleep(0, []),
            _tavily_hits(oa, question, cfg, 10) if question else asyncio.sleep(0, []),
            question_context(question, web_cfg) if question and web_cfg.get("enabled") and web_cfg.get("api_key")
            else asyncio.sleep(0, {}),
        )
        by_id = {short_id(w["id"]): w for w in seed_works}
        seeds = [s for s in seed_ids if s in by_id]
        if oa.limited and not seeds:
            raise OpenAlexLimited(LIMIT_MSG)

        # Citation graph: what each seed cites, and what cites each seed.
        links: Dict[str, Dict[str, Set[int]]] = {}
        citing_lists = await asyncio.gather(*[oa.citing(s, 25) for s in seeds])
        extra: Dict[str, Dict] = {}
        for i, s in enumerate(seeds):
            for r in to_paper(by_id[s])["refs"]:
                links.setdefault(r, {}).setdefault("cited_by", set()).add(i)   # candidate is cited by seed i
            for w in citing_lists[i]:
                wid = short_id(w["id"])
                extra[wid] = w
                links.setdefault(wid, {}).setdefault("cites", set()).add(i)    # candidate cites seed i
        graph_ids = sorted(links, key=lambda k: -sum(len(v) for v in links[k].values()))[:150]

        kw_ids = [short_id(w["id"]) for w in keyword]
        web_ids = [short_id(h["work"]["id"]) if h["work"] else f"web:{n}" for n, h in enumerate(web)]
        known = {**{short_id(w["id"]): w for w in keyword}, **extra, **by_id,
                 **{short_id(h["work"]["id"]): h["work"] for h in web if h["work"]}}
        missing = [i for i in dict.fromkeys(graph_ids) if i not in known]
        for w in await oa.batch(missing):
            known[short_id(w["id"])] = w

    pool: Dict[str, Dict[str, Any]] = {}
    for wid in dict.fromkeys(seeds + graph_ids + kw_ids + [i for i in web_ids if not i.startswith("web:")]):
        if wid in known:
            pool[wid] = to_paper(known[wid])
    for n, h in enumerate(web):
        if not h["work"]:
            pool[f"web:{n}"] = {"id": f"web:{n}", "t": h["title"] or h["url"], "au": "", "y": None,
                                "v": h["url"].split("/")[2] if "//" in h["url"] else "", "type": "Web",
                                "peer": 0, "oa": 1, "cit": 0, "doi": "", "pmid": "", "url": h["url"],
                                "abstract": h["snippet"], "topic": "", "flag": None, "refs": []}
    web_by_id = {wid: h for wid, h in zip(web_ids, web)}

    # Semantic: TF-IDF over the pool, against the centroid of (question + seeds).
    vecs = _tfidf({**{k: f"{p['t']} {p['t']} {p['abstract']}" for k, p in pool.items()}, "__q": question or ""})
    qv = vecs.pop("__q")

    def sem(use: Set[int]) -> Dict[str, float]:
        cen = _centroid([qv] * max(1, len(use) // 2) + [vecs[seeds[i]] for i in use if seeds[i] in vecs])
        return {k: _cos(v, cen) for k, v in vecs.items()}

    all_idx = set(range(len(seeds)))
    g_ev = _graph(pool, seeds, links, all_idx)
    g = _norm({k: v["raw"] for k, v in g_ev.items() if k not in seeds})
    s_raw = sem(all_idx)
    s = _norm({k: v for k, v in s_raw.items() if k not in seeds})
    cands = [k for k in pool if k not in seeds]
    sem_order = sorted(cands, key=lambda k: -s.get(k, 0))

    judge_on = llm is not None and llm_available(cfg)
    l: Dict[str, float] = {}
    if judge_on:
        pre = sorted(cands, key=lambda k: -combine(g.get(k, 0), s.get(k, 0), None))[:30]
        l = await _judge(llm, cfg, question, [pool[x]["t"] for x in seeds], [pool[k] for k in pre])

    out = []
    for k in cands:
        ev = g_ev[k]
        ch = []
        if ev["direct"] or any(n >= 3 for n in ev["shared"].values()):
            ch.append("graph")
        if k in sem_order[:30]:
            ch.append("semantic")
        if k in kw_ids:
            ch.append("keyword")
        if k in web_by_id:
            ch.append("web")
        if not ch:
            continue
        p = pool[k]
        lk = links.get(k, {})
        out.append({**{x: p[x] for x in ("id", "t", "au", "y", "v", "type", "peer", "oa", "cit", "doi", "pmid", "url", "flag", "topic")},
                    "abstract": p["abstract"][:600],
                    "g": round(g.get(k, 0), 4), "s": round(s.get(k, 0), 4),
                    "l": round(l[k], 4) if k in l else None, "ch": ch,
                    "sl": sorted({i + 1 for i in ev["direct"]} | {i + 1 for i, n in ev["shared"].items() if n >= 3}),
                    "links": {"cites": sorted(i + 1 for i in lk.get("cites", set())),
                              "cited_by": sorted(i + 1 for i in lk.get("cited_by", set())),
                              "shared": {str(i + 1): n for i, n in ev["shared"].items()}},
                    "kw_rank": _rank(kw_ids, k), "web_rank": _rank(web_ids, k),
                    "web_hit": ({x: web_by_id[k][x] for x in ("url", "title", "snippet")} if k in web_by_id else None),
                    "web": []})
    out.sort(key=lambda p: -combine(p["g"], p["s"], p["l"]))
    out = out[:limit]

    # Clusters: top OpenAlex subfields among returned candidates.
    top_topics = [t for t, _ in Counter(p["topic"] for p in out if p["topic"]).most_common(4)]
    clusters = {f"c{i}": t for i, t in enumerate(top_topics)}
    inv = {t: k for k, t in clusters.items()}
    for p in out:
        p["cl"] = inv.get(p["topic"], "other")
    if any(p["cl"] == "other" for p in out):
        clusters["other"] = "Other topics"

    # Supporting web sources for the top papers.
    supported = 0
    if web_cfg.get("enabled") and web_cfg.get("api_key"):
        targets = [{"doc_type": "paper", "title": p["t"], "doi": p["doi"], "url": p["url"], "_p": p}
                   for p in out if not p["id"].startswith("web:")]
        supported = await add_supporting_sources(targets, web_cfg)
        for t in targets:
            t["_p"]["web"] = t.get("supporting_sources", [])

    loo = _loo(pool, seeds, links, kw_ids, web_ids, sem, l)
    seed_list = [{"n": i + 1, **_brief(pool[s]), "flag": pool[s]["flag"], "topic": pool[s]["topic"]}
                 for i, s in enumerate(seeds)]
    queries = [{"channel": "graph", "q": f"references + citing works of {len(seeds)} seeds"}]
    if question:
        queries += [{"channel": "keyword", "q": question}, {"channel": "web", "q": question}]
    return {
        "question": question,
        "seeds": seed_list,
        "papers": out,
        "clusters": clusters,
        "loo": loo,
        "queries": queries,
        "web_context": web_ctx or {},
        "judge": judge_on,
        "stats": {"seeds": len(seeds), "graph_candidates": len(links), "keyword_hits": len(kw_ids),
                  "web_hits": len(web), "pool": len(pool), "returned": len(out), "judged": len(l),
                  "papers_with_supporting_sources": supported},
    }


def _loo(pool, seeds, links, kw_ids, web_ids, sem, l) -> List[Dict[str, Any]]:
    """Leave-one-out: hide each seed, rank the pool with the others, record where the hidden seed lands."""
    rows = []
    for h, hid in enumerate(seeds):
        use = set(range(len(seeds))) - {h}
        others = {seeds[i] for i in use}
        cands = [k for k in pool if k not in others]
        g_ev = _graph(pool, seeds, links, use)
        g = _norm({k: g_ev[k]["raw"] for k in cands})
        s = _norm({k: v for k, v in sem(use).items() if k in cands})
        graph_order = [k for k in sorted(cands, key=lambda k: -g[k]) if g[k] > 0]
        sem_order = sorted(cands, key=lambda k: -s.get(k, 0))
        hyb_order = sorted(cands, key=lambda k: -combine(g.get(k, 0), s.get(k, 0), l.get(k)))
        rows.append({"seed": h + 1,
                     "keyword": _rank([k for k in kw_ids if k not in others], hid),
                     "web": _rank([k for k in web_ids if k not in others], hid),
                     "graph": _rank(graph_order, hid),
                     "semantic": _rank(sem_order, hid),
                     "hybrid": _rank(hyb_order, hid)})
    return rows
