"""Stage 5 — Answer Generation.

Build context → strong LLM answer → citation reindex → Markdown formatting.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Tuple
from urllib.parse import quote

from openoe.core.models import Document
from openoe.llm.client import LLMClient
from openoe.llm.prompts import PromptManager
from openoe.utils.text import truncate

log = logging.getLogger(__name__)
_BARE_NUMERIC_CITATION_RE = re.compile(r"(?<![\^\w])\[\d+(?:\s*[,\-–]\s*\d+)*\]")


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

async def generate_answer(
    question: str,
    papers: List[Document],
    guidelines: List[Document],
    cfg: Dict[str, Any],
    llm: LLMClient,
    prompts: PromptManager,
    history: List[Dict[str, str]] | None = None,
) -> Dict[str, Any]:
    """Returns ``{"answer": ..., "markdown": ..., "citations": [...], "trace": LLMTrace}``."""
    # Build context + source list
    context, sources = _build_context(papers, guidelines)
    log.info("Stage 5 context: %d papers + %d guidelines = %d sources", len(papers), len(guidelines), len(sources))
    rag_json = json.dumps(context, ensure_ascii=False, indent=2)

    system = prompts.get(cfg.get("system_prompt", "medical_expert"))
    prompt = prompts.render(
        cfg.get("prompt_template", "generate_answer_v4"),
        question=question,
        rag_context=rag_json,
    )

    result = await llm.chat(
        step="answer",
        prompt=prompt,
        system=system,
        model=cfg["model"],
        provider=cfg.get("provider"),
        max_tokens=int(cfg.get("max_tokens", 16384)),
        history=history,
    )
    log.info("Stage 5 raw answer: %d chars, preview=%s", len(result.content), result.content[:100])

    answer = _fix_chinese_citations(result.content.strip())
    answer = _remove_trailing_references(answer)
    answer = _BARE_NUMERIC_CITATION_RE.sub("", answer)
    cited_ids = _extract_cited_ids(answer)
    log.info("Stage 5 cited_ids=%s from %d sources", cited_ids, len(sources))
    answer, cited = _reindex(answer, sources, cited_ids)
    log.info("Stage 5 after reindex: %d cited sources", len(cited))
    markdown = _format_markdown(question, answer, cited)

    return {
        "answer": answer,
        "markdown": markdown,
        "citations": cited,
        "trace": result.trace,
    }


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------

def _build_context(
    papers: List[Document], guidelines: List[Document],
) -> Tuple[List[Dict], List[Dict]]:
    ctx: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    # Use 1-based IDs — LLMs naturally cite as [^1], [^2], ...
    idx = 1
    for d in papers:
        ctx.append({"id": idx, "type": "paper", "title": d.title, "abstract": truncate(d.content, 900), "metadata": d.metadata})
        sources.append({"doc_type": "paper", "title": d.title, **d.metadata})
        idx += 1
    for d in guidelines:
        chunks = d.relevant_chunks or [truncate(d.content, 2000)]
        ctx.append({"id": idx, "type": "guideline", "title": d.title, "relevant_chunks": chunks})
        sources.append({"doc_type": "guideline", "title": d.title, "relevant_chunks": chunks, **d.metadata})
        idx += 1
    return ctx, sources


# ---------------------------------------------------------------------------
# Citation post-processing
# ---------------------------------------------------------------------------

def _fix_chinese_citations(text: str) -> str:
    return re.sub(r"【\^(\d+)】", r"[^\1]", text) if text else text


def _extract_cited_ids(text: str) -> List[int]:
    seen: set[int] = set()
    out: List[int] = []
    for s in re.findall(r"\[\^(\d+)\]", text):
        i = int(s)
        if i not in seen:
            out.append(i)
            seen.add(i)
    return out


def _reindex(answer: str, sources: List[Dict], cited_ids: List[int]) -> Tuple[str, List[Dict]]:
    """Reindex [^N] citations to sequential 1-based and collect cited sources.

    Context IDs are 1-based (id=1,2,...), sources list is 0-based (index 0,1,...).
    So cited_id N maps to sources[N-1].
    """
    # Map cited IDs to source indices, filter out-of-range
    valid_ids = [i for i in cited_ids if 1 <= i <= len(sources)]
    # New sequential 1-based mapping
    old2new = {old: new_idx + 1 for new_idx, old in enumerate(valid_ids)}
    log.info("Citation reindex: cited_ids=%s, valid=%s, sources_len=%d, old2new=%s",
             cited_ids, valid_ids, len(sources), old2new)

    def _replace(m: re.Match) -> str:
        old = int(m.group(1))
        if old in old2new:
            return f"[^{old2new[old]}]"
        return ""

    new_answer = re.sub(r"\[\^(\d+)\]", _replace, answer)
    cited = [sources[i - 1] for i in valid_ids]  # 1-based → 0-based index
    return new_answer, cited


def _remove_trailing_references(text: str) -> str:
    """Strip reference lists LLMs sometimes append at the end."""
    if not text:
        return text
    lines = text.split("\n")
    heading_re = re.compile(r"^\s*#{1,4}\s*(参考文献|references|bibliography)", re.I)
    ref_re = re.compile(r"^\s*\[\^?\d+\]\s*[:：]")
    cut = None
    consec = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        if heading_re.match(s):
            if cut is None:
                cut = i
            continue
        if ref_re.match(s):
            if cut is None:
                cut = i
            consec += 1
        else:
            if consec < 2:
                cut = None
                consec = 0
    if cut is not None and consec >= 2:
        log.info("Removed trailing references from line %d (cut %d lines)", cut, len(lines) - cut)
        return "\n".join(lines[:cut]).rstrip()
    return text


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------

_EVIDENCE_LEVEL = {
    "Guideline": 1, "Meta Analysis": 1, "Systematic Review": 1,
    "Randomized Controlled Trial": 2, "Clinical Trial": 3, "Multicenter Study": 3,
    "Review": 4, "Case Reports": 5, "Journal Article": 5,
}
_BADGE_COLORS = {1: "brightgreen", 2: "green", 3: "yellow", 4: "orange", 5: "red"}


def _format_markdown(question: str, answer: str, cited: List[Dict]) -> str:
    md = [f"## Answer\n\n{answer}\n\n---\n\n"]
    if cited:
        md.append("## References\n\n")
        for i, s in enumerate(cited, 1):
            if s.get("doc_type") == "guideline":
                md.append(_fmt_guideline(i, s))
            else:
                md.append(_fmt_paper(i, s))
    return "".join(md)


def _fmt_paper(i: int, s: Dict) -> str:
    pmid = str(s.get("pmid", "")).strip()
    title = s.get("title", "N/A")
    doi = str(s.get("doi", "")).strip()
    journal = s.get("journal_iso", "")
    if_ = s.get("impact_factor", "")
    date = s.get("pub_date", "")
    pub_types = s.get("pub_types", [])
    if isinstance(pub_types, str):
        pub_types = [pub_types]

    pmid_link = f"[PMID: {pmid}](https://pubmed.ncbi.nlm.nih.gov/{quote(pmid)}/)" if pmid else ""
    doi_norm = doi
    for p in ("doi:", "DOI:", "https://doi.org/", "http://doi.org/"):
        if doi_norm.lower().startswith(p.lower()):
            doi_norm = doi_norm[len(p):].strip()
            break
    doi_link = f" | [DOI](https://doi.org/{quote(doi_norm)})" if doi_norm else ""

    info = journal or ""
    if if_:
        info += f" (IF: {if_})"
    if date:
        info += f" · {date}"

    # Evidence badge
    for pt in (pub_types if isinstance(pub_types, list) else []):
        if pt in _EVIDENCE_LEVEL:
            lv = _EVIDENCE_LEVEL[pt]
            badge = f"![{pt}](https://img.shields.io/badge/{quote(pt)}-{_BADGE_COLORS[lv]})"
            info = f"{badge} {info}"
            break

    return f"#### ^{i}: {title}\n\n{pmid_link}{doi_link} | {info}\n\n"


def _fmt_guideline(i: int, s: Dict) -> str:
    title = s.get("title", "N/A")
    n = len(s.get("relevant_chunks", []))
    badge = "![Guideline](https://img.shields.io/badge/Guideline-red)"
    return f"#### ^{i}: {title}\n\nClinical guideline {badge}, relevant passages: {n}\n\n"
