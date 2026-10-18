# MedicUp

**From the papers you trust to the evidence you need.**

MedicUp is a medical research discovery engine for doctors and researchers. You start with a clinical question, a few papers you already trust, or both. MedicUp follows citations, meaning and the web, ranks what it finds, explains why each paper matters and where it was found, and lets you keep or reject every recommendation. Follow-up questions build on what you kept.

Built for the techeurope hackathon on [Nebius Token Factory](https://tokenfactory.nebius.com) (LLM inference and embeddings) and [Tavily](https://www.tavily.com) (agentic web search). Foundation: [OpenOE-Lite](https://github.com/wizardlancet/OpenOE-Lite) (MIT).

---

## The vision

A doctor has a patient question, two papers they trust, and under five minutes to find what else matters. Keyword search alone does not get there: it returns thousands of hits, none of them connected to what the doctor already knows.

MedicUp treats the problem as abduction: the clues are the papers you trust plus your question; the best explanation is the set of papers that connect them, with a stated reason. The hackathon brief says it plainly:

> "help a researcher move from known papers to a ranked, traceable set of new ones. It should make clear why each recommendation is relevant and where the supporting information came from."

Three principles follow from that:

1. **Start from what you know.** Seed papers and your question are the input, not a bag of keywords.
2. **Every recommendation is traceable.** Each paper shows *why* it is here, *which signals* found it, and links to the source. Several independent signals have to agree before a paper ranks high.
3. **AI proposes, doctors decide.** You keep or reject papers; kept papers become the seeds of your next question. If no sources are found, there is no answer.

Background numbers from the team's research, to be checked against the original studies before quoting them: doctors spend under 5 minutes per clinical question, about 60% of their questions go unanswered, and PubMed answers fewer questions than curated tools such as UpToDate (63% vs 83%).

---

## What you get

A chat interface in the style of Open WebUI. Type anything and press Enter. Seed papers are optional and live behind the **+** / **Seed papers** button (DOI, PMID, PubMed or OpenAlex link, or a title, one per line).

Every answer has the same shape:

1. **Cited answer**, written only from the retrieved papers, with footnote citations.
2. **Suggested follow-ups**: three questions a clinician would ask next, generated from the content. Click one and MedicUp runs a new search with your kept papers as seeds and the conversation as context.
3. **Paper panel** (scrollable): for each paper, title, authors, journal, year, study design, *Why* it is relevant, *Found via* which signals, the % match with its breakdown, DOI / PubMed / full-text links, the abstract, a citation export, and **Keep / Reject**.
4. **Citation graph**: your seeds (or your question) in the middle row, the ranked papers branching above and below on curved links. Dot size is the score; branches are citation links; hover a dot to light up its chain and figures; click a dot to jump to the paper.

Further tabs: **Clusters** (grouped by research topic), **Web sources** (background found by Tavily), **Trace** (where the results came from and how long each step took). The reading list exports as BibTeX, RIS, Markdown or CSV. Light and dark mode, mobile layout, saved chats.

The original Medicap dashboard (`/app`) is still available with the leave-one-out evaluation: hide one seed, re-run, and check whether the graph recovers it.

---

## How the ranking works

Candidate papers come from independent signals:

| Signal | Source | What it finds |
|---|---|---|
| Cites / cited by / related | OpenAlex citation graph | Papers connected to several of your seeds |
| Recommendations | Semantic Scholar | Papers recommended for the whole seed set |
| Search | OpenAlex, PubMed | Papers for queries the LLM derives from your question and seeds |
| Web | Tavily (scholarly domains) | Hits resolved to a real DOI or PMID and ranked like any other paper |

Every candidate is scored in two steps:

- **Pre-score** (signals only): 50% embedding similarity to the question and the seed centroid (Nebius `Qwen3-Embedding-8B`), 30% share of seeds linked by citation, 10% agreement across independent signals, 10% Semantic Scholar recommendation.
- **AI relevance** (0–10): the top 25 candidates by pre-score are read by Nebius `Qwen3-235B`, which grades relevance against the question and seed titles, writes the one-sentence *why*, names the related seeds and extracts the study design as stated in the abstract.

**% match = 60% AI relevance + 40% pre-score.** The 20 best are shown; the breakdown is visible on every card.

Claims we keep honest, following the team's ranking analysis:

- Relevance is a gate. Quality only re-orders papers that are already relevant.
- Citation counts reflect study type, not study quality; they are shown, not scored.
- Automated quality or bias grading is not reliable enough to stand alone, so MedicUp says "traceable" and "explained", not "best evidence" or "verified quality".
- Study-design labels are AI-extracted from the abstract and can be wrong; the retraction flag comes from OpenAlex metadata.

---

## Knowledge graph

Today the citation layer is built and visible. The intended schema is a small scholarly ontology with the MeSH tree as taxonomy layer:

- **Node types:** Paper, Concept (MeSH or topic), Author, Venue, WebSource, Figure, Run
- **Link types:** CITES, SHARES_REFS_WITH, ABOUT, BROADER, SUPPORTED_BY, HAS_FIGURE, FOUND_BY, RANKED_IN

The concept / MeSH view is proposed, not implemented.

---

## Quick start

```bash
git clone https://github.com/Fredo220/MedicUp.git
cd MedicUp
pip install -e .
cp .env_template .env    # add NEBIUS_API_KEY and TAVILY_API_KEY
openoe --port 8000       # open http://localhost:8000
```

Routes: `/` landing page, `/chat` the MedicUp chat, `/app` the Medicap dashboard, `/qa` the original OpenOE question-answering page, `/docs` the API.

### Configuration

| Variable | Required | Description |
|---|:---:|---|
| `NEBIUS_API_KEY` | yes | Nebius Token Factory key: chat answers, follow-ups, embeddings, AI judge |
| `TAVILY_API_KEY` | yes | Tavily: web discovery, web context, supporting sources, figures |
| `OPENALEX_EMAIL` | | OpenAlex polite pool (also sent to PubMed E-utilities) |
| `OPENALEX_API_KEY` | recommended | Without a key OpenAlex allows about 100 searches per day; after that MedicUp falls back to PubMed, and the citation signals are unavailable until midnight UTC |
| `MEDICUP_SMALL_MODEL`, `MEDICUP_STRONG_MODEL`, `MEDICUP_EMBED_MODEL` | | Any Token Factory model ids (defaults: Qwen3-30B-A3B, Qwen3-235B-A22B, Qwen3-Embedding-8B) |
| `OPENOE_DEBUG` | | `true` logs full LLM requests to `logs/llm/` |

Models and stages are configured in `openoe/config/pipeline.yaml`, prompts in `openoe/config/prompts.yaml`.

### API

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/discover` | Discovery: `{question, seeds, disliked, limit}`; question or seeds required |
| `WS` | `/ws/discover` | Same, with progress events |
| `POST` | `/api/followup` | Cited answer + suggested follow-ups from a list of papers |
| `POST` | `/api/query`, `WS /ws/query` | Original OpenOE question answering |
| `POST` | `/api/medicap/resolve`, `/api/medicap/discover`, `/api/medicap/figures` | Medicap dashboard endpoints |
| `GET` | `/api/config`, `/api/traces/{id}`, `/health` | Config (redacted), execution trace, health |

### Tests

```bash
pip install -e ".[dev]"
pytest -q
```

---

## Deploy on Nebius

The app is one container (FastAPI + static frontend, port 8000, no credentials inside). The model endpoint is Nebius Token Factory, so the interface and backend run separately from the model, as the hackathon requires.

```bash
docker build -t medicup .
docker run -p 8000:8000 --env-file .env medicup
```

On Nebius: push the image to a Container Registry, create a serverless container or VM from it with port `8000`, and set `NEBIUS_API_KEY`, `TAVILY_API_KEY`, `OPENALEX_EMAIL` and `OPENALEX_API_KEY` as secret environment variables.

---

## Project structure

```
openoe/
├── discovery/          # Seed-based discovery
│   ├── seeds.py        #   Resolve DOI / PMID / title / URL (OpenAlex, PubMed fallback)
│   ├── candidates.py   #   Citation graph, Semantic Scholar, search, Tavily web signal
│   ├── rank.py         #   Embeddings, pre-score, AI judge, study design
│   ├── followup.py     #   Cited answers and suggested follow-ups
│   ├── openalex.py / pubmed.py
│   └── service.py      #   Orchestration
├── medicap/            # Medicap dashboard backend (resolve, discover, LOO evaluation, figures)
├── pipeline/           # Original OpenOE question-answering stages
├── sources/            # OpenAlex and Tavily retrieval routes
├── llm/                # Async Token Factory client, prompt loading
├── api/app.py          # FastAPI, WebSockets, static routes
├── static/
│   ├── landing.html    # Landing page
│   ├── index.html      # MedicUp chat
│   └── medicap.html    # Medicap dashboard
└── config/             # pipeline.yaml, prompts.yaml
```

---

## Status and known limitations

- Not yet deployed on Nebius; deployment needs a Nebius AI Cloud project and service account.
- OpenAlex free quota is small; add `OPENALEX_API_KEY` for demos.
- Figures depend on open-access sources; many papers have none.
- The product name is MedicUp; the dashboard interface is still called Medicap.

---

## Credits and licence

Built on [OpenOE-Lite](https://github.com/wizardlancet/OpenOE-Lite) by wizardlancet (MIT). Data from [OpenAlex](https://openalex.org), [PubMed](https://pubmed.ncbi.nlm.nih.gov), [Semantic Scholar](https://www.semanticscholar.org) and [Tavily](https://www.tavily.com). Models served by [Nebius Token Factory](https://tokenfactory.nebius.com).

MIT License. MedicUp is a research tool: it can make mistakes, check the sources, and it is not a substitute for clinical judgement.
