# Omni — NVIDIA NIM Documentation Assistant

Omni answers questions about deploying NVIDIA NIM by combining two kinds of data in one agent: **unstructured documentation** (retrieval-augmented generation over the official NIM docs) and **structured data** (a SQL table of supported models, GPUs, VRAM and precision). A hand-written ReAct loop lets the LLM decide which tool to call, call both when needed, and ground its answer in what the tools return.

> **Status (Nov 2026): work in progress, runs locally.** Not deployed yet. A coverage audit of the knowledge base is under way; see [Known issues](#known-issues-and-in-progress).

**Example question:** *"Which models need an H100, and what deployment steps does NIM require for them?"*
The agent queries the model catalog for H100 models, searches the docs for deployment steps, and returns one answer with its sources and the reasoning steps it took.

## Architecture

```
Streamlit chat UI (app.py)
        |  POST /ask
        v
FastAPI + Pydantic (main.py)
        |
        v
ReAct agent loop (agent.py)  <---->  Gemini (function calling, manual loop)
        |                |
        v                v
search_documentation   query_model_catalog
(retrieve.py)          (tools.py)
        |                |
        v                v
Postgres + pgvector (Docker)
  doc_chunks: 384-d embeddings + full-text index
  model_catalog: structured model facts
```

## How it works

**Data pipeline**
1. `ingest.py` scrapes the NIM LLM docs asynchronously (aiohttp + BeautifulSoup).
2. `chunk.py` splits each page into ~400-word chunks with overlap, keeping title and URL as metadata.
3. `embed.py` embeds chunks with `all-MiniLM-L6-v2` (384 dimensions, runs locally) and stores them in pgvector.
4. `clean_catalog.py` / `load_catalog.py` clean a hand-built CSV of 8 NIM-supported models with Pandas and load it into a Postgres table (`model_name`, `required_gpu`, `min_vram_gb`, `supported_precision`, `deployment_type`).

**Hybrid retrieval** (`retrieve.py`): semantic search (pgvector cosine distance) plus keyword search (Postgres full-text search), merged, de-duplicated and cut to the top 5. Keyword search catches exact terms such as model and flag names that embeddings can blur.

**Agent** (`agent.py`): automatic function calling is switched off and the loop runs step by step: generate, check for tool calls, run the tools, feed results back, repeat (max 5 turns). Two tools: `search_documentation` and `query_model_catalog`. The system instruction forces the catalog tool for any question that names a model, GPU, VRAM or precision, and says to call both tools when unsure. Each response returns the answer, sources and the list of reasoning steps. The loop is implemented explicitly rather than through the SDK's automatic function calling, so every step can be seen and logged.

## Evaluation

15 questions across three types: 5 catalog-only, 5 docs-only, 5 compound (need both tools). Run with `src/evaluate.py`.

| Metric | Result |
|---|---|
| Agent completion rate | 15/15 (100%) |
| Average faithfulness | 0.93 (13 answers scored 1.0, 2 scored 0.5) |

Faithfulness is scored by a Gemini judge that checks whether each answer is supported by the retrieved context. I used a small custom judge because RAGAS (0.4.3 and 0.3.9) failed to import with my dependency versions.

**Limitations, stated plainly:**
- Faithfulness only. I do not yet measure retrieval precision/recall or answer correctness against a gold answer.
- The judge is Gemini scoring Gemini, which can be lenient.
- 15 questions is a small set.
- These scores were measured before the knowledge-base audit below. I am re-running them after the fix and will report the new numbers.

## What went wrong and what I fixed

- **Routing bug.** The agent was answering model/GPU questions from documentation only and skipping the catalog tool. The evaluation exposed it; I tightened the system instruction so model-specific questions always call the catalog first.
- **Self-correction.** When a search for "Llama-3-70B" returned nothing, the agent retried with "Llama" on its own and found the row.
- **Upstream instability.** Gemini returned 503s, request timeouts and a model deprecation (twice). I added a 45-second timeout, automatic retries (after a pause) in the evaluation runner, a reusable `diagnose.py` that tests Postgres, embeddings and Gemini independently, and migrated models (now `gemini-3.1-flash-lite`).
- **Evaluation bug.** An earlier run saved error messages as if they were answers, which corrupted the scores. I changed the runner to score only successful answers and to report completion rate separately.

## Known issues and in progress

- **Knowledge-base coverage.** The live NIM docs index lists 59 pages; Omni has 51. Missing: the docs overview page, the Get Started "advanced" pages (including three new model guides), the request-timeouts troubleshooting page, the multimodal input page, and the benchmarking page (its URL moved). Fix: read the page list from the docs' own index instead of the sidebar.
- **Possible stale rows.** `embed.py` inserts and updates but never deletes, so pages removed from the docs could still sit in the vector table. I am checking and will rebuild the table from scratch.
- **Catalog and eval questions that depend on removed content.** The "Nemotron 3 Ultra 550B / Day 0" row and two eval questions refer to content that no longer appears in the current docs; I am reviewing them.
- **Chunker flattens structure.** Code blocks and tables lose their line breaks when chunked, which hurts the support-matrix page most.
- **Hybrid merge is simple.** Cosine similarity and full-text rank are on different scales and are sorted together; reciprocal rank fusion would be more principled.
- **Catalog is small and hand-built** (8 models) and has not yet been cross-checked against the official support matrix.
- `requirements.txt` needs cleanup (a typo, unused packages, missing ones). Use the install line below for now.

## Run locally

Requires Python 3.12, Docker and a Gemini API key.

```bash
git clone git@github.com:Rahulkulkarni2002/OMNI.git && cd OMNI
python -m venv venv && source venv/bin/activate
pip install aiohttp certifi beautifulsoup4 pandas python-dotenv sentence-transformers \
            psycopg2-binary fastapi uvicorn google-genai streamlit requests

docker compose up -d        # pgvector/pgvector:pg16 as container "omni_db"
```

Create a `.env` file in the project root (it is git-ignored):

```
DATABASE_URL=postgresql://omni:omni_local_dev@localhost:5432/omni
GEMINI_API_KEY=your_key_here
```

Build the data, then start the app (scripts use relative paths, so run them from `src/`):

```bash
cd src
python ingest.py && python chunk.py && python embed.py
python load_catalog.py
python diagnose.py          # checks Postgres, embeddings and Gemini
python main.py              # terminal 1: API on :8000
streamlit run app.py        # terminal 2: UI on :8501
```

The Docker container stops when the machine sleeps; restart it with `docker start omni_db`.

## Roadmap

- [x] Scraping and chunking pipeline
- [x] pgvector store and hybrid retrieval
- [x] FastAPI backend and Streamlit chat UI
- [x] Tool calling with a hand-written ReAct loop
- [x] Evaluation harness (faithfulness)
- [ ] Knowledge-base audit and rebuild, then re-run evaluation
- [ ] Retrieval-quality metrics and a larger question set
- [ ] Hosted demo (Streamlit Community Cloud + hosted Postgres)
- [ ] Dockerfiles, Docker Compose for the full stack, CI
- [ ] Later: MCP server, charts, multiple LLM providers

## License

MIT (see `LICENSE`).
