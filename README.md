# DevOne - AI Code Assistant

A retrieval-augmented generation (RAG) system for understanding, searching, and explaining source code repositories. DevOne connects directly to GitHub repos, indexes them locally, and lets you ask questions, generate documentation, find issues, and get a whole-project overview - all through a single Streamlit app.

---

## Features

- Connect any GitHub repo via a personal access token - clones/pulls locally via git, no GitHub API rate limits involved
- Incremental sync - only re-embeds files that actually changed since the last sync, using git diff against the last recorded commit
- Structural code parsing for Python, Java, JavaScript, C++, and C - chunks at the function/class level instead of arbitrary fixed-size splitting
- Agentic retrieval - the chat model chooses between semantic search, exact-text search, and full-file reads depending on the question, instead of always doing static top-k retrieval
- Conversation memory - follow-up questions build on prior turns in the same session
- Token usage tracking - every LLM call logs input/output/total tokens, viewable per-project and per-model in a Usage Stats tab
- Documentation generation - per-function docs, per-class docs, per-file summaries, issue finding, improvement suggestions, and a whole-project overview via map-reduce summarization
- README generator - builds a README.md from the ingested codebase
- Quick single-file upload for fast sanity checks, kept separate from real connected repos

---

## Technology Stack

| Technology | Purpose |
| --- | --- |
| Python | Core application logic |
| Streamlit | User interface |
| LangChain / LangGraph | RAG orchestration and agentic tool-calling |
| Groq API | Hosted LLM for chat and documentation generation |
| Ollama | Local embedding generation (mxbai-embed-large, 1024-dimensional) |
| SQLite + sqlite-vec | Unified storage for vectors, repo metadata, and usage logs |
| git | Repo cloning/pulling and incremental diffing, called directly rather than via the GitHub REST API |

---

## Project Structure

```text
code_rag_github/
- app.py                Streamlit UI and orchestration
- vector.py             SQLite + sqlite-vec storage layer
- repo_manager.py       GitHub repo cloning, syncing, incremental diffing
- agent_tools.py        LangGraph agentic retrieval loop and tools
- src/
  - parser.py           Structural code chunking
  - doc_generator.py    Documentation generation (functions, classes, files, README, project overview)
- requirements.txt
- run.bat
- screenshots/          UI screenshots
- data/                  Local storage (gitignored) - devone.db, cloned repos
```

---

## Installation

### 1. Install Ollama (used for local embeddings)

Download from https://ollama.com and pull the embedding model:

```bash
ollama pull mxbai-embed-large
```

### 2. Get a Groq API key

Sign up at https://console.groq.com and generate an API key.

### 3. Create a GitHub Personal Access Token

Fine-grained token, read-only "Contents" permission, scoped to the repos you want to index.

### 4. Set up your environment

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
GITHUB_PAT=your_token_here
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run

```bash
streamlit run app.py
```

---

## How It Works

### Ingestion

1. Paste a GitHub repo URL and click Connect / Sync
2. The app clones the repo locally via git, or pulls the latest changes if already connected
3. Every supported file is parsed into function/class-level chunks
4. Chunks are embedded locally via Ollama and stored in SQLite + sqlite-vec
5. On future syncs, only files that changed - per git diff against the last recorded commit - get re-embedded

### Chat

1. Ask a question about the connected repo
2. A LangGraph agent decides whether to run a semantic search, an exact-text search, a full-file read, or some combination, based on the question
3. The Groq-hosted LLM generates an answer grounded in whatever the tools returned
4. Token usage is logged, and the tool activity used to answer is shown alongside the response

---

## Design Decisions

A few choices worth calling out, since they were deliberate rather than defaults:

- **SQLite + sqlite-vec over Postgres/pgvector** - both were evaluated; the embedded option was chosen since this is a single-user, local-first tool, and a separate database service wasn't worth the overhead.
- **git clone/pull over the GitHub REST API** - avoids API rate limits entirely, and git already handles diffing for free on repeat syncs.
- **Agentic tool-calling over static top-k retrieval** - plain vector search can miss the right chunk when a question's phrasing does not line up semantically with the code. Giving the model grep and read_file tools lets it recover from that.
- **No invented cost estimates** - token usage is tracked precisely; dollar-cost estimates are not shown, since accurate per-model API pricing is not something to guess at reliably.

---

## Known Limitations

- Occasional tool-call formatting errors from the underlying model - a known reliability quirk with function-calling LLMs, not a bug in this app. The app retries automatically and shows a clean message if it still fails.
- Ambiguous vocabulary - the same word meaning different things in different files - can occasionally lead the agent to the wrong file.
- Project overview generation is capped at 20 files by default to keep response time reasonable for large repos.

---

## Screenshots

Please check the `screenshots/` folder for UI screenshots.

---

## Author

Arya Barsode
