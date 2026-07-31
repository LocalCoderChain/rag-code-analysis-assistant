"""
app.py  —  Local AI Code Analysis & Documentation Assistant
============================================================
Same core patterns from eng_rag project:
  - @st.cache_resource for LLM + retriever (prevents slow reruns)
  - chain = prompt | model  (LangChain LCEL)
  - st.session_state.messages  (persistent chat)
  - OllamaLLM + OllamaEmbeddings + ChromaDB

New features for code analysis:
  - Structural code parsing (functions, classes, imports)
  - Documentation generation tab
  - Issues & improvements finder
  - README generator

Run:
    streamlit run app.py
"""

import os

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from vector import (
    ingest_code_file,
    get_all_chunks,
    list_collections,
    delete_collection,
    get_usage_stats,
)
from src.doc_generator import CodeDocGenerator, build_codebase_summary
from repo_manager import ingest_repo, list_repos, check_remote_status
from agent_tools import answer_with_tools

load_dotenv()

QUICK_TEST_COLLECTION = "quick-test"
LONG_SESSION_TOKEN_WARNING = 50_000


def get_available_models() -> list[str]:
    """Chat models actually available to this Groq API key."""
    from groq import Groq
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        models = client.models.list()
        return sorted(m.id for m in models.data if getattr(m, "active", True))
    except Exception:
        return []


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DevOne",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; }

[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #30363d;
}
.brand-block {
    text-align: center; padding: 1.2rem 0.5rem 1rem;
    border-bottom: 1px solid #30363d; margin-bottom: 0.8rem;
}
.brand-title { font-family: 'Consolas', 'Cascadia Code', monospace;
               font-size: 1.7rem; font-weight: 700; color: #58a6ff;
               letter-spacing: 0.5px; display: block; }
.chunk-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 0.8rem 1rem; margin-bottom: 0.6rem;
}
.chunk-tag {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 0.75rem; font-weight: 600; margin-bottom: 0.4rem;
}
.tag-function { background: #1f6feb; color: white; }
.tag-class    { background: #388bfd22; color: #58a6ff; border: 1px solid #388bfd; }
.tag-imports  { background: #2ea04326; color: #3fb950; border: 1px solid #2ea043; }
.tag-code     { background: #6e40c926; color: #bc8cff; border: 1px solid #6e40c9; }

[data-testid="stChatMessage"] {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; margin-bottom: 0.6rem; padding: 0.8rem 1rem;
}
.stButton > button {
    background: #238636;
    color: white; border: none; border-radius: 6px; font-weight: 600;
}
.stButton > button:hover { background: #2ea043; }
.stTabs [data-baseweb="tab"] { color: #8b949e; }
.stTabs [aria-selected="true"] { color: #58a6ff; border-bottom-color: #58a6ff; }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ─────────────────────────────────────────────────────
for key, default in {
    "messages":     [],
    "collection":   QUICK_TEST_COLLECTION,
    "model":        "llama-3.3-70b-versatile",
    "embed_model":  "mxbai-embed-large",
    "top_k":        6,
    "show_sources": True,
    "ingested_files": [],
    "session_tokens": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════════════
# CACHED RESOURCES  (same @st.cache_resource pattern as eng_rag)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_llm(model_name: str) -> ChatGroq:
    """Load LLM once, via Groq's API."""
    return ChatGroq(model=model_name, temperature=0.1, max_tokens=2048, api_key=os.getenv("GROQ_API_KEY"))


@st.cache_resource
def load_doc_generator(model_name: str) -> CodeDocGenerator:
    """Cache the documentation generator LLM wrapper."""
    return CodeDocGenerator(model_name=model_name)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class='brand-block'>
        <span class='brand-title'>DevOne</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Model config ───────────────────────────────────────────────────────────
    st.markdown("### Model")
    available_models = get_available_models()
    if not available_models:
        st.warning("No Groq models found — check `GROQ_API_KEY` in your .env file.")
        available_models = ["llama-3.3-70b-versatile"]

    model_index = (available_models.index(st.session_state.model)
                   if st.session_state.model in available_models else 0)
    new_model = st.selectbox("LLM (via Groq)", available_models, index=model_index)
    if new_model != st.session_state.model:
        st.session_state.model = new_model
        load_llm.clear()
        load_doc_generator.clear()

    new_embed = st.selectbox(
        "Embedding model",
        ["mxbai-embed-large"],  # fixed: sqlite-vec's table dimension is locked in at creation
    )
    if new_embed != st.session_state.embed_model:
        st.session_state.embed_model = new_embed

    new_topk = st.slider("Top-K retrieved chunks", 2, 12, st.session_state.top_k)
    if new_topk != st.session_state.top_k:
        st.session_state.top_k = new_topk

    st.divider()

    # ── GitHub repo ────────────────────────────────────────────────────────────
    st.markdown("### Connect GitHub Repo")
    repo_url = st.text_input("Repo URL", placeholder="https://github.com/owner/repo")

    if st.button("Connect / Sync", use_container_width=True) and repo_url.strip():
        token = os.getenv("GITHUB_PAT")
        if not token:
            st.error("GITHUB_PAT not found — check your .env file.")
        else:
            progress = st.empty()

            def _progress(done, total, path):
                progress.markdown(f"_Ingesting {done}/{total}: {path}…_")

            try:
                with st.spinner("Cloning and indexing repository…"):
                    result = ingest_repo(repo_url.strip(), token,
                                         st.session_state.embed_model,
                                         progress_callback=_progress)
                progress.empty()
                st.session_state.collection = result["collection"]

                if result["status"] == "up_to_date":
                    st.info(
                        f"**{result['collection']}** already up to date "
                        f"(commit `{result['commit_sha'][:7]}`, {result['elapsed_seconds']}s)"
                    )
                else:
                    verb = "connected" if result["status"] == "first_ingest" else "synced"
                    deleted_note = (f", {result['files_deleted']} removed"
                                    if result["files_deleted"] else "")
                    st.success(
                        f"**{result['collection']}** {verb} — "
                        f"{result['files_ingested']} files updated{deleted_note}, "
                        f"{result['total_chunks']} chunks "
                        f"(commit `{result['commit_sha'][:7]}`, {result['elapsed_seconds']}s)"
                    )
            except Exception as e:
                progress.empty()
                st.error(f"Failed to connect repo: {e}")

    st.divider()

    # ── Active project ─────────────────────────────────────────────────────────
    st.markdown("### Active Project")
    tracked_repos = list_repos()
    repo_names    = [r["name"] for r in tracked_repos]
    ad_hoc        = [c for c in list_collections() if c not in repo_names]
    all_projects  = repo_names + ad_hoc

    if all_projects:
        current_index = (all_projects.index(st.session_state.collection)
                         if st.session_state.collection in all_projects else 0)
        selected = st.selectbox("Working on", all_projects, index=current_index)
        if selected != st.session_state.collection:
            st.session_state.collection = selected

        if st.session_state.collection in repo_names and st.button(
            "Check for updates", use_container_width=True
        ):
            token = os.getenv("GITHUB_PAT")
            repo_info = next(r for r in tracked_repos if r["name"] == st.session_state.collection)
            status = check_remote_status(repo_info["url"], token)
            if status["status"] == "up_to_date":
                st.info("Up to date.")
            elif status["status"] == "behind":
                st.warning("New commits available — reconnect above to sync.")
            else:
                st.warning("Could not check — repo not found locally.")
    else:
        st.caption("No projects yet — connect a repo above or upload a file below.")

    if st.session_state.collection and st.button("Delete active project", use_container_width=True):
        delete_collection(st.session_state.collection)
        st.session_state.ingested_files = []
        st.success("Deleted.")

    st.divider()

    # ── File upload ────────────────────────────────────────────────────────────
    st.markdown("### Upload Source Code (quick test)")
    st.caption("Supported: `.py` `.java` `.js` `.cpp` `.c`")

    uploaded_files = st.file_uploader(
        "Select code files",
        type=["py", "java", "js", "cpp", "c"],
        accept_multiple_files=True,
        key="code_uploader",
    )

    if uploaded_files and st.button("Ingest Files", use_container_width=True):
        for f in uploaded_files:
            with st.spinner(f"Parsing & embedding {f.name}…"):
                source_code = f.read().decode("utf-8", errors="replace")
                summary = ingest_code_file(
                    filename=f.name,
                    source_code=source_code,
                    collection=QUICK_TEST_COLLECTION,
                    embed_model=st.session_state.embed_model,
                )
            if f.name not in st.session_state.ingested_files:
                st.session_state.ingested_files.append(f.name)
            st.success(
                f"**{f.name}** ingested — "
                f"{summary['functions']} functions, "
                f"{summary['classes']} classes, "
                f"{summary['total_chunks']} chunks"
            )
        st.session_state.collection = QUICK_TEST_COLLECTION

    # Show ingested files
    if st.session_state.ingested_files:
        st.divider()
        st.markdown("### Ingested Files")
        for fname in st.session_state.ingested_files:
            st.caption(fname)

    st.divider()

    # ── Options ────────────────────────────────────────────────────────────────
    st.markdown("### Options")
    st.session_state.show_sources = st.toggle("Show tool activity",
                                               value=st.session_state.show_sources)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_tokens = 0
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PANEL — TABS
# ══════════════════════════════════════════════════════════════════════════════

if not list_collections():
    st.info("Upload source code files from the sidebar to get started.")

tab_chat, tab_docs, tab_issues, tab_readme, tab_usage = st.tabs([
    "Chat",
    "Generate Docs",
    "Issues & Improvements",
    "README Generator",
    "Usage Stats",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("#### Ask anything about your codebase")
    st.caption("Examples: *Explain the main function* · *What does the Parser class do?* · *How is data stored?*")

    if st.session_state.session_tokens > LONG_SESSION_TOKEN_WARNING:
        st.info(
            f"This conversation has used ~{st.session_state.session_tokens:,} tokens so far — "
            "each new question resends the full history. Consider **Clear chat** in the "
            "sidebar to keep responses faster and cheaper."
        )

    # Replay history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources") and st.session_state.show_sources:
                with st.expander("Tool Activity"):
                    st.markdown(msg["sources"])

    # Chat input
    if question := st.chat_input("Ask about your code…"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            status = st.empty()
            status.markdown("_Thinking…_")
            sources_md = ""

            try:
                llm = load_llm(st.session_state.model)
                # Prior turns only — question + final answer, not each turn's
                # tool-calling scaffolding (keeps re-sent history lightweight).
                history = [
                    (m["role"], m["content"])
                    for m in st.session_state.messages[:-1]
                ]
                result = answer_with_tools(
                    question, llm, st.session_state.model,
                    collection=st.session_state.collection,
                    embed_model=st.session_state.embed_model,
                    top_k=st.session_state.top_k,
                    history=history,
                )
                response = result["answer"]
                st.session_state.session_tokens += result["tokens"]["total"]

                status.empty()
                st.markdown(response)

                if result["tool_calls"]:
                    sources_md = "**Tools used:**\n\n"
                    for call in result["tool_calls"]:
                        sources_md += (
                            f"<div class='chunk-card'>"
                            f"<span class='chunk-tag tag-code'>{call['name']}</span><br>"
                            f"<small>{call['output'][:200].strip()}…</small>"
                            f"</div>\n"
                        )

                if sources_md and st.session_state.show_sources:
                    with st.expander("Tool Activity", expanded=False):
                        st.markdown(sources_md, unsafe_allow_html=True)

            except Exception as e:
                if "tool calling is not supported" in str(e).lower():
                    response = (
                        f"**Error:** `{st.session_state.model}` doesn't support tool calling, "
                        "which this agentic chat mode requires.\n\n"
                        "Switch to **llama-3.3-70b-versatile** (or another tool-calling-capable "
                        "model) in the sidebar and try again."
                    )
                elif "tool_use_failed" in str(e).lower():
                    response = (
                        "**The model had trouble forming a tool call for that question.** "
                        "This is an occasional model reliability quirk, not a bug in the app.\n\n"
                        "Try rephrasing with a shorter, more specific search term, or just ask again."
                    )
                else:
                    response = (
                        f"**Error:** `{e}`\n\n"
                        "**Checklist:**\n"
                        "- Is `GROQ_API_KEY` set correctly in your `.env`?\n"
                        f"- Is `{st.session_state.model}` a currently available Groq model?\n"
                        "- Is Ollama running? (`ollama serve`) — still needed for embeddings\n"
                        f"- Is `{st.session_state.embed_model}` pulled? "
                        f"(`ollama pull {st.session_state.embed_model}`)\n"
                        "- Have you connected a repo or uploaded code files from the sidebar?"
                    )
                status.empty()
                st.markdown(response)
                sources_md = ""

        st.session_state.messages.append({
            "role": "assistant", "content": response, "sources": sources_md
        })


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: DOCUMENTATION GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_docs:
    st.markdown("#### Generate Documentation")
    st.caption("Auto-generate docstrings and summaries for your code.")

    col1, col2 = st.columns([2, 1])

    with col1:
        all_docs = get_all_chunks(
            collection=st.session_state.collection,
            embed_model=st.session_state.embed_model,
        )

        # File selector
        files_in_kb = list(set(d.metadata.get("source", "") for d in all_docs if d.metadata.get("source")))
        if not files_in_kb:
            st.info("No files ingested yet. Upload code from the sidebar first.")
        else:
            selected_file = st.selectbox("Select file to document", files_in_kb)
            doc_type = st.radio(
                "Documentation type",
                ["File Summary", "Function Docs", "Class Docs", "Find Issues", "Suggest Improvements"],
                horizontal=True,
            )

            if st.button("Generate Documentation", use_container_width=True):
                file_docs = [d for d in all_docs if d.metadata.get("source") == selected_file]
                full_code  = "\n\n".join(d.page_content for d in file_docs)
                language   = file_docs[0].metadata.get("language", "") if file_docs else ""

                gen = load_doc_generator(st.session_state.model)

                proj = st.session_state.collection

                with st.spinner("Generating…"):
                    if doc_type == "File Summary":
                        result = gen.summarize_file(full_code, selected_file, language, collection=proj)

                    elif doc_type == "Function Docs":
                        fn_docs = [d for d in file_docs if d.metadata.get("chunk_type") == "function"]
                        if not fn_docs:
                            result = "No functions found in this file."
                        else:
                            parts = []
                            for fd in fn_docs[:8]:  # cap at 8 to avoid timeout
                                name   = fd.metadata.get("name", "unknown")
                                parts.append(f"### `{name}()`\n\n"
                                             + gen.document_function(fd.page_content, name, selected_file, collection=proj))
                            result = "\n\n---\n\n".join(parts)

                    elif doc_type == "Class Docs":
                        cls_docs = [d for d in file_docs if d.metadata.get("chunk_type") == "class"]
                        if not cls_docs:
                            result = "No classes found in this file."
                        else:
                            parts = []
                            for cd in cls_docs[:5]:
                                name   = cd.metadata.get("name", "unknown")
                                parts.append(f"### Class `{name}`\n\n"
                                             + gen.document_class(cd.page_content, name, selected_file, collection=proj))
                            result = "\n\n---\n\n".join(parts)

                    elif doc_type == "Find Issues":
                        result = gen.find_issues(full_code, selected_file, collection=proj)

                    else:  # Suggest Improvements
                        result = gen.suggest_improvements(full_code, selected_file, collection=proj)

                st.markdown("---")
                st.markdown(result)

                # Download button
                st.download_button(
                    "Download as Markdown",
                    data=result,
                    file_name=f"{selected_file}_{doc_type.lower().replace(' ','_')}.md",
                    mime="text/markdown",
                )

    with col2:
        if files_in_kb:
            st.markdown("#### Codebase Stats")
            funcs   = sum(1 for d in all_docs if d.metadata.get("chunk_type") == "function")
            classes = sum(1 for d in all_docs if d.metadata.get("chunk_type") == "class")
            total   = len(all_docs)
            st.metric("Files", len(files_in_kb))
            st.metric("Functions", funcs)
            st.metric("Classes", classes)
            st.metric("Total Chunks", total)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: ISSUES & IMPROVEMENTS (quick access)
# ══════════════════════════════════════════════════════════════════════════════
with tab_issues:
    st.markdown("#### Code Review Assistant")
    st.caption("Paste any code snippet for instant AI review.")

    pasted_code = st.text_area(
        "Paste code here",
        height=280,
        placeholder="Paste any function, class, or code block here…",
    )
    fname_input = st.text_input("Filename (optional)", value="snippet.py")
    review_type = st.radio("Review type", ["Find Issues", "Suggest Improvements"], horizontal=True)

    if st.button("Run Code Review", use_container_width=True) and pasted_code.strip():
        gen = load_doc_generator(st.session_state.model)
        with st.spinner("Reviewing code…"):
            if review_type == "Find Issues":
                result = gen.find_issues(pasted_code, fname_input, collection="pasted-snippet")
            else:
                result = gen.suggest_improvements(pasted_code, fname_input, collection="pasted-snippet")
        st.markdown("---")
        st.markdown(result)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: README GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_readme:
    st.markdown("#### README Generator")
    st.caption("Generate a professional README.md from your uploaded codebase.")

    all_docs_readme = get_all_chunks(
        collection=st.session_state.collection,
        embed_model=st.session_state.embed_model,
    )

    if not all_docs_readme:
        st.info("No files ingested yet. Upload code from the sidebar first.")
    else:
        files_list = list(set(
            d.metadata.get("source", "") for d in all_docs_readme
            if d.metadata.get("source")
        ))
        st.info(f"Will generate README based on **{len(files_list)} file(s)**: "
                + ", ".join(f"`{f}`" for f in files_list))

        if st.button("Generate README.md", use_container_width=True):
            gen = load_doc_generator(st.session_state.model)
            with st.spinner("Analysing codebase and generating README…"):
                summary   = build_codebase_summary(all_docs_readme)
                file_list = "\n".join(f"- {f}" for f in files_list)
                readme    = gen.generate_readme(summary, file_list, collection=st.session_state.collection)

            st.markdown("---")
            st.markdown(readme)
            st.download_button(
                "Download README.md",
                data=readme,
                file_name="README.md",
                mime="text/markdown",
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: USAGE STATS
# ══════════════════════════════════════════════════════════════════════════════
with tab_usage:
    st.markdown("#### Token Usage")
    st.caption("Real token counts per call — no dollar-cost estimate, since current "
               "per-model API pricing isn't something to guess at reliably.")

    scope = st.radio("Scope", ["All projects", "Active project only"], horizontal=True)
    stats = get_usage_stats(st.session_state.collection if scope == "Active project only" else None)

    if stats["total_queries"] == 0:
        st.info("No usage logged yet — ask something in Chat or generate some docs first.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Queries", stats["total_queries"])
        col2.metric("Input Tokens", f"{stats['total_input_tokens']:,}")
        col3.metric("Output Tokens", f"{stats['total_output_tokens']:,}")
        col4.metric("Total Tokens", f"{stats['total_tokens']:,}")

        st.markdown("##### By Model")
        for model, data in stats["by_model"].items():
            st.markdown(
                f"<div class='chunk-card'>"
                f"<span class='chunk-tag tag-code'>{model}</span> "
                f"{data['queries']} calls — {data['input']:,} in / {data['output']:,} out / "
                f"{data['total']:,} total tokens"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("##### Recent Activity")
        for entry in stats["recent"]:
            st.caption(
                f"{entry['timestamp']} · {entry['source']} · {entry['model']} · "
                f"{entry['total_tokens']:,} tokens"
            )
