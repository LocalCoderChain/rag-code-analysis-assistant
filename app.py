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

import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from vector import (
    get_retriever,
    ingest_code_file,
    get_all_chunks,
    list_collections,
    delete_collection,
)
from src.doc_generator import CodeDocGenerator, build_codebase_summary

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CodeRAG Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', sans-serif; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #30363d;
}
.brand-block {
    text-align: center; padding: 1.2rem 0.5rem 1rem;
    border-bottom: 1px solid #30363d; margin-bottom: 0.8rem;
}
.brand-icon  { font-size: 2.4rem; display: block; }
.brand-title { font-size: 1.5rem; font-weight: 800; color: #58a6ff;
               letter-spacing: 1px; display: block; }
.brand-sub   { font-size: 0.7rem; color: #8b949e; letter-spacing: 2px;
               text-transform: uppercase; display: block; margin-top: 3px; }

.main-header {
    background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
    border: 1px solid #30363d; border-radius: 12px;
    padding: 1.5rem 2rem; margin-bottom: 1.2rem;
}
.main-header h1 { color: #58a6ff; margin: 0 0 0.3rem; font-size: 1.7rem; }
.main-header p  { color: #8b949e; margin: 0; font-size: 0.9rem; }

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
    background: linear-gradient(135deg, #238636, #196127);
    color: white; border: none; border-radius: 8px; font-weight: 600;
}
.stButton > button:hover { background: linear-gradient(135deg, #2ea043, #238636); }
.stTabs [data-baseweb="tab"] { color: #8b949e; }
.stTabs [aria-selected="true"] { color: #58a6ff; border-bottom-color: #58a6ff; }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ─────────────────────────────────────────────────────
for key, default in {
    "messages":     [],
    "collection":   "code_kb",
    "model":        "llama3.2",
    "embed_model":  "mxbai-embed-large",
    "top_k":        6,
    "show_sources": True,
    "ingested_files": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════════════
# CACHED RESOURCES  (same @st.cache_resource pattern as eng_rag)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_llm(model_name: str) -> OllamaLLM:
    """Load LLM once. Same: OllamaLLM(model=...) from eng_rag."""
    return OllamaLLM(model=model_name, temperature=0.1)


@st.cache_resource
def load_retriever(collection: str, embed_model: str, top_k: int):
    """
    Cache retriever. Same: vector_store.as_retriever(search_kwargs={"k":k})
    """
    return get_retriever(collection=collection, embed_model=embed_model, top_k=top_k)


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
        <span class='brand-title'>CodeRAG</span>
        <span class='brand-sub'>AI Code Assistant</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Model config ───────────────────────────────────────────────────────────
    st.markdown("### 🤖 Model")
    new_model = st.selectbox(
        "LLM (via Ollama)",
        ["llama3.2", "llama3.1", "llama3", "mistral", "codellama",
         "deepseek-coder", "gemma2", "phi3", "qwen2.5-coder"],
        index=0,
    )
    if new_model != st.session_state.model:
        st.session_state.model = new_model
        load_llm.clear()
        load_doc_generator.clear()

    new_embed = st.selectbox(
        "Embedding model",
        ["mxbai-embed-large", "nomic-embed-text", "all-minilm"],
    )
    if new_embed != st.session_state.embed_model:
        st.session_state.embed_model = new_embed
        load_retriever.clear()

    new_topk = st.slider("Top-K retrieved chunks", 2, 12, st.session_state.top_k)
    if new_topk != st.session_state.top_k:
        st.session_state.top_k = new_topk
        load_retriever.clear()

    st.divider()

    # ── Collection ─────────────────────────────────────────────────────────────
    st.markdown("### 🗄️ Project Collection")
    existing = list_collections()
    if existing:
        st.caption("Existing: " + ", ".join(f"`{c}`" for c in existing))

    col_input = st.text_input("Collection name", value=st.session_state.collection)
    if col_input and col_input != st.session_state.collection:
        st.session_state.collection = col_input
        load_retriever.clear()

    if st.button("🗑️ Delete collection", use_container_width=True):
        delete_collection(st.session_state.collection)
        st.session_state.ingested_files = []
        load_retriever.clear()
        st.success("Collection deleted.")

    st.divider()

    # ── File upload ────────────────────────────────────────────────────────────
    st.markdown("### 📁 Upload Source Code")
    st.caption("Supported: `.py` `.java` `.js` `.cpp` `.c`")

    uploaded_files = st.file_uploader(
        "Select code files",
        type=["py", "java", "js", "cpp", "c"],
        accept_multiple_files=True,
        key="code_uploader",
    )

    if uploaded_files and st.button("⚙️ Ingest Files", use_container_width=True):
        for f in uploaded_files:
            with st.spinner(f"Parsing & embedding {f.name}…"):
                source_code = f.read().decode("utf-8", errors="replace")
                summary = ingest_code_file(
                    filename=f.name,
                    source_code=source_code,
                    collection=st.session_state.collection,
                    embed_model=st.session_state.embed_model,
                )
            if f.name not in st.session_state.ingested_files:
                st.session_state.ingested_files.append(f.name)
            st.success(
                f"✅ **{f.name}** → "
                f"{summary['functions']} functions, "
                f"{summary['classes']} classes, "
                f"{summary['total_chunks']} chunks"
            )
        load_retriever.clear()

    # Show ingested files
    if st.session_state.ingested_files:
        st.divider()
        st.markdown("### 📋 Ingested Files")
        for fname in st.session_state.ingested_files:
            ext = fname.rsplit(".", 1)[-1] if "." in fname else ""
            icons = {"py": "🐍", "java": "☕", "js": "⚡", "cpp": "⚙️", "c": "🔧"}
            st.caption(f"{icons.get(ext, '📄')} {fname}")

    st.divider()

    # ── Options ────────────────────────────────────────────────────────────────
    st.markdown("### ⚙️ Options")
    st.session_state.show_sources = st.toggle("Show retrieved chunks",
                                               value=st.session_state.show_sources)
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES  (engineering-quality prompts for code)
# ══════════════════════════════════════════════════════════════════════════════

CODE_QA_TEMPLATE = """
You are a senior software engineer helping a developer understand a codebase.

Use ONLY the retrieved code context below to answer the question.
If the code context doesn't contain enough information, say so — do not invent code.

Retrieved code context:
{context}

Developer's question:
{question}

Provide a clear, precise technical answer.
Reference specific function names, class names, or line patterns from the context.
Use markdown code blocks when showing code examples.
"""

CODE_EXPLAIN_TEMPLATE = """
You are a senior software engineer explaining code to a developer.

Explain the following code context in detail:
- What it does (high level)
- How it works (step by step)
- Key variables or data structures used
- Any patterns or design decisions worth noting

Retrieved code context:
{context}

Question / focus area:
{question}

Be clear and educational. Use markdown formatting.
"""


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PANEL — TABS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class='main-header'>
    <h1>🧑‍💻 Code Analysis & Documentation Assistant</h1>
    <p>Upload source code files and ask questions, generate docs,
    find issues — all locally with no data leaving your machine.</p>
</div>
""", unsafe_allow_html=True)

if not list_collections():
    st.info("👈 Upload source code files from the sidebar to get started.")

tab_chat, tab_docs, tab_issues, tab_readme = st.tabs([
    "💬 Chat",
    "📝 Generate Docs",
    "🔍 Issues & Improvements",
    "📄 README Generator",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("#### Ask anything about your codebase")
    st.caption("Examples: *Explain the main function* · *What does the Parser class do?* · *How is data stored?*")

    # Replay history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources") and st.session_state.show_sources:
                with st.expander("📎 Retrieved Code Chunks"):
                    st.markdown(msg["sources"])

    # Chat input
    if question := st.chat_input("Ask about your code…"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            status = st.empty()
            status.markdown("_🔍 Searching codebase…_")
            sources_md = ""

            try:
                # Retrieve relevant code chunks
                retriever = load_retriever(
                    collection=st.session_state.collection,
                    embed_model=st.session_state.embed_model,
                    top_k=st.session_state.top_k,
                )
                docs = retriever.invoke(question)

                # Build context with code metadata
                context_parts = []
                for doc in docs:
                    meta   = doc.metadata
                    header = (f"[File: {meta.get('source','?')} | "
                              f"Type: {meta.get('chunk_type','?')} | "
                              f"Name: {meta.get('name','?')}]")
                    context_parts.append(f"{header}\n```{meta.get('language','')}\n"
                                         f"{doc.page_content}\n```")
                context = "\n\n---\n\n".join(context_parts)

                # Choose template based on question type
                explain_keywords = ["explain", "what does", "how does", "describe",
                                    "summarize", "summarise", "what is"]
                template = CODE_EXPLAIN_TEMPLATE if any(
                    k in question.lower() for k in explain_keywords
                ) else CODE_QA_TEMPLATE

                # Build and invoke chain (same pattern as eng_rag)
                status.markdown("_⚡ Generating answer…_")
                llm    = load_llm(st.session_state.model)
                prompt = ChatPromptTemplate.from_template(template)
                chain  = prompt | llm
                response = chain.invoke({"context": context, "question": question})

                status.empty()
                st.markdown(response)

                # Format source citations
                if docs:
                    sources_md = "**Retrieved code chunks:**\n\n"
                    for doc in docs:
                        meta  = doc.metadata
                        ctype = meta.get("chunk_type", "code")
                        name  = meta.get("name", "")
                        src   = meta.get("source", "?")
                        lang  = meta.get("language", "")
                        tag_class = {
                            "function": "tag-function",
                            "class":    "tag-class",
                            "imports":  "tag-imports",
                        }.get(ctype, "tag-code")
                        label = f"{ctype}: {name}" if name else ctype
                        sources_md += (
                            f"<div class='chunk-card'>"
                            f"<span class='chunk-tag {tag_class}'>{label}</span> "
                            f"<code>{src}</code><br>"
                            f"<small>{doc.page_content[:150].strip()}…</small>"
                            f"</div>\n"
                        )

                if sources_md and st.session_state.show_sources:
                    with st.expander("📎 Retrieved Code Chunks", expanded=False):
                        st.markdown(sources_md, unsafe_allow_html=True)

            except Exception as e:
                response = (
                    f"⚠️ **Error:** `{e}`\n\n"
                    "**Checklist:**\n"
                    "- Is Ollama running? (`ollama serve`)\n"
                    f"- Is `{st.session_state.model}` pulled? "
                    f"(`ollama pull {st.session_state.model}`)\n"
                    f"- Is `{st.session_state.embed_model}` pulled? "
                    f"(`ollama pull {st.session_state.embed_model}`)\n"
                    "- Have you uploaded and ingested code files from the sidebar?"
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

            if st.button("🧠 Generate Documentation", use_container_width=True):
                file_docs = [d for d in all_docs if d.metadata.get("source") == selected_file]
                full_code  = "\n\n".join(d.page_content for d in file_docs)
                language   = file_docs[0].metadata.get("language", "") if file_docs else ""

                gen = load_doc_generator(st.session_state.model)

                with st.spinner("Generating…"):
                    if doc_type == "File Summary":
                        result = gen.summarize_file(full_code, selected_file, language)

                    elif doc_type == "Function Docs":
                        fn_docs = [d for d in file_docs if d.metadata.get("chunk_type") == "function"]
                        if not fn_docs:
                            result = "⚠️ No functions found in this file."
                        else:
                            parts = []
                            for fd in fn_docs[:8]:  # cap at 8 to avoid timeout
                                name   = fd.metadata.get("name", "unknown")
                                parts.append(f"### `{name}()`\n\n"
                                             + gen.document_function(fd.page_content, name, selected_file))
                            result = "\n\n---\n\n".join(parts)

                    elif doc_type == "Class Docs":
                        cls_docs = [d for d in file_docs if d.metadata.get("chunk_type") == "class"]
                        if not cls_docs:
                            result = "⚠️ No classes found in this file."
                        else:
                            parts = []
                            for cd in cls_docs[:5]:
                                name   = cd.metadata.get("name", "unknown")
                                parts.append(f"### Class `{name}`\n\n"
                                             + gen.document_class(cd.page_content, name, selected_file))
                            result = "\n\n---\n\n".join(parts)

                    elif doc_type == "Find Issues":
                        result = gen.find_issues(full_code, selected_file)

                    else:  # Suggest Improvements
                        result = gen.suggest_improvements(full_code, selected_file)

                st.markdown("---")
                st.markdown(result)

                # Download button
                st.download_button(
                    "⬇️ Download as Markdown",
                    data=result,
                    file_name=f"{selected_file}_{doc_type.lower().replace(' ','_')}.md",
                    mime="text/markdown",
                )

    with col2:
        if files_in_kb:
            st.markdown("#### 📊 Codebase Stats")
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

    if st.button("🔍 Run Code Review", use_container_width=True) and pasted_code.strip():
        gen = load_doc_generator(st.session_state.model)
        with st.spinner("Reviewing code…"):
            if review_type == "Find Issues":
                result = gen.find_issues(pasted_code, fname_input)
            else:
                result = gen.suggest_improvements(pasted_code, fname_input)
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

        if st.button("📄 Generate README.md", use_container_width=True):
            gen = load_doc_generator(st.session_state.model)
            with st.spinner("Analysing codebase and generating README…"):
                summary   = build_codebase_summary(all_docs_readme)
                file_list = "\n".join(f"- {f}" for f in files_list)
                readme    = gen.generate_readme(summary, file_list)

            st.markdown("---")
            st.markdown(readme)
            st.download_button(
                "⬇️ Download README.md",
                data=readme,
                file_name="README.md",
                mime="text/markdown",
            )
