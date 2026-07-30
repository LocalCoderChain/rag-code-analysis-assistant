"""
agent_tools.py  —  Agentic Retrieval Loop
============================================
Instead of always doing plain top-k vector search, the chat model gets
three tools and decides which to call:
  - vector_search : semantic search over embedded chunks (broad/conceptual)
  - grep          : exact text search across the project's files on disk
  - read_file     : full content of one specific file

grep/read_file only work for repo-based projects — they need real files on
disk, which quick-test single-file uploads never have (see repo_manager.py's
get_local_path_for_collection). vector_search always works.

Built on LangGraph's prebuilt ReAct agent rather than a hand-rolled graph,
since the tool-calling loop (call tools, feed results back, repeat until a
final answer) is a well-trodden, already-solved pattern.
"""

import os
import re

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from vector import search as vector_search_impl
from repo_manager import get_local_path_for_collection

SKIP_DIRS = {".git", "node_modules", "__pycache__", "venv", "myvenv", ".venv", "dist", "build"}

SYSTEM_PROMPT = """You are a senior software engineer helping a developer understand their codebase.

You have three tools available:
- vector_search: semantic search over embedded code chunks — best for broad or conceptual questions.
- grep: exact text search across the project's files — best for finding where something specific is defined or used by name.
- read_file: read a file's full content — best when you need to see an entire file, not just a chunk (e.g. "what does X file do").

Use whichever tool(s) fit the question, and call more than one if the first doesn't give you enough to answer well. Reference specific files, functions, and line patterns in your answer. If the tools don't turn up enough information, say so clearly rather than guessing."""


def grep(pattern: str, collection: str, max_matches: int = 30) -> str:
    """Exact text search across every file in a repo-based project."""
    local_path = get_local_path_for_collection(collection)
    if not local_path:
        return "grep is only available for connected GitHub repos, not quick-test uploads."

    regex   = re.compile(re.escape(pattern), re.IGNORECASE)
    matches = []
    for root, dirs, files in os.walk(local_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            full_path = os.path.join(root, fname)
            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, start=1):
                        if regex.search(line):
                            rel = os.path.relpath(full_path, local_path)
                            matches.append(f"{rel}:{i}: {line.strip()}")
                            if len(matches) >= max_matches:
                                break
            except OSError:
                continue
            if len(matches) >= max_matches:
                break
        if len(matches) >= max_matches:
            break

    return "\n".join(matches) if matches else f"No matches found for '{pattern}'."


def read_file(path: str, collection: str, max_chars: int = 8000) -> str:
    """Read one file's full content from a repo-based project."""
    local_path = get_local_path_for_collection(collection)
    if not local_path:
        return "read_file is only available for connected GitHub repos, not quick-test uploads."

    full_path = os.path.normpath(os.path.join(local_path, path))
    if not full_path.startswith(os.path.normpath(local_path)):
        return "Invalid path."
    if not os.path.isfile(full_path):
        return f"File not found: {path}"

    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    if len(content) > max_chars:
        content = content[:max_chars] + "\n... (truncated)"
    return content


def _build_tools(collection: str, embed_model: str, top_k: int) -> list:
    """
    Wrap the three tools as closures over the current project/embed model,
    so the LLM only ever has to supply the part it should actually decide
    (query/pattern/path) — not app-level context like which project is active.
    """

    @tool
    def vector_search(query: str) -> str:
        """Semantic search over embedded code chunks in the current project.
        Use for broad or conceptual questions like "how does X work" or
        "explain the retrieval flow" — finds chunks similar in meaning to
        the query, even without exact wording overlap."""
        docs = vector_search_impl(query, collection, embed_model, top_k)
        if not docs:
            return "No relevant chunks found."
        parts = []
        for doc in docs:
            meta   = doc.metadata
            header = (f"[File: {meta.get('source','?')} | "
                      f"Type: {meta.get('chunk_type','?')} | Name: {meta.get('name','?')}]")
            parts.append(f"{header}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    @tool
    def grep_tool(pattern: str) -> str:
        """Exact text search across every file in the current project.
        Use to find where a specific function, class, or string is defined
        or used by name — e.g. "where is ingest_repo defined"."""
        return grep(pattern, collection)

    @tool
    def read_file_tool(path: str) -> str:
        """Read a file's full content by its path (e.g. "app.py" or
        "src/parser.py"). Use when you need the whole file, not just a
        chunk — e.g. "what does app.py do"."""
        return read_file(path, collection)

    return [vector_search, grep_tool, read_file_tool]


def answer_with_tools(question: str, llm, collection: str, embed_model: str, top_k: int = 6) -> dict:
    """
    Runs the agentic loop: the model can call vector_search/grep/read_file
    as needed before producing a final answer. Returns the answer plus
    which tools were actually used (name + output preview) for UI display.
    """
    tools = _build_tools(collection, embed_model, top_k)
    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

    result   = agent.invoke({"messages": [("user", question)]})
    messages = result["messages"]

    answer = messages[-1].content
    tool_calls = [
        {"name": m.name, "output": m.content}
        for m in messages
        if type(m).__name__ == "ToolMessage"
    ]

    return {"answer": answer, "tool_calls": tool_calls}
