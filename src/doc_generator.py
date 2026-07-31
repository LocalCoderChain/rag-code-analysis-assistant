"""
doc_generator.py  —  AI Documentation Generator
=================================================
Uses the same chain = prompt | model pattern from eng_rag.
Generates:
  - Per-function docstrings / summaries
  - Per-class summaries
  - Per-file summaries
  - Full project README

All generation is RAG-grounded: it reads chunks from the local store
rather than hallucinating structure.
"""

import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from vector import log_usage


# ── Prompt templates ───────────────────────────────────────────────────────────

FUNCTION_DOC_TEMPLATE = """
You are a senior software engineer writing professional documentation.

Analyze the following function/method code and generate:
1. A one-line summary
2. Parameters description (if any)
3. Return value description (if any)
4. What the function does (2-3 sentences)
5. Any important notes or edge cases

Source file: {filename}
Function name: {name}

Code:
{code}

Generate clean, professional documentation. Use markdown formatting.
"""

CLASS_DOC_TEMPLATE = """
You are a senior software engineer writing professional documentation.

Analyze the following class code and generate:
1. A one-line summary of what this class represents
2. Its purpose and responsibility
3. Key methods and their roles
4. Usage context (when would you use this class?)

Source file: {filename}
Class name: {name}

Code:
{code}

Generate clean, professional documentation. Use markdown formatting.
"""

FILE_SUMMARY_TEMPLATE = """
You are a senior software engineer writing technical documentation.

Analyze the following source code file and generate:
1. File purpose (1-2 sentences)
2. Key components (functions, classes, constants)
3. Dependencies / imports used
4. How this file fits into a larger project
5. Overall code quality observations

Filename: {filename}
Language: {language}

Code:
{code}

Generate a clear, professional file summary. Use markdown formatting.
"""

README_TEMPLATE = """
You are a senior software engineer creating project documentation.

Based on the following codebase summary, generate a professional README.md that includes:

1. Project title and description
2. Features list
3. Tech stack / dependencies detected
4. Installation instructions (generic)
5. Usage instructions
6. Project structure explanation
7. Key functions/classes overview
8. Contributing guidelines (brief)
9. License placeholder

Codebase summary:
{codebase_summary}

Files analyzed:
{file_list}

Generate a complete, professional README.md in markdown format.
"""

ISSUE_FINDER_TEMPLATE = """
You are a senior code reviewer performing a thorough code review.

Analyze the following code and identify:
1. Potential bugs or logic errors
2. Security vulnerabilities (SQL injection, unvalidated input, etc.)
3. Performance issues
4. Code style / best practice violations
5. Missing error handling
6. Memory leaks or resource management issues

Source file: {filename}

Code:
{code}

Be specific. Reference line numbers or function names where possible.
Use markdown formatting with severity labels: [Critical], [Warning], [Suggestion]
"""

IMPROVEMENT_TEMPLATE = """
You are a senior software architect providing code improvement suggestions.

Review the following code and suggest specific improvements for:
1. Code structure and organization
2. Performance optimizations
3. Better variable/function naming
4. Design pattern opportunities
5. Refactoring opportunities
6. Test coverage suggestions

Source file: {filename}

Code:
{code}

Provide actionable, specific suggestions. Use markdown formatting.
"""

REPO_OVERVIEW_TEMPLATE = """
You are a senior software engineer writing a project overview for a new developer joining the team.

You've been given individual summaries of each file in this project (not the raw code, since a whole repo doesn't fit in one context window). Based on these, write a cohesive overview that explains:
1. What this project actually does, in plain language (2-3 sentences)
2. The overall architecture — how the pieces fit together
3. The main technologies/frameworks used
4. Anything notable about how it's designed

Project: {project_name}

Per-file summaries:
{file_summaries}

Write a clear, well-organized overview. Use markdown formatting. Synthesize what the project as a whole is for and how it works — don't just restate the file list.
"""


# ══════════════════════════════════════════════════════════════════════════════
# GENERATOR CLASS
# ══════════════════════════════════════════════════════════════════════════════

class CodeDocGenerator:
    """
    Wraps the LLM chain for all documentation generation tasks.
    Same chain = prompt | model pattern as eng_rag.
    """

    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.model_name = model_name
        self.model = ChatGroq(model=model_name, temperature=0.1, max_tokens=2048, api_key=os.getenv("GROQ_API_KEY"))

    def _run(self, template: str, collection: str = None, **kwargs) -> str:
        """Generic chain runner: chain = prompt | model → chain.invoke(...)"""
        prompt   = ChatPromptTemplate.from_template(template)
        chain    = prompt | self.model
        response = chain.invoke(kwargs)

        usage = getattr(response, "usage_metadata", None) or {}
        log_usage(
            collection, kwargs.get("filename") or kwargs.get("name") or "", "doc_generation",
            self.model_name, usage.get("input_tokens", 0), usage.get("output_tokens", 0),
            usage.get("total_tokens", 0),
        )
        return response.content

    def document_function(self, code: str, name: str, filename: str, collection: str = None) -> str:
        return self._run(FUNCTION_DOC_TEMPLATE, collection=collection, code=code, name=name, filename=filename)

    def document_class(self, code: str, name: str, filename: str, collection: str = None) -> str:
        return self._run(CLASS_DOC_TEMPLATE, collection=collection, code=code, name=name, filename=filename)

    def summarize_file(self, code: str, filename: str, language: str, collection: str = None) -> str:
        return self._run(FILE_SUMMARY_TEMPLATE, collection=collection,
                         code=code[:4000], filename=filename, language=language)

    def generate_readme(self, codebase_summary: str, file_list: str, collection: str = None) -> str:
        return self._run(README_TEMPLATE, collection=collection,
                         codebase_summary=codebase_summary, file_list=file_list)

    def find_issues(self, code: str, filename: str, collection: str = None) -> str:
        return self._run(ISSUE_FINDER_TEMPLATE, collection=collection, code=code[:4000], filename=filename)

    def suggest_improvements(self, code: str, filename: str, collection: str = None) -> str:
        return self._run(IMPROVEMENT_TEMPLATE, collection=collection, code=code[:4000], filename=filename)

    def summarize_repo(self, file_summaries: str, project_name: str, collection: str = None) -> str:
        return self._run(REPO_OVERVIEW_TEMPLATE, collection=collection,
                         file_summaries=file_summaries, project_name=project_name)


def build_codebase_summary(docs: list[Document]) -> str:
    """
    Build a structured summary of all ingested code chunks.
    Used as input for README generation.
    """
    files     = {}
    functions = []
    classes   = []

    for doc in docs:
        src   = doc.metadata.get("source", "unknown")
        ctype = doc.metadata.get("chunk_type", "")
        name  = doc.metadata.get("name", "")
        lang  = doc.metadata.get("language", "")

        files[src] = lang
        if ctype == "function" and name:
            functions.append(f"  - {name}() in {src}")
        if ctype == "class" and name:
            classes.append(f"  - {name} in {src}")

    lines = []
    lines.append(f"Files ({len(files)}):")
    for f, l in files.items():
        lines.append(f"  - {f} ({l})")

    if functions:
        lines.append(f"\nFunctions ({len(functions)}):")
        lines.extend(functions[:30])  # cap to avoid token overflow

    if classes:
        lines.append(f"\nClasses ({len(classes)}):")
        lines.extend(classes[:20])

    return "\n".join(lines)


def generate_project_overview(all_docs: list[Document], generator: CodeDocGenerator,
                              project_name: str, collection: str = None,
                              max_files: int = 20, progress_callback=None) -> str:
    """
    Map-reduce summarization for "what is this project?" questions: a whole
    repo's raw source won't fit in one context window, but a short summary
    per file will. Map: summarize each file individually. Reduce: summarize
    those summaries into one cohesive overview.
    """
    files = {}
    for doc in all_docs:
        src = doc.metadata.get("source", "unknown")
        files.setdefault(src, []).append(doc)

    filenames  = list(files.keys())[:max_files]
    truncated  = len(files) > max_files

    file_summaries = []
    for i, filename in enumerate(filenames):
        if progress_callback:
            progress_callback(i + 1, len(filenames), filename)
        docs     = files[filename]
        code     = "\n\n".join(d.page_content for d in docs)
        language = docs[0].metadata.get("language", "")
        summary  = generator.summarize_file(code, filename, language, collection=collection)
        file_summaries.append(f"### {filename}\n{summary}")

    overview = generator.summarize_repo("\n\n".join(file_summaries), project_name, collection=collection)

    if truncated:
        overview += f"\n\n*(Overview based on the first {max_files} of {len(files)} files.)*"

    return overview
