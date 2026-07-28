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

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


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


# ══════════════════════════════════════════════════════════════════════════════
# GENERATOR CLASS
# ══════════════════════════════════════════════════════════════════════════════

class CodeDocGenerator:
    """
    Wraps the LLM chain for all documentation generation tasks.
    Same chain = prompt | model pattern as eng_rag.
    """

    def __init__(self, model_name: str = "llama3.2"):
        # Same as eng_rag: model = OllamaLLM(model=...)
        self.model = OllamaLLM(model=model_name, temperature=0.1)

    def _run(self, template: str, **kwargs) -> str:
        """Generic chain runner. Same: chain = prompt | model → chain.invoke(...)"""
        prompt = ChatPromptTemplate.from_template(template)
        chain  = prompt | self.model
        return chain.invoke(kwargs)

    def document_function(self, code: str, name: str, filename: str) -> str:
        return self._run(FUNCTION_DOC_TEMPLATE, code=code, name=name, filename=filename)

    def document_class(self, code: str, name: str, filename: str) -> str:
        return self._run(CLASS_DOC_TEMPLATE, code=code, name=name, filename=filename)

    def summarize_file(self, code: str, filename: str, language: str) -> str:
        return self._run(FILE_SUMMARY_TEMPLATE,
                         code=code[:4000], filename=filename, language=language)

    def generate_readme(self, codebase_summary: str, file_list: str) -> str:
        return self._run(README_TEMPLATE,
                         codebase_summary=codebase_summary, file_list=file_list)

    def find_issues(self, code: str, filename: str) -> str:
        return self._run(ISSUE_FINDER_TEMPLATE, code=code[:4000], filename=filename)

    def suggest_improvements(self, code: str, filename: str) -> str:
        return self._run(IMPROVEMENT_TEMPLATE, code=code[:4000], filename=filename)


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
