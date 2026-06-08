"""
parser.py  —  Intelligent Source Code Parser
=============================================
Parses .py / .java / .js / .cpp / .c files and extracts:
  - imports / includes
  - class definitions
  - function / method definitions
  - docstrings / comments
  - raw code blocks

Returns LangChain Document objects with rich metadata so the
retriever and LLM know exactly what each chunk represents.

Design principle: keep functions intact as single chunks wherever
possible. Splitting mid-function destroys semantic meaning.
"""

import re
from langchain_core.documents import Document


# ══════════════════════════════════════════════════════════════════════════════
# LANGUAGE DISPATCH
# ══════════════════════════════════════════════════════════════════════════════

def parse_code_file(filename: str, source_code: str) -> list[Document]:
    """
    Entry point. Detects language from extension and routes to the
    appropriate parser. Returns a list of LangChain Documents.
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    parsers = {
        "py":   _parse_python,
        "java": _parse_java_js_cpp,
        "js":   _parse_java_js_cpp,
        "cpp":  _parse_java_js_cpp,
        "c":    _parse_java_js_cpp,
    }

    parser_fn = parsers.get(ext, _parse_generic)
    docs = parser_fn(filename, source_code)

    # Always add a "full file" document for file-level questions
    docs.append(Document(
        page_content=source_code[:6000],   # cap at 6000 chars for context
        metadata={
            "source":    filename,
            "chunk_type": "full_file",
            "language":  ext,
        }
    ))
    return docs


# ══════════════════════════════════════════════════════════════════════════════
# PYTHON PARSER
# ══════════════════════════════════════════════════════════════════════════════

def _parse_python(filename: str, source: str) -> list[Document]:
    """
    Extracts from Python files:
    - import blocks
    - top-level functions (def)
    - class definitions (class + all methods inside)
    - standalone methods
    """
    docs = []
    lines = source.splitlines()

    # ── Imports ────────────────────────────────────────────────────────────────
    import_lines = [l for l in lines if l.strip().startswith(("import ", "from "))]
    if import_lines:
        docs.append(Document(
            page_content="\n".join(import_lines),
            metadata={"source": filename, "chunk_type": "imports", "language": "py"}
        ))

    # ── Functions & Classes via indentation-aware extraction ──────────────────
    current_block  = []
    current_type   = None
    current_name   = ""
    brace_depth    = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Detect function definition
        fn_match = re.match(r'^(async\s+)?def\s+(\w+)\s*\(', line)
        # Detect class definition
        cls_match = re.match(r'^class\s+(\w+)', line)

        if (fn_match or cls_match) and not line.startswith((" ", "\t")):
            # Save previous block if any
            if current_block and current_type:
                docs.append(Document(
                    page_content="\n".join(current_block),
                    metadata={
                        "source":     filename,
                        "chunk_type": current_type,
                        "name":       current_name,
                        "language":   "py",
                        "line_start": i - len(current_block),
                    }
                ))
            current_block = [line]
            current_name  = (fn_match.group(2) if fn_match else cls_match.group(1))
            current_type  = "function" if fn_match else "class"

        elif current_block:
            # Continue collecting block (indented lines belong to it)
            if line.startswith((" ", "\t")) or stripped == "" or stripped.startswith("#"):
                current_block.append(line)
            else:
                # New top-level statement — save block
                docs.append(Document(
                    page_content="\n".join(current_block),
                    metadata={
                        "source":     filename,
                        "chunk_type": current_type,
                        "name":       current_name,
                        "language":   "py",
                    }
                ))
                current_block = [line]
                current_type  = "code_block"
                current_name  = f"line_{i}"

    # Save last block
    if current_block and current_type:
        docs.append(Document(
            page_content="\n".join(current_block),
            metadata={
                "source":     filename,
                "chunk_type": current_type,
                "name":       current_name,
                "language":   "py",
            }
        ))

    return docs


# ══════════════════════════════════════════════════════════════════════════════
# JAVA / JS / C++ / C PARSER  (brace-counting)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_java_js_cpp(filename: str, source: str) -> list[Document]:
    """
    Extracts from brace-delimited languages:
    - import / include / using statements
    - class blocks
    - function / method blocks (brace-balanced extraction)
    """
    ext  = filename.rsplit(".", 1)[-1].lower()
    docs = []
    lines = source.splitlines()

    # ── Imports / includes ─────────────────────────────────────────────────────
    import_patterns = {
        "java": r'^\s*(import|package)\s+',
        "js":   r'^\s*(import|require|export)\s+',
        "cpp":  r'^\s*#\s*(include|define|pragma)\s+',
        "c":    r'^\s*#\s*(include|define)\s+',
    }
    pattern = import_patterns.get(ext, r'^\s*import\s+')
    import_lines = [l for l in lines if re.match(pattern, l)]
    if import_lines:
        docs.append(Document(
            page_content="\n".join(import_lines),
            metadata={"source": filename, "chunk_type": "imports", "language": ext}
        ))

    # ── Function / class block extraction via brace counting ──────────────────
    # Detect start of a function or class declaration
    fn_patterns = {
        "java": r'(public|private|protected|static|void|class|interface)\s+\w+',
        "js":   r'(function\s+\w+|class\s+\w+|\w+\s*=\s*(function|\())',
        "cpp":  r'(\w[\w\s\*&]+)\s+(\w+)\s*\(',
        "c":    r'(\w[\w\s\*]+)\s+(\w+)\s*\(',
    }
    fn_re = re.compile(fn_patterns.get(ext, r'\w+\s*\('))

    i = 0
    while i < len(lines):
        line = lines[i]
        if fn_re.search(line) and "{" in line:
            # Start collecting a brace-delimited block
            block  = [line]
            depth  = line.count("{") - line.count("}")
            name   = _extract_name(line)
            ctype  = "class" if "class" in line.lower() else "function"
            j      = i + 1

            while j < len(lines) and depth > 0:
                block.append(lines[j])
                depth += lines[j].count("{") - lines[j].count("}")
                j += 1

            docs.append(Document(
                page_content="\n".join(block),
                metadata={
                    "source":     filename,
                    "chunk_type": ctype,
                    "name":       name,
                    "language":   ext,
                    "line_start": i,
                }
            ))
            i = j
        else:
            i += 1

    # If no blocks found, fall back to generic chunking
    if len(docs) <= 1:
        return _parse_generic(filename, source)

    return docs


# ══════════════════════════════════════════════════════════════════════════════
# GENERIC FALLBACK PARSER  (sliding window chunks)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_generic(filename: str, source: str) -> list[Document]:
    """
    Fallback for unsupported languages or files where structural
    parsing finds nothing. Splits into 60-line overlapping windows.
    """
    ext   = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"
    lines = source.splitlines()
    docs  = []

    window  = 60
    overlap = 10

    for i in range(0, len(lines), window - overlap):
        chunk_lines = lines[i : i + window]
        if not any(l.strip() for l in chunk_lines):
            continue
        docs.append(Document(
            page_content="\n".join(chunk_lines),
            metadata={
                "source":     filename,
                "chunk_type": "code_block",
                "name":       f"lines_{i}_{i+window}",
                "language":   ext,
                "line_start": i,
            }
        ))

    return docs


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _extract_name(line: str) -> str:
    """Try to pull a function/class name from a declaration line."""
    # Match word before parenthesis
    m = re.search(r'\b(\w+)\s*\(', line)
    if m:
        return m.group(1)
    # Match word after class/function keyword
    m = re.search(r'(?:class|function|def)\s+(\w+)', line)
    if m:
        return m.group(1)
    return "unknown"


def get_supported_extensions() -> list[str]:
    return ["py", "java", "js", "cpp", "c"]
