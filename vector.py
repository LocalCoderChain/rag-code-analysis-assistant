"""
vector.py  —  Code RAG Storage Layer
=======================================
SQLite + sqlite-vec, single file: data/devone.db

One file holds everything (no separate stores to keep in sync):
  - chunks       chunk text + code metadata (source, chunk_type, name, language...)
  - chunks_vec   embeddings, sqlite-vec virtual table (vec0)
  - repos        connected-repo registry (used from Phase 2 onward)

Adapted for source code:
  - Code files → parser.py → Documents
  - Documents stored with code-specific metadata
    {source, chunk_type, name, language, line_start}
  - "collection" = a named group of chunks for isolation (maps to a repo later)
"""

import os
import sqlite3

import sqlite_vec
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.parser import parse_code_file

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "data", "devone.db")

# Fixed for now: the vec0 table's dimension is locked in at creation time,
# so mixing embedding models with different dimensions isn't supported yet.
EMBED_MODEL = "mxbai-embed-large"


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════

def get_embeddings(model_name: str = EMBED_MODEL) -> OllamaEmbeddings:
    return OllamaEmbeddings(model=model_name)


# ══════════════════════════════════════════════════════════════════════════════
# CONNECTION + SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    _init_schema(db)
    return db


def _init_schema(db: sqlite3.Connection) -> None:
    db.execute("""
        CREATE TABLE IF NOT EXISTS repos (
            name TEXT PRIMARY KEY,
            url TEXT,
            last_commit_sha TEXT,
            last_synced_at TEXT
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collection TEXT NOT NULL,
            source TEXT NOT NULL,
            chunk_type TEXT,
            name TEXT,
            language TEXT,
            line_start INTEGER,
            sub_chunk INTEGER,
            text TEXT NOT NULL
        )
    """)
    db.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_collection_source
        ON chunks(collection, source)
    """)

    exists = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_vec'"
    ).fetchone()
    if not exists:
        # Dimension isn't known until we actually embed something — probe once.
        dim = len(get_embeddings().embed_query("dimension probe"))
        db.execute(f"CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding FLOAT[{dim}])")
    db.commit()


# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVER  (kept invoke()-shaped so app.py doesn't need to change yet)
# ══════════════════════════════════════════════════════════════════════════════

class _SqliteRetriever:
    def __init__(self, collection: str, embed_model: str, top_k: int):
        self.collection  = collection
        self.embed_model = embed_model
        self.top_k       = top_k

    def invoke(self, query: str) -> list[Document]:
        return search(query, self.collection, self.embed_model, self.top_k)


def get_retriever(collection: str = "code_kb",
                  embed_model: str = EMBED_MODEL,
                  top_k: int = 6) -> _SqliteRetriever:
    return _SqliteRetriever(collection, embed_model, top_k)


def search(query: str,
          collection: str = "code_kb",
          embed_model: str = EMBED_MODEL,
          top_k: int = 6) -> list[Document]:
    """
    Semantic search: embed the query, find nearest chunks via sqlite-vec,
    filter by collection.

    Over-fetches candidates from the vector index and filters by collection
    in Python rather than pushing the filter into the vec0 query — keeps this
    correct regardless of which sqlite-vec version ends up installed, instead
    of relying on newer/less common partition-key SQL syntax.
    """
    embedder     = get_embeddings(embed_model)
    query_vector = sqlite_vec.serialize_float32(embedder.embed_query(query))

    db = get_connection()
    candidates = db.execute(
        "SELECT rowid FROM chunks_vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        (query_vector, max(top_k * 10, 50)),
    ).fetchall()

    results = []
    for (rowid,) in candidates:
        row = db.execute(
            "SELECT text, source, chunk_type, name, language, line_start, sub_chunk "
            "FROM chunks WHERE id = ? AND collection = ?",
            (rowid, collection),
        ).fetchone()
        if row:
            text, source, chunk_type, name, language, line_start, sub_chunk = row
            results.append(Document(
                page_content=text,
                metadata={
                    "source": source, "chunk_type": chunk_type,
                    "name": name, "language": language, "line_start": line_start,
                    "sub_chunk": sub_chunk,
                },
            ))
        if len(results) >= top_k:
            break
    db.close()
    return results


# ══════════════════════════════════════════════════════════════════════════════
# COLLECTION MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def list_collections() -> list[str]:
    db = get_connection()
    rows = db.execute("SELECT DISTINCT collection FROM chunks ORDER BY collection").fetchall()
    db.close()
    return [r[0] for r in rows]


def delete_file_chunks(collection: str, filename: str) -> None:
    """Remove all stored chunks for one file (used when it's deleted/renamed upstream)."""
    db = get_connection()
    ids = [r[0] for r in db.execute(
        "SELECT id FROM chunks WHERE collection = ? AND source = ?", (collection, filename)
    ).fetchall()]
    if ids:
        db.executemany("DELETE FROM chunks_vec WHERE rowid = ?", [(i,) for i in ids])
        db.execute(f"DELETE FROM chunks WHERE id IN ({','.join('?' * len(ids))})", ids)
    db.commit()
    db.close()


def delete_collection(collection: str) -> None:
    db = get_connection()
    ids = [r[0] for r in db.execute(
        "SELECT id FROM chunks WHERE collection = ?", (collection,)
    ).fetchall()]
    if ids:
        db.executemany("DELETE FROM chunks_vec WHERE rowid = ?", [(i,) for i in ids])
        db.execute(f"DELETE FROM chunks WHERE id IN ({','.join('?' * len(ids))})", ids)
    db.commit()
    db.close()


# ══════════════════════════════════════════════════════════════════════════════
# CODE FILE INGESTION
# ══════════════════════════════════════════════════════════════════════════════

def ingest_code_file(filename: str,
                     source_code: str,
                     collection: str,
                     embed_model: str = EMBED_MODEL) -> dict:
    """
    Parse a code file → Documents → embed → store in SQLite + sqlite-vec.

    Returns a summary dict: {functions, classes, imports, total_chunks}

    Re-ingesting the same filename replaces its old chunks entirely (delete
    then insert) rather than overwriting by position — avoids leaving orphan
    chunks behind if the file's chunk count changed since last ingest.
    """
    # Step 1: Structural parsing
    docs = parse_code_file(filename, source_code)

    # Step 2: Secondary split for any chunk > 1200 chars (safety net)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    final_docs = []
    for doc in docs:
        if len(doc.page_content) > 1200:
            sub = splitter.split_documents([doc])
            for i, s in enumerate(sub):
                s.metadata["sub_chunk"] = i
            final_docs.extend(sub)
        else:
            final_docs.append(doc)

    # Step 3: Embed
    embedder = get_embeddings(embed_model)
    vectors  = embedder.embed_documents([d.page_content for d in final_docs])

    # Step 4: Replace old chunks for this file, insert new ones
    db = get_connection()
    old_ids = [r[0] for r in db.execute(
        "SELECT id FROM chunks WHERE collection = ? AND source = ?",
        (collection, filename),
    ).fetchall()]
    if old_ids:
        db.executemany("DELETE FROM chunks_vec WHERE rowid = ?", [(i,) for i in old_ids])
        db.execute(f"DELETE FROM chunks WHERE id IN ({','.join('?' * len(old_ids))})", old_ids)

    for doc, vector in zip(final_docs, vectors):
        meta = doc.metadata
        cur = db.execute(
            "INSERT INTO chunks (collection, source, chunk_type, name, language, "
            "line_start, sub_chunk, text) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (collection, filename, meta.get("chunk_type"), meta.get("name"),
             meta.get("language"), meta.get("line_start"), meta.get("sub_chunk"),
             doc.page_content),
        )
        chunk_id = cur.lastrowid
        db.execute(
            "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
            (chunk_id, sqlite_vec.serialize_float32(vector)),
        )
    db.commit()
    db.close()

    return {
        "functions":    sum(1 for d in final_docs if d.metadata.get("chunk_type") == "function"),
        "classes":      sum(1 for d in final_docs if d.metadata.get("chunk_type") == "class"),
        "imports":      sum(1 for d in final_docs if d.metadata.get("chunk_type") == "imports"),
        "total_chunks": len(final_docs),
    }


def get_all_chunks(collection: str,
                   embed_model: str = EMBED_MODEL,
                   filename: str = None) -> list[Document]:
    """
    Retrieve all stored chunks (optionally filtered by filename).
    Used by the documentation generator to read all code.
    """
    db = get_connection()
    if filename:
        rows = db.execute(
            "SELECT text, source, chunk_type, name, language, line_start, sub_chunk "
            "FROM chunks WHERE collection = ? AND source = ?",
            (collection, filename),
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT text, source, chunk_type, name, language, line_start, sub_chunk "
            "FROM chunks WHERE collection = ?",
            (collection,),
        ).fetchall()
    db.close()

    return [
        Document(
            page_content=text,
            metadata={
                "source": source, "chunk_type": chunk_type,
                "name": name, "language": language, "line_start": line_start,
                "sub_chunk": sub_chunk,
            },
        )
        for text, source, chunk_type, name, language, line_start, sub_chunk in rows
    ]
