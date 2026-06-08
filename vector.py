"""
vector.py  —  Code RAG Vector Pipeline
=======================================
Same architecture as the Engineering RAG project:
  OllamaEmbeddings → Chroma → as_retriever()

Adapted for source code:
  - Code files → parser.py → Documents
  - Documents stored with code-specific metadata
    {source, chunk_type, name, language, line_start}
  - Collection per project for isolation
"""

import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.parser import parse_code_file

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDINGS  (same as eng_rag)
# ══════════════════════════════════════════════════════════════════════════════

def get_embeddings(model_name: str = "mxbai-embed-large") -> OllamaEmbeddings:
    return OllamaEmbeddings(model=model_name)


# ══════════════════════════════════════════════════════════════════════════════
# VECTOR STORE  (same Chroma pattern as eng_rag)
# ══════════════════════════════════════════════════════════════════════════════

def get_vectorstore(collection: str = "code_kb",
                    embed_model: str = "mxbai-embed-large") -> Chroma:
    """
    Open or create a persistent Chroma collection.
    Same call as eng_rag:
        Chroma(collection_name=..., persist_directory=..., embedding_function=...)
    """
    return Chroma(
        collection_name=collection,
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings(embed_model),
    )


def get_retriever(collection: str = "code_kb",
                  embed_model: str = "mxbai-embed-large",
                  top_k: int = 6):
    """
    Same pattern as eng_rag:
        vector_store.as_retriever(search_kwargs={"k": top_k})
    k=6 because code chunks tend to be smaller, so more context helps.
    """
    vs = get_vectorstore(collection, embed_model)
    return vs.as_retriever(search_kwargs={"k": top_k})


def list_collections() -> list[str]:
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        return [c.name for c in client.list_collections()]
    except Exception:
        return []


def delete_collection(collection: str) -> None:
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        client.delete_collection(collection)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# CODE FILE INGESTION
# ══════════════════════════════════════════════════════════════════════════════

def ingest_code_file(filename: str,
                     source_code: str,
                     collection: str,
                     embed_model: str) -> dict:
    """
    Parse a code file → Documents → embed → store in ChromaDB.

    Returns a summary dict:
      {functions, classes, imports, total_chunks}

    Steps (same upload phase as eng_rag):
      1. parse_code_file() → list of Documents with code metadata
      2. For large chunks, apply RecursiveCharacterTextSplitter as fallback
      3. OllamaEmbeddings → vectors
      4. Chroma.add_documents() → persists to disk
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

    # Step 3+4: Embed and store
    vs  = get_vectorstore(collection, embed_model)
    ids = [f"{filename}_chunk_{i}" for i in range(len(final_docs))]
    vs.add_documents(documents=final_docs, ids=ids)

    # Build summary for UI display
    summary = {
        "functions":    sum(1 for d in final_docs if d.metadata.get("chunk_type") == "function"),
        "classes":      sum(1 for d in final_docs if d.metadata.get("chunk_type") == "class"),
        "imports":      sum(1 for d in final_docs if d.metadata.get("chunk_type") == "imports"),
        "total_chunks": len(final_docs),
    }
    return summary


def get_all_chunks(collection: str,
                   embed_model: str,
                   filename: str = None) -> list[Document]:
    """
    Retrieve all stored chunks (optionally filtered by filename).
    Used by the documentation generator to read all code.
    """
    vs = get_vectorstore(collection, embed_model)
    try:
        result = vs.get()
        docs = []
        for i, text in enumerate(result["documents"]):
            meta = result["metadatas"][i] if result["metadatas"] else {}
            if filename and meta.get("source") != filename:
                continue
            docs.append(Document(page_content=text, metadata=meta))
        return docs
    except Exception:
        return []
