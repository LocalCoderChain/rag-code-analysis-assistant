"""
repo_manager.py  —  GitHub Repo Ingestion
============================================
Clones/pulls a GitHub repo locally via `git` (subprocess), not the GitHub
REST API — no API rate limits to manage, and git already handles diffing
for free on subsequent syncs.

The PAT is only ever passed inline to a single git command (clone/pull/
fetch) and is never written into the cloned repo's .git/config — it's
stripped back out immediately after cloning.
"""

import os
import subprocess
import time
from urllib.parse import urlparse

from src.parser import get_supported_extensions
from vector import ingest_code_file, delete_file_chunks, get_connection

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
REPOS_DIR = os.path.join(BASE_DIR, "data", "repos")

SKIP_DIRS = {".git", "node_modules", "__pycache__", "venv", "myvenv", ".venv", "dist", "build"}


def parse_repo_url(url: str) -> tuple[str, str]:
    """Extract (owner, repo) from a GitHub URL like https://github.com/owner/repo(.git)"""
    path = urlparse(url).path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    parts = path.split("/")
    if len(parts) < 2:
        raise ValueError(f"Not a valid GitHub repo URL: {url}")
    return parts[0], parts[1]


def _authed_url(owner: str, repo: str, token: str) -> str:
    return f"https://{token}@github.com/{owner}/{repo}.git"


def _clean_url(owner: str, repo: str) -> str:
    return f"https://github.com/{owner}/{repo}.git"


def _local_path(owner: str, repo: str) -> str:
    return os.path.join(REPOS_DIR, f"{owner}_{repo}")


def get_local_path_for_collection(collection: str) -> str | None:
    """
    Return the local cloned directory for a repo-based collection (name
    format "owner/repo"), or None if it's not a repo (e.g. quick-test
    uploads) or hasn't actually been cloned to disk.
    """
    if "/" not in collection:
        return None
    owner, repo = collection.split("/", 1)
    path = _local_path(owner, repo)
    return path if os.path.isdir(path) else None


def clone_or_pull(url: str, token: str) -> str:
    """
    Clone the repo on first connect, or pull latest if it already exists
    locally. Returns the local path to the repo's working copy.
    """
    owner, repo = parse_repo_url(url)
    local_path  = _local_path(owner, repo)
    authed_url  = _authed_url(owner, repo, token)

    os.makedirs(REPOS_DIR, exist_ok=True)

    if os.path.isdir(os.path.join(local_path, ".git")):
        # Pass the token only for this one pull — never stored in .git/config
        subprocess.run(["git", "pull", authed_url],
                       cwd=local_path, check=True, capture_output=True, text=True)
    else:
        # Full clone (not shallow) — incremental sync needs old commits to
        # still be reachable locally so `git diff <old_sha> <new_sha>` works.
        subprocess.run(["git", "clone", authed_url, local_path],
                       check=True, capture_output=True, text=True)
        # Strip the token back out of the stored remote immediately after cloning
        subprocess.run(["git", "remote", "set-url", "origin", _clean_url(owner, repo)],
                       cwd=local_path, check=True, capture_output=True, text=True)

    return local_path


def get_current_commit_sha(local_path: str) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"],
                            cwd=local_path, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def walk_supported_files(local_path: str) -> list[str]:
    """Return relative paths of all files with a supported extension."""
    exts    = set(get_supported_extensions())
    matches = []
    for root, dirs, files in os.walk(local_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            ext = f.rsplit(".", 1)[-1].lower() if "." in f else ""
            if ext in exts:
                rel = os.path.relpath(os.path.join(root, f), local_path)
                matches.append(rel)
    return matches


def get_repo_record(collection: str) -> dict | None:
    for r in list_repos():
        if r["name"] == collection:
            return r
    return None


def get_changed_files(local_path: str, old_sha: str, new_sha: str) -> tuple[list[str], list[str]]:
    """
    Diff two commits and return (files_to_ingest, files_to_delete), both
    relative paths filtered to supported extensions. Added/modified files
    (and the new side of a rename) go in files_to_ingest; deleted files
    (and the old side of a rename) go in files_to_delete.
    """
    result = subprocess.run(
        ["git", "diff", "--name-status", old_sha, new_sha],
        cwd=local_path, check=True, capture_output=True, text=True,
    )
    exts = set(get_supported_extensions())

    def _supported(path: str) -> bool:
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        return ext in exts

    to_ingest, to_delete = [], []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        status, *paths = line.split("\t")
        if status.startswith("A") or status.startswith("M"):
            if _supported(paths[0]):
                to_ingest.append(paths[0])
        elif status.startswith("D"):
            if _supported(paths[0]):
                to_delete.append(paths[0])
        elif status.startswith("R"):
            old_path, new_path = paths[0], paths[1]
            if _supported(old_path):
                to_delete.append(old_path)
            if _supported(new_path):
                to_ingest.append(new_path)

    return to_ingest, to_delete


def ingest_repo(url: str, token: str, embed_model: str, progress_callback=None) -> dict:
    """
    Clone/pull a repo and sync it into storage. First connect ingests every
    supported file; every later sync uses `git diff` against the last
    recorded commit to re-embed only files that actually changed, and
    removes chunks for anything deleted/renamed away. If nothing changed
    since last sync, skips re-ingestion entirely.
    """
    start = time.time()
    owner, repo_name = parse_repo_url(url)
    collection = f"{owner}/{repo_name}"

    existing = get_repo_record(collection)
    old_sha  = existing["last_commit_sha"] if existing else None

    local_path = clone_or_pull(url, token)
    new_sha    = get_current_commit_sha(local_path)

    if old_sha == new_sha:
        return {
            "collection":     collection,
            "status":         "up_to_date",
            "files_ingested": 0,
            "files_deleted":  0,
            "total_chunks":   0,
            "commit_sha":     new_sha,
            "elapsed_seconds": round(time.time() - start, 1),
        }

    if old_sha is None:
        files_to_ingest = walk_supported_files(local_path)
        files_to_delete = []
    else:
        files_to_ingest, files_to_delete = get_changed_files(local_path, old_sha, new_sha)

    total_chunks = 0
    for i, rel_path in enumerate(files_to_ingest):
        if progress_callback:
            progress_callback(i + 1, len(files_to_ingest), rel_path)
        full_path = os.path.join(local_path, rel_path)
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            source_code = f.read()
        summary = ingest_code_file(
            filename=rel_path,
            source_code=source_code,
            collection=collection,
            embed_model=embed_model,
        )
        total_chunks += summary["total_chunks"]

    for rel_path in files_to_delete:
        delete_file_chunks(collection, rel_path)

    _record_repo(collection, url, new_sha)

    return {
        "collection":      collection,
        "status":          "first_ingest" if old_sha is None else "synced",
        "files_ingested":  len(files_to_ingest),
        "files_deleted":   len(files_to_delete),
        "total_chunks":    total_chunks,
        "commit_sha":      new_sha,
        "elapsed_seconds": round(time.time() - start, 1),
    }


def _record_repo(collection: str, url: str, sha: str) -> None:
    db = get_connection()
    db.execute(
        "INSERT INTO repos (name, url, last_commit_sha, last_synced_at) "
        "VALUES (?, ?, ?, datetime('now')) "
        "ON CONFLICT(name) DO UPDATE SET "
        "url=excluded.url, last_commit_sha=excluded.last_commit_sha, "
        "last_synced_at=excluded.last_synced_at",
        (collection, url, sha),
    )
    db.commit()
    db.close()


def list_repos() -> list[dict]:
    db = get_connection()
    rows = db.execute(
        "SELECT name, url, last_commit_sha, last_synced_at FROM repos ORDER BY name"
    ).fetchall()
    db.close()
    return [
        {"name": n, "url": u, "last_commit_sha": s, "last_synced_at": t}
        for n, u, s, t in rows
    ]


def check_remote_status(url: str, token: str) -> dict:
    """
    Lightweight check: is the locally-ingested version of this repo behind
    the remote? Does a git fetch (no merge, no re-ingest) and compares SHAs.
    Full incremental re-embedding of just the changed files is a separate,
    later improvement — this only answers "is it stale."
    """
    owner, repo_name = parse_repo_url(url)
    local_path = _local_path(owner, repo_name)

    if not os.path.isdir(os.path.join(local_path, ".git")):
        return {"status": "not_ingested"}

    authed_url = _authed_url(owner, repo_name, token)
    subprocess.run(["git", "fetch", authed_url],
                   cwd=local_path, check=True, capture_output=True, text=True)
    remote_sha = subprocess.run(["git", "rev-parse", "FETCH_HEAD"],
                                cwd=local_path, check=True, capture_output=True, text=True).stdout.strip()
    local_sha = get_current_commit_sha(local_path)

    return {
        "status":     "up_to_date" if remote_sha == local_sha else "behind",
        "local_sha":  local_sha,
        "remote_sha": remote_sha,
    }
