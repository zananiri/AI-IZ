"""The corpus vector store: one Chroma collection per category under legal.corpus.vectordb_dir,
separate from the Legal tab's signed index, plus a sqlite state file for incremental indexing.

State (<vectordb_dir>/_corpus_state.sqlite):
  records(category, record_id, record_hash, chunk_ids, indexed_at) -- a record whose hash is
      unchanged is skipped; a changed one has its old chunks deleted before the new ones go in;
      one no longer in the input can be pruned.
  chunks(chunk_id, category, record_id, lexical_text) -- the keyword-index copy of each chunk
      (final letters folded), exported with --export-lexical rather than stored in Chroma twice.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from docslides.legal_data.corpus_chunking import CorpusChunk


@dataclass
class IndexedRecord:
    record_hash: str
    chunk_ids: list[str]


class CorpusState:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS records (
                category TEXT NOT NULL, record_id TEXT NOT NULL, record_hash TEXT NOT NULL,
                chunk_ids TEXT NOT NULL, indexed_at TEXT NOT NULL, PRIMARY KEY (category, record_id));
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY, category TEXT NOT NULL, record_id TEXT NOT NULL, lexical_text TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS chunks_by_record ON chunks (category, record_id);
            """
        )

    def get(self, category: str, record_id: str) -> IndexedRecord | None:
        row = self.db.execute("SELECT record_hash, chunk_ids FROM records WHERE category=? AND record_id=?",
                              (category, record_id)).fetchone()
        return IndexedRecord(row[0], json.loads(row[1])) if row else None

    def put(self, category: str, record_id: str, record_hash: str, chunks: list[CorpusChunk]) -> None:
        self.db.execute("DELETE FROM chunks WHERE category=? AND record_id=?", (category, record_id))
        self.db.execute(
            "INSERT OR REPLACE INTO records VALUES (?, ?, ?, ?, ?)",
            (category, record_id, record_hash, json.dumps([c.chunk_id for c in chunks]),
             datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        self.db.executemany("INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?)",
                            [(c.chunk_id, category, record_id, c.lexical_text) for c in chunks])

    def delete(self, category: str, record_id: str) -> None:
        self.db.execute("DELETE FROM chunks WHERE category=? AND record_id=?", (category, record_id))
        self.db.execute("DELETE FROM records WHERE category=? AND record_id=?", (category, record_id))

    def record_ids(self, category: str) -> set[str]:
        return {row[0] for row in self.db.execute("SELECT record_id FROM records WHERE category=?", (category,))}

    def lexical(self, category: str) -> Iterator[tuple[str, str, str]]:
        yield from self.db.execute("SELECT chunk_id, record_id, lexical_text FROM chunks WHERE category=?", (category,))

    def commit(self) -> None:
        self.db.commit()

    def close(self) -> None:
        self.db.commit()
        self.db.close()


class CorpusCollection:
    def __init__(self, vectordb_dir: Path, name: str) -> None:
        import chromadb

        self.client = chromadb.PersistentClient(path=str(vectordb_dir))
        # embedding_function=None: vectors are always passed in (bge-m3, as the Legal tab's index);
        # never let Chroma fall back to, and download, its own default embedder.
        self.collection = self.client.get_or_create_collection(
            name, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )
        try:
            self.batch = int(self.client.get_max_batch_size())
        except Exception:  # noqa: BLE001 -- older/newer clients: a safe default
            self.batch = 4000

    def upsert(self, chunks: list[CorpusChunk], vectors) -> None:
        for start in range(0, len(chunks), self.batch):
            part = chunks[start : start + self.batch]
            self.collection.upsert(
                ids=[c.chunk_id for c in part],
                embeddings=vectors[start : start + self.batch],  # a numpy array: no per-float conversion
                documents=[c.text for c in part],
                metadatas=[c.metadata for c in part],
            )

    def delete(self, chunk_ids: list[str]) -> None:
        for start in range(0, len(chunk_ids), self.batch):
            self.collection.delete(ids=chunk_ids[start : start + self.batch])

    def count(self) -> int:
        return self.collection.count()

    def query(self, vector, n: int, where: dict | None = None) -> dict:
        return self.collection.query(query_embeddings=[vector], n_results=n, where=where,
                                     include=["documents", "metadatas", "distances"])
