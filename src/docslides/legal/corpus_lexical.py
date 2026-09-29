"""BM25 over the bulk corpus's lexical copy, for legal/corpus_retrieval.retrieve_planned.

Dense search ranks exact terms of art loosely -- "עושק", "דרישת הסבירות", a law's own name -- and the
29 Sept eval review found most wrong answers began with the governing section never being
retrieved. scripts/legal_data/vectorize.py --export-lexical already writes every chunk's lexical
text (points stripped, quotes unified, final letters folded) to <vectordb>/<category>/
lexical_<category>.jsonl; this indexes it with legal/keyword.py's Hebrew-aware `terms` (each word
plus its prefix-stripped forms), so query and corpus tokenize the same way.

legal/keyword.KeywordIndex keeps a Counter per chunk, which suits the signed bundle's few hundred
chunks but would need several GB for the corpus's ~200k. Here the BM25 weight of every (chunk,
term) pair is computed once into one sparse matrix -- a query is a sum over its terms' columns --
and cached beside the source file (bm25_<category>.npz / .json), rebuilt when the source changes.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from docslides.legal.keyword import terms
from docslides.legal_data.hebrew import normalize_for_index
from docslides.logging_setup import get_logger

logger = get_logger(__name__)

K1, B = 1.2, 0.75
INDEX_VERSION = 1


class CorpusLexicalIndex:
    def __init__(self, ids: list[str], vocabulary: dict[str, int], weights) -> None:
        self.ids = ids
        self.vocabulary = vocabulary
        self.weights = weights  # scipy CSC: chunk x term BM25 weights

    @classmethod
    def build(cls, ids: list[str], texts: list[str]) -> CorpusLexicalIndex:
        import numpy as np
        from sklearn.feature_extraction.text import CountVectorizer

        counts = CountVectorizer(analyzer=terms, dtype=np.float32)
        matrix = counts.fit_transform(texts).tocsr()
        n = matrix.shape[0]
        lengths = np.asarray(matrix.sum(axis=1)).ravel()
        average = float(lengths.mean()) if n else 0.0
        df = np.bincount(matrix.indices, minlength=matrix.shape[1])
        idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
        rows = np.repeat(np.arange(n), np.diff(matrix.indptr))
        tf = matrix.data
        norm = K1 * (1.0 - B + B * lengths[rows] / (average or 1.0))
        matrix.data = (idf[matrix.indices] * tf * (K1 + 1.0) / (tf + norm)).astype(np.float32)
        vocabulary = {term: int(i) for term, i in counts.vocabulary_.items()}
        return cls(ids, vocabulary, matrix.tocsc())

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """(chunk_id, BM25 score) of the best `k` chunks sharing any term with `query`."""
        import numpy as np

        columns = sorted({self.vocabulary[t] for t in terms(normalize_for_index(query)) if t in self.vocabulary})
        if not columns or k <= 0:
            return []
        scores = np.asarray(self.weights[:, columns].sum(axis=1)).ravel()
        nonzero = int(np.count_nonzero(scores))
        if not nonzero:
            return []
        k = min(k, nonzero)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top]

    def save(self, npz: Path, meta: Path, source_size: int) -> None:
        from scipy import sparse

        sparse.save_npz(npz, self.weights)
        meta.write_text(json.dumps({"version": INDEX_VERSION, "source_size": source_size, "ids": self.ids,
                                    "vocabulary": self.vocabulary}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, npz: Path, meta: Path) -> CorpusLexicalIndex:
        from scipy import sparse

        info = json.loads(meta.read_text(encoding="utf-8"))
        return cls(info["ids"], info["vocabulary"], sparse.load_npz(npz).tocsc())


def _cache_is_current(npz: Path, meta: Path, source_size: int) -> bool:
    if not (npz.exists() and meta.exists()):
        return False
    try:
        with open(meta, encoding="utf-8") as f:
            head = f.read(200)  # version and size come first; no need to parse the vocabulary
        return f'"version": {INDEX_VERSION}' in head and f'"source_size": {source_size}' in head
    except OSError:
        return False


@lru_cache(maxsize=4)
def lexical_index(vectordb_dir: str, category: str) -> CorpusLexicalIndex | None:
    """The category's index: from the cache if it matches the source, else built (a minute or two
    for the full corpus) and cached. None when there is no lexical file or sklearn/scipy is missing
    -- retrieval then runs on dense search alone, as before."""
    directory = Path(vectordb_dir) / category
    source = directory / f"lexical_{category}.jsonl"
    if not source.exists():
        return None
    npz, meta = directory / f"bm25_{category}.npz", directory / f"bm25_{category}.json"
    size = source.stat().st_size
    try:
        if _cache_is_current(npz, meta, size):
            return CorpusLexicalIndex.load(npz, meta)
        ids, texts = [], []
        with open(source, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    ids.append(row["chunk_id"])
                    texts.append(row["text"])
        logger.info("corpus_lexical_building", category=category, chunks=len(ids))
        index = CorpusLexicalIndex.build(ids, texts)
        try:
            index.save(npz, meta, size)
        except OSError as exc:  # a read-only corpus dir: keep the in-memory index
            logger.warning("corpus_lexical_cache_not_saved", category=category, error=str(exc))
        return index
    except ImportError as exc:
        logger.warning("corpus_lexical_unavailable", error=str(exc))
        return None
