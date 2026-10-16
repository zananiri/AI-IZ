"""Spanish-aware keyword search (BM25) over the Legal index, for retrieval.py.

Dense embeddings match meaning but are weak on exact identifiers -- article numbers ("1902",
"14 bis"), terms of art ("usucapión", "retracto de colindantes"), law numbers ("1/2000"), amounts --
which is exactly what a lawyer's question names. Keyword search is the second signal; retrieval.py
fuses both rankings.

Every word is lowercased and accent-folded ("Artículo" = "articulo"), indexed together with a light
stem (plural and gender endings: "arrendamientos" -> "arrendamiento", "obligada" -> "obligad"),
and stopwords are dropped. Law numbers ("29/1994") and article numbers with their suffix
("14 bis" -> "14bis") are single tokens. The index is small (one law is a few dozen chunks) and is
rebuilt only when the signed bundle changes.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from aicodex.legal_data.spanish import fold_accents

_SUFFIX_RE = re.compile(r"\b(\d{1,4})\s+(bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies)\b")
_TOKEN_RE = re.compile(r"\d{1,4}/\d{2,4}|\d+(?:bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies)\b|\d+(?:[.,]\d+)*|[a-zñü]+")
_STOPWORDS = {
    "a", "al", "ante", "bajo", "con", "contra", "de", "del", "desde", "durante", "en", "entre", "hacia", "hasta",
    "mediante", "para", "por", "segun", "sin", "sobre", "tras", "el", "la", "los", "las", "lo", "un", "una", "unos",
    "unas", "y", "e", "o", "u", "ni", "que", "se", "su", "sus", "es", "son", "ser", "sera", "seran", "este", "esta",
    "estos", "estas", "ese", "esa", "dicho", "dicha", "cual", "cuales", "como", "cuando", "donde", "si", "no", "mas",
    "muy", "ya", "le", "les", "me", "mi", "tu", "nos", "ha", "han", "haber", "hay", "puede", "pueden", "podra",
    "podran", "debe", "deben", "otro", "otra", "otros", "otras", "todo", "toda", "todos", "todas", "cada", "qué",
    "que", "articulo", "articulos", "art", "arts", "ley", "apartado", "presente", "caso", "casos",
}
def stem(token: str) -> str:
    """A light Spanish stem: the plural -s/-es, then a final gender/number vowel, never below four
    characters -- "arrendamientos" and "arrendamiento" -> "arrendamient", "obligadas" -> "obligad"."""
    if token[0].isdigit() or len(token) <= 4:
        return token
    if token.endswith("es") and len(token) > 5 and token[-3] not in "aeiou":
        token = token[:-2]
    elif token.endswith("s"):
        token = token[:-1]
    if len(token) > 4 and token[-1] in "aoe":
        token = token[:-1]
    return token


def _variants(token: str) -> set[str]:
    forms = {token}
    stemmed = stem(token)
    if stemmed != token:
        forms.add(stemmed)
    return forms


def term_groups(text: str) -> list[set[str]]:
    """One set per word of `text`: the word and its stem."""
    text = _SUFFIX_RE.sub(r"\1\2", fold_accents(text).lower())
    groups = []
    for token in _TOKEN_RE.findall(text):
        if token in _STOPWORDS:
            continue
        forms = _variants(token) - _STOPWORDS
        if forms:
            groups.append(forms)
    return groups


def terms(text: str) -> list[str]:
    """Index terms of `text`: every word with its stem."""
    return [form for group in term_groups(text) for form in group]


def query_groups(query: str) -> list[frozenset[str]]:
    """The distinct words of a query, each as the set of its forms. A query word scores once, by
    its best-matching form, so a word and its stem never count twice."""
    return list(dict.fromkeys(frozenset(group) for group in term_groups(query)))


_K1, _B = 1.2, 0.75


class KeywordIndex:
    def __init__(self, docs: list[tuple[str, str]]) -> None:  # (chunk_id, text)
        self.ids = [chunk_id for chunk_id, _ in docs]
        self.tf = [Counter(terms(text)) for _, text in docs]
        self.lengths = [sum(tf.values()) for tf in self.tf]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter = Counter()
        for tf in self.tf:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {term: math.log(1 + (n - count + 0.5) / (count + 0.5)) for term, count in df.items()}

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """(chunk_id, BM25 score) of the best `k` chunks with any query term; each query word
        counts once, by its best-matching form (query_groups)."""
        groups = query_groups(query)
        scored = []
        for i, tf in enumerate(self.tf):
            score = 0.0
            norm_base = _K1 * (1 - _B + _B * self.lengths[i] / (self.avg_length or 1.0))
            for group in groups:
                score += max((self.idf[term] * tf[term] * (_K1 + 1) / (tf[term] + norm_base)
                              for term in group if tf.get(term)), default=0.0)
            if score > 0:
                scored.append((self.ids[i], score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]
