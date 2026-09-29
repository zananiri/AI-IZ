"""Hebrew-aware keyword search (BM25) over the Legal index, for retrieval.py.

Dense embeddings match meaning but are weak on exact identifiers -- section
numbers ("116יז10", "132א"), coined terms ("נחזות עמוקה"), amounts -- which is
exactly what a lawyer's question names. Keyword search is the second signal;
retrieval.py fuses both rankings.

Hebrew glues prefixes to words (ו/ה/ב/ל/מ/ש/כ: "בהיוועדות", "והמפונים"), so
every word is indexed and queried together with its prefix-stripped forms;
quote marks inside abbreviations are dropped (כ"ב, כ״ב -> כב) and a maqaf
splits words the way a space does. The index is small (one law is a few dozen
chunks) and is rebuilt only when the signed bundle changes.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from docslides.legal.insertions import join_spaced_section_numbers

_POINTS_RE = re.compile("[֑-ׇ]")
_QUOTES_RE = re.compile("[\"'׳״‘’“”]")
_TOKEN_RE = re.compile(r"\d+[א-ת]{1,3}\d*|\d+(?:[.,]\d+)*|[א-תa-zA-Z]+")
_PREFIXES = sorted(
    ["ו", "ה", "ב", "ל", "מ", "ש", "כ", "וה", "וב", "ול", "ומ", "וש", "וכ", "שה", "שב", "של", "מה", "בה", "לה",
     "כש", "וכש", "ושה", "ובה", "ולה", "ומה", "משה"],
    key=len, reverse=True,
)
_STOPWORDS = {
    "של", "על", "את", "או", "לא", "אם", "כי", "זה", "זו", "הוא", "היא", "לפי", "בו", "בה", "מה", "מי", "כל", "גם",
    "אשר", "עם", "יש", "אין", "אל", "כך", "הם", "הן", "אחר", "אחרי", "לרבות", "חוק", "סעיף", "סעיפים", "קטן",
}
_K1, _B = 1.2, 0.75


def _variants(token: str) -> set[str]:
    forms = {token}
    if token[0].isdigit():
        return forms
    for prefix in _PREFIXES:
        if token.startswith(prefix) and len(token) - len(prefix) >= 3:
            forms.add(token[len(prefix):])
    return forms


def term_groups(text: str) -> list[set[str]]:
    """One set per word of `text`: the word and its prefix-stripped forms."""
    text = join_spaced_section_numbers(_POINTS_RE.sub("", text))
    text = _QUOTES_RE.sub("", text).replace("־", " ").lower()
    groups = []
    for token in _TOKEN_RE.findall(text):
        if token in _STOPWORDS:
            continue
        forms = _variants(token) - _STOPWORDS
        if forms:
            groups.append(forms)
    return groups


def terms(text: str) -> list[str]:
    """Index terms of `text`: every word with its prefix-stripped forms."""
    return [form for group in term_groups(text) for form in group]


def query_groups(query: str) -> list[frozenset[str]]:
    """The distinct words of a query, each as the set of its forms. A query word scores once, by
    its best-matching form: summing over the forms counted "ביטול" twice (as ביטול and as יטול, a
    ב that isn't a prefix), so any word that happens to start with a prefix letter outweighed
    a rarer term of art."""
    return list(dict.fromkeys(frozenset(group) for group in term_groups(query)))


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
