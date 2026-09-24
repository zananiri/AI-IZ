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


def terms(text: str) -> list[str]:
    """Index terms of `text`: every word with its prefix-stripped forms."""
    text = join_spaced_section_numbers(_POINTS_RE.sub("", text))
    text = _QUOTES_RE.sub("", text).replace("־", " ").lower()
    out: list[str] = []
    for token in _TOKEN_RE.findall(text):
        if token in _STOPWORDS:
            continue
        out.extend(_variants(token) - _STOPWORDS)
    return out


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
        """(chunk_id, BM25 score) of the best `k` chunks with any query term."""
        query_terms = set(terms(query))
        scored = []
        for i, tf in enumerate(self.tf):
            score = 0.0
            for term in query_terms:
                freq = tf.get(term)
                if not freq:
                    continue
                norm = freq + _K1 * (1 - _B + _B * self.lengths[i] / (self.avg_length or 1.0))
                score += self.idf[term] * freq * (_K1 + 1) / norm
            if score > 0:
                scored.append((self.ids[i], score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]
