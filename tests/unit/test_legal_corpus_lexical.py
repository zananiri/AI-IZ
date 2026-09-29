"""BM25 over the bulk corpus's lexical copy (legal/corpus_lexical.py)."""

import json

from docslides.legal.corpus_lexical import lexical_index
from docslides.legal_data.hebrew import normalize_for_index


def _write_lexical(directory, rows):
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / f"lexical_{directory.name}.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps({"chunk_id": chunk_id, "record_id": chunk_id, "text": normalize_for_index(text)},
                                ensure_ascii=False) + "\n" for chunk_id, text in rows)


def test_the_rare_term_of_art_ranks_its_chunk_first_and_the_index_is_cached(tmp_path):
    _write_lexical(tmp_path / "laws", [
        ("a", "חוזה שנכרת עקב עושק ניתן לבטלו"),
        ("b", "חוזה שנכרת עקב טעות"),
        ("c", "הודעת ביטול תינתן תוך זמן סביר"),
    ])
    lexical_index.cache_clear()

    index = lexical_index(str(tmp_path), "laws")

    assert [chunk_id for chunk_id, _ in index.search("עושק", 5)] == ["a"]
    assert index.search("ביטול חוזה בשל עושק", 3)[0][0] == "a"
    assert index.search("מילה שאינה בשום מקום", 5) == []
    assert (tmp_path / "laws" / "bm25_laws.npz").exists()

    lexical_index.cache_clear()
    assert lexical_index(str(tmp_path), "laws").search("שנכרת עקב עושק", 1)[0][0] == "a"  # from the cache


def test_prefixed_forms_match_and_a_missing_file_turns_bm25_off(tmp_path):
    _write_lexical(tmp_path / "laws", [("a", "המעביד ישלם פיצויי פיטורים"), ("b", "העובד יקבל הודעה מוקדמת")])
    lexical_index.cache_clear()

    index = lexical_index(str(tmp_path), "laws")

    assert index.search("לפיצויי פיטורים", 1)[0][0] == "a"  # "ל" + "פיצויי": the prefix is stripped
    assert lexical_index(str(tmp_path), "procedural_rules") is None
