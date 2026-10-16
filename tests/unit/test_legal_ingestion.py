import hashlib
import hmac
import json

import pytest

from docslides.config import get_config
from docslides.legal import bundle, chunking
from docslides.legal.chunking import SourceMeta, chunk_sections
from docslides.legal.models import ChunkMetadata, content_hash
from docslides.legal.structure import extract_cross_references, parse_sections

STATUTE = """\
חוק החוזים (חלק כללי), תשל"ג-1973
פרק א': כריתת החוזה
הצעה
1. חוזה נכרת בדרך של הצעה וקיבול לפי הוראות פרק זה.
3. (א) הצעה ניתנת לחזרה כל עוד לא הגיעה הודעת הקיבול למציע, והכל בכפוף לסעיף 4.
(ב) הודעת החזרה צריכה להגיע לניצע לפני שנשלחה הודעת הקיבול.
4. הצעה שנקבעה לה תקופה, אינה ניתנת לחזרה בתוך אותה תקופה; ראו גם סעיף 5 לחוק המכר.
פרק ב': ביטול החוזה
טעות
14. (א) מי שהתקשר בחוזה עקב טעות רשאי לבטל את החוזה.
(ב) רשאי בית המשפט לבטל את החוזה.
"""

META = SourceMeta(
    law_id="contracts-1973",
    law_name='חוק החוזים (חלק כללי), תשל"ג-1973',
    effective_date_start="1973-06-01",
    status="current",
    source_type="statute",
    source_origin="knesset",
)


@pytest.fixture(autouse=True)
def word_token_count(monkeypatch):
    # Deterministic budgets without loading a real tokenizer.
    monkeypatch.setattr(chunking, "count_tokens", lambda text: len(text.split()))


def test_parses_chapters_sections_titles_and_subsections():
    sections = {s.number: s for s in parse_sections(STATUTE)}

    assert list(sections) == ["preamble", "1", "3", "4", "14"]
    assert sections["1"].title == "הצעה"
    assert sections["1"].chapter == "פרק א': כריתת החוזה"
    assert sections["14"].chapter == "פרק ב': ביטול החוזה"
    assert sections["14"].title == "טעות"
    assert [s.label for s in sections["3"].subsections] == ["א", "ב"]


def test_body_text_starting_with_perek_is_not_a_chapter_heading():
    sections = parse_sections("1. טקסט ראשון.\nפרק זמן סביר ייקבע בתקנות.\n2. טקסט שני.")
    assert [s.number for s in sections] == ["1", "2"]
    assert "פרק זמן סביר" in sections[0].text


def test_year_or_list_numbers_do_not_open_sections():
    sections = parse_sections("1. ראשון.\n1973. שנה\n2. שני.\n1. פריט ברשימה")
    assert [s.number for s in sections] == ["1", "2"]


def test_internal_cross_references_resolve_and_external_ones_are_skipped():
    assert extract_cross_references("בכפוף לסעיף 4 ולסעיפים 6 עד 8", own_number="3") == ["4", "6", "8", "7"]
    assert extract_cross_references("ראו סעיף 5 לחוק המכר") == []
    assert extract_cross_references("כאמור בסעיף 5 לחוק זה") == ["5"]
    sections = {s.number: s for s in parse_sections(STATUTE)}
    assert sections["3"].cross_refs == ["4"]
    assert sections["4"].cross_refs == []  # "section 5 of the Sale Law" is external


def test_references_joined_by_or_to_another_law_are_external():
    # "סעיפים 6 או 7 לחוק־יסוד" once linked this law's section 6 (a whole amendment of it).
    assert extract_cross_references("מועמד אינו כשיר, לפי סעיפים 6 או 7 לחוק־יסוד: הכנסת") == []
    assert extract_cross_references("לפי סעיפים 6 או 7") == ["6", "7"]
    assert extract_cross_references("הוראות סעיפים 21א ו־24(ט1)") == ["21א", "24"]


AMENDING_LAW = """\
חוק הבחירות (הוראות מיוחדות), התשפ"ו-2026
1. הוראות פרק זה יחולו לעניין הבחירות.
2. כאמור בסעיף 1, יושב ראש הוועדה רשאי לקבוע הוראות.
⟦תיקון חוק מיסוי תשלומים - מס' 11⟧
3. בחוק מיסוי תשלומים בתקופת בחירות, התשנ"ו-1996, בסעיף 2(ב), במקום "25%" יבוא "18%".
4. בחוק המפלגות, התשנ"ב-1992, בסעיף 2, אחרי "של ראש הרשות" יבוא "לרבות צילום".
"""


def test_amending_sections_link_no_cross_references(monkeypatch):
    monkeypatch.setattr(get_config().legal.ingestion, "chunk_max_tokens", 500)
    meta = SourceMeta(law_id="elections", law_name='חוק הבחירות (הוראות מיוחדות), התשפ"ו-2026',
                      effective_date_start="2026-07-16", status="current", source_type="statute",
                      source_origin="knesset")
    by_id = {c.metadata.chunk_id: c for c in chunk_sections(parse_sections(AMENDING_LAW), meta, "2026-09-23")}

    assert by_id["elections@2026-07-16:2"].metadata.cross_references == ["elections@2026-07-16:1"]
    # "בסעיף 2" in an amendment is section 2 of the amended law, not of this one --
    # whether the section is marked by its "תיקון" title (3) or only by its opening words (4).
    assert by_id["elections@2026-07-16:3"].metadata.cross_references == []
    assert by_id["elections@2026-07-16:4"].metadata.cross_references == []


def test_hebrew_chunks_store_gershayim_instead_of_ascii_quotes(monkeypatch):
    monkeypatch.setattr(get_config().legal.ingestion, "chunk_max_tokens", 500)
    chunks = chunk_sections(parse_sections(AMENDING_LAW), SourceMeta(
        law_id="elections", law_name='חוק הבחירות (הוראות מיוחדות), התשפ"ו-2026', effective_date_start="2026-07-16",
        status="current", source_type="statute", source_origin="knesset"), "2026-09-23")
    amendment = next(c for c in chunks if c.metadata.section_number == "3")

    assert all('"' not in c.text and '"' not in c.metadata.breadcrumb for c in chunks)
    assert 'במקום ״25%״ יבוא ״18%״' in amendment.text and "התשנ״ו-1996" in amendment.text
    assert amendment.metadata.law_name == 'חוק הבחירות (הוראות מיוחדות), התשפ"ו-2026'  # identity field untouched

    english = SourceMeta(law_id="en", law_name="Contracts Law", effective_date_start="1973-06-01", status="current",
                         source_type="statute", source_origin="knesset", language="en")
    (chunk,) = chunk_sections(parse_sections('1. The term "offer" means a proposal.'), english, "2026-09-23")
    assert '"offer"' in chunk.text


def test_short_sections_are_one_chunk_with_breadcrumb_and_metadata(monkeypatch):
    monkeypatch.setattr(get_config().legal.ingestion, "chunk_max_tokens", 500)
    chunks = chunk_sections(parse_sections(STATUTE), META, "2026-09-23")
    by_id = {c.metadata.chunk_id: c for c in chunks}

    chunk = by_id["contracts-1973@1973-06-01:14"]
    # Stored Hebrew text carries ״ where the source had an ASCII double quote (normalize_hebrew_quotes).
    assert chunk.text.startswith("חוק החוזים (חלק כללי), תשל״ג-1973 > פרק ב': ביטול החוזה > סעיף 14 — טעות")
    assert chunk.metadata.source_type == "statute"
    assert chunk.metadata.source_origin == "knesset"
    assert by_id["contracts-1973@1973-06-01:3"].metadata.cross_references == ["contracts-1973@1973-06-01:4"]


def test_long_section_splits_at_subsections_then_into_sibling_parts(monkeypatch):
    monkeypatch.setattr(get_config().legal.ingestion, "chunk_max_tokens", 20)
    chunks = chunk_sections(parse_sections(STATUTE), META, "2026-09-23")
    ids = [c.metadata.chunk_id for c in chunks]

    assert "contracts-1973@1973-06-01:3(א)" in ids
    assert "contracts-1973@1973-06-01:3(ב)" in ids
    parts = [c for c in chunks if c.metadata.source_id == "contracts-1973@1973-06-01:4"]
    assert len(parts) == 2
    assert [p.metadata.chunk_id for p in parts] == [
        "contracts-1973@1973-06-01:4#p1",
        "contracts-1973@1973-06-01:4#p2",
    ]
    assert all(p.metadata.part_count == 2 for p in parts)
    # Nothing truncated: both parts together hold the whole provision.
    assert "אינה ניתנת לחזרה" in parts[0].text and "לחוק המכר" in parts[1].text


def test_metadata_round_trips_through_chroma_form_and_hash_is_stable():
    meta = ChunkMetadata(
        chunk_id="x:1", source_id="x:1", section_key="x:1", law_id="x", law_name="חוק", chapter=None, part=None,
        section_number="1", subsection_number=None, breadcrumb="חוק > סעיף 1", effective_date_start="2000-01-01",
        effective_date_end=None, status="current", source_type="statute", source_origin="knesset",
        ingestion_date="2026-09-23", language="he", cross_references=["x:2"],
    )
    flat = meta.to_chroma()
    assert flat["effective_date_end"] == "" and json.loads(flat["cross_references"]) == ["x:2"]
    assert ChunkMetadata.from_chroma(flat) == meta
    assert content_hash("text", flat) == content_hash("text", ChunkMetadata.from_chroma(flat).to_chroma())
    assert content_hash("text", flat) != content_hash("text!", flat)


def test_bundle_signature_detects_tampering(tmp_path, monkeypatch):
    manifest = tmp_path / "bundle.json"
    monkeypatch.setenv(bundle.BUNDLE_KEY_ENV, "secret")
    bundle.save_entries({"a": {"sha256": "1"}}, path=manifest)
    entries, level = bundle.verify(manifest)
    assert level == "signed" and entries == {"a": {"sha256": "1"}}

    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["entries"]["b"] = {"sha256": "2"}
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(bundle.BundleError):
        bundle.verify(manifest)

    monkeypatch.delenv(bundle.BUNDLE_KEY_ENV)
    assert bundle.verify(manifest)[1] == "hashes_only"
    with pytest.raises(bundle.BundleError):
        bundle.save_entries({}, path=manifest)


def test_bundle_signature_is_hmac_sha256_over_canonical_json(tmp_path, monkeypatch):
    manifest = tmp_path / "bundle.json"
    monkeypatch.setenv(bundle.BUNDLE_KEY_ENV, "secret")
    bundle.save_entries({"a": {"sha256": "1"}}, path=manifest)
    expected = hmac.new(b"secret", b'{"a":{"sha256":"1"}}', hashlib.sha256).hexdigest()
    assert json.loads(manifest.read_text(encoding="utf-8"))["signature"] == expected
