from docslides.canon.chunking import ProvisionRecord, chunk_provision, citation_label


def test_short_canon_produces_single_chunk_with_header():
    record = ProvisionRecord(
        code="cic",
        number="1055",
        paragraph=None,
        breadcrumb="Book IV: Function of the Church > Title VII: Marriage",
        text="Marriage is ordered toward the good of the spouses.",
        source_url="https://www.vatican.va/archive/cod-iuris-canonici/eng/documents/cic_lib4-cann1055-1062_en.html",
        language="en",
    )

    chunks = chunk_provision(record)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.id == "cic:1055"
    assert chunk.text.startswith("CIC Can. 1055 — Book IV: Function of the Church > Title VII: Marriage")
    assert "Marriage is ordered" in chunk.text
    assert chunk.code == "cic"
    assert chunk.source_url == record.source_url


def test_canon_with_paragraph_includes_section_mark_and_id():
    record = ProvisionRecord(
        code="cic", number="1167", paragraph="1", breadcrumb="Book IV",
        text="The Apostolic See alone can establish new sacramentals.",
        source_url="https://example.org/cic", language="en",
    )

    chunks = chunk_provision(record)

    assert len(chunks) == 1
    assert chunks[0].id == "cic:1167:1"
    assert "Can. 1167 §1" in chunks[0].text


def test_oversized_provision_splits_into_multiple_chunks_with_shared_header():
    long_text = " ".join(f"Canon prohibet actum numero {i}." for i in range(2000))
    record = ProvisionRecord(
        code="cceo", number="7", paragraph="1", breadcrumb="TITULUS I",
        text=long_text, source_url="https://example.org/cceo", language="la",
    )

    chunks = chunk_provision(record)

    assert len(chunks) > 1
    for idx, chunk in enumerate(chunks, start=1):
        assert chunk.id == f"cceo:7:1:{idx}"
        assert chunk.text.startswith("CCEO Can. 7 §1 — TITULUS I (part")
    # no sentence content lost or duplicated across the split
    rejoined = " ".join(chunk.text.split("\n\n", 1)[1] for chunk in chunks)
    assert "Canon prohibet actum numero 0" in rejoined
    assert "Canon prohibet actum numero 1999" in rejoined


def test_citation_label_formatting():
    assert citation_label("cic", "1055", "1") == "CIC Can. 1055 §1"
    assert citation_label("cceo", "7") == "CCEO Can. 7"
