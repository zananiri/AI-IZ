"""The attorney's case files (legal/case_files.py): classification, structure-aware chunking per
document type, and index -> search -> answer with a stub embedder and model."""

import asyncio
import hashlib

import numpy as np
import pytest

from docslides.config import get_config
from docslides.legal import case_files
from docslides.legal.case_files import CaseDocument, PageText, build_units, chunk_document, classify, find_date


def _doc(text_pages, doc_type, rel_path="Cohen v. Levi/claim.pdf", title="כתב תביעה", wrapped=True, doc_date="2024-03-01"):
    pages = [PageText(n, t, wrapped) for n, t in enumerate(text_pages, 1)]
    from pathlib import Path

    return CaseDocument(Path("/x") / rel_path, rel_path, case_files.matter_of(rel_path), doc_type, title, doc_date, pages)


def test_classify_by_name_and_opening():
    assert classify("כתב תביעה.pdf", "") == "pleading"
    assert classify("claim.pdf", "בית משפט השלום בתל אביב\nכתב הגנה\n1. הנתבע מכחיש") == "pleading"
    assert classify("agreement.docx", "") == "contract"
    assert classify("x.pdf", "פרוטוקול דיון\n" + "ש: איפה היית?\nת: בבית.\n" * 4) == "transcript"
    assert classify("x.txt", "From: a@b.com\nTo: c@d.com\nSubject: offer\nDate: 1 March 2024\n\nHi") == "email"
    assert classify("mail.eml", "") == "email"
    assert classify("פסק דין.pdf", "") == "judgment"
    assert classify("scan.pdf", "random words") == "document"


def test_find_date_formats():
    assert find_date("נחתם ביום 16 ביולי 2024 בתל אביב") == "2024-07-16"
    assert find_date("Dated 01/03/2024") == "2024-03-01"  # day first
    assert find_date("Signed March 5, 2023") == "2023-03-05"
    assert find_date("2022-11-30 hearing") == "2022-11-30"
    assert find_date("see clause 1.2.10 and 3.4.12") == ""  # clause numbers, not dates
    assert find_date("no date here") == ""


def test_wrapped_pdf_lines_rejoin_into_numbered_paragraphs_with_headings():
    page = "\n".join([
        "העובדות",
        "1. התובע התקשר עם הנתבעת בהסכם לאספקת סחורה ביום 1.3.2024, והנתבעת התחייבה",
        "לספק את הסחורה תוך 30 יום ממועד החתימה על ההסכם.",
        "2. הנתבעת לא סיפקה את הסחורה במועד, ואף לא הודיעה לתובע על העיכוב בכתב או",
        "בעל פה.",
        "הנזק",
        "3. התובע נאלץ לרכוש סחורה חלופית במחיר גבוה יותר.",
    ])
    units = build_units([PageText(4, page)], "pleading")
    paras = [u for u in units if u.kind == "para"]
    assert [u.label for u in paras] == ["1", "2", "3"]
    assert paras[0].text.endswith("על ההסכם.")  # the wrapped line joined its paragraph
    assert paras[0].section == "העובדות" and paras[2].section == "הנזק"
    assert all(u.page == 4 for u in units)


def test_numbered_paragraph_is_never_split_and_header_names_pages_and_paragraphs():
    para = "{n}. " + "הנתבעת הפרה את ההסכם הפרה יסודית ולא תיקנה את ההפרה למרות התראה. " * 6
    pages = ["\n".join(para.format(n=n) for n in range(1 + 5 * p, 6 + 5 * p)) for p in range(4)]
    chunks = chunk_document(_doc(pages, "pleading", wrapped=False), budget=400, overlap=0)
    assert len(chunks) > 1
    for chunk in chunks:
        body = chunk.text.split("\n\n", 1)[1]
        for line in body.splitlines():
            assert line.endswith("למרות התראה.")  # whole paragraphs only
        assert chunk.text.startswith("Matter: Cohen v. Levi | File: Cohen v. Levi/claim.pdf | Type: pleading")
        assert "Para. " in chunk.text.splitlines()[0]
        assert chunk.metadata["page_start"] <= chunk.metadata["page_end"]
    assert chunks[0].metadata["labels"].startswith("1")
    assert chunks[-1].metadata["page_end"] == 4
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_transcript_keeps_question_with_answer():
    lines = []
    for n in range(40):
        lines += [f"ש: שאלה מספר {n} על אירועי יום החתימה ומה אמר לך המנהל באותו יום?",
                  f"ת: תשובה מספר {n}: הוא אמר שהסחורה תגיע בזמן ושאין סיבה לדאגה."]
    chunks = chunk_document(_doc(["\n".join(lines)], "transcript", "Cohen v. Levi/hearing.pdf"), budget=200, overlap=40)
    assert len(chunks) > 3
    for chunk in chunks:
        body = chunk.text.split("\n\n", 1)[1].splitlines()
        assert body[0].startswith("ש:") and body[-1].startswith("ת:")  # never cut between Q and A


def test_email_thread_splits_per_message_with_its_header():
    thread = "\n".join([
        "From: Dan Cohen <dan@example.com>",
        "To: Ruth Levi <ruth@example.com>",
        "Date: 5 March 2024",
        "Subject: Termination notice",
        "",
        "Ruth, we are terminating the agreement effective immediately due to the delays.",
        "",
        "-----Original Message-----",
        "From: Ruth Levi <ruth@example.com>",
        "Sent: 1 March 2024",
        "Subject: Delivery",
        "",
        "Dan, the delivery will be two weeks late because of the strike at the port.",
    ])
    doc = _doc([thread], "email", "Cohen v. Levi/mail/thread.eml", title="Termination notice")
    doc.pages[0].number = None
    chunks = chunk_document(doc, budget=400, overlap=60)
    assert len(chunks) == 2
    assert "From: Dan Cohen" in chunks[0].text and "terminating" in chunks[0].text
    assert "strike at the port" not in chunks[0].text
    assert "From: Ruth Levi" in chunks[1].text and "strike at the port" in chunks[1].text
    assert "terminating" not in chunks[1].text.split("\n\n", 1)[1]  # no overlap across messages


def test_docx_auto_numbering_is_restored(tmp_path):
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_heading("Statement of Claim", level=1)
    for text in ("The plaintiff signed the agreement.", "The defendant did not deliver.", "Damages followed."):
        document.add_paragraph(text, style="List Number")
    path = tmp_path / "Cohen v. Levi" / "claim.docx"
    path.parent.mkdir()
    document.save(path)
    doc = case_files.load_document(path, tmp_path)
    assert doc.matter == "Cohen v. Levi" and doc.doc_type == "pleading"
    units = [u for u in build_units(doc.pages, doc.doc_type) if u.kind == "para"]
    assert [u.label for u in units] == ["1", "2", "3"]
    assert units[0].section == "Statement of Claim"


def _fake_embed(texts):
    """Bag-of-words hashing vectors: enough for dense search to prefer chunks sharing words."""
    out = np.zeros((len(texts), 256), dtype=np.float32)
    for i, text in enumerate(texts):
        for word in text.lower().split():
            out[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1.0
        out[i] /= np.linalg.norm(out[i]) or 1.0
    return out


@pytest.fixture
def case_folder(tmp_path, monkeypatch):
    pytest.importorskip("chromadb")
    pytest.importorskip("sklearn")
    cfg = get_config()
    monkeypatch.setattr(cfg.legal.case_files, "folder", str(tmp_path / "legal_data"))
    monkeypatch.setattr(cfg.legal.case_files, "vectordb_dir", str(tmp_path / "vectordb"))
    monkeypatch.setattr(cfg.legal.retrieval, "reranker_model", None)
    monkeypatch.setattr(case_files, "_embed", _fake_embed)
    case_files._lexical_index.cache_clear()
    root = tmp_path / "legal_data"
    (root / "Cohen v. Levi").mkdir(parents=True)
    (root / "Estate of Mizrahi").mkdir()
    (root / "Cohen v. Levi" / "statement of claim.txt").write_text(
        "Statement of Claim\n\n1. The plaintiff ordered 500 steel beams on 1 March 2024.\n\n"
        "2. The defendant promised delivery within 30 days and never delivered the beams.\n", encoding="utf-8")
    (root / "Cohen v. Levi" / "statement of defence.txt").write_text(
        "Statement of Defence\n\n1. The defendant denies any delivery deadline was agreed.\n\n"
        "2. The strike at Haifa port delayed all shipments, a force majeure event.\n", encoding="utf-8")
    (root / "Estate of Mizrahi" / "will.txt").write_text(
        "Last will\n\n1. I leave the apartment on Herzl street to my daughter Maya.\n", encoding="utf-8")
    (root / "old.doc").write_bytes(b"binary")
    return root


def test_index_search_and_answer_end_to_end(case_folder, monkeypatch):
    reports = case_files.index_folder()
    actions = {r.rel_path: r.action for r in reports}
    assert actions["Cohen v. Levi/statement of claim.txt"] == "indexed"
    assert actions["old.doc"] == "skipped"
    assert case_files.matters() == ["Cohen v. Levi", "Estate of Mizrahi"]

    again = {r.rel_path: r.action for r in case_files.index_folder()}
    assert again["Estate of Mizrahi/will.txt"] == "unchanged"

    hits = case_files.search(["What caused the delay of the shipments? strike port"], matter="Cohen v. Levi")
    assert hits and all(h.metadata["matter"] == "Cohen v. Levi" for h in hits)
    assert any("strike at Haifa port" in h.text for h in hits)

    class FakeLLM:
        model = "fake"

        async def complete_text(self, messages, call_site, sampling=None, enable_thinking=None):
            assert call_site.name == "legal_case_files_chat"
            assert "<excerpts>" in messages[-1].content
            return "The defendant attributes the delay to a strike at Haifa port [1]."

    import docslides.llm.client as client

    monkeypatch.setattr(client, "get_legal_orchestrator_client", lambda: FakeLLM())
    reply, excerpts = asyncio.run(case_files.answer("Why were the beams late? strike", matter="Cohen v. Levi"))
    assert case_files.cited(reply, excerpts)[0][0] == 1

    (case_folder / "Estate of Mizrahi" / "will.txt").unlink()
    assert {r.rel_path: r.action for r in case_files.index_folder()}["Estate of Mizrahi/will.txt"] == "missing"
    assert {r.rel_path: r.action for r in case_files.index_folder(prune=True)}["Estate of Mizrahi/will.txt"] == "pruned"
    assert case_files.matters() == ["Cohen v. Levi"]
