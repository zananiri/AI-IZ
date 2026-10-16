"""Corpus chunking and index state without a model or vector store: law chunks carry the law name,
section numbers and filterable metadata; judgment chunks repeat the case header; overlap works
(and is off by default in the shared section chunker); the incremental state tracks hashes.
Also vectorize.py's guards, with fakes for the model, torch and Chroma: fp16 NaN vectors re-encoded in
fp32, a missing GPU refused up front, an empty collection after indexing an error."""

import importlib.util
import json
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest

from docslides.cleaning.tokens import count_tokens
from docslides.legal import chunking
from docslides.legal_data import corpus_index, wikisource
from docslides.legal_data.corpus_chunking import CorpusChunk, chunk_record, pack_paragraphs
from docslides.legal_data.corpus_index import CorpusState
from docslides.legal_data.records import CorpusRecord

WIKITEXT = """{{ח:התחלה}}
{{ח:כותרת|חוק החוזים (חלק כללי), תשל״ג–1973}}
{{ח:קטע2|פרק א|פרק א׳: כריתת החוזה}}
{{ח:סעיף|1|כריתת חוזה – כיצד}}
{{ח:ת}} חוזה נכרת בדרך של הצעה וקיבול.
{{ח:סעיף|3|חזרה מן ההצעה}}
{{ח:תת|(א)}} המציע רשאי לחזור בו מן ההצעה.
{{ח:תת|(ב)}} קבע המציע שהצעתו היא ללא חזרה, אין הוא רשאי לחזור בו.
{{ח:קטע2|תוספת|התוספת הראשונה}}
{{ח:ת}} טופס בקשה.
{{ח:סוף}}
"""


def _record(**fields) -> dict:
    base = {"id": "r1", "source": "s", "source_url": "u", "license": "l", "retrieved_at": "2026-09-25T00:00:00+00:00"}
    return CorpusRecord(**{**base, **fields}).finalize().model_dump(mode="json")


def test_law_chunks_carry_the_law_name_sections_and_scalar_metadata():
    parsed = wikisource.parse_law_page(WIKITEXT, "חוק החוזים")
    record = _record(id="wikisource:1", category="laws", title=parsed.full_title, authority_level="law",
                     status="in_force", effective_date="1973-06-01", text=parsed.text, sections=parsed.sections)
    chunks = chunk_record(record, "hash", 500, 64, False)
    numbers = {c.metadata.get("section_number") for c in chunks}
    assert {"1", "3"} <= numbers and any(n.startswith("schedule:") for n in numbers)
    assert all(c.text.startswith(record["title"]) for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(isinstance(v, (str, int, float, bool)) for c in chunks for v in c.metadata.values())
    assert all(c.metadata["status"] == "in_force" and c.metadata["effective_ymd"] == 19730601 for c in chunks)
    assert not any(ch in "ךםןףץ" for c in chunks for ch in c.lexical_text)


def test_judgment_chunks_repeat_the_case_header_and_overlap():
    paragraph = "בית המשפט דן בטענות הצדדים ומצא כי יש לדחות את הערעור"
    record = _record(id="supreme_court:x", category="supreme_court", title="t", authority_level="judgment",
                     court="בית המשפט העליון", case_number='ע"א 1/20', case_name="א נ' ב", decision_date="2021-01-01",
                     text="\n".join(f"{paragraph} {i}" for i in range(40)))
    size = count_tokens(f"{paragraph} 10")
    chunks = chunk_record(record, "hash", size * 6 + 60, size + 2, False)
    assert len(chunks) > 1
    assert all(c.text.startswith('בית המשפט העליון | ע"א 1/20 | א נ\' ב | 2021-01-01') for c in chunks)
    bodies = [c.text.split("\n\n", 1)[1].splitlines() for c in chunks]
    assert all(a[-1] in b for a, b in zip(bodies, bodies[1:]))
    assert chunks[0].metadata["decision_ymd"] == 20210101 and chunks[0].metadata["case_number"] == 'ע"א 1/20'


def test_paragraph_packing_overlaps_only_when_asked():
    paragraphs = [f"פסקה מספר {i} בפסק הדין" for i in range(20)]
    size = max(count_tokens(p) for p in paragraphs)
    plain = pack_paragraphs(paragraphs, size * 4, 0)
    assert [line for group in plain for line in group.splitlines()] == paragraphs
    overlapped = pack_paragraphs(paragraphs, size * 4, size)
    assert len(overlapped) > 1
    assert all(a.splitlines()[-1] in b.splitlines() for a, b in zip(overlapped, overlapped[1:]))


def test_section_chunker_overlap_is_off_by_default():
    text = " ".join(f"משפט מספר {i}." for i in range(30))
    unit = count_tokens("משפט מספר 10.")
    parts = chunking._pack(text, unit * 5)
    assert sum(len(p.splitlines()) for p in parts) == 30  # every sentence exactly once
    overlapped = chunking._pack(text, unit * 5, unit * 2)
    assert all(a.splitlines()[-1] in b.splitlines() for a, b in zip(overlapped, overlapped[1:]))


def test_corpus_state_tracks_hashes_chunks_and_the_lexical_copy(tmp_path):
    state = CorpusState(tmp_path / "state.sqlite")
    chunks = [CorpusChunk("r1#1", "t", "t", "lex one", {}), CorpusChunk("r1#2", "t", "t", "lex two", {})]
    state.put("laws", "r1", "h1", chunks)
    state.commit()
    indexed = state.get("laws", "r1")
    assert (indexed.record_hash, indexed.chunk_ids) == ("h1", ["r1#1", "r1#2"])
    state.put("laws", "r1", "h2", chunks[:1])  # the record changed: fewer chunks now
    assert [row[0] for row in state.lexical("laws")] == ["r1#1"]
    assert state.record_ids("laws") == {"r1"} and state.record_ids("supreme_court") == set()
    state.delete("laws", "r1")
    assert state.get("laws", "r1") is None
    state.close()


def _vectorize():
    path = Path(__file__).resolve().parents[2] / "scripts" / "legal_data" / "vectorize.py"
    spec = importlib.util.spec_from_file_location("vectorize_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nan_vectors_from_fp16_are_re_encoded_in_fp32(monkeypatch):
    np = pytest.importorskip("numpy")
    from docslides.rag import embedding

    class Model:
        def __init__(self, bad: bool) -> None:
            self.bad, self.calls = bad, []

        def encode(self, texts, **kwargs):
            self.calls.append(list(texts))
            vectors = np.ones((len(texts), 4), dtype=np.float16 if self.bad else np.float32)
            if self.bad:
                vectors[1] = np.nan
            return vectors

    half, full = Model(bad=True), Model(bad=False)
    monkeypatch.setattr(embedding, "get_embedder",
                        lambda name, device=None, full_precision=False: full if full_precision else half)
    vectors = _vectorize().Embedder("m", ["cuda"], 8).encode(["a", "b", "c"])
    assert np.isfinite(vectors).all() and vectors.dtype == np.float32
    assert full.calls == [["b"]]  # only the NaN row is redone


def test_a_missing_gpu_is_refused_before_any_work(monkeypatch):
    cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    vectorize = _vectorize()
    vectorize.check_devices(["cuda"])
    vectorize.check_devices(["cpu"])
    with pytest.raises(SystemExit, match="Accelerator"):
        vectorize.check_devices(["cuda:0", "cuda:1"])
    cuda.is_available = lambda: False
    with pytest.raises(SystemExit, match="0 CUDA"):
        vectorize.check_devices(["cuda"])


def test_records_read_into_an_empty_collection_is_an_error(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    parsed = wikisource.parse_law_page(WIKITEXT, "חוק החוזים")
    record = _record(id="wikisource:1", category="laws", title=parsed.full_title, authority_level="law",
                     status="in_force", text=parsed.text, sections=parsed.sections)
    (tmp_path / "in" / "laws").mkdir(parents=True)
    (tmp_path / "in" / "laws" / "laws.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    class Collection:  # upserts that silently store nothing
        def __init__(self, *args) -> None:
            pass

        def upsert(self, chunks, vectors) -> None:
            pass

        def count(self) -> int:
            return 0

    class Embedder:
        def encode(self, texts):
            return np.ones((len(texts), 4), dtype=np.float32)

        def close(self) -> None:
            pass

    monkeypatch.setattr(corpus_index, "CorpusCollection", Collection)
    args = Namespace(input=str(tmp_path / "in"), vectordb=str(tmp_path / "db"), dry_run=False, shard=None, sample=None,
                     rebuild=False, prune=False, chunk_tokens=500, overlap_tokens=64)
    cfg = types.SimpleNamespace(collection_prefix="t", fold_final_letters_for_embedding=False)
    with pytest.raises(RuntimeError, match="collection is empty"):
        _vectorize().index_category("laws", args, cfg, "m", Embedder)
