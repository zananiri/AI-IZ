"""The RAM guard in front of the in-process model loads: out of RAM, the Rust tokenizer aborts
the whole app instead of raising, so the reranker is skipped and the embedder raises first."""

import pytest

from docslides import system_memory
from docslides.config import get_config
from docslides.legal import retrieval
from docslides.rag import embedding


def test_available_bytes_is_a_number_or_unknown():
    free = system_memory.available_bytes()
    assert free is None or free > 0


def test_has_room_passes_when_unknown_or_unset(monkeypatch):
    monkeypatch.setattr(system_memory, "available_bytes", lambda: None)
    assert system_memory.has_room(3.0)[0]
    monkeypatch.setattr(system_memory, "available_bytes", lambda: 1)
    assert system_memory.has_room(None)[0]
    assert not system_memory.has_room(3.0)[0]


def test_reranker_skipped_with_little_ram_and_tried_again_later(monkeypatch):
    loads = []
    monkeypatch.setattr(retrieval, "_load_reranker", lambda name, device=None: loads.append(name) or "model")
    retrieval._reranker_room_checked.clear()
    monkeypatch.setattr(get_config().legal.retrieval, "min_free_ram_gb_for_reranker", 3.0)
    monkeypatch.setattr(system_memory, "available_bytes", lambda: 1024**3)
    assert retrieval._reranker("m", "cpu") is None and loads == []
    monkeypatch.setattr(system_memory, "available_bytes", lambda: 8 * 1024**3)
    assert retrieval._reranker("m", "cpu") == "model"
    monkeypatch.setattr(system_memory, "available_bytes", lambda: 1024**3)  # loaded already: no recheck
    assert retrieval._reranker("m", "cpu") == "model"
    retrieval._reranker_room_checked.clear()


def test_embedder_raises_instead_of_crashing_with_little_ram(monkeypatch):
    pytest.importorskip("sentence_transformers")
    monkeypatch.setattr(system_memory, "available_bytes", lambda: 1024**3)
    embedding._get_embedder.cache_clear()
    with pytest.raises(MemoryError, match="Not enough free RAM"):
        embedding.get_embedder("not/a-model", "cpu")
