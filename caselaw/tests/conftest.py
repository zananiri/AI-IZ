import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


@pytest.fixture
def cfg(tmp_path):
    """A config writing under tmp_path, with the offline tokenizer and hashing encoder."""
    from israeli_caselaw_ingest.config import load_config

    return load_config(root=str(tmp_path), overrides={
        "chunk": {"tokenizer": "regex", "min_tokens": 120, "max_tokens": 180, "overlap_tokens": 30,
                  "doc_batch_size": 50},
        "embed": {"model": "hashing-test:64", "shard_size": 97, "estimate_sample": 50, "save_model": False},
        "bm25": {"shard_size": 150},
        "read": {"batch_size": 64},
    })
