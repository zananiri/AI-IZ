"""The signed bundle: the manifest of every chunk approved into the Legal
tab's index, keyed by chunk_id -> content hash + who approved it and when.

The manifest is HMAC-SHA256-signed with the key in DOCSLIDES_LEGAL_BUNDLE_KEY.
Approving a batch (legal/staging.py) requires that key. At query time,
legal/retrieval.py drops any chunk whose id isn't in the manifest or whose
recomputed hash differs from the approved one, so text can't get into an
answer by being written straight into the vector store.

If the key isn't set in the process running the API, the signature can't be
checked. Hashes are still enforced, the audit log records
"hashes_only", and a warning is logged. A manifest whose signature fails
against a configured key raises BundleError.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from docslides.config import get_config
from docslides.logging_setup import get_logger

logger = get_logger(__name__)

BUNDLE_KEY_ENV = "DOCSLIDES_LEGAL_BUNDLE_KEY"

VerificationLevel = Literal["signed", "hashes_only"]


class BundleError(Exception):
    pass


def _manifest_path() -> Path:
    return Path(get_config().legal.ingestion.bundle_manifest)


def _key() -> bytes | None:
    key = os.environ.get(BUNDLE_KEY_ENV)
    return key.encode("utf-8") if key else None


def require_key() -> bytes:
    key = _key()
    if key is None:
        raise BundleError(
            f"Set {BUNDLE_KEY_ENV} to sign the bundle (any long random secret, e.g. "
            "`python -c \"import secrets; print(secrets.token_hex(32))\"`), and set the same "
            "value for the API process so it can verify the signature."
        )
    return key


def _sign(entries: dict, key: bytes) -> str:
    canonical = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hmac.new(key, canonical.encode("utf-8"), hashlib.sha256).hexdigest()


def load_entries(path: Path | None = None) -> dict[str, dict]:
    path = path or _manifest_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("entries", {})


def save_entries(entries: dict[str, dict], path: Path | None = None) -> None:
    key = require_key()
    path = path or _manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "signing": "hmac-sha256",
        "entries": entries,
        "signature": _sign(entries, key),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)
    _cache.clear()


def verify(path: Path | None = None) -> tuple[dict[str, dict], VerificationLevel]:
    """Returns (entries, verification level); raises BundleError if a key is
    configured and the signature doesn't match."""
    path = path or _manifest_path()
    if not path.exists():
        return {}, "signed"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = manifest.get("entries", {})
    key = _key()
    if key is None:
        logger.warning("legal_bundle_signature_unchecked", reason=f"{BUNDLE_KEY_ENV} not set")
        return entries, "hashes_only"
    signature = manifest.get("signature") or ""
    if not hmac.compare_digest(signature, _sign(entries, key)):
        raise BundleError(f"Signed bundle {path} failed signature verification -- refusing to serve from it")
    return entries, "signed"


_cache: dict[tuple[str, float], tuple[dict[str, dict], VerificationLevel]] = {}


def verified_entries() -> tuple[dict[str, dict], VerificationLevel]:
    """`verify()`, cached until the manifest file changes."""
    path = _manifest_path()
    cache_key = (str(path), path.stat().st_mtime if path.exists() else 0.0)
    if cache_key not in _cache:
        _cache.clear()
        _cache[cache_key] = verify(path)
    return _cache[cache_key]
