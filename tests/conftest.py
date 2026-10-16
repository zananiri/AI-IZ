import os
from pathlib import Path

os.environ.setdefault("CODEX_SENTINEL_CONFIG", str(Path(__file__).parent.parent / "config" / "config.yaml"))
