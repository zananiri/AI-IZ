import os
from pathlib import Path

os.environ.setdefault("DOCSLIDES_CONFIG", str(Path(__file__).parent.parent / "config" / "config.yaml"))
