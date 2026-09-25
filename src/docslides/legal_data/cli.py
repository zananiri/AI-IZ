"""What every fetcher script shares: the common flags, where output goes, the run manifest and the
stop-and-report rule.

    --dry-run    only metadata requests (robots.txt, sizes, counts); reports what a run would
                 download and its estimated size; writes nothing but the manifest
    --sample N   a small real run: N items per source, written under <output>/_sample/
    --output D   output root (default legal_data.output_dir, ./legal_txt)

Exit codes: 0 ok, 1 failed, 2 stopped (403 / access denied / robots.txt / host unavailable /
missing contact email), 130 interrupted. A stop is never retried or worked around.
"""

from __future__ import annotations

import argparse
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from docslides.config import LegalDataConfig, get_config
from docslides.legal_data.http import DownloadLedger, FetchStopped, PoliteClient
from docslides.legal_data.manifest import RunManifest


@dataclass
class RunContext:
    cfg: LegalDataConfig
    args: argparse.Namespace
    root: Path  # where this run writes (<output>/_sample in a sample run)
    main_root: Path  # the real output root (shared inputs, e.g. the downloaded dump, live here)
    manifest: RunManifest
    client: PoliteClient
    ledger: DownloadLedger

    @property
    def dry_run(self) -> bool:
        return bool(self.args.dry_run)

    @property
    def sample(self) -> int | None:
        return self.args.sample

    def log(self, message: str) -> None:
        self.manifest.log(message)


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="report what would be downloaded; write nothing")
    parser.add_argument("--sample", type=int, metavar="N", help="small real run, written under <output>/_sample/")
    parser.add_argument("--output", help="output root (default: legal_data.output_dir)")
    return parser


def run(script: str, args: argparse.Namespace, body: Callable[[RunContext], None]) -> int:
    cfg = get_config().legal_data
    main_root = Path(args.output or cfg.output_dir)
    root = main_root / "_sample" if args.sample else main_root
    manifest = RunManifest(script, root, vars(args), dry_run=args.dry_run, sample=args.sample)
    client: PoliteClient | None = None
    status, code = "ok", 0
    try:
        client = PoliteClient(cfg, log=manifest.log)
        ledger = DownloadLedger(main_root / "_manifests" / "_downloads.json", read_only=args.dry_run)
        manifest.log(f"{script}: output {root.resolve()}{' (dry run)' if args.dry_run else ''}")
        body(RunContext(cfg, args, root, main_root, manifest, client, ledger))
    except FetchStopped as exc:
        manifest.error(f"STOPPED: {exc}")
        status, code = "stopped", 2
    except KeyboardInterrupt:
        manifest.error("interrupted")
        status, code = "interrupted", 130
    except Exception as exc:  # noqa: BLE001 -- recorded in the manifest with its traceback
        manifest.error(f"{type(exc).__name__}: {exc}")
        manifest.log(traceback.format_exc())
        status, code = "failed", 1
    finally:
        manifest.finish(status, client)
        if client is not None:
            client.close()
    return code
