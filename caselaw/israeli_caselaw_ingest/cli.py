"""python -m israeli_caselaw_ingest download|filter|clean|chunk|embed|bm25|validate|peek|all"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from .config import load_config, paths_for

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Israeli Supreme Court case law (LevMuchnik/SupremeCourtOfIsrael) -> chunks, embeddings, indexes.")

ConfigOpt = Annotated[Optional[Path], typer.Option("--config", "-c", help="config.yaml to merge over the shipped one")]
RootOpt = Annotated[Optional[str], typer.Option("--root", help="output root (default: paths.root, or $CASELAW_ROOT)")]
SampleOpt = Annotated[Optional[int], typer.Option("--sample-n", help="keep only the first N filtered documents (0 = all)")]
ForceOpt = Annotated[bool, typer.Option("--force", help="redo the stage even if it is done")]


def _cfg(config, root, sample_n) -> dict:
    return load_config(config, root=root, sample_n=sample_n)


@app.command()
def download(config: ConfigOpt = None, root: RootOpt = None, force: ForceOpt = False):
    """Download cases_all.parquet from the Hugging Face Hub (skipped when present and complete)."""
    from .download import download as run

    run(_cfg(config, root, None), force=force)


@app.command("filter")
def filter_(config: ConfigOpt = None, root: RootOpt = None, sample_n: SampleOpt = None, force: ForceOpt = False):
    """Keep judgments decided before max_decision_date; write reports/filter_report.md."""
    from .filter import run_filter

    run_filter(_cfg(config, root, sample_n), force=force)


@app.command()
def clean(config: ConfigOpt = None, root: RootOpt = None, sample_n: SampleOpt = None, force: ForceOpt = False):
    """Repair encoding, normalise, strip boilerplate, parse headers -> documents.parquet."""
    from .clean import run_clean

    run_clean(_cfg(config, root, sample_n), force=force)


@app.command()
def chunk(config: ConfigOpt = None, root: RootOpt = None, sample_n: SampleOpt = None, force: ForceOpt = False):
    """Structure-aware chunks -> chunks/year=YYYY/*.parquet."""
    from .chunk import run_chunk

    run_chunk(_cfg(config, root, sample_n), force=force)


@app.command()
def embed(config: ConfigOpt = None, root: RootOpt = None, sample_n: SampleOpt = None, force: ForceOpt = False,
          yes: Annotated[bool, typer.Option("--yes", help="go ahead even if the estimate exceeds the limit")] = False,
          store: Annotated[Optional[str], typer.Option("--store", help="lancedb | faiss")] = None,
          shard_stride: Annotated[int, typer.Option("--shard-stride", help="GPU workers: how many run at once")] = 1,
          shard_offset: Annotated[int, typer.Option("--shard-offset", help="GPU workers: this worker's index")] = 0,
          device: Annotated[Optional[str], typer.Option("--device", help="e.g. cuda:1 (default: embed.device)")] = None,
          no_store: Annotated[bool, typer.Option("--no-store", help="only encode shards (GPU workers)")] = False):
    """Embed chunks (resumable shards) and build the vector store; also builds BM25.

    Two GPUs: run `embed --shard-stride 2 --shard-offset K --device cuda:K --no-store` for K = 0, 1
    at the same time, then a plain `embed` to build the store."""
    from .embed import run_embed
    from .search import run_bm25

    cfg = _cfg(config, root, sample_n)
    run_embed(cfg, yes=yes, force=force, store=store, shard_stride=shard_stride, shard_offset=shard_offset,
              device=device, build_store=not no_store)
    if cfg["bm25"].get("enabled", True) and not no_store:
        run_bm25(cfg, force=force)


@app.command()
def bm25(config: ConfigOpt = None, root: RootOpt = None, sample_n: SampleOpt = None, force: ForceOpt = False):
    """Build the BM25 index (no GPU needed)."""
    from .search import run_bm25

    run_bm25(_cfg(config, root, sample_n), force=force)


@app.command()
def validate(config: ConfigOpt = None, root: RootOpt = None, sample_n: SampleOpt = None):
    """Write reports/validation_report.md and run the sanity queries."""
    from .validate import run_validate

    cfg = _cfg(config, root, sample_n)
    run_validate(cfg)
    typer.echo((paths_for(cfg).reports / "validation_report.md").read_text(encoding="utf-8"))


@app.command()
def peek(config: ConfigOpt = None, root: RootOpt = None, sample_n: SampleOpt = None,
         n: Annotated[int, typer.Option("--n")] = 5, seed: Annotated[int, typer.Option("--seed")] = 0):
    """Print N random chunks."""
    from .validate import peek as run

    typer.echo(run(_cfg(config, root, sample_n), n=n, seed=seed))


@app.command("all")
def all_(config: ConfigOpt = None, root: RootOpt = None, sample_n: SampleOpt = None, force: ForceOpt = False,
         embed_: Annotated[bool, typer.Option("--embed", help="also embed and build the vector store")] = False,
         yes: Annotated[bool, typer.Option("--yes")] = False,
         skip_download: Annotated[bool, typer.Option("--skip-download")] = False):
    """download -> filter -> clean -> chunk -> bm25 -> [embed] -> validate. Each stage resumes."""
    from .chunk import run_chunk
    from .clean import run_clean
    from .download import download as run_download
    from .embed import run_embed
    from .filter import run_filter
    from .search import run_bm25
    from .validate import run_validate

    cfg = _cfg(config, root, sample_n)
    if not skip_download:
        run_download(cfg)
    run_filter(cfg, force=force)
    run_clean(cfg, force=force)
    run_chunk(cfg, force=force)
    if cfg["bm25"].get("enabled", True):
        run_bm25(cfg, force=force)
    if embed_:
        run_embed(cfg, yes=yes, force=force)
    run_validate(cfg)
    typer.echo((paths_for(cfg).reports / "validation_report.md").read_text(encoding="utf-8"))


def main() -> None:
    app()
