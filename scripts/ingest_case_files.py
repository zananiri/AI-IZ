#!/usr/bin/env python3
"""Chunk and index the attorney's case files in ./legal_data (config legal.case_files.folder), then
chat with them.

    python scripts/ingest_case_files.py                       # index new/changed files (the default)
    python scripts/ingest_case_files.py index --dry-run       # classify + chunk only, write nothing
    python scripts/ingest_case_files.py index --prune         # also drop files deleted from the folder
    python scripts/ingest_case_files.py index --watch 60      # keep indexing, polling every 60 s
    python scripts/ingest_case_files.py show "legal_data/Cohen v. Levi/claim.pdf"   # print a file's chunks
    python scripts/ingest_case_files.py chat                  # chat with all the case files
    python scripts/ingest_case_files.py chat --matter "Cohen v. Levi"
    python scripts/ingest_case_files.py ask "When was the notice of termination sent?" --matter "Cohen v. Levi"
    python scripts/ingest_case_files.py stats                 # matters, files and chunks indexed

Put one subfolder per matter (client / case) in legal_data/; files directly in it belong to no
matter. Supported: PDF (scans are OCRed when an OCR engine is installed), DOCX, XLSX, EML, TXT/MD,
images. Convert .doc/.rtf/.msg first. Unchanged files are skipped on re-runs, so run it again
whenever files are added or edited.

The chunking strategy (structure-aware, per document type, with a context header on every chunk)
is described in src/docslides/legal/case_files.py. Answers come from the Legal tab's model
(legal.orchestrator) and cite the excerpts they use: file, page and paragraph.

Needs the `legal` extra (chromadb, sentence-transformers): pip install -e ".[legal]"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)  # config paths (./legal_data, ./data/...) are relative to the project root

from docslides.legal import case_files  # noqa: E402


def _print_report(reports: list[case_files.FileReport], verbose: bool) -> int:
    failures = 0
    totals: dict[str, int] = {}
    for r in reports:
        totals[r.action] = totals.get(r.action, 0) + 1
        if r.action == "unchanged" and not verbose:
            continue
        line = f"[{r.action}] {r.rel_path}"
        if r.doc_type:
            line += f"  ({r.doc_type}, {r.chunks} chunks)"
        if r.message:
            line += f" -- {r.message}"
        print(line, flush=True)
        for note in r.notes:
            print(f"    note: {note}")
        failures += r.action == "failed"
    if reports:
        print("Summary: " + ", ".join(f"{n} {action}" for action, n in sorted(totals.items())))
    return failures


def cmd_index(args) -> int:
    def progress(report: case_files.FileReport) -> None:
        if report.action != "unchanged":
            print(f"  ... {report.rel_path}: {report.action}", flush=True)

    while True:
        try:
            reports = case_files.index_folder(dry_run=args.dry_run, prune=args.prune, force=args.force,
                                              on_file=None if args.dry_run else progress)
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 2
        failures = _print_report(reports, args.verbose)
        if not args.watch:
            return 1 if failures else 0
        time.sleep(args.watch)


def cmd_show(args) -> int:
    path = Path(args.file)
    if not path.is_absolute():
        path = (Path.cwd() / path) if path.exists() else ROOT / path
    for chunk in case_files.preview(path):
        print(f"===== {chunk.chunk_id}  ({case_files.count_tokens(chunk.text)} tokens) =====")
        print(chunk.text)
        print()
    return 0


def cmd_stats(_args) -> int:
    collection, state = case_files.open_store()
    try:
        records = sorted(state.record_ids(case_files.CATEGORY))
    finally:
        state.close()
    print(f"{collection.count()} chunks from {len(records)} files")
    by_matter: dict[str, int] = {}
    for rel in records:
        by_matter[case_files.matter_of(rel)] = by_matter.get(case_files.matter_of(rel), 0) + 1
    for matter, count in sorted(by_matter.items()):
        print(f"  {matter}: {count} files")
    return 0


def _print_answer(reply: str, excerpts: list[case_files.Excerpt], show_excerpts: bool) -> None:
    print(f"\n{reply}\n")
    used = case_files.cited(reply, excerpts)
    if used:
        print("Sources:")
        for number, excerpt in used:
            print(f"  [{number}] {excerpt.citation()}")
    elif excerpts:
        print("(The answer cites none of the excerpts -- treat it with care.)")
    if show_excerpts:
        print("\nExcerpts given to the model:")
        for number, excerpt in enumerate(excerpts, 1):
            print(f"\n--- [{number}] {excerpt.citation()} ({excerpt.via}) ---\n{excerpt.text}")
    print()


def _check_matter(matter: str | None) -> bool:
    if matter and matter not in case_files.matters():
        known = ", ".join(case_files.matters()) or "none indexed yet"
        print(f"Unknown matter {matter!r}. Indexed matters: {known}", file=sys.stderr)
        return False
    return True


def cmd_ask(args) -> int:
    if not _check_matter(args.matter):
        return 2
    reply, excerpts = asyncio.run(case_files.answer(args.question, matter=args.matter, doc_type=args.type))
    _print_answer(reply, excerpts, args.excerpts)
    return 0


def cmd_chat(args) -> int:
    if not _check_matter(args.matter):
        return 2
    matter = args.matter
    history: list[case_files.Turn] = []
    print("Chat with your case files. Commands: /matter <name> | /matter (all) | /matters | /reset | /quit")
    print(f"Scope: {matter or 'all matters'}")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        if question in ("/quit", "/exit", "/q"):
            return 0
        if question == "/reset":
            history.clear()
            print("Conversation cleared.")
            continue
        if question == "/matters":
            print("\n".join(case_files.matters()) or "No matters indexed yet.")
            continue
        if question.startswith("/matter"):
            wanted = question[len("/matter"):].strip() or None
            if _check_matter(wanted):
                matter = wanted
                history.clear()
                print(f"Scope: {matter or 'all matters'} (conversation cleared)")
            continue
        try:
            reply, excerpts = asyncio.run(case_files.answer(question, history, matter, args.type))
        except Exception as exc:  # noqa: BLE001 -- keep the chat alive (model server down, ...)
            print(f"Error: {exc}", file=sys.stderr)
            continue
        _print_answer(reply, excerpts, args.excerpts)
        history.append(case_files.Turn(question, reply))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    index = sub.add_parser("index", help="chunk and index new/changed case files (default)")
    index.add_argument("--dry-run", action="store_true", help="classify and chunk only; write nothing")
    index.add_argument("--prune", action="store_true", help="remove files deleted from the folder from the index")
    index.add_argument("--force", action="store_true", help="re-index every file, changed or not")
    index.add_argument("--watch", type=int, metavar="SECONDS", help="keep running, re-checking every SECONDS")
    index.add_argument("--verbose", action="store_true", help="also list unchanged files")

    show = sub.add_parser("show", help="print one file's chunks (nothing is indexed)")
    show.add_argument("file")

    sub.add_parser("stats", help="what is indexed")

    for name, help_text in (("ask", "answer one question"), ("chat", "interactive chat")):
        p = sub.add_parser(name, help=help_text)
        if name == "ask":
            p.add_argument("question")
        p.add_argument("--matter", help="only this matter (its subfolder name in legal_data/)")
        p.add_argument("--type", choices=case_files.DOC_TYPES, help="only this document type")
        p.add_argument("--excerpts", action="store_true", help="also print the excerpts the model was given")

    args = parser.parse_args()
    if args.command is None:
        args = parser.parse_args(["index", *sys.argv[1:]])
    return {"index": cmd_index, "show": cmd_show, "stats": cmd_stats, "ask": cmd_ask, "chat": cmd_chat}[
        args.command](args)


if __name__ == "__main__":
    sys.exit(main())
