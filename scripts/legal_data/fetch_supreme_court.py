#!/usr/bin/env python3
"""Supreme Court judgments and non-technical decisions from the Hugging Face dataset
LevMuchnik/SupremeCourtOfIsrael (2022 snapshot, license OpenRAIL), with strict privacy exclusions.

    python scripts/legal_data/fetch_supreme_court.py --dry-run            # file size, whether it's present
    python scripts/legal_data/fetch_supreme_court.py --sample 200         # rows via the Hub's rows API -> legal_txt/_sample/
    python scripts/legal_data/fetch_supreme_court.py                      # the 1.5 GB parquet (sha256-verified, resumable)
    python scripts/legal_data/fetch_supreme_court.py --anonymize          # also replace private parties' names

Output: legal_txt/supreme_court/supreme_court-00001.jsonl ... (50,000 records per shard) and
filter_report.json: how many documents each keep rule and each privacy rule removed, plus the
distinct case types (meta_inyan_nm) and departments (meta_mador_nm) with their counts -- check them
against legal_data.privacy.family_case_prefixes / family_subject_values. Rules:
src/docslides/legal_data/supreme_court.py.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from docslides.legal_data import cli
from docslides.legal_data import supreme_court as sc
from docslides.legal_data.progress import Progress
from docslides.legal_data.records import JsonlWriter

SHARD_SIZE = 50_000


def body(ctx: cli.RunContext) -> None:
    sources = ctx.cfg.sources
    repo, revision, filename = sources.supreme_court_repo, sources.supreme_court_revision, sources.supreme_court_file
    privacy = sc.PrivacyFilter(ctx.cfg.privacy)  # refuses to run with empty privacy lists
    info = sc.hub_file_info(ctx.client, repo, revision, filename)
    snapshot = f"revision {info['commit'] or revision}, last modified {info['last_modified']}"
    ctx.manifest.source("Hugging Face dataset", f"https://huggingface.co/datasets/{repo}", sc.LICENSE,
                        f"{repo} (Lev Muchnik et al.)", file=filename, bytes=info["size"], sha256=info["sha256"],
                        commit=info["commit"])
    parquet = ctx.main_root / "supreme_court" / "raw" / filename
    if ctx.dry_run:
        present = parquet.exists()
        ctx.manifest.set("estimates", "parquet", {"file": filename, "mb": round((info["size"] or 0) / 1e6, 1),
                                                  "already_present": present})
        ctx.log(f"{filename}: {(info['size'] or 0) / 1e6:.0f} MB{' (already downloaded)' if present else ''}")
        return

    if ctx.sample:
        rows = sc.sample_rows(ctx.client, repo, ctx.sample)
        ctx.log(f"sample: {len(rows)} rows from the datasets-server rows API")
        total_rows = len(rows)
    else:
        url = sc.resolve_url(repo, revision, filename)
        result = ctx.client.download(url, parquet, ctx.ledger, expected_sha256=info["sha256"],
                                     progress_label=f"download {filename}")
        ctx.manifest.file(url, result.path, result.bytes, result.sha256, result.status)
        ctx.log(f"{filename} {result.status}")
        rows = sc.iter_parquet(parquet)
        total_rows = sc.parquet_rows(parquet)

    anonymizer = sc.Anonymizer(ctx.cfg.privacy.public_body_patterns) if ctx.args.anonymize else None
    writer = JsonlWriter(ctx.root / "supreme_court", "supreme_court", shard_size=SHARD_SIZE)
    report = {"input_rows": 0, "by_type": Counter(), "keep_buckets": Counter(), "excluded_first_rule": Counter(),
              "excluded_any_rule": Counter(), "written": Counter(), "encoding": Counter(),
              "case_types": Counter(), "departments": Counter(), "anonymization_replacements": 0}
    progress = Progress("supreme_court rows", total=total_rows, unit="rows", log=ctx.log)
    try:
        for row in rows:
            report["input_rows"] += 1
            progress.update(1, written=writer.count, excluded_privacy=sum(report["excluded_first_rule"].values()))
            report["by_type"][str(row.get("Type"))] += 1
            report["case_types"][str(row.get("meta_inyan_nm"))] += 1
            report["departments"][str(row.get("meta_mador_nm"))] += 1
            level, bucket = sc.classify(row, ctx.cfg.judgment_types)
            report["keep_buckets"][bucket] += 1
            if level is None:
                continue
            hits = privacy.hits(row)
            if hits:
                report["excluded_first_rule"][hits[0]] += 1
                for rule in hits:
                    report["excluded_any_rule"][rule] += 1
                continue
            record, replaced = sc.to_record(row, level, repo, snapshot, anonymizer)
            report["anonymization_replacements"] += replaced
            report["written"][level] += 1
            report["encoding"][record.quality.encoding] += 1
            writer.write(record)
    except BaseException:
        writer.close(commit=False)
        raise
    progress.done(written=writer.count)
    paths = writer.close()
    report = {k: dict(v.most_common()) if isinstance(v, Counter) else v for k, v in report.items()}
    report["removed_by_keep_rule"] = sum(n for b, n in report["keep_buckets"].items() if b not in ("judgment", "decision_non_technical"))
    report["removed_by_privacy"] = sum(report["excluded_first_rule"].values())
    report["records_written"] = writer.count
    (ctx.root / "supreme_court" / "filter_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    ctx.manifest.data["filters"] = {k: report[k] for k in ("keep_buckets", "excluded_first_rule", "excluded_any_rule",
                                                           "removed_by_keep_rule", "removed_by_privacy")}
    ctx.manifest.count("input_rows", report["input_rows"])
    ctx.manifest.count("records_written", writer.count)
    ctx.log(f"supreme_court: {report['input_rows']} rows -> {writer.count} records "
            f"({report['removed_by_keep_rule']} not kept by type, {report['removed_by_privacy']} removed for privacy); "
            f"{len(paths)} shard(s)")


def main() -> int:
    parser = cli.base_parser(__doc__)
    parser.add_argument("--anonymize", action="store_true", help="replace private parties' names with [צד N]")
    args = parser.parse_args()
    return cli.run("fetch_supreme_court", args, body)


if __name__ == "__main__":
    raise SystemExit(main())
