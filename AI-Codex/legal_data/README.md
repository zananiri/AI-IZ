# legal_data — the attorney's case files

Put the case files here, **one subfolder per matter** (client / case). Any depth below that is fine:

```
legal_data/
  Cohen v. Levi/
    pleadings/statement of claim.pdf
    pleadings/statement of defence.docx
    correspondence/2024-03-05 termination.eml
    hearing 2024-05-12 protocol.pdf
  Estate of Mizrahi/
    will.pdf
```

Then, from the project root:

```bash
python scripts/ingest_case_files.py                 # chunk + index new/changed files
python scripts/ingest_case_files.py chat            # chat with them (/matter <name> to scope)
```

Supported: PDF (scanned pages are OCRed when an OCR engine is installed), DOCX, XLSX, EML, TXT/MD,
images. Convert `.doc`, `.rtf` and `.msg` to DOCX/PDF/EML first; the indexer lists the ones it skipped.

Everything in this folder except this README is git-ignored: case files are privileged client
material and must not be committed or pushed.
