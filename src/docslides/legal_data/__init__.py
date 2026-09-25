"""Bulk Israeli legal corpus: acquisition from the permitted sources, normalization to JSONL
under legal_txt/, and vectorization into a Chroma store separate from the Legal tab's signed
index. Entry points are the scripts in scripts/legal_data/ (see its README.md).

Permitted sources only: the Knesset OData API (+ its fs.knesset.gov.il files), the Hebrew
Wikisource Open Law Book via the official Wikimedia dumps, the Hugging Face dataset
LevMuchnik/SupremeCourtOfIsrael, and ISCD metadata files. Every request goes through
legal_data/http.py's PoliteClient.
"""
