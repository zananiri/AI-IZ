"""Bulk Spanish legal corpus: acquisition from the permitted open-data sources, normalization to
JSONL under legal_txt/, and vectorization into a Chroma store separate from the Legal tab's signed
index. Entry points are the scripts in scripts/legal_data/ (see its README.md).

Permitted sources only: the BOE open-data API (consolidated state and autonomous-community
legislation), the Tribunal Constitucional's public case-law system (hj.tribunalconstitucional.es)
and the EU Publications Office's CELLAR / SPARQL services (EU law in Spanish). Every request goes
through legal_data/http.py's PoliteClient.
"""
