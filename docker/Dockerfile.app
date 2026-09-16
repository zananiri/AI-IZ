# App service: FastAPI backend + Gradio UI (mounted at /ui) + the
# ingestion/OCR(CPU)/cleaning/translation/slides pipeline.
#
# OCR runs on CPU here by default (PaddleOCR PP-OCRv6, Tesseract). The
# GPU-based OCR fallback (PaddleOCR-VL / Surya) shares the same GPU as vLLM
# opportunistically -- see ocr/gpu_arbiter.py -- so this container must have
# GPU access too when that fallback is enabled (see docker-compose.yml).
#
# spaCy's small pipelines are baked in at build time (they're small, pure
# pip-installed model packages, not runtime weight downloads) -- this is the
# one online step for this image; everything else (fastText, Stanza,
# PaddleOCR, Surya, Qwen weights) is pulled at runtime from the bind-mounted
# caches populated by scripts/download_models.sh on the host. See README.md.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng tesseract-ocr-fra tesseract-ocr-spa tesseract-ocr-ita \
    tesseract-ocr-deu tesseract-ocr-ara tesseract-ocr-heb \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY config /app/config
COPY assets /app/assets

RUN pip install --no-cache-dir -e ".[ocr,lang,dev]"

RUN python -m spacy download en_core_web_sm \
    && python -m spacy download fr_core_news_sm \
    && python -m spacy download es_core_news_sm \
    && python -m spacy download it_core_news_sm \
    && python -m spacy download de_core_news_sm

ENV DOCSLIDES_CONFIG=/app/config/config.yaml
EXPOSE 8456

CMD ["python", "-m", "docslides.api.main"]
