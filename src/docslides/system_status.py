"""Health of everything the app leans on: the backend API, each LLM server and model (downloaded?
loaded? answering?), LibreOffice, MinerU, PyMuPDF, the OCR engines, the GPU, the retrieval
models, the language models and the legal corpus.

Nothing here runs at import or at app start -- the System status tab (ui/gradio_app.py) calls
run_checks() when it's opened. Each check is cheap (a version probe, a file lookup, one HTTP
request with a short timeout) and they run in parallel. Heavy libraries are never imported:
installed packages are found with importlib metadata, so the check itself can't crash the app.
The one costly step, a one-token generation, is sent only to models already in memory unless
`test_generation` asks for every model (which loads the idle ones).
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx

from docslides.config import LLMConfig, get_config

OK, WARN, FAIL, OFF = "ok", "warn", "fail", "off"
_ICONS = {OK: "🟢 OK", WARN: "🟡 Warning", FAIL: "🔴 Down", OFF: "⚪ Not used"}
_GROUPS = ["App", "LLMs", "Documents & PDF", "OCR", "Hardware", "Retrieval & language", "Legal data"]

_HTTP_TIMEOUT_S = 4.0
_PING_TIMEOUT_S = 30.0  # a model already in memory answers one token well within this
_LOAD_PING_TIMEOUT_S = 300.0  # ...one that has to load first (31B from disk) may not


@dataclass
class Check:
    group: str
    name: str
    state: str
    detail: str


def _version(*dists: str) -> str | None:
    for dist in dists:
        try:
            return importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _run_all(cmd: list[str], timeout: float = 20.0) -> str:
    """A command's whole output (stdout, else stderr); raises on a non-zero exit."""
    out = subprocess.run(  # noqa: PLW1510 -- the exit code is checked by hand
        cmd, capture_output=True, text=True, timeout=timeout,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    text = (out.stdout or out.stderr).strip()
    if out.returncode != 0:
        raise RuntimeError(text.splitlines()[-1] if text else f"exit code {out.returncode}")
    return text


def _run(cmd: list[str], timeout: float = 20.0) -> str:
    """The first line of a command's output -- a version banner."""
    text = _run_all(cmd, timeout)
    return text.splitlines()[0] if text else ""


# --- App --------------------------------------------------------------------------------------


def _check_api() -> list[Check]:
    # The same default as ui/gradio_app.API_BASE_URL -- the server the chat tabs post to.
    api_url = os.environ.get("DOCSLIDES_API_URL", "http://localhost:8456")
    try:
        r = httpx.get(f"{api_url}/health", timeout=_HTTP_TIMEOUT_S)
        r.raise_for_status()
        return [Check("App", "Backend API", OK, f"{api_url} answers /health")]
    except Exception as exc:  # noqa: BLE001 -- any failure is a status to report
        return [Check("App", "Backend API", FAIL, f"{api_url}: {_short(exc)} -- chats can't run without it")]


def _check_template() -> list[Check]:
    path = Path(get_config().paths.template_potx)
    if path.exists():
        return [Check("App", "Slide template", OK, str(path))]
    return [Check("App", "Slide template", WARN, f"{path} missing -- decks use python-pptx's blank default")]


# --- LLMs -------------------------------------------------------------------------------------


@dataclass
class _Deployment:
    roles: list[str]
    cfg: LLMConfig


def _deployments() -> list[_Deployment]:
    """Every model the app talks to, one row per (backend, server, model): the general chat and the
    Legal tab share Gemma 4 on one server, so they share a row."""
    cfg = get_config()
    candidates: list[tuple[str, LLMConfig]] = [
        ("General chat", cfg.llm),
        ("Legal GPT", cfg.legal.orchestrator),
    ]
    if cfg.translation.translator is not None:
        candidates.append(("Translator", cfg.translation.translator))
    judge = os.environ.get("DOCSLIDES_LEGAL_JUDGE_MODEL") or cfg.legal.judge_model
    if judge:
        candidates.append(("Eval judge (evals only)", cfg.legal.orchestrator.model_copy(update={"model": judge})))

    merged: dict[tuple[str, str, str], _Deployment] = {}
    for role, llm_cfg in candidates:
        key = (llm_cfg.backend, llm_cfg.base_url.rstrip("/"), llm_cfg.model)
        if key in merged:
            merged[key].roles.append(role)
        else:
            merged[key] = _Deployment([role], llm_cfg)
    return list(merged.values())


def _ollama_has(names: list[str], model: str) -> str | None:
    """The installed tag matching `model` ("gemma4:31b"; a bare "llama3" means ":latest")."""
    wanted = model if ":" in model else f"{model}:latest"
    for name in names:
        if name in (model, wanted):
            return name
    return None


def _servers(deployments: list[_Deployment]) -> list[Check]:
    checks: list[Check] = []
    seen: set[tuple[str, str]] = set()
    for dep in deployments:
        backend, base = dep.cfg.backend, dep.cfg.base_url.rstrip("/")
        if (backend, base) in seen:
            continue
        seen.add((backend, base))
        name = f"{'Ollama' if backend == 'ollama' else 'vLLM'} server"
        try:
            if backend == "ollama":
                version = httpx.get(f"{base}/api/version", timeout=_HTTP_TIMEOUT_S).json().get("version", "?")
                checks.append(Check("LLMs", name, OK, f"{base} · Ollama {version}"))
            else:
                r = httpx.get(f"{base}/models", timeout=_HTTP_TIMEOUT_S,
                              headers={"Authorization": f"Bearer {dep.cfg.api_key}"})
                r.raise_for_status()
                checks.append(Check("LLMs", name, OK, f"{base} is serving"))
        except Exception as exc:  # noqa: BLE001 -- any failure is a status to report
            hint = " -- start it with `ollama serve`" if backend == "ollama" else ""
            checks.append(Check("LLMs", name, FAIL, f"{base}: {_short(exc)}{hint}"))
    return checks


def _ping(http: httpx.Client, dep: _Deployment, timeout: float) -> str:
    """One-token generation; returns the latency, raises if the model doesn't answer."""
    cfg = dep.cfg
    messages = [{"role": "user", "content": "Reply with OK."}]
    start = time.perf_counter()
    if cfg.backend == "ollama":
        # num_ctx as the app sends it, so a loaded model isn't reloaded at another context size.
        body = {"model": cfg.model, "messages": messages, "stream": False,
                "options": {"num_predict": 1, "num_ctx": cfg.max_model_len}}
        r = http.post("/api/chat", json=body, timeout=timeout)
    else:
        r = http.post("/chat/completions", json={"model": cfg.model, "messages": messages, "max_tokens": 1},
                      timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:160]}")
    return f"answered in {time.perf_counter() - start:.1f}s"


def _check_model(dep: _Deployment, test_generation: bool) -> Check:
    cfg, base = dep.cfg, dep.cfg.base_url.rstrip("/")
    name = f"{cfg.model} ({', '.join(dep.roles)})"
    optional = all("evals only" in role for role in dep.roles)
    missing_state = WARN if optional else FAIL
    state = "served"
    # One connection for all of a model's requests: on a host busy generating, each new one costs seconds.
    with httpx.Client(base_url=base, timeout=_HTTP_TIMEOUT_S,
                      headers={"Authorization": f"Bearer {cfg.api_key}"}) as http:
        try:
            if cfg.backend == "ollama":
                tags = http.get("/api/tags").json().get("models", [])
                tag = _ollama_has([m.get("name", "") for m in tags], cfg.model)
                if tag is None:
                    return Check("LLMs", name, missing_state, f"not downloaded -- `ollama pull {cfg.model}`")
                size_gb = next((m.get("size", 0) for m in tags if m.get("name") == tag), 0) / 1e9
                loaded = http.get("/api/ps").json().get("models", [])
                in_memory = _ollama_has([m.get("name", "") for m in loaded], cfg.model) is not None
                state = f"downloaded ({size_gb:.1f} GB), {'in memory' if in_memory else 'loading'}"
                if not in_memory and not test_generation:
                    return Check("LLMs", name, OK, f"downloaded ({size_gb:.1f} GB), idle -- loads on first use")
                timeout = _PING_TIMEOUT_S if in_memory else _LOAD_PING_TIMEOUT_S
            else:
                served = [m.get("id") for m in http.get("/models").json().get("data", [])]
                if cfg.model not in served:
                    return Check("LLMs", name, missing_state,
                                 f"not served here (serving: {', '.join(served) or 'nothing'})")
                timeout = _PING_TIMEOUT_S
            latency = _ping(http, dep, timeout)
            return Check("LLMs", name, OK, f"{state.replace('loading', 'loaded')}, {latency}")
        except httpx.TimeoutException:
            # Requests to one model queue behind each other: a long answer in progress (a Legal turn,
            # an eval) holds the ping back.
            return Check("LLMs", name, WARN, f"{state}, but no answer in time -- busy with another request, or stuck")
        except httpx.TransportError:
            return Check("LLMs", name, missing_state, f"server at {base} not reachable")
        except Exception as exc:  # noqa: BLE001 -- any failure is a status to report
            return Check("LLMs", name, missing_state, f"no answer: {_short(exc)}")


# --- Documents & PDF --------------------------------------------------------------------------


def _find_soffice() -> tuple[str | None, bool]:
    """(path, on PATH?) -- the same lookup MinerU's office_to_pdf.get_soffice_command does."""
    on_path = shutil.which("soffice")
    if on_path:
        return on_path, True
    roots = [os.environ.get("PROGRAMFILES", "C:/Program Files"), os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")]
    roots += [f"{drive}:/" for drive in "CDEFGH"]
    for root in roots:
        candidate = Path(root) / "LibreOffice/program/soffice.exe"
        if candidate.exists():
            return str(candidate), False
    for candidate in ("/Applications/LibreOffice.app/Contents/MacOS/soffice", "/usr/bin/libreoffice"):
        if Path(candidate).exists():
            return candidate, False
    return None, False


def _check_libreoffice() -> list[Check]:
    path, _ = _find_soffice()
    if path is None:
        return [Check("Documents & PDF", "LibreOffice", FAIL,
                      "soffice not found -- DOCX/PPTX/XLSX attachments can't be converted")]
    # soffice.exe has no console output on Windows; soffice.com (next to it) prints the version.
    probe = str(Path(path).with_suffix(".com")) if path.lower().endswith(".exe") and Path(path).with_suffix(".com").exists() else path
    try:
        version = _run([probe, "--version"], timeout=30)
        return [Check("Documents & PDF", "LibreOffice", OK, f"{version or 'responds'} · {path}")]
    except Exception as exc:  # noqa: BLE001 -- any failure is a status to report
        return [Check("Documents & PDF", "LibreOffice", WARN, f"found at {path} but `--version` failed: {_short(exc)}")]


def _check_mineru() -> list[Check]:
    version = _version("magic-pdf")
    if version is None or importlib.util.find_spec("magic_pdf") is None:
        return [Check("Documents & PDF", "MinerU (magic-pdf)", WARN,
                      "not installed -- PDFs fall back to PyMuPDF; DOCX/PPTX/XLSX/images can't be parsed")]
    cfg_path = Path(os.environ.get("MINERU_TOOLS_CONFIG_JSON", Path.home() / "magic-pdf.json"))
    if not cfg_path.exists():
        return [Check("Documents & PDF", "MinerU (magic-pdf)", FAIL, f"{version} installed, but {cfg_path} is missing")]
    try:
        mineru_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 -- any failure is a status to report
        return [Check("Documents & PDF", "MinerU (magic-pdf)", FAIL, f"{cfg_path} unreadable: {_short(exc)}")]
    models_dir = Path(mineru_cfg.get("models-dir", ""))
    if not models_dir.is_dir() or not any(models_dir.iterdir()):
        return [Check("Documents & PDF", "MinerU (magic-pdf)", FAIL, f"{version} installed, models missing at {models_dir}")]
    device = mineru_cfg.get("device-mode", "cpu")
    return [Check("Documents & PDF", "MinerU (magic-pdf)", OK, f"{version} · models in {models_dir} · device {device}")]


def _check_pymupdf() -> list[Check]:
    version = _version("pymupdf", "PyMuPDF")
    if version is None:
        return [Check("Documents & PDF", "PyMuPDF", FAIL, "not installed -- no PDF text extraction or page rendering")]
    return [Check("Documents & PDF", "PyMuPDF", OK, f"{version} (PDF text layer, scan detection, page rendering)")]


# --- OCR --------------------------------------------------------------------------------------


def _check_tesseract() -> list[Check]:
    if _version("pytesseract") is None:
        return [Check("OCR", "Tesseract", FAIL, "pytesseract not installed")]
    exe = shutil.which("tesseract")
    if exe is None:
        default = Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Tesseract-OCR/tesseract.exe"
        exe = str(default) if default.exists() else None
    if exe is None:
        return [Check("OCR", "Tesseract", FAIL, "tesseract binary not found on PATH")]
    try:
        version = _run([exe, "--version"])
        langs = _run_all([exe, "--list-langs"])
        installed = {line.strip() for line in langs.splitlines()[1:] if line.strip()}
    except Exception as exc:  # noqa: BLE001 -- any failure is a status to report
        return [Check("OCR", "Tesseract", FAIL, f"{exe}: {_short(exc)}")]
    needed = set(get_config().ocr.tesseract.lang_codes.values())
    missing = sorted(needed - installed)
    if missing:
        return [Check("OCR", "Tesseract", WARN, f"{version} · missing language data: {', '.join(missing)}")]
    return [Check("OCR", "Tesseract", OK, f"{version} · languages {', '.join(sorted(needed))}")]


def _check_paddle() -> list[Check]:
    ocr_version = _version("paddleocr")
    paddle_version = _version("paddlepaddle", "paddlepaddle-gpu")
    if ocr_version is None or paddle_version is None:
        missing = [n for n, v in (("paddleocr", ocr_version), ("paddlepaddle", paddle_version)) if v is None]
        return [Check("OCR", "PaddleOCR / PaddleOCR-VL", FAIL, f"not installed: {', '.join(missing)}")]
    models_root = Path(os.environ.get("PADDLE_PDX_CACHE_HOME", Path.home() / ".paddlex")) / "official_models"
    models = sorted(p.name for p in models_root.iterdir() if p.is_dir()) if models_root.is_dir() else []
    has_vl = any("VL" in m for m in models)
    detail = f"paddleocr {ocr_version}, paddlepaddle {paddle_version}"
    if not models:
        return [Check("OCR", "PaddleOCR / PaddleOCR-VL", WARN, f"{detail} · no models downloaded yet (fetched on first OCR)")]
    vl_note = "" if has_vl else " · PaddleOCR-VL not downloaded yet (fetched on first Arabic OCR)"
    return [Check("OCR", "PaddleOCR / PaddleOCR-VL", OK, f"{detail} · {len(models)} models{vl_note}")]


def _check_surya() -> list[Check]:
    version = _version("surya-ocr")
    if version is None:
        return [Check("OCR", "Surya (Hebrew)", FAIL, "surya-ocr not installed -- Hebrew scans fall back to Tesseract")]
    return [Check("OCR", "Surya (Hebrew)", OK, f"{version} (models download on first use)")]


# --- Hardware ---------------------------------------------------------------------------------


def _check_gpu() -> list[Check]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return [Check("Hardware", "NVIDIA GPU", OFF, "nvidia-smi not found -- models and OCR run on the CPU")]
    try:
        out = subprocess.run(  # noqa: PLW1510 -- the exit code is checked by hand
            [exe, "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        rows = [line.split(", ") for line in out.stdout.strip().splitlines() if line.strip()]
    except Exception as exc:  # noqa: BLE001 -- any failure is a status to report
        return [Check("Hardware", "NVIDIA GPU", WARN, f"nvidia-smi failed: {_short(exc)}")]
    return [
        Check("Hardware", f"GPU {i}", OK, f"{name} · {int(used) / 1024:.1f} / {int(total) / 1024:.1f} GB used · {util}% busy")
        for i, (name, used, total, util) in enumerate(rows)
    ] or [Check("Hardware", "NVIDIA GPU", WARN, "nvidia-smi lists no GPU")]


# --- Retrieval & language ---------------------------------------------------------------------


def _hf_cached(repo: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(repo, "config.json"), str)


def _check_hf_models() -> list[Check]:
    retrieval = get_config().legal.retrieval
    checks = []
    for label, repo in (("Embedding model", retrieval.embedding_model), ("Reranker", retrieval.reranker_model)):
        if not repo:
            checks.append(Check("Retrieval & language", label, OFF, "disabled in config"))
        elif _hf_cached(repo):
            checks.append(Check("Retrieval & language", label, OK, f"{repo} downloaded"))
        else:
            checks.append(Check("Retrieval & language", label, WARN, f"{repo} not in the Hugging Face cache -- downloads on first Legal question"))
    return checks


def _check_language_models() -> list[Check]:
    langs = get_config().languages
    checks = []
    fasttext = Path(langs.fasttext_model_path)
    if fasttext.exists():
        checks.append(Check("Retrieval & language", "fastText language ID", OK, f"{fasttext} ({fasttext.stat().st_size / 1e6:.0f} MB)"))
    else:
        checks.append(Check("Retrieval & language", "fastText language ID", WARN, f"{fasttext} missing -- language detection falls back"))

    spacy_missing = [m for m in langs.spacy_models.values() if importlib.util.find_spec(m) is None]
    checks.append(Check(
        "Retrieval & language", "spaCy pipelines",
        WARN if spacy_missing else OK,
        f"missing: {', '.join(spacy_missing)}" if spacy_missing else ", ".join(langs.spacy_models.values()),
    ))

    stanza_root = Path(os.environ.get("STANZA_RESOURCES_DIR", Path.home() / "stanza_resources"))
    stanza_missing = [lang for lang in langs.stanza_models.values() if not (stanza_root / lang).is_dir()]
    checks.append(Check(
        "Retrieval & language", "Stanza pipelines",
        WARN if stanza_missing else OK,
        f"missing in {stanza_root}: {', '.join(stanza_missing)}" if stanza_missing else ", ".join(langs.stanza_models.values()),
    ))
    return checks


# --- Legal data -------------------------------------------------------------------------------


def _check_legal_data() -> list[Check]:
    from docslides.legal.caselaw import caselaw_stats
    from docslides.legal.corpus_retrieval import corpus_stats

    legal = get_config().legal
    checks = []
    corpus = corpus_stats()
    in_use = legal.retrieval.source == "corpus"
    if corpus:
        checks.append(Check("Legal data", "Legal corpus", OK,
                            f"{corpus['chunks']:,} chunks from {corpus['records']:,} documents · built {(corpus.get('built_at') or '?')[:10]}"))
    else:
        checks.append(Check("Legal data", "Legal corpus", FAIL if in_use else OFF,
                            f"not installed at {legal.corpus.vectordb_dir} -- see scripts/legal_data/install_corpus.py"))

    if legal.corpus.caselaw_dir:
        caselaw = caselaw_stats()
        if caselaw:
            checks.append(Check("Legal data", "Case law index", OK,
                                f"{caselaw['judgments']:,} judgments · built {(caselaw.get('built_at') or '?')[:10]}"))
        else:
            checks.append(Check("Legal data", "Case law index", FAIL, f"not built at {legal.corpus.caselaw_dir}"))
    else:
        checks.append(Check("Legal data", "Case law index", OFF, "legal.corpus.caselaw_dir is null"))

    if legal.retrieval.source == "signed_index":
        exists = Path(legal.retrieval.vectordb_dir).is_dir()
        checks.append(Check("Legal data", "Signed index", OK if exists else FAIL, legal.retrieval.vectordb_dir))
    return checks


# --- Running and rendering --------------------------------------------------------------------


def _short(exc: Exception) -> str:
    text = str(exc).strip() or type(exc).__name__
    return text if len(text) <= 160 else text[:157] + "..."


def run_checks(test_generation: bool = False) -> list[Check]:
    """Every check, in parallel. `test_generation` sends a one-token prompt to every model,
    loading the idle ones (slow, and may evict another model from memory); without it only
    models already in memory (and vLLM's, always resident) are asked."""
    deployments = _deployments()
    tasks: list[Callable[[], list[Check]]] = [
        _check_api, _check_template, lambda: _servers(deployments),
        _check_libreoffice, _check_mineru, _check_pymupdf,
        _check_tesseract, _check_paddle, _check_surya, _check_gpu,
        _check_hf_models, _check_language_models, _check_legal_data,
    ]
    tasks += [lambda dep=dep: [_check_model(dep, test_generation)] for dep in deployments]

    def safe(task: Callable[[], list[Check]]) -> list[Check]:
        try:
            return task()
        except Exception as exc:  # noqa: BLE001 -- a broken check reports itself, never the whole page
            name = getattr(task, "__name__", "check").removeprefix("_check_")
            return [Check("App", name, FAIL, f"check crashed: {_short(exc)}")]

    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        results = [check for batch in pool.map(safe, tasks) for check in batch]
    return sorted(results, key=lambda c: _GROUPS.index(c.group) if c.group in _GROUPS else len(_GROUPS))


def render_markdown(checks: list[Check], elapsed_s: float) -> str:
    counts = {state: sum(c.state == state for c in checks) for state in (OK, WARN, FAIL)}
    lines = [
        (
            f"_Checked {datetime.now().astimezone():%Y-%m-%d %H:%M:%S} in {elapsed_s:.1f}s -- "
            f"{counts[OK]} OK · {counts[WARN]} warnings · {counts[FAIL]} down_"
        ),
    ]
    for group in dict.fromkeys(c.group for c in checks):
        lines += ["", f"### {group}", "", "| Component | Status | Details |", "|---|---|---|"]
        for c in (c for c in checks if c.group == group):
            detail = c.detail.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {c.name} | {_ICONS[c.state]} | {detail} |")
    return "\n".join(lines)


def status_report(test_generation: bool = False) -> str:
    start = time.perf_counter()
    checks = run_checks(test_generation)
    return render_markdown(checks, time.perf_counter() - start)
