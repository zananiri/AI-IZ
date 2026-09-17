<#
.SYNOPSIS
    One-shot setup for Windows: creates the venv, installs every Python
    dependency (core + OCR + language + MinerU ingestion extras), sets up the
    LLM backend appropriate for THIS machine's hardware (vLLM on an NVIDIA
    GPU, Ollama everywhere else -- CPU or AMD), and downloads every model
    weight the app needs (the chat model, fastText, spaCy, Stanza, PaddleOCR,
    Surya, MinerU).

    This is the ONLY online step. Nothing under src/docslides/** calls out to
    the internet at request time -- see README.md.

.PARAMETER ModelsDir
    Where to put non-Hugging-Face model files (fastText, MinerU). Default: ./models

.PARAMETER SkipMineru
    Skip the ingestion-mineru extra (magic-pdf). It has a known dependency
    conflict with gradio's huggingface-hub pin (see the warning this script
    prints). PDF ingestion still works without it via the PyMuPDF fallback;
    DOCX/PPTX/XLSX/image ingestion will not.

.PARAMETER SkipHeavyOcr
    Skip paddleocr/paddlepaddle/surya-ocr (large, slow to build).

.PARAMETER ForceBackend
    "vllm" or "ollama" -- skip NVIDIA GPU auto-detection and use this backend.

.PARAMETER OllamaModel
    Model tag to pull when the Ollama backend is selected (general chat model
    + Legal tab orchestrator). Default: qwen3:32b

.PARAMETER OllamaDictalmModel
    Model tag to pull for the Legal tab's Hebrew analyst when the Ollama
    backend is selected. Default: dicta-il/DictaLM-3.0-24B-Thinking

.EXAMPLE
    .\scripts\setup.ps1
.EXAMPLE
    .\scripts\setup.ps1 -ModelsDir D:\models -SkipMineru
.EXAMPLE
    .\scripts\setup.ps1 -ForceBackend ollama -OllamaModel qwen3:8b
#>
param(
    [string]$ModelsDir = "./models",
    [switch]$SkipMineru,
    [switch]$SkipHeavyOcr,
    [string]$QwenModelRepo = "Qwen/Qwen3-32B-AWQ",
    [ValidateSet("", "vllm", "ollama")]
    [string]$ForceBackend = "",
    [string]$OllamaModel = "qwen3:32b",
    [string]$OllamaDictalmModel = "dicta-il/DictaLM-3.0-24B-Thinking"
)

$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")
$RepoRoot = Get-Location
$Skipped = New-Object System.Collections.Generic.List[string]

New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

# ---------------------------------------------------------------------------
Write-Host "== [1/7] Locating Python (3.10-3.12) ==" -ForegroundColor Cyan
$PythonBin = $null
foreach ($cand in @("py -3.11", "py -3.10", "py -3.12", "python", "python3")) {
    $parts = $cand.Split(" ")
    $exe = $parts[0]
    $exeArgs = $parts[1..($parts.Length - 1)]
    $cmd = Get-Command $exe -ErrorAction SilentlyContinue
    if ($cmd) {
        try {
            $ver = & $exe @exeArgs -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null
            if ($ver -in @("3.10", "3.11", "3.12")) {
                $PythonBin = $cand
                break
            }
        } catch {}
    }
}
if (-not $PythonBin) {
    Write-Host "[fatal] No Python 3.10-3.12 found. Install from https://python.org and re-run." -ForegroundColor Red
    exit 1
}
Write-Host "using $PythonBin"
Write-Host ""

# ---------------------------------------------------------------------------
Write-Host "== [2/7] Creating virtual environment (.venv) ==" -ForegroundColor Cyan
if (-not (Test-Path ".venv")) {
    $parts = $PythonBin.Split(" ")
    & $parts[0] $parts[1..($parts.Length - 1)] -m venv .venv
}
$VenvPy = Join-Path $RepoRoot ".venv\Scripts\python.exe"
# Repair pip via ensurepip rather than "pip install --upgrade pip": upgrading
# pip in-place while pip.exe itself is running can hit a Windows file-lock
# (WinError 32) that corrupts the install. ensurepip is idempotent and safe.
& $VenvPy -m ensurepip --upgrade *>$null
Write-Host "venv ready: $(& $VenvPy --version)"
Write-Host ""

function Invoke-Pip {
    # Deliberately NOT an advanced function (no [Parameter()]/[CmdletBinding()]):
    # those make PowerShell bind common parameters like -ErrorAction/-ErrorVariable,
    # and pip's own "-e" (editable install) is then rejected as an ambiguous
    # abbreviation of those two -- silently no-op'ing this whole call instead of
    # raising. Using the plain $args automatic variable skips parameter binding
    # entirely, so "-e" reaches pip untouched.
    & $VenvPy -m pip @args
    return $LASTEXITCODE -eq 0
}

# ---------------------------------------------------------------------------
Write-Host "== [3/7] Installing Python dependencies ==" -ForegroundColor Cyan
Write-Host "-- core + dev extras --"
$extras = "dev"
if (-not $SkipHeavyOcr) { $extras = "ocr,lang,dev" } else { $extras = "lang,dev" }
if (-not (Invoke-Pip install -e ".[$extras]")) {
    Write-Host "[fatal] core dependency install failed" -ForegroundColor Red
    exit 1
}
Invoke-Pip install -U "huggingface_hub[cli]" -q | Out-Null

if (-not $SkipMineru) {
    Write-Host "-- ingestion-mineru extra (magic-pdf) --"
    Write-Host "NOTE: magic-pdf's own deps pin huggingface-hub<1.0, which conflicts" -ForegroundColor Yellow
    Write-Host "      with gradio's huggingface-hub>=1.16 requirement. pip will" -ForegroundColor Yellow
    Write-Host "      install both packages but print a dependency-conflict" -ForegroundColor Yellow
    Write-Host "      warning -- this is expected (see" -ForegroundColor Yellow
    Write-Host "      src/docslides/ingestion/parser.py adapter notes). Pass" -ForegroundColor Yellow
    Write-Host "      -SkipMineru to skip it (PDF still works via the PyMuPDF" -ForegroundColor Yellow
    Write-Host "      fallback; DOCX/PPTX/XLSX/image ingestion needs it)." -ForegroundColor Yellow
    if (-not (Invoke-Pip install -e ".[ingestion-mineru]")) {
        $Skipped.Add("magic-pdf (MinerU) package")
    } else {
        # Clean up any stray *.dist-info left behind by a Windows file-lock
        # during the huggingface-hub downgrade pip just performed.
        Get-ChildItem ".venv\Lib\site-packages" -Filter "~*" -Directory -ErrorAction SilentlyContinue |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    }
} else {
    Write-Host "-- skipping ingestion-mineru extra (-SkipMineru) --"
    $Skipped.Add("magic-pdf (MinerU) package [skipped by request]")
}
Write-Host ""

# ---------------------------------------------------------------------------
Write-Host "== [4/7] Detecting LLM backend for this machine ==" -ForegroundColor Cyan
$Backend = $ForceBackend
if (-not $Backend) {
    $HasNvidia = $false
    try {
        $gpus = Get-CimInstance Win32_VideoController -ErrorAction Stop | Select-Object -ExpandProperty Name
        $HasNvidia = ($gpus -join " ") -match "NVIDIA"
    } catch {}
    $Backend = if ($HasNvidia) { "vllm" } else { "ollama" }
}
Write-Host "selected backend: $Backend"
Write-Host ""

# ---------------------------------------------------------------------------
if ($Backend -eq "vllm") {
    Write-Host "== [5/7] vLLM path: pulling serving image + downloading the model ==" -ForegroundColor Cyan
    if (Get-Command docker -ErrorAction SilentlyContinue) {
        docker pull vllm/vllm-openai:latest
        if ($LASTEXITCODE -ne 0) { $Skipped.Add("docker pull vllm/vllm-openai:latest") }
    } else {
        Write-Host "[skip] docker not found on PATH -- install Docker Desktop, then run:"
        Write-Host "       docker pull vllm/vllm-openai:latest"
        $Skipped.Add("docker pull vllm/vllm-openai:latest")
    }
    Write-Host "[note] vLLM needs an NVIDIA GPU with CUDA support (via Docker Desktop's" -ForegroundColor Yellow
    Write-Host "       WSL2 GPU passthrough on Windows). Confirm 'docker run --gpus all" -ForegroundColor Yellow
    Write-Host "       nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi' works before" -ForegroundColor Yellow
    Write-Host "       relying on 'docker compose up'." -ForegroundColor Yellow
    Write-Host ""

    & $VenvPy scripts/verify_vllm_launch.py
    Write-Host ""

    $HfCmd = Get-Command hf -ErrorAction SilentlyContinue
    if (-not $HfCmd) { $HfCmd = Get-Command huggingface-cli -ErrorAction SilentlyContinue }
    if ($HfCmd) {
        & $HfCmd.Source download $QwenModelRepo
        if ($LASTEXITCODE -ne 0) { $Skipped.Add("Qwen weights: $QwenModelRepo") }
    } else {
        Write-Host "[skip] Neither 'hf' nor 'huggingface-cli' found on PATH." -ForegroundColor Yellow
        $Skipped.Add("Qwen weights: $QwenModelRepo")
    }
    # config/config.yaml already defaults to backend: vllm -- no override file needed.
    Remove-Item ".env.local" -ErrorAction SilentlyContinue
} else {
    Write-Host "== [5/7] Ollama path: installing Ollama + pulling the model ==" -ForegroundColor Cyan
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        if (Get-Command winget -ErrorAction SilentlyContinue) {
            winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements
            if ($LASTEXITCODE -ne 0) { $Skipped.Add("ollama (winget)") }
            # winget installs land ollama.exe under LocalAppData; refresh PATH for this session.
            $env:Path = "$env:LOCALAPPDATA\Programs\Ollama;$env:Path"
        } else {
            Write-Host "[skip] winget not found. Install manually from https://ollama.com/download" -ForegroundColor Yellow
            $Skipped.Add("ollama (manual install)")
        }
    } else {
        Write-Host "ollama already installed: $(ollama --version 2>&1 | Select-Object -First 1)"
    }

    if (Get-Command ollama -ErrorAction SilentlyContinue) {
        $ollamaUp = $false
        try { Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 2 | Out-Null; $ollamaUp = $true } catch {}
        if (-not $ollamaUp) {
            Write-Host "Starting 'ollama serve' in the background..."
            Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
            for ($i = 0; $i -lt 15 -and -not $ollamaUp; $i++) {
                Start-Sleep -Seconds 1
                try { Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 2 | Out-Null; $ollamaUp = $true } catch {}
            }
        }
        Write-Host "Pulling $OllamaModel (this is a large download, comparable to the vLLM weights)..."
        ollama pull $OllamaModel
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[warn] 'ollama pull $OllamaModel' failed. Check the exact tag at" -ForegroundColor Yellow
            Write-Host "       https://ollama.com/library/qwen3 and retry: ollama pull <tag>" -ForegroundColor Yellow
            $Skipped.Add("ollama pull $OllamaModel")
        }

        Write-Host "Pulling $OllamaDictalmModel (Legal tab's Hebrew analyst, ~13-20GB)..."
        ollama pull $OllamaDictalmModel
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[warn] 'ollama pull $OllamaDictalmModel' failed. Check the exact tag at" -ForegroundColor Yellow
            Write-Host "       https://ollama.com/dicta-il/DictaLM-3.0-24B-Thinking and retry: ollama pull <tag>" -ForegroundColor Yellow
            $Skipped.Add("ollama pull $OllamaDictalmModel")
        }
    }

    # Read by the GUI launcher / any local (non-Docker) run of the app so
    # config/config.yaml's vLLM defaults (general llm: + Legal orchestrator/
    # hebrew_analyst) are overridden without editing it. docker-compose.
    # portable.yml sets the same vars itself, so it does not read this file.
    @"
DOCSLIDES_LLM_BACKEND=ollama
DOCSLIDES_LLM_BASE_URL=http://localhost:11434
DOCSLIDES_LLM_MODEL=$OllamaModel
DOCSLIDES_LEGAL_ORCHESTRATOR_BACKEND=ollama
DOCSLIDES_LEGAL_ORCHESTRATOR_BASE_URL=http://localhost:11434
DOCSLIDES_LEGAL_ORCHESTRATOR_MODEL=$OllamaModel
DOCSLIDES_LEGAL_HEBREW_BACKEND=ollama
DOCSLIDES_LEGAL_HEBREW_BASE_URL=http://localhost:11434
DOCSLIDES_LEGAL_HEBREW_MODEL=$OllamaDictalmModel
"@ | Set-Content -Path ".env.local" -Encoding utf8
    Write-Host "wrote $RepoRoot\.env.local (backend=ollama, model=$OllamaModel, legal hebrew_analyst=$OllamaDictalmModel)"
}
Write-Host ""

# ---------------------------------------------------------------------------
Write-Host "== [6/7] Downloading language/OCR model weights ==" -ForegroundColor Cyan

Write-Host "-- fastText language-id model (lid.176, ~125MB) --"
$LidPath = Join-Path $ModelsDir "lid.176.bin"
if (Test-Path $LidPath) {
    Write-Host "already present: $LidPath"
} else {
    try {
        Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin" `
            -OutFile "$LidPath.part" -UseBasicParsing
        Move-Item "$LidPath.part" $LidPath -Force
        Write-Host "done: $LidPath"
    } catch {
        Write-Host "[skip] fastText download failed: $_" -ForegroundColor Yellow
        Remove-Item "$LidPath.part" -ErrorAction SilentlyContinue
        $Skipped.Add("fastText lid.176")
    }
}
Write-Host ""

Write-Host "-- spaCy small pipelines (en, fr, es, it, de) --"
$spacyOk = & $VenvPy -c "import spacy" 2>$null; $spacyAvailable = $LASTEXITCODE -eq 0
if ($spacyAvailable) {
    foreach ($model in @("en_core_web_sm", "fr_core_news_sm", "es_core_news_sm", "it_core_news_sm", "de_core_news_sm")) {
        & $VenvPy -m spacy download $model
        if ($LASTEXITCODE -ne 0) { $Skipped.Add("spacy:$model") }
    }
} else {
    Write-Host "[skip] spacy not installed in this environment." -ForegroundColor Yellow
    $Skipped.Add("spacy (all models)")
}
Write-Host ""

Write-Host "-- Stanza pipelines (ar, he) --"
& $VenvPy -c "import stanza" 2>$null
if ($LASTEXITCODE -eq 0) {
    & $VenvPy -c "import stanza; stanza.download('ar'); stanza.download('he')"
    if ($LASTEXITCODE -ne 0) { $Skipped.Add("stanza:ar/he") }
} else {
    Write-Host "[skip] stanza not installed in this environment." -ForegroundColor Yellow
    $Skipped.Add("stanza (ar/he)")
}
Write-Host ""

if (-not $SkipHeavyOcr) {
    Write-Host "-- PaddleOCR PP-OCRv6 / PaddleOCR-VL model weights --"
    & $VenvPy -c "import paddleocr" 2>$null
    if ($LASTEXITCODE -eq 0) {
        # paddleocr 3.x removed use_angle_cls/show_log/use_gpu (now
        # use_textline_orientation/device, or just rely on defaults) --
        # kwargs it doesn't recognize raise ValueError rather than being
        # ignored, so keep this call minimal and version-tolerant.
        $paddleScript = @"
from paddleocr import PaddleOCR, PaddleOCRVL
for lang in ["en", "fr", "es", "it", "de", "german"]:
    try:
        PaddleOCR(lang=lang)
    except Exception as exc:
        print(f"[warn] PaddleOCR(lang={lang!r}) failed: {exc}")
try:
    PaddleOCRVL()
except Exception as exc:
    print(f"[warn] PaddleOCRVL() failed: {exc}")
"@
        $paddleScript | & $VenvPy -
        if ($LASTEXITCODE -ne 0) { $Skipped.Add("paddleocr/paddleocr-vl weights") }
    } else {
        Write-Host "[skip] paddleocr not installed in this environment." -ForegroundColor Yellow
        $Skipped.Add("paddleocr/paddleocr-vl weights")
    }
    Write-Host ""

    Write-Host "-- Surya OCR (Hebrew primary, Arabic GPU fallback) --"
    & $VenvPy -c "import surya" 2>$null
    if ($LASTEXITCODE -eq 0) {
        # surya-ocr's API was rewritten around its "Predictor" classes
        # (surya.detection.DetectionPredictor / surya.recognition.
        # RecognitionPredictor); the old surya.model.* module path is gone.
        # Predictor.__init__ loads (and thus downloads) weights eagerly.
        $suryaScript = @"
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
DetectionPredictor()
RecognitionPredictor()
"@
        $suryaScript | & $VenvPy -
        if ($LASTEXITCODE -ne 0) { $Skipped.Add("surya weights") }
    } else {
        Write-Host "[skip] surya (surya-ocr) not installed in this environment." -ForegroundColor Yellow
        $Skipped.Add("surya weights")
    }
    Write-Host ""
} else {
    Write-Host "-- skipping PaddleOCR/Surya weights (-SkipHeavyOcr) --"
    $Skipped.Add("paddleocr/surya weights [skipped by request]")
    Write-Host ""
}

Write-Host "-- Tesseract language packs --"
if (Get-Command tesseract -ErrorAction SilentlyContinue) {
    Write-Host "tesseract already installed: $(tesseract --version 2>&1 | Select-Object -First 1)"
} elseif (Get-Command winget -ErrorAction SilentlyContinue) {
    Write-Host "Installing via winget (UB-Mannheim build) -- select additional"
    Write-Host "language packs (fra/spa/ita/deu/ara/heb) in the installer if prompted."
    winget install --id UB-Mannheim.TesseractOCR -e --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) { $Skipped.Add("tesseract (winget)") }
} else {
    Write-Host "[skip] winget not found. Install manually from the UB-Mannheim build:" -ForegroundColor Yellow
    Write-Host "       https://github.com/UB-Mannheim/tesseract/wiki"
    Write-Host "       and select the fra/spa/ita/deu/ara/heb language packs."
    $Skipped.Add("tesseract (manual)")
}
Write-Host ""

Write-Host "-- LibreOffice (needed by MinerU for DOCX/PPTX/XLSX -> PDF conversion) --"
$sofficeFound = (Get-Command soffice -ErrorAction SilentlyContinue) -or (Test-Path "$env:PROGRAMFILES\LibreOffice\program\soffice.exe")
if ($sofficeFound) {
    Write-Host "soffice already installed."
} elseif ($SkipMineru) {
    Write-Host "[skip] MinerU extra was skipped (-SkipMineru) -- not needed."
} elseif (Get-Command winget -ErrorAction SilentlyContinue) {
    winget install --id TheDocumentFoundation.LibreOffice -e --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) { $Skipped.Add("libreoffice (winget)") }
} else {
    Write-Host "[skip] winget not found. Install manually from https://www.libreoffice.org/download/" -ForegroundColor Yellow
    $Skipped.Add("libreoffice (manual)")
}
Write-Host ""

# ---------------------------------------------------------------------------
Write-Host "== [7/7] MinerU (magic-pdf) layout/table/formula model weights ==" -ForegroundColor Cyan
& $VenvPy -c "import magic_pdf" 2>$null
if ($LASTEXITCODE -eq 0) {
    # PDF-Extract-Kit bundles many nested OCR/table/formula model files;
    # combined with a long repo path this can exceed Windows' 260-char
    # MAX_PATH and fail with "[Errno 2] No such file or directory" on the
    # deepest files. Best-effort fix it at the source (needs admin; silently
    # no-ops otherwise) and shorten our own path prefix as a second line of
    # defense either way.
    try {
        $lp = Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -ErrorAction Stop
        if ($lp.LongPathsEnabled -ne 1) {
            Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -ErrorAction Stop
            Write-Host "[info] Enabled Windows long-path support (HKLM LongPathsEnabled=1)." -ForegroundColor Yellow
        }
    } catch {
        Write-Host "[note] Could not verify/enable Windows long-path support (needs admin)." -ForegroundColor Yellow
        Write-Host "       If the download below fails with 'No such file or directory' on a" -ForegroundColor Yellow
        Write-Host "       long path, re-run this script as Administrator, or enable it manually:" -ForegroundColor Yellow
        Write-Host "       https://learn.microsoft.com/windows/win32/fileio/maximum-file-path-limitation" -ForegroundColor Yellow
    }

    $mineruDir = Join-Path (Resolve-Path $ModelsDir) "mineru"
    New-Item -ItemType Directory -Force -Path $mineruDir | Out-Null
    $mineruScript = @"
import json, os
from pathlib import Path
from huggingface_hub import snapshot_download

models_dir = Path(r"$mineruDir")
try:
    # Shortened folder names (not the upstream repo names) to leave more of
    # the 260-char Windows path budget for the repos' own deep subfolders.
    snapshot_download(repo_id="opendatalab/PDF-Extract-Kit-1.0", local_dir=str(models_dir / "pek"))
    snapshot_download(repo_id="hantian/layoutreader", local_dir=str(models_dir / "lr"))
except OSError as exc:
    if os.name == "nt" and getattr(exc, "errno", None) == 2:
        print(f"[warn] MinerU model download hit a Windows path-length limit: {exc}")
        print("       Re-run this script as Administrator so it can enable long-path")
        print("       support (HKLM...FileSystem LongPathsEnabled), or move this repo")
        print("       to a shorter path (e.g. C:\\docslides) and re-run.")
    else:
        print(f"[warn] MinerU model download failed: {exc}")
        print("       Repo IDs drift across magic-pdf releases -- check the current")
        print("       instructions at https://github.com/opendatalab/MinerU for the")
        print("       installed version and download manually.")
    raise SystemExit(1)
except Exception as exc:
    print(f"[warn] MinerU model download failed: {exc}")
    print("       Repo IDs drift across magic-pdf releases -- check the current")
    print("       instructions at https://github.com/opendatalab/MinerU for the")
    print("       installed version and download manually.")
    raise SystemExit(1)

config_path = Path.home() / "magic-pdf.json"
config = {}
if config_path.exists():
    try:
        config = json.loads(config_path.read_text())
    except Exception:
        config = {}
# "models-dir" must point at the "models" subfolder INSIDE the
# PDF-Extract-Kit-1.0 download (it expects Layout/, MFD/, MFR/, TabRec/
# directly inside it -- see resources/model_config/model_configs.yaml --
# and the repo itself nests those one level under its own "models/" dir,
# not at its root). layoutreader is a separate, optional key: magic_pdf
# falls back to downloading hantian/layoutreader from Hugging Face at
# runtime if this path doesn't exist, so it's a soft dependency.
config["models-dir"] = str(models_dir / "pek" / "models")
config["layoutreader-model-dir"] = str(models_dir / "lr")
config.setdefault("device-mode", "cpu")
# magic_pdf's default layout model (layoutlmv3) needs detectron2, which has
# no Windows wheel and is painful to build from source there. doclayout_yolo
# is bundled in the [full] extra and needs no detectron2 -- use it instead.
config.setdefault("layout-config", {"model": "doclayout_yolo"})
config_path.write_text(json.dumps(config, indent=2))
print(f"wrote {config_path}")
"@
    $mineruScript | & $VenvPy -
    if ($LASTEXITCODE -ne 0) { $Skipped.Add("MinerU model weights (see warning above)") }

    Write-Host "-- Patching known magic-pdf/fasttext upstream bugs (see scripts/patch_mineru.py) --"
    & $VenvPy scripts/patch_mineru.py
} else {
    Write-Host "[skip] magic_pdf not installed -- skipped MinerU extra above." -ForegroundColor Yellow
    $Skipped.Add("MinerU model weights [magic_pdf not installed]")
}
Write-Host ""

# ---------------------------------------------------------------------------
if ($Skipped.Count -eq 0) {
    Write-Host "Done. Everything installed and downloaded successfully." -ForegroundColor Green
} else {
    Write-Host "Finished with $($Skipped.Count) step(s) skipped/failed:" -ForegroundColor Yellow
    foreach ($item in $Skipped) { Write-Host "  - $item" }
    Write-Host "Re-run this script after addressing them -- it's safe to re-run."
}
Write-Host ""
Write-Host "Next steps ($Backend backend selected):"
if ($Backend -eq "vllm") {
    Write-Host "  1. GPU host: docker compose up -d          # starts vllm + app"
    Write-Host "  2. Or dev-run the app only:  .venv\Scripts\python.exe -m uvicorn docslides.api.main:run --factory"
} else {
    Write-Host "  1. Docker:   docker compose -f docker-compose.portable.yml up -d"
    Write-Host "  2. Or locally: Get-Content .env.local | ForEach-Object { if (`$_ -match '^([^=]+)=(.*)$') { Set-Item -Path Env:`$(`$Matches[1]) -Value `$Matches[2] } }; .venv\Scripts\python.exe -m uvicorn docslides.api.main:app --host 0.0.0.0 --port 8456"
    Write-Host "  3. Or just use gui\DocSlides.bat -- it reads .env.local automatically."
}
