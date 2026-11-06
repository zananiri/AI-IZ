"""docslides desktop launcher -- Windows and macOS.

A single-file, dependency-free (stdlib-only: tkinter, subprocess, urllib)
GUI that starts/stops the system and shows live status for the vLLM and
app services, plus a streaming log pane. "Start app locally" also starts the
Agentic RAG tab's service (the LPJ pipeline, `python -m lpj_rag serve` in its
own venv) and Ollama, which that service answers with. Works the same on both platforms;
see gui/DocSlides.bat (Windows) and gui/DocSlides.command (macOS) for
double-clickable entry points.

Run directly:
    python gui/launcher.py
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

import tkinter as tk
from tkinter import ttk

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_PORT = 8456  # keep in sync with DOCSLIDES_APP_PORT default in src/docslides/api/main.py
APP_HEALTH_URL = f"http://localhost:{APP_PORT}/health"
APP_UI_URL = f"http://localhost:{APP_PORT}/ui"
VLLM_HEALTH_URL = "http://localhost:8000/health"
OLLAMA_HEALTH_URL = "http://localhost:11434/api/tags"
ENV_LOCAL_PATH = REPO_ROOT / ".env.local"
DOCKER_COMPOSE_VLLM = ["docker-compose.yml"]
DOCKER_COMPOSE_PORTABLE = ["docker-compose.portable.yml"]
POLL_INTERVAL_SECONDS = 3.0
# Agentic RAG tab: the LPJ pipeline (package lpj_rag, its own venv), served on loopback.
# Keep the URL in sync with LPJ_RAG_URL in src/docslides/ui/gradio_app.py / api/routes_lpj.py.
LPJ_RAG_URL = os.environ.get("LPJ_RAG_URL", "http://127.0.0.1:8765")
LPJ_HEALTH_URL = f"{LPJ_RAG_URL}/health"
LPJ_START_TIMEOUT_SECONDS = 60.0


def venv_python() -> Path:
    """Path to the project venv's python, falling back to the current interpreter."""
    candidates = [
        REPO_ROOT / ".venv" / "Scripts" / "python.exe",  # Windows
        REPO_ROOT / ".venv" / "bin" / "python",  # macOS/Linux
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(sys.executable)


def read_env_local() -> dict[str, str]:
    """Parse the simple KEY=VALUE lines scripts/setup.* writes to .env.local
    when the Ollama backend is selected. Absent file -> vLLM (the default in
    config/config.yaml, nothing to override)."""
    env: dict[str, str] = {}
    if not ENV_LOCAL_PATH.exists():
        return env
    # utf-8-sig: tolerate a BOM from older setup.ps1 runs (Windows PowerShell
    # 5.1 `Set-Content -Encoding utf8` writes one), which would otherwise hide
    # the first key and silently fall back to the vllm backend.
    for line in ENV_LOCAL_PATH.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def lpj_pipeline_dir() -> Path:
    """The LPJ pipeline folder: LPJ_RAG_DIR (environment or .env.local), else
    ../RAG_LPJ/pipeline next to this repo -- the same default scripts/setup.* uses."""
    configured = os.environ.get("LPJ_RAG_DIR") or read_env_local().get("LPJ_RAG_DIR")
    return Path(configured) if configured else REPO_ROOT.parent / "RAG_LPJ" / "pipeline"


def lpj_python(pipeline: Path) -> Path | None:
    """The pipeline venv's python (setup.* creates it), or None when it isn't set up."""
    for candidate in (pipeline / ".venv" / "Scripts" / "python.exe", pipeline / ".venv" / "bin" / "python"):
        if candidate.exists():
            return candidate
    return None


def detected_backend() -> str:
    return read_env_local().get("DOCSLIDES_LLM_BACKEND", "vllm")


def docker_compose_cmd() -> list[str] | None:
    """Return the base ['docker', 'compose'] or ['docker-compose'] command, or None."""
    if shutil.which("docker"):
        try:
            subprocess.run(
                ["docker", "compose", "version"],
                cwd=REPO_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            return ["docker", "compose"]
        except Exception:
            pass
    if shutil.which("docker-compose"):
        return ["docker-compose"]
    return None


def check_url(url: str, timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


class ProcessRunner:
    """Runs one subprocess at a time in a background thread, streaming output
    into a queue that the Tk main loop drains (Tkinter widgets are not
    thread-safe, so worker threads never touch them directly)."""

    def __init__(self, out_queue: "queue.Queue[tuple[str, object]]"):
        self.out_queue = out_queue
        self.proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    @property
    def busy(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def run(self, cmd: list[str], label: str, on_done_msg: str | None = None) -> bool:
        with self._lock:
            if self.busy:
                self.out_queue.put(("log", f"[busy] still running a previous command, ignoring: {label}\n"))
                return False
            self.out_queue.put(("log", f"$ {' '.join(cmd)}\n"))
            self.out_queue.put(("status", "running"))
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    cwd=REPO_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except FileNotFoundError as exc:
                self.out_queue.put(("log", f"[error] {exc}\n"))
                self.out_queue.put(("status", "idle"))
                return False
        threading.Thread(target=self._pump, args=(self.proc, on_done_msg), daemon=True).start()
        return True

    def _pump(self, proc: subprocess.Popen, on_done_msg: str | None) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            self.out_queue.put(("log", line))
        code = proc.wait()
        self.out_queue.put(("log", f"[exit {code}]\n"))
        if on_done_msg:
            self.out_queue.put(("log", f"{on_done_msg}\n"))
        self.out_queue.put(("status", "idle"))

    def stop(self) -> None:
        with self._lock:
            if self.proc is not None and self.proc.poll() is None:
                self.out_queue.put(("log", "[stopping current command]\n"))
                self.proc.terminate()


class Launcher(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("docslides launcher")
        self.geometry("860x560")
        self.minsize(700, 440)

        self.out_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self.runner = ProcessRunner(self.out_queue)
        self.compose_cmd = docker_compose_cmd()
        self.local_app_proc: subprocess.Popen | None = None
        self.lpj_proc: subprocess.Popen | None = None
        self.ollama_proc: subprocess.Popen | None = None
        self.backend = detected_backend()  # "vllm" or "ollama", from .env.local (see scripts/setup.*)

        self._build_ui()
        self._poll_health()
        self._drain_queue()

    # -- UI construction ---------------------------------------------------
    def _build_ui(self) -> None:
        status_frame = ttk.LabelFrame(self, text="Status")
        status_frame.pack(fill="x", padx=10, pady=(10, 5))

        self.backend_var = tk.StringVar(value=self.backend)
        self.llm_var = tk.StringVar(value="checking...")
        self.app_var = tk.StringVar(value="checking...")
        self.lpj_var = tk.StringVar(value="checking...")
        self.docker_var = tk.StringVar(
            value=f"found ({' '.join(self.compose_cmd)})" if self.compose_cmd else "not found"
        )

        self._status_row(status_frame, "LLM backend (from .env.local)", self.backend_var, 0)
        self._status_row(status_frame, self._llm_status_label(), self.llm_var, 1)
        self._status_row(status_frame, f"App / UI (localhost:{APP_PORT})", self.app_var, 2)
        self._status_row(status_frame, f"Agentic RAG ({LPJ_RAG_URL.split('//')[-1]})", self.lpj_var, 3)
        self._status_row(status_frame, "Docker Compose", self.docker_var, 4)

        actions = ttk.LabelFrame(self, text="Actions")
        actions.pack(fill="x", padx=10, pady=5)

        row1 = ttk.Frame(actions)
        row1.pack(fill="x", pady=3)
        ttk.Button(row1, text="Run setup", command=self.run_setup).pack(side="left", padx=4)
        ttk.Button(row1, text="Start (docker compose up)", command=self.start_docker).pack(side="left", padx=4)
        ttk.Button(row1, text="Stop (docker compose down)", command=self.stop_docker).pack(side="left", padx=4)
        ttk.Button(row1, text="Tail logs", command=self.tail_logs).pack(side="left", padx=4)

        row2 = ttk.Frame(actions)
        row2.pack(fill="x", pady=3)
        ttk.Button(row2, text="Start app locally (no Docker, uses .env.local's backend)",
                   command=self.start_local_app).pack(side="left", padx=4)
        ttk.Button(row2, text="Stop local app", command=self.stop_local_app).pack(side="left", padx=4)
        ttk.Button(row2, text="Open UI", command=lambda: webbrowser.open(APP_UI_URL)).pack(side="left", padx=4)
        ttk.Button(row2, text="Stop running command", command=self.runner.stop).pack(side="left", padx=4)

        row3 = ttk.Frame(actions)
        row3.pack(fill="x", pady=3)
        ttk.Button(row3, text="Start Agentic RAG service", command=self.start_rag_services).pack(side="left", padx=4)
        ttk.Button(row3, text="Stop Agentic RAG service", command=self.stop_rag_services).pack(side="left", padx=4)

        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        self.log_text = tk.Text(log_frame, wrap="word", state="disabled", font=("Consolas", 9))
        scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        clear_btn = ttk.Button(self, text="Clear log", command=self._clear_log)
        clear_btn.pack(anchor="e", padx=10, pady=(0, 8))

    def _status_row(self, parent: ttk.LabelFrame, label: str, var: tk.StringVar, row: int) -> None:
        ttk.Label(parent, text=label, width=28).grid(row=row, column=0, sticky="w", padx=8, pady=3)
        ttk.Label(parent, textvariable=var).grid(row=row, column=1, sticky="w", padx=8, pady=3)

    def _llm_status_label(self) -> str:
        return "Ollama (localhost:11434)" if self.backend == "ollama" else "vLLM (localhost:8000)"

    # -- health polling ------------------------------------------------------
    def _poll_health(self) -> None:
        # Re-read in case setup.* just (re)wrote .env.local, e.g. after "Run setup".
        self.backend = detected_backend()

        def worker() -> None:
            llm_url = OLLAMA_HEALTH_URL if self.backend == "ollama" else VLLM_HEALTH_URL
            llm_ok = check_url(llm_url)
            app_ok = check_url(APP_HEALTH_URL)
            lpj_ok = check_url(LPJ_HEALTH_URL)
            self.out_queue.put(("llm", llm_ok))
            self.out_queue.put(("app", app_ok))
            self.out_queue.put(("lpj", lpj_ok))

        threading.Thread(target=worker, daemon=True).start()
        self.after(int(POLL_INTERVAL_SECONDS * 1000), self._poll_health)

    # -- queue draining (runs on the Tk main thread) -------------------------
    def _drain_queue(self) -> None:
        try:
            while True:
                kind, payload = self.out_queue.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "llm":
                    self.backend_var.set(self.backend)
                    self.llm_var.set("up" if payload else "down / not reachable")
                elif kind == "app":
                    self.app_var.set("up" if payload else "down / not reachable")
                elif kind == "lpj":
                    starting = self.lpj_proc is not None and self.lpj_proc.poll() is None
                    self.lpj_var.set("up" if payload else ("starting..." if starting else "down / not running"))
                elif kind == "status":
                    pass  # reserved for future busy-state UI
        except queue.Empty:
            pass
        self.after(150, self._drain_queue)

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    # -- actions ---------------------------------------------------------
    def run_setup(self) -> None:
        if sys.platform == "win32":
            cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(REPO_ROOT / "scripts" / "setup.ps1")]
        else:
            cmd = ["bash", str(REPO_ROOT / "scripts" / "setup.sh")]
        self.runner.run(cmd, "setup", on_done_msg="Setup finished -- see above for any skipped steps.")

    def _compose_files(self) -> list[str]:
        """vLLM path uses docker-compose.yml (NVIDIA GPU, vllm+app services);
        the Ollama path uses docker-compose.portable.yml (app only -- Ollama
        runs natively on the host, see that file's header comment)."""
        return DOCKER_COMPOSE_PORTABLE if self.backend == "ollama" else DOCKER_COMPOSE_VLLM

    def _compose_args(self) -> list[str]:
        args: list[str] = []
        for f in self._compose_files():
            args += ["-f", f]
        return args

    def start_docker(self) -> None:
        if not self.compose_cmd:
            self._append_log("[error] docker/docker compose not found on PATH.\n")
            return
        if self.backend == "ollama":
            self._append_log(
                "[info] Ollama backend: make sure 'ollama serve' is running on the HOST "
                "first (scripts/setup.* starts it) -- this only starts the app container.\n"
            )
        self.runner.run(self.compose_cmd + self._compose_args() + ["up", "-d"], "docker compose up",
                         on_done_msg="Containers started (or already running). Check Status above.")
        self._append_log(
            "[info] Agentic RAG: its service listens on 127.0.0.1 only, so the app container can't reach it "
            "(see README). Use 'Start app locally' for that tab.\n"
        )

    def stop_docker(self) -> None:
        if not self.compose_cmd:
            self._append_log("[error] docker/docker compose not found on PATH.\n")
            return
        self.runner.run(self.compose_cmd + self._compose_args() + ["down"], "docker compose down")

    def tail_logs(self) -> None:
        if not self.compose_cmd:
            self._append_log("[error] docker/docker compose not found on PATH.\n")
            return
        self.runner.run(self.compose_cmd + self._compose_args() + ["logs", "-f", "--tail", "100"], "docker compose logs")

    def start_local_app(self) -> None:
        if self.local_app_proc is not None and self.local_app_proc.poll() is None:
            self._append_log("[info] local app is already running.\n")
            return
        py = venv_python()
        cmd = [str(py), "-m", "uvicorn", "docslides.api.main:app", "--host", "0.0.0.0", "--port", str(APP_PORT)]
        env_overrides = read_env_local()
        self._append_log(f"$ {' '.join(cmd)}\n")
        if env_overrides:
            self._append_log(f"(applying .env.local: {env_overrides})\n")
        else:
            self._append_log("(no .env.local -- using config/config.yaml's vllm backend as-is)\n")
        env = {**os.environ, **env_overrides}
        try:
            self.local_app_proc = subprocess.Popen(
                cmd, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
        except FileNotFoundError as exc:
            self._append_log(f"[error] {exc}\n")
            return
        threading.Thread(target=self._pump_local_app, daemon=True).start()
        self.start_rag_services()

    def _pump_local_app(self) -> None:
        proc = self.local_app_proc
        assert proc is not None and proc.stdout is not None
        for line in proc.stdout:
            self.out_queue.put(("log", line))
        code = proc.wait()
        self.out_queue.put(("log", f"[local app exited {code}]\n"))

    def stop_local_app(self) -> None:
        if self.local_app_proc is not None and self.local_app_proc.poll() is None:
            self.local_app_proc.terminate()
            self._append_log("[stopping local app]\n")
        else:
            self._append_log("[info] local app is not running.\n")
        self.stop_rag_services()

    # -- Agentic RAG services (LPJ pipeline + Ollama) ----------------------
    def start_rag_services(self) -> None:
        """Start what the Agentic RAG tab needs and isn't already up: Ollama (the
        service answers with it, whatever the app's own backend), then `lpj_rag serve`
        from the pipeline's venv. Runs in a thread; the Status row shows the result."""
        threading.Thread(target=self._start_rag_services, daemon=True).start()

    def _start_rag_services(self) -> None:
        def log(text: str) -> None:
            self.out_queue.put(("log", text))

        if check_url(LPJ_HEALTH_URL):
            log("[agentic rag] service already running.\n")
            return
        if self.lpj_proc is not None and self.lpj_proc.poll() is None:
            log("[agentic rag] service is still starting.\n")
            return
        pipeline = lpj_pipeline_dir()
        if not (pipeline / "lpj_rag").is_dir():
            log(f"[agentic rag] no LPJ pipeline at {pipeline} -- set LPJ_RAG_DIR in .env.local; tab stays offline.\n")
            return
        py = lpj_python(pipeline)
        if py is None:
            log(f"[agentic rag] {pipeline} has no .venv yet -- click 'Run setup' "
                "(installs requirements-agentic-rag.txt into it).\n")
            return

        if not check_url(OLLAMA_HEALTH_URL):
            ollama = shutil.which("ollama")
            if ollama is None:
                log("[agentic rag] Ollama isn't running and isn't on PATH -- answers will fail until it is.\n")
            else:
                log(f"$ {ollama} serve\n")
                self.ollama_proc = self._spawn([ollama, "serve"], REPO_ROOT, "ollama")
                for _ in range(20):
                    if check_url(OLLAMA_HEALTH_URL):
                        break
                    time.sleep(0.5)

        cmd = [str(py), "-m", "lpj_rag", "serve"]
        log(f"$ {' '.join(cmd)}   (in {pipeline})\n")
        proc = self._spawn(cmd, pipeline, "agentic rag")
        self.lpj_proc = proc
        if proc is None:
            return
        deadline = time.monotonic() + LPJ_START_TIMEOUT_SECONDS
        while time.monotonic() < deadline and proc.poll() is None:
            if check_url(LPJ_HEALTH_URL):
                log(f"[agentic rag] service up at {LPJ_RAG_URL}.\n")
                return
            time.sleep(1)
        if proc.poll() is None:
            log("[agentic rag] still starting (no /health answer yet) -- see its log lines above.\n")

    def _spawn(self, cmd: list[str], cwd: Path, tag: str) -> subprocess.Popen | None:
        """Start a long-running service, streaming its output into the log with a tag."""
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        try:
            proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace", bufsize=1, creationflags=flags)
        except OSError as exc:
            self.out_queue.put(("log", f"[{tag}] [error] {exc}\n"))
            return None

        def pump() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                self.out_queue.put(("log", f"[{tag}] {line}"))
            self.out_queue.put(("log", f"[{tag} exited {proc.wait()}]\n"))

        threading.Thread(target=pump, daemon=True).start()
        return proc

    def stop_rag_services(self) -> None:
        """Stop the services this launcher started (never ones it found already running)."""
        for attr, tag in (("lpj_proc", "agentic rag"), ("ollama_proc", "ollama")):
            proc = getattr(self, attr)
            if proc is not None and proc.poll() is None:
                proc.terminate()
                self._append_log(f"[stopping {tag}]\n")
            setattr(self, attr, None)

    def on_close(self) -> None:
        for proc in (self.local_app_proc, self.lpj_proc, self.ollama_proc):
            if proc is not None and proc.poll() is None:
                proc.terminate()
        self.destroy()


def main() -> None:
    app = Launcher()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    main()
