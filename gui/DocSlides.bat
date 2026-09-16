@echo off
REM Double-click entry point for the docslides launcher GUI on Windows.
REM Prefers the project venv's python (has tkinter from the system Python
REM install); falls back to whatever "python" resolves to on PATH.

setlocal
set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%.."
set "VENV_PY=%REPO_ROOT%\.venv\Scripts\python.exe"

if exist "%VENV_PY%" (
    "%VENV_PY%" "%SCRIPT_DIR%launcher.py"
) else (
    python "%SCRIPT_DIR%launcher.py"
)

if errorlevel 1 (
    echo.
    echo Launcher exited with an error. If this is the first run, try:
    echo   scripts\setup.ps1
    echo to create the virtual environment first.
    pause
)
