@echo off
setlocal

cd /d "%~dp0"
set "PYTHON=%~dp0.venv\Scripts\python.exe"
set "PYTHONPATH=%~dp0src"
set "TIYA_PROJECT_ROOT=%~dp0"

if not exist "%PYTHON%" (
    echo [ERROR] Virtual environment Python not found: "%PYTHON%"
    exit /b 1
)

rem Refuse to start if a copied virtual environment still imports another checkout.
"%PYTHON%" -P -c "import os, pathlib, sys, TIYA; expected=(pathlib.Path(os.environ['TIYA_PROJECT_ROOT'])/'src'/'TIYA').resolve(); actual=pathlib.Path(TIYA.__file__).resolve().parent; sys.exit(f'TIYA source mismatch: expected {expected}, got {actual}') if expected != actual else None"
if errorlevel 1 (
    echo [ERROR] Refusing to start with a non-local TIYA source tree.
    exit /b 1
)

"%PYTHON%" -P -m TIYA.QQ_bot
exit /b %ERRORLEVEL%
