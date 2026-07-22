@echo off
setlocal

cd /d "%~dp0"
set "PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo [ERROR] Virtual environment Python not found: "%PYTHON%"
    exit /b 1
)

"%PYTHON%" -m TIYA.QQ_bot
exit /b %ERRORLEVEL%
