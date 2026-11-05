@echo off
setlocal
if exist "%~dp0python.local.bat" (
    call "%~dp0python.local.bat" "%~dp0app.py"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0app.py"
) else (
    python "%~dp0app.py"
)
if errorlevel 1 pause
