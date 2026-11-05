@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0python.local.bat" (
    call "%~dp0python.local.bat" scripts\package_release.py %*
) else if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" scripts\package_release.py %*
) else (
    python scripts\package_release.py %*
)
if errorlevel 1 pause
