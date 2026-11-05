@echo off
setlocal
if exist "%~dp0python.local.bat" (
    call "%~dp0python.local.bat" "%~dp0app.py"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0app.py"
) else (
    python "%~dp0app.py"
)
if errorlevel 1 (
    echo.
    echo 如果提示缺少 PyQt5 或 ultralytics，请先双击 安装依赖.bat。
    pause
)
