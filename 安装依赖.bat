@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
    echo 未找到 Python。请先安装 Python 3.10 或更新版本，再运行本脚本。
    pause
    exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 goto :failed
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :failed
echo 依赖安装完成，请双击 启动上位机.bat。
pause
exit /b 0
:failed
echo 安装失败，请检查上面的报错信息。
pause
exit /b 1
