@echo off
setlocal
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
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
set "REQUIREMENTS=requirements.txt"
if /i "%~1"=="build" set "REQUIREMENTS=requirements-build.txt"
".venv\Scripts\python.exe" -m pip install -r "%REQUIREMENTS%"
if errorlevel 1 goto :failed
if /i "%~1"=="build" (
    echo 打包依赖已安装。现在可以运行 构建exe.bat。
) else (
    echo 运行依赖已安装。现在可以运行 启动上位机.bat。
)
pause
exit /b 0
:failed
echo 安装失败，请检查上面的报错信息。
pause
exit /b 1
