@echo off
setlocal
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
if exist "%~dp0python.local.bat" (
    call "%~dp0python.local.bat" -c "import PyInstaller, PyQt5, ultralytics, PIL" >nul 2>nul
    if errorlevel 1 goto :missing
    call "%~dp0python.local.bat" scripts\package_release.py %*
) else if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" -c "import PyInstaller, PyQt5, ultralytics, PIL" >nul 2>nul
    if errorlevel 1 goto :missing
    "%~dp0.venv\Scripts\python.exe" scripts\package_release.py %*
) else (
    where python >nul 2>nul
    if errorlevel 1 goto :missing
    python -c "import PyInstaller, PyQt5, ultralytics, PIL" >nul 2>nul
    if errorlevel 1 goto :missing
    python scripts\package_release.py %*
)
if errorlevel 1 goto :failed
exit /b 0
:missing
echo 缺少打包依赖。请在命令提示符中运行 安装依赖.bat build。
pause
exit /b 1
:failed
echo 打包失败，请查看上面的详细错误。
pause
exit /b 1
