@echo off
echo ========================================
echo 启动流萤前端界面
echo ========================================

cd /d "%~dp0..\webui"

echo 启动本地 HTTP 服务器...
echo 地址: http://localhost:8080
echo.

REM 使用 Python 内置 HTTP 服务器
python -m http.server 8080

pause
