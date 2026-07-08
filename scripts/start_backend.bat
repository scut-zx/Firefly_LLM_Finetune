@echo off
echo ========================================
echo 流萤角色助手 - 后端 API 服务
echo ========================================
echo.
echo 模式: 直接模型推理 (Direct Mode)
echo 地址: http://localhost:7860
echo 文档: http://localhost:7860/docs
echo 前端: http://localhost:7860/
echo.

cd /d "%~dp0.."
python -m backend.app --direct --port 7860

pause
