@echo off
echo ╔══════════════════════════════════════════════════════════════╗
echo ║         流萤角色助手 — 一键启动脚本                              ║
echo ╠══════════════════════════════════════════════════════════════╣
echo ║                                                              ║
echo ║  步骤：                                                        ║
echo ║  1. 生成训练数据 (已完成则跳过)                                     ║
echo ║  2. 构建 RAG 知识库                                              ║
echo ║  3. 训练 LoRA (已完成则跳过)                                      ║
echo ║  4. 启动 vLLM 推理服务                                           ║
echo ║  5. 启动后端 API 服务                                            ║
echo ║  6. 启动前端界面                                                 ║
echo ║                                                              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

echo 请选择要执行的操作:
echo   [1] 完整流程（数据生成 + RAG + 启动服务）
echo   [2] 仅启动服务（vLLM + 后端 + 前端）
echo   [3] 仅训练模型
echo   [4] 仅构建 RAG
echo   [5] 重新生成训练数据
echo.

set /p choice="请输入选项 (1-5): "

if "%choice%"=="1" goto full
if "%choice%"=="2" goto serve
if "%choice%"=="3" goto train
if "%choice%"=="4" goto rag
if "%choice%"=="5" goto data
echo 无效选项 & pause & exit /b

:data
echo.
echo [1/3] 生成训练数据...
python scripts\02_generate_training_pairs.py
echo [2/3] 清洗训练数据...
python scripts\03_data_cleaning.py
echo.
echo ✅ 数据准备完成！
echo    训练数据: data\firefly_training_cleaned.json
goto ask_rag

:rag
echo.
echo 构建 RAG 知识库...
python scripts\04_build_rag_db.py
echo.
echo ✅ RAG 构建完成！
goto ask_train

:train
echo.
echo 开始训练...
call scripts\train.bat
goto serve

:full
call :data
call :rag
call :train
goto serve

:serve
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║  启动服务（请在单独的终端窗口中分别运行）:                        ║
echo ║                                                              ║
echo ║  终端1: scripts\start_vllm.bat     (vLLM 推理 :8000)          ║
echo ║  终端2: scripts\start_backend.bat  (后端API  :7860)           ║
echo ║  终端3: scripts\start_frontend.bat (前端界面 :8080)           ║
echo ║                                                              ║
echo ║  然后访问: http://localhost:8080                              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

pause
exit /b

:ask_rag
set /p rag_choice="是否构建 RAG？(y/n): "
if /i "%rag_choice%"=="y" goto rag
goto ask_train

:ask_train
set /p train_choice="是否进行训练？(y/n): "
if /i "%train_choice%"=="y" goto train
goto serve
