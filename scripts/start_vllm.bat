@echo off
echo ========================================
echo 启动 vLLM 推理服务
echo ========================================

cd /d "%~dp0.."
set MODEL_PATH=%CD%\model
set LORA_PATH=%CD%\output\Firefly_LoRA
set PORT=8000

echo 模型路径: %MODEL_PATH%
echo LoRA路径: %LORA_PATH%
echo 端口: %PORT%
echo.

REM 检查 LoRA 权重是否存在
if not exist "%LORA_PATH%" (
    echo [WARN] LoRA 权重不存在: %LORA_PATH%
    echo 将启动基础模型（无LoRA），角色风格会受影响
    echo.
    vllm serve "%MODEL_PATH%" ^
        --port %PORT% ^
        --served-model-name firefly-assistant ^
        --max-model-len 4096 ^
        --dtype bfloat16 ^
        --trust-remote-code ^
        --gpu-memory-utilization 0.85
) else (
    echo [OK] LoRA 权重已找到，启动带LoRA的服务...
    echo.
    vllm serve "%MODEL_PATH%" ^
        --port %PORT% ^
        --served-model-name firefly-assistant ^
        --enable-lora ^
        --lora-modules firefly-lora=%LORA_PATH% ^
        --max-loras 4 ^
        --max-lora-rank 64 ^
        --max-model-len 4096 ^
        --dtype bfloat16 ^
        --trust-remote-code ^
        --gpu-memory-utilization 0.85
)

echo.
echo vLLM 服务已停止
pause
