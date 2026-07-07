@echo off
echo ========================================
echo 启动 vLLM 推理服务
echo ========================================

set MODEL_PATH=C:\Users\Admin\Desktop\Firefly_LLM_Finetune\model
set LORA_PATH=C:\Users\Admin\Desktop\Firefly_LLM_Finetune\output\Firefly_LoRA
set PORT=8000

echo 模型路径: %MODEL_PATH%
echo LoRA路径: %LORA_PATH%
echo 端口: %PORT%
echo.

REM 检查 LoRA 权重是否存在
if not exist "%LORA_PATH%" (
    echo ⚠️ LoRA 权重不存在: %LORA_PATH%
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
    echo ✅ LoRA 权重已找到，启动带LoRA的服务...
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
