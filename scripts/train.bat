@echo off
echo ========================================
echo 流萤 LoRA 微调训练
echo ========================================
echo.
echo 环境: conda activate firefly
echo 框架: LlamaFactory
echo 模型: Qwen3-4B-Instruct-2507
echo 方法: LoRA (r=16, alpha=32)
echo.
echo 确保已完成:
echo   1. 运行 02_generate_training_pairs.py 生成训练数据
echo   2. 运行 03_data_cleaning.py 清洗数据
echo   3. 运行 04_build_rag_db.py 构建RAG数据库
echo.

call conda activate firefly
if %ERRORLEVEL% NEQ 0 (
    echo ❌ conda activate firefly 失败！
    echo 请确保已创建 firefly 环境
    pause
    exit /b 1
)

set PYTHONPATH=C:\Users\Admin\Desktop\Firefly_LLM_Finetune\LlamaFactory;%PYTHONPATH%
cd /d C:\Users\Admin\Desktop\Firefly_LLM_Finetune\LlamaFactory

echo.
echo 开始训练...
echo 配置文件: C:\Users\Admin\Desktop\Firefly_LLM_Finetune\configs\firefly_lora_sft.yaml
echo 输出目录: C:\Users\Admin\Desktop\Firefly_LLM_Finetune\output\Firefly_LoRA
echo.

llamafactory-cli train C:\Users\Admin\Desktop\Firefly_LLM_Finetune\configs\firefly_lora_sft.yaml

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ 训练完成！
    echo LoRA 权重: C:\Users\Admin\Desktop\Firefly_LLM_Finetune\output\Firefly_LoRA
) else (
    echo.
    echo ❌ 训练失败！请检查错误信息。
)

pause
