"""
集中配置管理 (Pydantic Settings)

通过环境变量管理所有路径和参数，消除硬编码。

用法:
    from backend.config import settings
    model_path = settings.model_path

环境变量（可选，有默认值）:
    FIREFLY_MODEL_PATH        模型路径 (默认: model/)
    FIREFLY_LORA_PATH         LoRA adapter 路径 (默认: output/Firefly_LoRA/)
    FIREFLY_RAG_DB_PATH       ChromaDB 路径 (默认: chroma_db/)
    FIREFLY_LOG_DIR           日志目录 (默认: logs/)
    FIREFLY_HOST              监听地址 (默认: 0.0.0.0)
    FIREFLY_PORT              端口号 (默认: 7860)
    FIREFLY_MAX_TOKENS        最大生成 token (默认: 512)
    FIREFLY_TEMPERATURE       温度 (默认: 0.7)
    FIREFLY_RATE_LIMIT        限流 (默认: 5/minute)
"""

import os
from pathlib import Path
from typing import Optional

# 无 pydantic 依赖的轻量实现
PROJECT_ROOT = Path(__file__).parent.parent


class Settings:
    """应用配置"""

    # 路径配置
    @property
    def model_path(self) -> str:
        return os.environ.get(
            "FIREFLY_MODEL_PATH",
            str(PROJECT_ROOT / "model"),
        )

    @property
    def lora_path(self) -> str:
        return os.environ.get(
            "FIREFLY_LORA_PATH",
            str(PROJECT_ROOT / "output" / "Firefly_LoRA"),
        )

    @property
    def dpo_merged_path(self) -> str:
        """DPO 合并后的模型路径（用于生产部署）"""
        return os.environ.get(
            "FIREFLY_DPO_MERGED_PATH",
            str(PROJECT_ROOT / "output" / "Firefly_DPO_Merged"),
        )

    @property
    def rag_db_path(self) -> str:
        return os.environ.get(
            "FIREFLY_RAG_DB_PATH",
            str(PROJECT_ROOT / "chroma_db"),
        )

    @property
    def log_dir(self) -> str:
        return os.environ.get(
            "FIREFLY_LOG_DIR",
            str(PROJECT_ROOT / "logs"),
        )

    @property
    def data_dir(self) -> str:
        return os.environ.get(
            "FIREFLY_DATA_DIR",
            str(PROJECT_ROOT / "data"),
        )

    # 服务配置
    @property
    def host(self) -> str:
        return os.environ.get("FIREFLY_HOST", "0.0.0.0")

    @property
    def port(self) -> int:
        return int(os.environ.get("FIREFLY_PORT", "7860"))

    # 模型推理配置
    @property
    def max_tokens(self) -> int:
        return int(os.environ.get("FIREFLY_MAX_TOKENS", "512"))

    @property
    def temperature(self) -> float:
        return float(os.environ.get("FIREFLY_TEMPERATURE", "0.7"))

    @property
    def top_p(self) -> float:
        return float(os.environ.get("FIREFLY_TOP_P", "0.9"))

    @property
    def top_k(self) -> int:
        return int(os.environ.get("FIREFLY_TOP_K", "50"))

    @property
    def repetition_penalty(self) -> float:
        return float(os.environ.get("FIREFLY_REPETITION_PENALTY", "1.1"))

    # 功能开关
    @property
    def enable_rag(self) -> bool:
        return os.environ.get("FIREFLY_ENABLE_RAG", "true").lower() == "true"

    @property
    def enable_validation(self) -> bool:
        return os.environ.get("FIREFLY_ENABLE_VALIDATION", "true").lower() == "true"

    @property
    def enable_rate_limit(self) -> bool:
        return os.environ.get("FIREFLY_ENABLE_RATE_LIMIT", "true").lower() == "true"

    @property
    def rate_limit(self) -> str:
        return os.environ.get("FIREFLY_RATE_LIMIT", "5/minute")

    # vLLM 配置
    @property
    def vllm_base_url(self) -> Optional[str]:
        return os.environ.get("FIREFLY_VLLM_URL", None)

    # 训练配置（供训练脚本使用）
    @property
    def lora_r(self) -> int:
        return int(os.environ.get("FIREFLY_LORA_R", "16"))

    @property
    def lora_alpha(self) -> int:
        return int(os.environ.get("FIREFLY_LORA_ALPHA", "32"))

    @property
    def training_epochs(self) -> int:
        return int(os.environ.get("FIREFLY_EPOCHS", "5"))

    @property
    def training_lr(self) -> float:
        return float(os.environ.get("FIREFLY_LR", "3e-5"))

    def display(self):
        """打印当前配置"""
        print(f"  Model Path:      {self.model_path}")
        print(f"  LoRA Path:       {self.lora_path}")
        print(f"  DPO Merged:      {self.dpo_merged_path} (exists: {Path(self.dpo_merged_path).exists()})")
        print(f"  RAG DB:          {self.rag_db_path} (exists: {Path(self.rag_db_path).exists()})")
        print(f"  Log Dir:         {self.log_dir}")
        print(f"  Host:Port:       {self.host}:{self.port}")
        print(f"  Max Tokens:      {self.max_tokens}")
        print(f"  Temperature:     {self.temperature}")
        print(f"  RAG Enabled:     {self.enable_rag}")
        print(f"  Validation:      {self.enable_validation}")
        print(f"  Rate Limit:      {self.rate_limit if self.enable_rate_limit else 'disabled'}")


# 全局配置实例
settings = Settings()
