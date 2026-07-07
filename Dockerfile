# ============================================================
# Firefly Character AI Assistant - Dockerfile
# ============================================================
# Multi-stage build for GPU-accelerated character LLM deployment
#
# Build:
#   docker build -t firefly-assistant .
#
# Run:
#   docker run --gpus all -p 7860:7860 \
#     -v $(pwd)/model:/app/model \
#     -v $(pwd)/output:/app/output \
#     -v $(pwd)/chroma_db:/app/chroma_db \
#     firefly-assistant
#
# Or use docker-compose:
#   docker-compose up -d
# ============================================================

FROM nvidia/cuda:12.4-runtime-ubuntu22.04

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY backend/requirements.txt /app/backend/requirements.txt

# 安装 Python 依赖
RUN pip install --no-cache-dir -r /app/backend/requirements.txt && \
    pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir \
    transformers==4.51.0 \
    peft \
    trl \
    accelerate \
    bitsandbytes

# 复制项目代码
COPY scripts/ /app/scripts/
COPY backend/ /app/backend/
COPY webui/ /app/webui/
COPY evaluation/ /app/evaluation/
COPY configs/ /app/configs/
COPY data/*.json /app/data/

# 创建必要目录
RUN mkdir -p /app/logs /app/output

# 暴露端口
EXPOSE 7860

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# 默认启动命令
# 模型和数据目录通过 volume 挂载
CMD ["python", "-m", "backend.app", "--direct", "--host", "0.0.0.0", "--port", "7860"]
