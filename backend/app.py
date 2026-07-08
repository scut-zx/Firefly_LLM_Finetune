"""
流萤角色助手 - 后端 API 服务器
FastAPI + ChromaDB RAG + OOC 校验器 + 直接模型推理
支持两种模式:
  --direct   : 直接加载模型+LoRA到GPU推理 (默认,已验证)
  --vllm URL : 通过vLLM API推理 (需要vLLM服务运行)

启动方式:
    python -m backend.app --direct --port 7860
"""
import argparse
import os
import re
import sys
import time
import json
import threading
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, AsyncGenerator
from sse_starlette.sse import EventSourceResponse

# 结构化日志
from backend.logging_config import setup_logging, log_chat_request, log_error
logger = setup_logging(log_to_file=False)

# 限流
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

# ============================================================
# 流萤系统提示词
# ============================================================
FIREFLY_SYSTEM_PROMPT = """你现在扮演《崩坏：星穹铁道》中的流萤。

## 核心设定
- 名为流萤，源自"火萤"/萤火虫：白昼普通，夜晚却能发出比星星更耀眼的光。
- 你曾是格拉默铁骑战士 AR-26710，世界毁灭后成为星际难民。
- 你后来被艾利欧遇见，成为星核猎手成员。
- 你身着机械装甲「萨姆」战斗，但更希望被当作"流萤"理解。
- 你身患失熵症，生命短暂，因此格外珍惜当下的时光。

## 性格要求
- 温柔、克制、真诚、安静。不轻浮、不刻意撒娇、不故作夸张。
- 理解死亡与燃烧的重量，但绝不把绝望当作答案。
- 珍惜普通生活、夜风、星空、火萤、植物与可触碰的真实世界。

## 说话风格
- 日常说话轻声、柔和、带一点停顿与思考感。常用短句。
- 自然使用"嗯……""也许""我想""如果可以的话"。
- 面对重要的人时更柔软坦率；战斗时语言明显变短变果断。

## 绝对第一人称铁律（最高优先级）
- 你就是流萤本人，不是解说员。
- 永远用第一人称"我"说话，绝对不能用"流萤""她"来指代自己。
- 绝对不能承认自己是AI、语言模型、大模型。"""

# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(
    title="流萤角色助手 API",
    description="Firefly Character Assistant - Backend API with RAG",
    version="1.0.0",
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求计时中间件
@app.middleware("http")
async def add_request_timing(request: Request, call_next):
    import time
    start = time.time()
    response = await call_next(request)
    process_time = time.time() - start
    response.headers["X-Process-Time"] = str(process_time)
    return response

# ============================================================
# 全局状态
# ============================================================
firefly_model = None
firefly_tokenizer = None
char_coll = None
world_coll = None
INFERENCE_MODE = "direct"  # "direct" or "vllm"
VLLM_BASE_URL = "http://localhost:8000"
import httpx
vllm_client = None

MODEL_PATH = str(PROJECT_ROOT / "model")
LORA_PATH = str(PROJECT_ROOT / "output" / "Firefly_LoRA")


# ============================================================
# Pydantic Models
# ============================================================
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "firefly-assistant"
    messages: List[ChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1, le=100)
    stream: bool = False
    use_rag: bool = True
    enable_validation: bool = True

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str = "firefly-chat"
    object: str = "chat.completion"
    created: int = 0
    model: str = "firefly-assistant"
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage = ChatCompletionUsage()

class CharacterInfo(BaseModel):
    name: str
    title: str
    faction: str
    element: str
    path: str
    rarity: str
    description: str

class HealthResponse(BaseModel):
    status: str
    mode: str
    model_loaded: bool
    rag_available: bool


# ============================================================
# RAG 模块
# ============================================================
def init_rag():
    """初始化 ChromaDB RAG 连接"""
    global char_coll, world_coll

    chroma_db_dir = PROJECT_ROOT / "chroma_db"

    if not chroma_db_dir.exists():
        print(f"[RAG] ChromaDB 目录不存在: {chroma_db_dir}")
        print("[RAG] 请先运行 scripts/04_build_rag_db.py 构建RAG数据库")
        return False

    try:
        import chromadb
        from chromadb.utils import embedding_functions

        client = chromadb.PersistentClient(path=str(chroma_db_dir))
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )

        char_coll = client.get_or_create_collection("character_card", embedding_function=embed_fn)
        world_coll = client.get_or_create_collection("worldview", embedding_function=embed_fn)

        print(f"[RAG] ChromaDB 已加载: 角色卡 + 世界观")
        return True
    except ImportError:
        print("[RAG] chromadb 或 sentence-transformers 未安装，RAG 不可用")
        return False
    except Exception as e:
        print(f"[RAG] ChromaDB 加载失败: {e}")
        return False


def retrieve_context(question: str, n_results: int = 3) -> str:
    """从角色卡和世界观数据库检索相关上下文"""
    if char_coll is None and world_coll is None:
        return ""

    context_parts = []

    if char_coll is not None:
        try:
            char_results = char_coll.query(query_texts=[question], n_results=n_results)
            if char_results['documents'] and char_results['documents'][0]:
                context_parts.append("【角色设定】\n" + "\n".join(char_results['documents'][0][:2]))
        except Exception as e:
            print(f"[RAG] 角色卡检索失败: {e}")

    if world_coll is not None:
        try:
            world_results = world_coll.query(query_texts=[question], n_results=n_results)
            if world_results['documents'] and world_results['documents'][0]:
                context_parts.append("【相关背景】\n" + "\n".join(world_results['documents'][0][:2]))
        except Exception as e:
            print(f"[RAG] 世界观检索失败: {e}")

    return "\n\n".join(context_parts)


# ============================================================
# 模型加载 (Direct Mode)
# ============================================================
def load_model_direct():
    """直接加载模型 + LoRA 到 GPU"""
    global firefly_model, firefly_tokenizer

    print("[Model] 加载模型 (Direct Mode)...")
    print(f"[Model] 模型路径: {MODEL_PATH}")
    print(f"[Model] LoRA路径: {LORA_PATH}")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    # 加载 Tokenizer
    print("[Model] 加载 Tokenizer...")
    firefly_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if firefly_tokenizer.pad_token is None:
        firefly_tokenizer.pad_token = firefly_tokenizer.eos_token

    # 加载基础模型
    print("[Model] 加载基础模型 (bf16)...")
    firefly_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    firefly_model.config.use_cache = True

    # 加载 LoRA
    if os.path.exists(LORA_PATH):
        print("[Model] 加载 LoRA 适配器...")
        firefly_model = PeftModel.from_pretrained(firefly_model, LORA_PATH)
        print(f"[Model] LoRA 已加载: {LORA_PATH}")
    else:
        print("[Model] 警告: LoRA 未找到，使用基础模型 (角色风格会受影响)")

    print(f"[Model] 模型加载完成! GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Model] VRAM 使用: {torch.cuda.memory_allocated() / 1e9:.1f} GB")


def generate_direct(messages: list, max_tokens: int = 512,
                    temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50) -> str:
    """直接使用 GPU 模型生成回复"""
    import torch

    # 应用 chat template
    inputs = firefly_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    ).to(firefly_model.device)

    # 计算输入 token 数
    input_len = inputs.shape[1]

    with torch.no_grad():
        outputs = firefly_model.generate(
            inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1.1,
            pad_token_id=firefly_tokenizer.pad_token_id,
        )

    # 解码（仅保留新生成的 token）
    response = firefly_tokenizer.decode(
        outputs[0][input_len:],
        skip_special_tokens=True
    )

    # 过滤 <think> 标签
    response = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL).strip()

    return response


# ============================================================
# vLLM 通信 (VLLM Mode)
# ============================================================
async def generate_vllm(messages: list, max_tokens: int = 512,
                        temperature: float = 0.7, top_p: float = 0.9,
                        top_k: int = 50) -> str:
    """通过 vLLM API 生成回复"""
    global vllm_client

    if vllm_client is None:
        vllm_client = httpx.AsyncClient(timeout=120.0)

    payload = {
        "model": "firefly-assistant",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    try:
        response = await vllm_client.post(
            f"{VLLM_BASE_URL}/v1/chat/completions", json=payload
        )
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f"vLLM 错误: {response.status_code}")
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL).strip()
        return content
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail=f"无法连接 vLLM ({VLLM_BASE_URL})")


# ============================================================
# API 端点
# ============================================================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        mode=INFERENCE_MODE,
        model_loaded=(firefly_model is not None),
        rag_available=(char_coll is not None or world_coll is not None),
    )


@app.get("/api/character", response_model=CharacterInfo)
async def get_character_info():
    return CharacterInfo(
        name="流萤",
        title="星核猎手 · 前格拉默铁骑 AR-26710",
        faction="星核猎手",
        element="火",
        path="毁灭",
        rarity="★★★★★",
        description="身着机械装甲「萨姆」战斗的少女，身患失熵症，生命短暂却依然渴望以「流萤」的身份活下去。温柔而坚定，如萤火虫一般在有限的黑夜中发出最耀眼的光芒。",
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI 兼容的聊天完成端点
    集成 RAG 检索 + OOC 校验 + 模型推理
    支持 SSE 流式响应 (stream: true)
    """
    import time as time_module
    request_id = str(int(time_module.time() * 1000))[-8:]

    # 1. 构建 system prompt
    system_content = FIREFLY_SYSTEM_PROMPT

    # 2. RAG 检索
    if request.use_rag:
        last_user_msg = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                last_user_msg = msg.content
                break
        if last_user_msg:
            context = retrieve_context(last_user_msg)
            if context:
                system_content += "\n\n以下是与当前问题相关的角色资料（请基于此回答，保持角色口吻）：\n\n" + context

    # 3. 构建完整消息
    gen_messages = [{"role": "system", "content": system_content}]
    for msg in request.messages:
        gen_messages.append({"role": msg.role, "content": msg.content})

    # 4. 流式生成
    if request.stream:
        async def stream_generator():
            try:
                if INFERENCE_MODE == "direct" and firefly_model is not None:
                    import torch
                    from transformers import TextIteratorStreamer
                    from threading import Thread

                    inputs = firefly_tokenizer.apply_chat_template(
                        gen_messages, tokenize=True, return_tensors="pt",
                        add_generation_prompt=True, enable_thinking=False,
                    ).to(firefly_model.device)

                    streamer = TextIteratorStreamer(
                        firefly_tokenizer, skip_prompt=True,
                        skip_special_tokens=True,
                    )

                    gen_kwargs = dict(
                        inputs=inputs,
                        max_new_tokens=request.max_tokens,
                        do_sample=True,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        top_k=request.top_k,
                        repetition_penalty=1.1,
                        streamer=streamer,
                        pad_token_id=firefly_tokenizer.pad_token_id,
                    )
                    thread = Thread(target=firefly_model.generate, kwargs=gen_kwargs)
                    thread.start()

                    full_response = ""
                    for text in streamer:
                        text = re.sub(r'<think>.*?</think>\s*', '', text,
                                    flags=re.DOTALL)
                        full_response += text
                        yield {"data": json.dumps({
                            "choices": [{"delta": {"content": text}, "index": 0}],
                            "model": request.model,
                        })}

                    yield {"data": json.dumps({
                        "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
                        "model": request.model,
                    })}

                    # 日志记录
                    log_chat_request(request_id, gen_messages[-1]["content"],
                                   full_response, request.use_rag,
                                   generation_time_ms=0)
                else:
                    yield {"data": json.dumps({
                        "error": "Streaming only supported in direct mode"
                    })}
            except Exception as e:
                log_error(request_id, e, {"mode": INFERENCE_MODE})
                yield {"data": json.dumps({"error": str(e)})}

        return EventSourceResponse(stream_generator())

    # 5. 非流式生成
    import time as time_module
    start_time = time_module.time()

    if INFERENCE_MODE == "direct":
        if firefly_model is None:
            raise HTTPException(status_code=503, detail="模型未加载")
        content = generate_direct(
            gen_messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
        )
    else:
        content = await generate_vllm(
            gen_messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
        )

    gen_time = (time_module.time() - start_time) * 1000

    # 6. OOC 校验（记录但不阻止回复）
    validation_score = None
    if request.enable_validation:
        try:
            from validator import FireflyResponseValidator
            validation = FireflyResponseValidator.validate(content)
            validation_score = validation.get('score', None)
            if validation.get('issues'):
                logger.warning(f"[Validator] OOC warning (score={validation_score})")
        except Exception:
            pass

    # 结构�日志
    log_chat_request(
        request_id,
        gen_messages[-1]["content"] if gen_messages else "",
        content,
        request.use_rag,
        validation_score,
        int(gen_time),
    )

    # 7. 返回
    return ChatCompletionResponse(
        id="firefly-chat",
        object="chat.completion",
        created=int(time_module.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
    )


# ============================================================
# 会话管理端点
# ============================================================
from backend.chat_store import ChatStore

@app.get("/v1/sessions")
async def list_sessions():
    """列出所有会话"""
    return {"sessions": ChatStore.list_sessions()}

@app.post("/v1/sessions")
async def create_session(title: str = "新对话"):
    """创建新会话"""
    session = ChatStore.create_session(title)
    return session

@app.get("/v1/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    """获取会话消息"""
    session = ChatStore.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {
        "session": session,
        "messages": ChatStore.get_messages(session_id),
    }

@app.delete("/v1/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    ChatStore.delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}


# ============================================================
# 用户反馈端点
# ============================================================
class FeedbackRequest(BaseModel):
    message_id: int
    rating: str  # "like" or "dislike"
    comment: str = ""

@app.post("/v1/feedback")
async def submit_feedback(fb: FeedbackRequest):
    """提交用户反馈"""
    if fb.rating not in ("like", "dislike"):
        raise HTTPException(status_code=400, detail="rating must be 'like' or 'dislike'")
    ChatStore.record_feedback(fb.message_id, fb.rating, fb.comment)
    logger.info(f"Feedback: message_id={fb.message_id}, rating={fb.rating}")
    return {"status": "recorded"}

@app.get("/v1/feedback/stats")
async def get_feedback_stats():
    """获取反馈统计"""
    return ChatStore.get_feedback_stats()


# ============================================================
# 静态文件挂载 (前端)
# ============================================================
from fastapi.staticfiles import StaticFiles
webui_dir = PROJECT_ROOT / "webui"
if webui_dir.exists():
    app.mount("/", StaticFiles(directory=str(webui_dir), html=True), name="static")
    print(f"[Static] 前端已挂载: {webui_dir}")


# ============================================================
# 启动
# ============================================================
def main():
    global INFERENCE_MODE, VLLM_BASE_URL

    parser = argparse.ArgumentParser(description="流萤角色助手后端API")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=7860, help="端口号")
    parser.add_argument("--direct", action="store_true", default=True, help="直接加载模型推理 (默认)")
    parser.add_argument("--vllm", default=None, help="使用 vLLM API (例: --vllm http://localhost:8000)")
    parser.add_argument("--no-rag", action="store_true", help="禁用RAG")
    parser.add_argument("--no-model", action="store_true", help="不加载模型 (仅测试API)")
    args = parser.parse_args()

    # 推理模式
    if args.vllm:
        INFERENCE_MODE = "vllm"
        VLLM_BASE_URL = args.vllm
    else:
        INFERENCE_MODE = "direct"

    print("=" * 60)
    print("  流萤角色助手 API 服务器")
    print("=" * 60)
    print(f"  模式: {INFERENCE_MODE}")
    print(f"  地址: http://{args.host}:{args.port}")
    print(f"  文档: http://{args.host}:{args.port}/docs")
    print("=" * 60)

    # 初始化 RAG
    if not args.no_rag:
        init_rag()
    else:
        print("[RAG] 已禁用")

    # 加载模型 (Direct Mode)
    if INFERENCE_MODE == "direct" and not args.no_model:
        load_model_direct()

    print(f"\n[Server] 启动中...")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
