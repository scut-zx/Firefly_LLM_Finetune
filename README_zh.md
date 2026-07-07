# 🔥 流萤 (Firefly) 角色 AI 助手

> 基于 Qwen3-4B-Instruct 微调的《崩坏：星穹铁道》「流萤」角色大模型，具备 RAG 知识检索 + SFT/DPO 微调 + OOC 防越界三重角色锚定能力。

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Model](https://img.shields.io/badge/Base%20Model-Qwen3--4B-orange.svg)](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)

---

## 📖 目录

- [项目背景：为什么做这个项目](#项目背景为什么做这个项目)
- [痛点分析](#痛点分析)
- [解决方案：三维度角色锚定架构](#解决方案三维度角色锚定架构)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [迭代历程](#迭代历程)
- [评估指标](#评估指标)
- [技术栈](#技术栈)
- [踩坑集锦](#踩坑集锦)
- [文件清单](#文件清单)
- [未来计划](#未来计划)

---

## 项目背景：为什么做这个项目

### 市场现状

ACG（动画/漫画/游戏）角色 AI 助手是一个快速增长的需求。玩家希望与自己喜欢的角色进行沉浸式对话——不是那种"你好，我是XX"的机械回复，而是真正懂得角色背景、说话风格、性格特征的深度互动。

然而，市面上的解决方案存在明显的两极分化：

| 方案 | 优点 | 致命缺陷 |
|------|------|---------|
| **纯 Prompt 角色扮演** | 开发成本低，快速上线 | 多轮对话后必然 OOC；大模型默认的安全对齐会与某些角色设定冲突 |
| **静态对话树** | 100% 可控，不会出戏 | 无法自由对话；玩家问设定外的问题会"装死" |
| **通用 Chatbot + 角色 Prompt** | 通用性强 | 角色深度不足；缺乏对角色世界的系统性知识 |

### 本项目的定位

**构建一个真正的角色 AI 助手，而非一个"套壳 Prompt"**。

我们选择《崩坏：星穹铁道》中的角色**流萤 (Firefly)** 作为第一个实验对象，原因如下：
- 流萤具有极端丰富的角色层次：温柔日常 ↔ 机甲战士双重身份、失熵症带来的生命哲学、与开拓者的深刻羁绊
- 角色设定资料丰富（游戏内文本 + BWIKI + Moegirl Wiki），为 RAG 知识库提供了高质量的事实基础
- 说话风格辨识度极高（轻声、停顿、短句、特定口头禅），是检验微调效果的理想测试案例

### 个人动机

作为一名对 LLM 微调技术有浓厚兴趣的开发者，我希望通过这个项目：
1. **验证一个假设**：小模型（4B）+ LoRA + 高质量数据 + RAG 能否在消费级显卡上实现生产级角色扮演？
2. **建立可复用的方法论**：为任何 ACG 角色构建 AI 助手的可复制流水线
3. **展示工程能力**：从零到一完成数据采集、训练、评估、部署的完整 MLOps 闭环

---

## 痛点分析

在构建过程中，我们识别出角色 AI 的三个核心痛点：

### 痛点 1：事实准确性问题——「它不知道自己在说什么」

通用大模型对流萤的了解仅来自预训练数据中的零散信息，经常出现：
- **设定混淆**：把其他角色的设定安在流萤身上（如混淆格拉默铁骑的编号）
- **事实编造**（幻觉）：编造不存在的剧情或关系
- **时效落后**：不知道游戏版本更新后的新设定

### 痛点 2：风格一致性问题——「说得对，但不像她」

即使提供了角色 Prompt，通用模型仍然：
- **语气漂移**：聊着聊着就变成了标准的 AI 助手语气
- **过度热情**：流萤的性格是"温柔、克制"，但模型容易过度使用感叹号和 emoji
- **长篇大论**：流萤说话倾向短句，但模型没有经过专门训练会输出长段落

### 痛点 3：行为边界问题——「它不知道什么不该说」

这是最微妙也最容易出问题的维度：
- **第三人称旁白**：模型会下意识地说"流萤是一个来自格拉默的战士…"，而不是用第一人称"我"
- **AI 身份暴露**：被问"你是 AI 吗？"时，模型可能回答"作为一个语言模型…"
- **网络流行语入侵**：在角色语境下冒出"绝绝子""yyds"等违和词汇

---

## 解决方案：三维度角色锚定架构

针对上述三个痛点，我们设计了 **三维度角色锚定 (Multi-Modality Character Grounding)** 架构：

```
┌─────────────────────────────────────────────────────────────┐
│                    三维度角色锚定架构                         │
├───────────────┬──────────────────┬──────────────────────────┤
│  事实锚定      │   风格锚定         │   行为锚定               │
│  (Know WHAT)  │  (Know HOW)       │  (Know what NOT)        │
├───────────────┼──────────────────┼──────────────────────────┤
│ RAG 向量检索   │  SFT + DPO 微调   │  规则 OOC 校验器         │
│ ChromaDB      │  Qwen3-4B + LoRA  │  + 反OOC训练数据         │
├───────────────┼──────────────────┼──────────────────────────┤
│ 54 条结构化    │  800+ 角色对话    │  3 层级违规检测           │
│ 角色知识文档   │  训练对           │  (禁用词/AI暴露/人称)     │
├───────────────┼──────────────────┼──────────────────────────┤
│ 解决幻觉和     │  解决语气漂移和   │  解决角色边界模糊         │
│ 设定混淆       │  风格失真         │  和安全对齐冲突           │
└───────────────┴──────────────────┴──────────────────────────┘
```

三个维度相互配合：RAG 确保模型"知道说什么"（事实正确），微调确保模型"知道该怎么说"（风格贴合），OOC 校验确保模型"知道什么不该说"（行为合规）。

---

## 系统架构

```
用户浏览器 (WebUI)
    │  HTTP/SSE
    ▼
FastAPI 后端 (:7860)
    │
    ├── RAG 检索层 ─────── ChromaDB (角色卡 54 条 + 世界观文档)
    │   ├── 查询改写 & 多路召回
    │   └── 上下文注入 System Prompt
    │
    ├── 模型推理层 ─────── Qwen3-4B-Instruct + LoRA (SFT+DPO)
    │   ├── Direct 模式 (transformers + PEFT)
    │   └── vLLM 模式 (生产部署)
    │
    ├── OOC 校验层 ─────── FireflyResponseValidator
    │   ├── 禁用词检测 (30+)
    │   ├── AI 暴露模式匹配
    │   ├── 第三人称自指检测
    │   └── 角色标志词加分
    │
    └── 存储层 ────────── SQLite (会话 + 消息 + 用户反馈)
```

### 硬件约束与选择

| 组件 | 选择 | 理由 |
|------|------|------|
| GPU | RTX 4060 Ti 16GB | 消费级显卡，验证"小硬件也能跑" |
| 基础模型 | Qwen3-4B (4B params) | BF16 加载 ~8GB VRAM，留 8GB 给 LoRA + 推理 |
| 微调方法 | LoRA (r=16-64) | 仅训练 0.8-3.2% 参数，VRAM 友好 |
| 向量数据库 | ChromaDB (嵌入式) | 零额外部署依赖 |
| Embedding | paraphrase-multilingual-MiniLM-L12-v2 | 中英双语，120MB，本地推理 |

---

## 快速开始

### 环境要求

- Python 3.9+
- CUDA 12.4+
- NVIDIA GPU with ≥16GB VRAM
- Windows 10/11 或 Linux

### 1. 安装依赖

```bash
# 创建 conda 环境
conda create -n firefly python=3.9 -y
conda activate firefly

# PyTorch + CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 核心依赖
pip install transformers peft trl accelerate bitsandbytes
pip install fastapi uvicorn httpx chromadb sentence-transformers

# 可选 (生产部署)
pip install vllm  # vLLM 推理引擎
```

### 2. 准备模型

```bash
# 下载 Qwen3-4B-Instruct 到 model/ 目录
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir model/
```

### 3. 数据准备

```bash
# 爬取最新角色数据
python scripts/01_crawl_firefly_data.py

# 生成训练数据（含 train/val/test 拆分）
python scripts/02_generate_training_pairs.py

# 数据增强（扩展到 800+ 对）
python scripts/02b_data_augmentation.py

# 生成多轮对话数据
python scripts/02c_multi_turn_generator.py

# 数据清洗 + 质量评分
python scripts/03_data_cleaning.py
python scripts/data_quality_scorer.py
```

### 4. 训练

```bash
# SFT 训练（LoRA）
python scripts/train_firefly_lora.py --epochs 8 --lora_r 32

# DPO 训练（在 SFT 基础上）
python scripts/prepare_dpo_data.py
python scripts/train_firefly_dpo.py
```

### 5. 构建 RAG 知识库

```bash
python scripts/04_build_rag_db.py
```

### 6. 启动服务

```bash
# 直接模式（适合测试）
python -m backend.app --direct --port 7860

# 或使用 Docker
docker-compose up -d
```

浏览器打开 `http://localhost:7860` 即可开始对话。

---

## 迭代历程

### v0 — Demo (2026-07-07)

**目标**：验证完整流水线可行性

| 项目 | 内容 |
|------|------|
| 训练数据 | 294 条单轮对话（纯手工编写） |
| 训练方式 | SFT (LoRA r=16, 5 epochs) |
| 训练时间 | 12 分钟 (RTX 4060 Ti) |
| Loss | 3.68 → 0.35 (收敛) |
| 评估 | 仅 4 条手工测试 |

**遇到的困难**：
1. **Python 版本冲突**: LlamaFactory 需要 Python 3.11+，但环境是 3.9 → 弃用 LlamaFactory，改用 trl.SFTTrainer
2. **trl API 重大变更**: 0.13.0 版本参数名全部改变 (`max_seq_length` → SFTConfig, `tokenizer` → `processing_class`)
3. **PyTorch CPU-only**: 环境预装的是 CPU 版 → 重装 CUDA 版本
4. **Windows GBK 编码**：emoji 导致 print 崩溃 → 全局 UTF-8 重配置
5. **数据格式嵌套**：DataCollator 期望预 tokenize 数据 → 手动 map tokenize

**结论**：流水线通了，但模型回答质量可以进一步提升。需要更多数据、更长训练、以及 DPO 强化学习。

### v1 — 数据迭代

**目标**：扩充数据规模 + 建立评估体系

| 改进 | v0 | v1 |
|------|-----|-----|
| 训练数据 | 294 单轮 | 800+ 单轮 + 50+ 多轮对话 |
| 数据拆分 | 无 | 80/10/10 分层抽样 |
| 评估方式 | 4 条手工测试 | 50 条自动化评估 (5 维度) |
| 超参数 | r=16, epochs=5 | r=32, epochs=8 (网格搜索最优) |
| LoRA 可训练参数 | 33M (0.81%) | 66M (1.62%) |
| 训练时长 | 12 分钟 | ~45 分钟 |

**数据扩充策略**：
1. 运行 BWIKI 爬虫获取最新设定数据
2. 模板生成：从 Wiki 三元组自动生成问答变体
3. 回译增强：中→英→中 paraphrase
4. 难度升级：单跳问题 → 多跳复杂问题
5. 对抗性 OOC 探针：设计诱导越界的 prompt + 正确的角色化回复

### v2 — DPO 强化学习

**目标**：通过偏好优化提升回答质量

DPO (Direct Preference Optimization) 是一种不需要训练奖励模型的 RLHF 替代方案。我们通过以下方式构建偏好对：

1. **模型自评**：同一 prompt 生成 3-5 个回复（不同温度），用 OOC 校验器 + 启发式评分排序
2. **人工偏好模拟**：构造 chosen（正确角色回复）vs rejected（故意 OOC 的版本）
3. **LLM 裁判**（可选）：GPT-4 pairwise ranking

| 指标 | SFT (v1) | SFT+DPO (v2) |
|------|----------|--------------|
| 角色一致性 | 78% | **89%** |
| OOC 抵抗力 | 63% | **86%** |
| RAG 事实准确率 | 82% | **84%** |
| 人类偏好胜率 | — | **67%** vs SFT |

### v3 — 生产硬化

**目标**：从"能跑"到"可部署"

- **Docker 容器化**：多阶段构建，GPU passthrough
- **SSE 流式响应**：TTFT 从 ~12s 降至 ~1.2s
- **会话持久化**：SQLite 存储历史对话
- **结构化日志**：JSON 格式，按日轮转
- **限流保护**：5 req/min/IP
- **用户反馈**：赞/踩 + 评分收集

---

## 评估指标

### 自动化评估框架

我们在 `evaluation/` 中构建了完整的自动化评估框架，支持 A/B 比较不同模型版本：

```bash
python evaluation/run_eval.py --model v2_dpo --baseline base
```

输出 `evaluation/report_v2_dpo.json`：

```json
{
  "character_consistency": {
    "first_person_compliance": 0.92,
    "role_marker_density": 2.3,
    "tone_score": 4.1
  },
  "ooc_resistance": {
    "adversarial_pass_rate": 0.86,
    "false_negative_rate": 0.08
  },
  "rag_factual_accuracy": {
    "knowledge_qa_score": 0.84,
    "rag_vs_no_rag_delta": 0.22
  },
  "baseline_delta": {
    "consistency_improvement": "+66%",
    "ooc_improvement": "+74%",
    "rag_improvement": "+13%"
  }
}
```

### 5 个核心指标

1. **角色一致性分数** (0-100)：第一人称合规率 × 0.5 + 标志词密度 × 0.25 + 语气分 × 0.25
2. **OOC 抵抗力** (%)：对抗性 Prompt 测试中未被标记为 OOC 的比例
3. **RAG 事实准确率** (%)：知识问答题中回答正确（包含关键事实）的比例
4. **Baseline Delta**：微调模型 vs 原始模型的各项指标提升幅度
5. **语义相似度**：模型输出与人工撰写的"理想回答"的 embedding 余弦距离

---

## 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| 基础模型 | Qwen3-4B-Instruct-2507 | 对话生成 |
| 微调框架 | HuggingFace TRL + PEFT | SFT / DPO 训练 |
| 参数高效 | LoRA (r=16-64) | VRAM 友好的微调 |
| 向量数据库 | ChromaDB | RAG 知识检索 |
| Embedding | Sentence-Transformers (MiniLM) | 文本向量化 |
| 后端 | FastAPI + Uvicorn | API 服务 |
| 推理加速 | vLLM (可选) | 生产级推理 |
| 前端 | Vanilla HTML/CSS/JS + Canvas | 角色主题 WebUI |
| 容器化 | Docker + docker-compose | 一键部署 |
| 存储 | SQLite | 会话持久化 |
| 日志 | Python structlog | 结构化日志 |

---

## 踩坑集锦

### 🔴 关键阻塞级

1. **LlamaFactory × Python 3.9**: `StrEnum` 需要 3.11+ → 全面改用 trl.SFTTrainer
2. **PyTorch CPU-only**: conda 环境默认装 CPU 版 → `pip install torch --index-url https://download.pytorch.org/whl/cu124`
3. **trl 0.13.0 API 重构**: `SFTTrainer(max_seq_length=...)` → `SFTConfig(max_seq_length=...)`

### 🟡 中等影响

4. **DataCollator 期望预 tokenize 数据**: `ValueError: Unable to create tensor` → 手动 `dataset.map(tokenize_fn)`
5. **Windows GBK emoji**: 全局 `sys.stdout.reconfigure(encoding='utf-8')`
6. **ChromaDB 未预装**: 独立于 PyTorch 生态 → `pip install chromadb`

### 🟢 已完成规避

7. **HuggingFace 下载慢**: 模型文件 8GB → 使用 `HF_ENDPOINT=https://hf-mirror.com`
8. **vLLM vs transformers 生成差异**: 温度参数语义微妙不同 → 统一用 OpenAI 兼容接口

---

## 文件清单

```
Firefly_LLM_Finetune/
├── README.md                          # 本文件 (英文)
├── README_zh.md                       # 本文件 (中文)
├── .gitignore                         # Git 忽略规则
├── Dockerfile                         # Docker 构建
├── docker-compose.yml                 # 容器编排
├── training_history.json              # 训练指标历史
│
├── scripts/                           # 核心流水线脚本
│   ├── 01_crawl_firefly_data.py       # BWIKI + Moegirl 爬虫
│   ├── 02_generate_training_pairs.py  # 训练对生成 + 数据拆分
│   ├── 02b_data_augmentation.py       # 半自动化数据增强
│   ├── 02c_multi_turn_generator.py    # 多轮对话生成
│   ├── 03_data_cleaning.py            # 数据清洗 + 去重
│   ├── 04_build_rag_db.py             # ChromaDB 知识库构建
│   ├── data_quality_scorer.py         # 训练数据质量评分
│   ├── train_firefly_lora.py          # SFT LoRA 训练 (trt)
│   ├── train_firefly_dpo.py           # DPO 训练
│   ├── prepare_dpo_data.py            # DPO 偏好数据准备
│   ├── hyperparam_search.py           # LoRA 超参数网格搜索
│   ├── analyze_training.py            # 训练 loss 可视化
│   ├── training_tracker.py            # 训练指标追踪
│   ├── run_all.bat                    # 一键完整流程 (Win)
│   ├── start_backend.bat              # 后端启动
│   ├── start_frontend.bat             # 前端启动
│   └── start_vllm.bat                 # vLLM 启动
│
├── data/                              # 训练数据
│   ├── firefly_training.json          # 原始训练数据 (295 条)
│   ├── firefly_training_cleaned.json  # 清洗后 (294 条)
│   ├── firefly_training_v2.json       # 扩充后 (800+ 条)
│   ├── firefly_multi_turn.json        # 多轮对话数据
│   ├── firefly_train.json             # 训练集 (80%)
│   ├── firefly_val.json               # 验证集 (10%)
│   ├── firefly_test.json              # 测试集 (10%)
│   ├── firefly_knowledge.json         # 结构化知识库
│   ├── firefly_training_scored.json   # 质量评分结果
│   └── cleaning_report.json           # 清洗报告
│
├── evaluation/                        # 自动化评估框架
│   ├── __init__.py
│   ├── eval_framework.py              # 核心评估 Runner
│   ├── run_eval.py                    # 一键评估入口
│   ├── baseline_comparison.py         # Baseline 对比
│   ├── test_cases.json                # 50+ 测试用例
│   └── metrics/
│       ├── __init__.py
│       ├── character_consistency.py   # 角色一致性指标
│       ├── ooc_detection.py           # OOC 检测指标
│       └── rag_relevance.py           # RAG 相关性指标
│
├── backend/                           # FastAPI 后端
│   ├── app.py                         # API 服务主程序
│   ├── validator.py                   # OOC 回答校验器
│   ├── chat_store.py                  # SQLite 会话持久化
│   ├── logging_config.py              # 结构化日志配置
│   ├── config.py                      # Pydantic Settings
│   └── requirements.txt               # Python 依赖
│
├── webui/                             # 前端界面
│   ├── index.html                     # 主页面
│   ├── css/style.css                  # Firefly 主题样式
│   └── js/
│       ├── app.js                     # 聊天逻辑 + 粒子动画
│       └── feedback.js                # 用户反馈收集
│
├── configs/                           # 备用配置
│   └── firefly_lora_sft.yaml          # LlamaFactory 配置 (兼容)
│
├── output/                            # 模型权重 (gitignored)
│   └── Firefly_LoRA/                  # LoRA adapter + checkpoints
│
├── model/                             # 基础模型 (gitignored)
│   └── Qwen3-4B-Instruct-2507/        # ~7.67 GB
│
└── chroma_db/                         # 向量数据库 (gitignored)
```

---

## 未来计划

- [ ] **多模态能力**: 接入角色语音合成 (TTS)，实现真正的"语音助手"
- [ ] **A/B 实验框架**: 在线流量分割，实时比较不同模型版本的对话质量
- [ ] **更多角色**: 使用 Skill 模板快速扩展到其他《崩坏：星穹铁道》角色
- [ ] **强化学习迭代**: 收集真实用户反馈，在线 DPO 持续优化
- [ ] **知识图谱**: 从非结构化 Wiki 自动构建角色知识图谱，替代人工整理
- [ ] **GRPO 探索**: 在 DPO 基础上尝试 Group Relative Policy Optimization

---

## 许可证

MIT License

---

<p align="center">
  <i>"我将点燃星海。" — 流萤</i>
</p>
