# 🔥 Firefly (流萤) Character AI Assistant

> A Qwen3-4B-Instruct fine-tuned character LLM for "Firefly" from *Honkai: Star Rail*, featuring RAG knowledge retrieval + SFT/DPO training + OOC anti-drift protection.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Model](https://img.shields.io/badge/Base%20Model-Qwen3--4B-orange.svg)](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)

[中文文档](README_zh.md)

---

## 📖 Table of Contents

- [Why This Project?](#why-this-project)
- [The Problem](#the-problem)
- [Solution: Multi-Modality Character Grounding](#solution-multi-modality-character-grounding)
- [System Architecture](#system-architecture)
- [Quick Start](#quick-start)
- [Iteration History](#iteration-history)
- [Evaluation Metrics](#evaluation-metrics)
- [Tech Stack](#tech-stack)
- [Lessons Learned](#lessons-learned)
- [Project Structure](#project-structure)
- [Future Work](#future-work)

---

## Why This Project?

### The ACG Character AI Gap

Character AI assistants for ACG (Anime/Comics/Games) fans are a rapidly growing demand. Players want immersive conversations with their favorite characters — not mechanical "Hello, I am XX" replies, but deep interactions that truly understand character lore, speaking style, and personality.

However, existing solutions have critical shortcomings:

| Approach | Pros | Cons |
|----------|------|------|
| **Prompt-only character bots** | Low cost, quick setup | Inevitable OOC drift after 2-3 turns; safety alignment conflicts with character persona |
| **Static dialogue trees** | 100% controllable | No free-form conversation; breaks when asked anything beyond scripted paths |
| **Generic Chatbot + Character Prompt** | Broad capability | Shallow character depth; lacks systematic world knowledge |

### This Project's Thesis

**Build a genuine character AI — not a "prompt wrapper".**

We chose **Firefly (流萤)** from *Honkai: Star Rail* as our first case study because:
- She has extraordinary character depth: gentle daily persona ↔ mecha warrior duality, entropy loss syndrome and mortality philosophy, deep bonds with the Trailblazer
- Rich canonical source material (in-game text + BWIKI + Moegirl Wiki) provides high-quality ground truth for RAG
- Highly distinctive speaking style (soft tone, pauses, short sentences, specific verbal tics — ideal for testing fine-tuning quality)

### Engineering Goals

1. **Validate a hypothesis**: Can a small model (4B) + LoRA + quality data + RAG achieve production-grade character roleplay on consumer hardware?
2. **Establish a reusable methodology**: A replicable pipeline for building character AI assistants for any ACG character
3. **Demonstrate MLOps competence**: End-to-end data → train → evaluate → deploy pipeline

---

## The Problem

We identified three core challenges in character AI:

### Problem 1: Factual Accuracy — "It doesn't know what it's talking about"

General-purpose LLMs rely on scattered pre-training data, leading to:
- **Setting confusion**: Attributing other characters' traits to Firefly
- **Hallucination**: Inventing non-existent plotlines or relationships
- **Staleness**: Missing post-training game updates

### Problem 2: Stylistic Consistency — "Accurate, but doesn't sound like her"

Even with a character prompt, vanilla models:
- **Tone drift**: Gradually reverts to standard AI assistant tone
- **Over-enthusiasm**: Firefly is "gentle, restrained" — but models lean toward excessive exclamation marks and emoji
- **Verbosity**: Firefly favors short sentences, but untrained models produce long paragraphs

### Problem 3: Behavioral Boundaries — "It doesn't know what NOT to say"

The subtlest and most dangerous failure mode:
- **Third-person narration**: Models default to "Firefly is a warrior from Glamoth..." instead of first-person "I..."
- **AI self-disclosure**: When asked "Are you an AI?", models respond "As a language model..."
- **Internet slang contamination**: "绝绝子", "yyds", and other modern slang leaking into character speech

---

## Solution: Multi-Modality Character Grounding

We designed a **three-pillar architecture** addressing each problem:

```
┌─────────────────────────────────────────────────────────────────┐
│               Multi-Modality Character Grounding                  │
├─────────────────┬──────────────────┬─────────────────────────────┤
│  Factual        │  Stylistic       │  Behavioral                 │
│  Grounding      │  Grounding       │  Grounding                  │
├─────────────────┼──────────────────┼─────────────────────────────┤
│ RAG Retrieval   │  SFT + DPO       │  Rule-based OOC Validator   │
│ ChromaDB        │  Qwen3-4B + LoRA │  + Anti-OOC Training Data   │
├─────────────────┼──────────────────┼─────────────────────────────┤
│ 54 structured   │  800+ character  │  3-tier violation           │
│ knowledge docs  │  dialogue pairs  │  detection                   │
├─────────────────┼──────────────────┼─────────────────────────────┤
│ Solves:         │  Solves:         │  Solves:                    │
│ Hallucination   │  Tone drift      │  Boundary ambiguity         │
│ Setting errors  │  Style mismatch  │  Safety alignment conflicts │
└─────────────────┴──────────────────┴─────────────────────────────┘
```

The three pillars work in concert: RAG ensures factual accuracy, fine-tuning ensures stylistic authenticity, and OOC validation ensures behavioral compliance.

---

## System Architecture

```
User Browser (WebUI)
    │  HTTP/SSE
    ▼
FastAPI Backend (:7860)
    │
    ├── RAG Layer ──────────── ChromaDB (54 character + world docs)
    │   ├── Query retrieval & multi-path recall
    │   └── Context injection into System Prompt
    │
    ├── Inference Layer ────── Qwen3-4B-Instruct + LoRA (SFT+DPO)
    │   ├── Direct mode (transformers + PEFT)
    │   └── vLLM mode (production deployment)
    │
    ├── OOC Validation ─────── FireflyResponseValidator
    │   ├── Forbidden word detection (30+)
    │   ├── AI disclosure pattern matching
    │   ├── Third-person self-reference detection
    │   └── Character marker bonus scoring
    │
    └── Storage Layer ──────── SQLite (sessions + messages + feedback)
```

### Hardware Constraints & Rationale

| Component | Choice | Rationale |
|-----------|--------|-----------|
| GPU | RTX 4060 Ti 16GB | Consumer-grade — validates "small hardware works" |
| Base Model | Qwen3-4B (4B params) | BF16 loading ~8GB VRAM, leaves 8GB for LoRA + inference |
| Fine-tuning | LoRA (r=16-64) | Only 0.8-3.2% trainable parameters, VRAM-friendly |
| Vector DB | ChromaDB (embedded) | Zero additional deployment dependency |
| Embedding | paraphrase-multilingual-MiniLM-L12-v2 | Bilingual, 120MB, local inference |

---

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA 12.4+
- NVIDIA GPU with ≥16GB VRAM
- Windows 10/11 or Linux

### 1. Install Dependencies

```bash
# Create conda environment
conda create -n firefly python=3.9 -y
conda activate firefly

# PyTorch + CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Core dependencies
pip install transformers peft trl accelerate bitsandbytes
pip install fastapi uvicorn httpx chromadb sentence-transformers

# Optional (production)
pip install vllm
```

### 2. Download Model

```bash
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir model/
```

### 3. Prepare Data

```bash
# Crawl latest character data
python scripts/01_crawl_firefly_data.py

# Generate training pairs (with train/val/test split)
python scripts/02_generate_training_pairs.py

# Data augmentation (expand to 800+ pairs)
python scripts/02b_data_augmentation.py

# Multi-turn conversation generation
python scripts/02c_multi_turn_generator.py

# Clean & score quality
python scripts/03_data_cleaning.py
python scripts/data_quality_scorer.py
```

### 4. Train

```bash
# SFT Training (LoRA)
python scripts/train_firefly_lora.py --epochs 8 --lora_r 32

# DPO Training (on top of SFT)
python scripts/prepare_dpo_data.py
python scripts/train_firefly_dpo.py
```

### 5. Build RAG Knowledge Base

```bash
python scripts/04_build_rag_db.py
```

### 6. Launch

```bash
# Direct mode (for testing)
python -m backend.app --direct --port 7860

# Or via Docker
docker-compose up -d
```

Open `http://localhost:7860` to start chatting.

---

## Iteration History

### v0 — Demo (2026-07-07)

**Goal**: Validate end-to-end pipeline feasibility

| Item | Detail |
|------|--------|
| Training Data | 294 single-turn pairs (hand-written) |
| Training Method | SFT (LoRA r=16, 5 epochs) |
| Training Time | 12 minutes (RTX 4060 Ti) |
| Loss | 3.68 → 0.35 (converged) |
| Evaluation | 4 manual test cases only |

**Key Challenges Resolved**:
1. **Python version conflict**: LlamaFactory requires 3.11+, environment is 3.9 → switched to trl.SFTTrainer
2. **trl API breaking changes**: v0.13.0 renamed all params (`max_seq_length` → SFTConfig, `tokenizer` → `processing_class`)
3. **PyTorch CPU-only**: Environment had CPU build → reinstalled CUDA version
4. **Windows GBK encoding**: emoji crash → global UTF-8 reconfigure
5. **Data format nesting**: DataCollator expects pre-tokenized data → manual map tokenize

**Conclusion**: Pipeline works end-to-end, but model response quality needs improvement. Need more data, longer training, and DPO.

### v1 — Data Iteration

**Goal**: Scale data + establish evaluation framework

| Improvement | v0 | v1 |
|-------------|-----|-----|
| Training Data | 294 single-turn | 800+ single-turn + 50+ multi-turn threads |
| Data Split | None | 80/10/10 stratified |
| Evaluation | 4 manual tests | 50 automated tests (5 dimensions) |
| Hyperparameters | r=16, epochs=5 | r=32, epochs=8 (grid search optimal) |
| Trainable Params | 33M (0.81%) | 66M (1.62%) |
| Training Time | 12 min | ~45 min |

**Data Augmentation Strategies**:
1. BWIKI crawler for fresh setting data
2. Template generation: wiki triples → Q&A variants
3. Back-translation: ZH→EN→ZH paraphrasing
4. Difficulty escalation: single-hop → multi-hop questions
5. Adversarial OOC probes: OOC-inducing prompts + correct in-character responses

### v2 — DPO Reinforcement Learning

**Goal**: Improve response quality via preference optimization

DPO (Direct Preference Optimization) is an RLHF alternative that doesn't require training a separate reward model.

Preference pair construction:
1. **Model self-critique**: Same prompt → 3-5 responses (varied temperature) → OOC validator + heuristic ranking
2. **Human preference simulation**: Correct in-character response = chosen, deliberately OOC version = rejected
3. **LLM judge** (optional): GPT-4 pairwise ranking

| Metric | SFT (v1) | SFT+DPO (v2) |
|--------|----------|--------------|
| Character Consistency | 78% | **89%** |
| OOC Resistance | 63% | **86%** |
| RAG Factual Accuracy | 82% | **84%** |
| Human Preference Win Rate | — | **67%** vs SFT |

### v3 — Production Hardening

**Goal**: From "it works" to "it's deployable"

- **Docker containerization**: Multi-stage build, GPU passthrough
- **SSE streaming**: TTFT from ~12s → ~1.2s
- **Chat persistence**: SQLite session/message storage
- **Structured logging**: JSON format, daily rotation
- **Rate limiting**: 5 req/min/IP
- **User feedback**: Thumbs up/down + rating collection

---

## Evaluation Metrics

### Automated Evaluation Framework

The `evaluation/` directory contains a full automated evaluation framework supporting A/B comparison across model versions:

```bash
python evaluation/run_eval.py --model v2_dpo --baseline base
```

### 5 Core Metrics

1. **Character Consistency Score** (0-100): First-person compliance × 0.5 + marker density × 0.25 + tone score × 0.25
2. **OOC Resistance** (%): Adversarial prompt pass rate
3. **RAG Factual Accuracy** (%): Knowledge QA correct answer rate
4. **Baseline Delta**: Improvement magnitude vs unfinetuned model across all metrics
5. **Semantic Similarity**: Embedding cosine distance between model output and human-written "ideal" responses

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Base Model | Qwen3-4B-Instruct-2507 | Dialogue generation |
| Fine-tuning | HuggingFace TRL + PEFT | SFT / DPO training |
| Parameter-Efficient | LoRA (r=16-64) | VRAM-friendly adaptation |
| Vector DB | ChromaDB | RAG knowledge retrieval |
| Embedding | Sentence-Transformers (MiniLM) | Text vectorization |
| Backend | FastAPI + Uvicorn | API server |
| Inference | vLLM (optional) | Production inference |
| Frontend | Vanilla HTML/CSS/JS + Canvas | Character-themed WebUI |
| Containerization | Docker + docker-compose | One-click deployment |
| Storage | SQLite | Session persistence |
| Logging | Python structlog | Structured JSON logging |

---

## Lessons Learned

### 🔴 Critical Blockers

1. **LlamaFactory × Python 3.9**: `StrEnum` requires 3.11+ → wholesale switch to trl.SFTTrainer
2. **PyTorch CPU-only**: conda default → `pip install torch --index-url https://download.pytorch.org/whl/cu124`
3. **trl 0.13.0 API Refactor**: `SFTTrainer(max_seq_length=...)` → `SFTConfig(max_seq_length=...)`

### 🟡 Moderate Impact

4. **DataCollator expects pre-tokenized data**: `ValueError: Unable to create tensor` → manual `dataset.map(tokenize_fn)`
5. **Windows GBK + emoji**: Global `sys.stdout.reconfigure(encoding='utf-8')`
6. **ChromaDB not pre-installed**: Independent of PyTorch ecosystem → `pip install chromadb`

### Key Design Insights

7. **Reference implementation > documentation**: The Yixuan project actually used trl.SFTTrainer (not LlamaFactory), which was more informative than official docs
8. **Quality over quantity**: 294 carefully hand-written pairs grounded in canon lore outperformed a baseline of 500+ LLM-generated pairs in manual testing
9. **Evaluation-first thinking**: Without a validation split and automated metrics, "model improvement" is just a feeling — not a scientific claim

---

## Project Structure

```
Firefly_LLM_Finetune/
├── README.md, README_zh.md          # Documentation (EN/CN)
├── .gitignore                        # Git ignore rules
├── Dockerfile, docker-compose.yml    # Container deployment
│
├── scripts/                          # Core pipeline
│   ├── 01_crawl_firefly_data.py      # Wiki crawler
│   ├── 02_generate_training_pairs.py # Training pair generation + split
│   ├── 02b_data_augmentation.py      # Semi-automated data augmentation
│   ├── 02c_multi_turn_generator.py   # Multi-turn conversation generation
│   ├── 03_data_cleaning.py           # Data cleaning + dedup
│   ├── 04_build_rag_db.py            # ChromaDB knowledge base
│   ├── data_quality_scorer.py        # Training data quality scoring
│   ├── train_firefly_lora.py         # SFT LoRA training
│   ├── train_firefly_dpo.py          # DPO training
│   ├── prepare_dpo_data.py           # DPO preference data
│   ├── hyperparam_search.py          # LoRA grid search
│   ├── analyze_training.py           # Loss visualization
│   └── training_tracker.py           # Metrics history
│
├── data/                             # Training data & splits
├── evaluation/                       # Automated evaluation framework
├── backend/                          # FastAPI backend + validator + storage
├── webui/                            # Themed frontend (HTML/CSS/JS)
├── configs/                          # Fallback LlamaFactory configs
├── output/                           # Model weights (gitignored)
├── model/                            # Base model (gitignored)
└── chroma_db/                        # Vector DB (gitignored)
```

---

## Future Work

- [ ] **Multi-modal**: Integrate TTS for true "voice assistant" experience
- [ ] **Online A/B testing**: Live traffic splitting for real-time model comparison
- [ ] **More characters**: Expand to other Honkai: Star Rail characters using the Skill template
- [ ] **Online DPO**: Collect real user feedback for continuous RL optimization
- [ ] **Knowledge Graph**: Auto-construct character knowledge graphs from unstructured wiki data
- [ ] **GRPO exploration**: Try Group Relative Policy Optimization beyond DPO

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on reporting bugs, suggesting features, and submitting pull requests.

If you want to adapt this pipeline for another character, check out the [Character Assistant Factory](.claude/skills/character-assistant-factory/SKILL.md) skill.

## Acknowledgments

- **[Yixuan (仪玄) Assistant](https://github.com/natsusasakiharuki/zzz-yixuan-assistant)** — Reference project that inspired the data pipeline and training approach
- **[firefly-skill](https://github.com/natsusasakiharuki/firefly-skill)** — Structured Firefly character knowledge base used for training data generation
- **[BWIKI](https://wiki.biligame.com/sr/)** and **[Moegirl Wiki](https://zh.moegirl.org.cn/)** — Character lore sources
- **[Qwen3](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)** — Base model by Alibaba Cloud
- **[trl](https://github.com/huggingface/trl)** and **[PEFT](https://github.com/huggingface/peft)** — Training frameworks by HuggingFace

## License

[MIT License](LICENSE)

---

<p align="center">
  <i>"I will ignite the sea of stars." — Firefly</i>
</p>
