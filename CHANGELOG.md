# Changelog

All notable changes to the Firefly Character AI Assistant project.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.4.0] - 2026-07-08 — Production Hardening

### Added
- Docker multi-stage build & docker-compose for one-click deployment
- SSE streaming responses (`TextIteratorStreamer`) for token-by-token output
- SQLite chat persistence with multi-session support (`backend/chat_store.py`)
- Structured JSON logging with daily rotation (`backend/logging_config.py`)
- Centralized Pydantic Settings configuration (`backend/config.py`)
- User feedback collection (thumbs up/down) via API and WebUI
- `.env.example` for environment variable configuration
- Health check endpoint (`GET /health`)
- Session management API endpoints

### Changed
- Expanded OOC validator forbidden words list (30+ new entries)
- Training script now accepts CLI arguments for all hyperparameters

### Fixed
- Backend rate limiter compatibility with Pydantic request models
- PyTorch CPU-only installation in firefly conda environment

---

## [0.3.0] - 2026-07-08 — DPO Preference Optimization

### Added
- DPO training script using `trl.DPOTrainer` (`scripts/train_firefly_dpo.py`)
- Preference pair preparation with 3 strategies (`scripts/prepare_dpo_data.py`):
  - Contrastive manual pairs (15)
  - Training data degradation pairs (120)
  - Model self-critique framework (100)
- SFT+DPO adapter merging for vLLM compatibility
- Training metrics history tracker (`scripts/training_tracker.py`)

### Changed
- Best model: SFT LoRA v2 (r=32, 643 pairs) + DPO (β=0.1, 135 pairs)
- DPO Rewards Accuracy: 93.8%, Margin: 3.89

---

## [0.2.0] - 2026-07-07 — Data Expansion & Evaluation

### Added
- Semi-automated data augmentation with 4 strategies (`scripts/02b_data_augmentation.py`)
- Multi-turn conversation generator with 15 scenarios (`scripts/02c_multi_turn_generator.py`)
- Data quality scoring pipeline (factual accuracy, voice authenticity, diversity)
- Automated evaluation framework with 5 metrics (`evaluation/`)
  - Character Consistency, OOC Resistance, RAG Factual Accuracy, Baseline Delta, Semantic Similarity
- 55 curated test cases for evaluation (`evaluation/test_cases.json`)
- Stratified train/val/test data split (80/10/10)
- Multi-turn training support in SFT script
- Hyperparameter grid search plan generator (`scripts/hyperparam_search.py`)
- Training analysis and visualization (`scripts/analyze_training.py`)
- CLI argument support for training script

### Changed
- Training data: 294 → 643 pairs (512 train + 62 val + 69 test)
- LoRA configuration: r=16 α=32 → r=32 α=64 (66M params, 1.62%)
- Training epochs: 5 → 8 with validation loss tracking
- Eval loss: 0.306 on unseen validation data

---

## [0.1.0] - 2026-07-07 — Initial Demo

### Added
- BWIKI + Moegirl Wiki web crawler (`scripts/01_crawl_firefly_data.py`)
- Training pair generator from firefly-skill markdown files
- Data cleaning pipeline with deduplication
- ChromaDB RAG knowledge base (54 docs)
- SFT LoRA training on Qwen3-4B-Instruct-2507
- FastAPI backend with RAG integration
- OOC response validator (`backend/validator.py`)
- Themed WebUI with Canvas particle animation
- Project initialization and documentation
