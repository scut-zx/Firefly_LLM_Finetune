# Contributing to Firefly Character AI Assistant

Thank you for your interest in contributing! This project aims to build high-quality character AI assistants using parameter-efficient fine-tuning on consumer hardware.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/scut-zx/Firefly_LLM_Finetune.git
cd Firefly_LLM_Finetune

# Create conda environment
conda create -n firefly python=3.10 -y
conda activate firefly

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies
pip install -r requirements.txt
```

## Prerequisites

- NVIDIA GPU with ≥16 GB VRAM (tested on RTX 4060 Ti)
- CUDA 12.4+
- Python 3.9+

## Project Structure

See the [README](README.md) for the full project structure and architecture overview.

## How to Contribute

### Reporting Bugs

Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md) to report issues. Include:
- Environment details (OS, Python version, GPU, VRAM)
- Steps to reproduce
- Expected vs actual behavior
- Error logs

### Suggesting Features

Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.md) to propose new ideas.

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make your changes
4. Run existing tests/scripts to verify nothing is broken
5. Commit with clear messages following [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation
   - `chore:` for maintenance
6. Push and create a Pull Request using the [PR template](.github/PULL_REQUEST_TEMPLATE.md)

### Code Style

- Python: Follow PEP 8
- Use type hints where practical
- Add docstrings for public functions
- Use `pathlib.Path` for file paths (not hardcoded strings)
- All scripts should handle Windows GBK encoding gracefully:
  ```python
  import sys
  if sys.platform == 'win32':
      sys.stdout.reconfigure(encoding='utf-8')
  ```

### Adding a New Character

This project was designed to be reusable. See the [Character Assistant Factory](.claude/skills/character-assistant-factory/SKILL.md) skill for a step-by-step guide to adapting the pipeline for any ACG character.

Quick steps:
1. Prepare character lore documents (markdown format)
2. Run data generation scripts (crawl, generate, augment, clean)
3. Build RAG knowledge base
4. Train SFT LoRA, then optionally DPO
5. Deploy with the shared backend + themed WebUI

## Testing

Currently testing is manual:
```bash
# Run evaluation framework
python evaluation/run_eval.py --model dpo --baseline base

# Start backend and test via WebUI
HF_HUB_OFFLINE=1 python -m backend.app --direct --port 7860
```

## Questions?

Open a [Discussion](https://github.com/scut-zx/Firefly_LLM_Finetune/discussions) or reach out via Issues.
