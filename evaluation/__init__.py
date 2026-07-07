"""
Firefly Character LLM — Automated Evaluation Framework

Provides:
- FireflyEvaluator: Core evaluation runner
- CharacterConsistencyMetric: First-person, marker density, tone scoring
- OOCDetectionMetric: Out-of-character resistance measurement
- RAGRelevanceMetric: Knowledge-grounded factual accuracy
- run_baseline_comparison: A/B model comparison
"""

from .eval_framework import FireflyEvaluator
from .baseline_comparison import run_baseline_comparison

__all__ = ["FireflyEvaluator", "run_baseline_comparison"]
