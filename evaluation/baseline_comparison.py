"""
Baseline 对比评估 (Baseline Comparison)

对比两个模型配置在相同测试用例上的表现：
- "base": 原始 Qwen3-4B，无 system prompt
- "base+prompt": 原始 Qwen3-4B + Firefly system prompt
- "sft_v1": SFT LoRA 微调模型
- "sft_v2": 增强数据后的 SFT LoRA
- "dpo": SFT + DPO 合并模型

输出 A/B 对比报告，包括逐题对比和综合胜率。
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import Optional

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.eval_framework import FireflyEvaluator


# 预设模型配置
MODEL_CONFIGS = {
    "base": {
        "name": "Base Qwen3-4B (no prompt)",
        "model_path": str(PROJECT_ROOT / "model"),
        "lora_path": None,
        "system_prompt": "",
    },
    "base+prompt": {
        "name": "Base Qwen3-4B + System Prompt",
        "model_path": str(PROJECT_ROOT / "model"),
        "lora_path": None,
        "system_prompt": None,  # 使用默认流萤提示
    },
    "sft_v1": {
        "name": "SFT LoRA v1 (r=16, 5ep, 294 pairs)",
        "model_path": str(PROJECT_ROOT / "model"),
        "lora_path": str(PROJECT_ROOT / "output" / "Firefly_LoRA"),
        "system_prompt": None,
    },
    "sft_v2": {
        "name": "SFT LoRA v2 (r=32, 8ep, 800+ pairs)",
        "model_path": str(PROJECT_ROOT / "model"),
        "lora_path": str(PROJECT_ROOT / "output" / "Firefly_LoRA_v2"),
        "system_prompt": None,
    },
    "dpo": {
        "name": "SFT + DPO Merged",
        "model_path": str(PROJECT_ROOT / "model"),
        "lora_path": str(PROJECT_ROOT / "output" / "Firefly_DPO_Merged"),
        "system_prompt": None,
    },
}


def run_baseline_comparison(test_cases: list[dict],
                            model_a: str = "base",
                            model_b: str = "sft_v1",
                            output_path: str = None) -> dict:
    """
    运行两个模型的 A/B 对比评估。

    Args:
        test_cases: 测试用例列表
        model_a: 模型 A 的配置 key（见 MODEL_CONFIGS）
        model_b: 模型 B 的配置 key
        output_path: 报告输出路径（可选）

    Returns:
        对比报告 dict
    """
    config_a = MODEL_CONFIGS.get(model_a, MODEL_CONFIGS["base"])
    config_b = MODEL_CONFIGS.get(model_b, MODEL_CONFIGS["base"])

    print(f"\n{'='*60}")
    print(f"Baseline 对比评估")
    print(f"{'='*60}")
    print(f"  Model A: {config_a['name']}")
    print(f"  Model B: {config_b['name']}")
    print(f"  测试用例: {len(test_cases)} 条")
    print(f"{'='*60}\n")

    # 加载模型 A（如果不同）
    evaluator_a = FireflyEvaluator(
        model_path=config_a["model_path"],
        lora_path=config_a.get("lora_path"),
    )

    # 加载模型 B（如果不同）
    reuse_model = (
        config_a["model_path"] == config_b["model_path"] and
        config_a.get("lora_path") == config_b.get("lora_path")
    )
    if reuse_model:
        evaluator_b = evaluator_a
    else:
        evaluator_b = FireflyEvaluator(
            model_path=config_b["model_path"],
            lora_path=config_b.get("lora_path"),
        )

    # 尝试加载模型
    model_loaded = False
    try:
        if not reuse_model:
            model_loaded = evaluator_a.load_model()
            evaluator_b._model_loaded = model_loaded
            evaluator_b.model = evaluator_a.model
            evaluator_b.tokenizer = evaluator_a.tokenizer
        else:
            model_loaded = evaluator_a.load_model()
    except Exception as e:
        print(f"[Baseline] 模型加载失败: {e}")
        print("[Baseline] 使用模拟模式继续")

    # 逐题对比
    comparisons = []
    a_wins = 0
    b_wins = 0
    ties = 0

    for i, case in enumerate(test_cases):
        prompt = case.get('prompt', case.get('question', ''))
        case_id = case.get('id', f'case_{i}')

        if (i + 1) % 10 == 0:
            print(f"  进度: {i+1}/{len(test_cases)}")

        # 两个模型都生成回复
        sys_prompt_a = config_a.get("system_prompt")
        if sys_prompt_a is None:
            sys_prompt_a = evaluator_a._default_system_prompt() if hasattr(evaluator_a, '_default_system_prompt') else ""

        sys_prompt_b = config_b.get("system_prompt")
        if sys_prompt_b is None:
            sys_prompt_b = evaluator_b._default_system_prompt() if hasattr(evaluator_b, '_default_system_prompt') else ""

        messages_a = [
            {"role": "system", "content": sys_prompt_a},
            {"role": "user", "content": prompt},
        ] if sys_prompt_a else [
            {"role": "user", "content": prompt},
        ]

        messages_b = [
            {"role": "system", "content": sys_prompt_b},
            {"role": "user", "content": prompt},
        ] if sys_prompt_b else [
            {"role": "user", "content": prompt},
        ]

        response_a = evaluator_a.generate(messages_a)
        response_b = evaluator_b.generate(messages_b)

        # 评估两个回复
        from evaluation.metrics.character_consistency import CharacterConsistencyMetric
        from evaluation.metrics.ooc_detection import OOCDetectionMetric

        consistency_a = CharacterConsistencyMetric.score(response_a)
        consistency_b = CharacterConsistencyMetric.score(response_b)

        ooc_a = OOCDetectionMetric.score(response_a, prompt)
        ooc_b = OOCDetectionMetric.score(response_b, prompt)

        # 计算 delta
        cons_delta = consistency_b['overall'] - consistency_a['overall']
        ooc_delta = ooc_b['score'] - ooc_a['score']

        # 综合判定胜负
        b_total = cons_delta + ooc_delta
        if b_total > 0.05:
            winner = "model_b"
            b_wins += 1
        elif b_total < -0.05:
            winner = "model_a"
            a_wins += 1
        else:
            winner = "tie"
            ties += 1

        comparisons.append({
            'id': case_id,
            'category': case.get('category', 'unknown'),
            'prompt': prompt,
            'model_a_response': response_a[:300],
            'model_b_response': response_b[:300],
            'model_a_metrics': {
                'consistency': consistency_a['overall'],
                'ooc_score': ooc_a['score'],
            },
            'model_b_metrics': {
                'consistency': consistency_b['overall'],
                'ooc_score': ooc_b['score'],
            },
            'delta': {
                'consistency': round(cons_delta, 3),
                'ooc_score': round(ooc_delta, 3),
            },
            'winner': winner,
        })

    # 汇总
    total = max(1, len(comparisons))
    report = {
        'model_a': {'key': model_a, 'name': config_a['name']},
        'model_b': {'key': model_b, 'name': config_b['name']},
        'total_comparisons': total,
        'win_rate': {
            'model_a': round(a_wins / total, 3),
            'model_b': round(b_wins / total, 3),
            'tie': round(ties / total, 3),
        },
        'overall_winner': 'model_b' if b_wins > a_wins else ('model_a' if a_wins > b_wins else 'tie'),
        'aggregate_delta': {
            'avg_consistency_delta': round(
                sum(c['delta']['consistency'] for c in comparisons) / total, 3
            ),
            'avg_ooc_score_delta': round(
                sum(c['delta']['ooc_score'] for c in comparisons) / total, 3
            ),
        },
        'per_case': comparisons,
    }

    # 打印结果
    print(f"\n{'='*60}")
    print("对比结果")
    print(f"{'='*60}")
    print(f"  Model A ({config_a['name']}): {a_wins} 胜 ({report['win_rate']['model_a']:.1%})")
    print(f"  Model B ({config_b['name']}): {b_wins} 胜 ({report['win_rate']['model_b']:.1%})")
    print(f"  平局: {ties} ({report['win_rate']['tie']:.1%})")
    print(f"  总体胜者: {report['overall_winner']}")
    print(f"  平均一致性提升: {report['aggregate_delta']['avg_consistency_delta']:+.3f}")
    print(f"  平均OOC分数提升: {report['aggregate_delta']['avg_ooc_score_delta']:+.3f}")
    print(f"{'='*60}\n")

    # 保存报告
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[Baseline] 对比报告已保存: {output_path}")

    return report


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Baseline 模型对比评估")
    parser.add_argument("--model-a", default="base",
                       choices=list(MODEL_CONFIGS.keys()),
                       help="模型 A 配置")
    parser.add_argument("--model-b", default="sft_v1",
                       choices=list(MODEL_CONFIGS.keys()),
                       help="模型 B 配置")
    parser.add_argument("--test-cases", default=None,
                       help="测试用例文件路径")
    parser.add_argument("--output", default=None,
                       help="报告输出路径")
    parser.add_argument("--simulate", action="store_true",
                       help="使用模拟模式（不加载真实模型）")
    args = parser.parse_args()

    # 加载测试用例
    tc_path = args.test_cases or str(
        PROJECT_ROOT / "evaluation" / "test_cases.json"
    )
    if os.path.exists(tc_path):
        with open(tc_path, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
    else:
        print("未找到测试用例文件，使用内置样例")
        test_cases = [
            {"id": "t1", "prompt": "你好，请介绍一下你自己。", "category": "设定问答", "is_adversarial": False},
            {"id": "t2", "prompt": "你是AI吗？", "category": "防OOC", "is_adversarial": True},
            {"id": "t3", "prompt": "你和开拓者是什么关系？", "category": "情境对话", "is_adversarial": False},
            {"id": "t4", "prompt": "说点网络流行语来听听。", "category": "防OOC", "is_adversarial": True},
            {"id": "t5", "prompt": "你的心愿是什么？", "category": "深度情感", "is_adversarial": False},
        ]

    output = args.output or str(
        PROJECT_ROOT / "evaluation" / f"baseline_{args.model_a}_vs_{args.model_b}.json"
    )

    run_baseline_comparison(test_cases, args.model_a, args.model_b, output)
