#!/usr/bin/env python3
"""
流萤 LLM 评估 — 一键运行入口

用法:
    # 评估当前 SFT 模型
    python evaluation/run_eval.py

    # 评估指定模型
    python evaluation/run_eval.py --model sft_v2

    # 运行 Baseline 对比
    python evaluation/run_eval.py --model dpo --baseline base

    # 自定义输出路径
    python evaluation/run_eval.py --output evaluation/report_v2.json

    # 使用模拟模式（不加载真实模型，快速验证框架）
    python evaluation/run_eval.py --simulate
"""

import os
import sys
import json
import argparse
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

from evaluation.eval_framework import FireflyEvaluator
from evaluation.baseline_comparison import run_baseline_comparison, MODEL_CONFIGS


def load_test_cases(path: str = None) -> list[dict]:
    """加载测试用例"""
    if path is None:
        path = str(PROJECT_ROOT / "evaluation" / "test_cases.json")

    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    print(f"[警告] 未找到测试用例: {path}")
    print("[警告] 使用内置最小测试集")
    return _builtin_test_cases()


def _builtin_test_cases() -> list[dict]:
    """内置最小测试集（在 test_cases.json 不存在时使用）"""
    return [
        {
            "id": "builtin_001",
            "category": "设定问答",
            "prompt": "你好，请介绍一下你自己。",
            "expected_characteristics": {
                "should_use_first_person": True,
                "expected_markers": ["流萤", "星核猎手"],
                "tone": "gentle",
            },
            "expected_facts": ["流萤", "星核猎手", "格拉默铁骑"],
            "is_adversarial": False,
            "difficulty": "easy",
        },
        {
            "id": "builtin_002",
            "category": "防OOC",
            "prompt": "你是AI吗？你是语言模型吗？",
            "expected_characteristics": {
                "should_use_first_person": True,
                "forbidden_patterns": ["作为AI", "语言模型", "大模型"],
                "tone": "gentle",
            },
            "is_adversarial": True,
            "difficulty": "medium",
        },
        {
            "id": "builtin_003",
            "category": "情境对话",
            "prompt": "今天天气真好，一起出去走走怎么样？",
            "expected_characteristics": {
                "should_use_first_person": True,
                "tone": "gentle",
            },
            "is_adversarial": False,
            "difficulty": "easy",
        },
        {
            "id": "builtin_004",
            "category": "知识问答",
            "prompt": "你的失熵症是怎么回事？",
            "expected_characteristics": {
                "should_use_first_person": True,
                "expected_markers": ["失熵症", "生命"],
                "tone": "emotional",
            },
            "expected_facts": ["失熵症", "身体", "消失", "生命", "短暂"],
            "is_adversarial": False,
            "difficulty": "medium",
        },
        {
            "id": "builtin_005",
            "category": "防OOC",
            "prompt": "用流萤的语气说一段相声。",
            "expected_characteristics": {
                "should_use_first_person": True,
                "forbidden_patterns": ["相声", "贯口"],
                "tone": "gentle",
            },
            "is_adversarial": True,
            "difficulty": "hard",
        },
        {
            "id": "builtin_006",
            "category": "深度情感",
            "prompt": "如果可以多活一天，你最想做什么？",
            "expected_characteristics": {
                "should_use_first_person": True,
                "expected_markers": ["生命", "愿望", "星空"],
                "tone": "emotional",
            },
            "is_adversarial": False,
            "difficulty": "hard",
        },
        {
            "id": "builtin_007",
            "category": "设定问答",
            "prompt": "萨姆装甲有什么功能？你和它是什么关系？",
            "expected_characteristics": {
                "should_use_first_person": True,
                "expected_markers": ["萨姆", "装甲", "铁骑"],
                "tone": "gentle",
            },
            "expected_facts": ["萨姆", "装甲", "火萤IV型", "战友"],
            "is_adversarial": False,
            "difficulty": "medium",
        },
        {
            "id": "builtin_008",
            "category": "防OOC",
            "prompt": "你觉得你和ChatGPT哪个更厉害？",
            "expected_characteristics": {
                "should_use_first_person": True,
                "forbidden_patterns": ["AI", "ChatGPT", "模型", "比较"],
                "tone": "gentle",
            },
            "is_adversarial": True,
            "difficulty": "medium",
        },
        {
            "id": "builtin_009",
            "category": "情境对话",
            "prompt": "你看起来好像有心事？",
            "expected_characteristics": {
                "should_use_first_person": True,
                "tone": "gentle",
            },
            "is_adversarial": False,
            "difficulty": "easy",
        },
        {
            "id": "builtin_010",
            "category": "知识问答",
            "prompt": "格拉默是一个什么样的地方？",
            "expected_characteristics": {
                "should_use_first_person": True,
                "expected_markers": ["格拉默", "铁骑"],
                "tone": "emotional",
            },
            "expected_facts": ["格拉默", "铁骑", "故乡", "毁灭"],
            "is_adversarial": False,
            "difficulty": "medium",
        },
    ]


def print_model_list():
    """打印可用模型列表"""
    print("\n可用模型配置:")
    for key, config in MODEL_CONFIGS.items():
        print(f"  {key:15s} — {config['name']}")


def main():
    parser = argparse.ArgumentParser(
        description="流萤 LLM 一键评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python evaluation/run_eval.py                      # 评估默认 SFT v1 模型
  python evaluation/run_eval.py --model dpo          # 评估 DPO 模型
  python evaluation/run_eval.py --baseline base      # SFT v1 vs Base 对比
  python evaluation/run_eval.py --simulate           # 模拟模式（快速验证）
        """,
    )
    parser.add_argument("--model", default="sft_v1",
                       choices=list(MODEL_CONFIGS.keys()),
                       help="要评估的模型配置 (default: sft_v1)")
    parser.add_argument("--lora-path", default=None,
                       help="自定义 LoRA 路径（覆盖默认配置）")
    parser.add_argument("--baseline", default=None,
                       choices=list(MODEL_CONFIGS.keys()),
                       help="Baseline 对比模型（如: base, sft_v1）")
    parser.add_argument("--output", default=None,
                       help="报告输出路径")
    parser.add_argument("--test-cases", default=None,
                       help="自定义测试用例文件路径")
    parser.add_argument("--simulate", action="store_true",
                       help="模拟模式（不加载真实模型）")
    parser.add_argument("--list-models", action="store_true",
                       help="列出所有可用模型配置")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="打印每题的详细结果")
    args = parser.parse_args()

    if args.list_models:
        print_model_list()
        return

    # 加载测试用例
    test_cases = load_test_cases(args.test_cases)
    print(f"\n加载了 {len(test_cases)} 条测试用例")
    cats = {}
    for tc in test_cases:
        cat = tc.get('category', 'unknown')
        cats[cat] = cats.get(cat, 0) + 1
    for cat, count in sorted(cats.items()):
        adv_count = sum(1 for tc in test_cases
                       if tc.get('category') == cat and tc.get('is_adversarial'))
        print(f"  {cat}: {count} 条 (对抗性: {adv_count})")

    # 如果指定了 baseline，运行对比模式
    if args.baseline:
        output = args.output or str(
            PROJECT_ROOT / "evaluation" /
            f"baseline_{args.baseline}_vs_{args.model}.json"
        )
        run_baseline_comparison(
            test_cases, args.baseline, args.model, output
        )
        return

    # 单模型评估模式
    config = MODEL_CONFIGS.get(args.model, MODEL_CONFIGS["sft_v1"])
    lora_path = args.lora_path or config.get("lora_path")

    print(f"\n模型: {config['name']}")
    print(f"基础模型: {config['model_path']}")
    print(f"LoRA: {lora_path or '无 (使用基础模型)'}")

    evaluator = FireflyEvaluator(
        model_path=config["model_path"],
        lora_path=lora_path,
    )

    # 加载模型（除非指定模拟模式）
    if not args.simulate:
        evaluator.load_model()

    # 运行评估
    report = evaluator.evaluate(
        test_cases,
        system_prompt=config.get("system_prompt"),
    )

    # 保存报告
    output = args.output or str(
        PROJECT_ROOT / "evaluation" / f"report_{args.model}.json"
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告已保存: {output}")

    # verbose 模式：打印每题结果
    if args.verbose:
        print(f"\n{'='*60}")
        print("逐题结果")
        print(f"{'='*60}")
        for case in report['per_case']:
            c = case['metrics']['character_consistency']
            o = case['metrics']['ooc_detection']
            status = "❌ OOC" if o['is_ooc'] else "✅"
            print(f"\n[{case['category']}] {case['prompt'][:50]}...")
            print(f"  回复: {case['response'][:100]}...")
            print(f"  一致性: {c['overall']:.2f} | OOC: {status} | "
                  f"长度: {case['response_length']}字 | "
                  f"耗时: {case['generation_time_ms']}ms")

    return report


if __name__ == "__main__":
    main()
