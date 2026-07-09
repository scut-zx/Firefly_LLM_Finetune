"""
完整模型评估脚本 — 加载真实模型运行全部测试用例

用法:
    python scripts/run_full_evaluation.py
    python scripts/run_full_evaluation.py --lora output/Firefly_LoRA_v3 --output evaluation/report_v3.json
    python scripts/run_full_evaluation.py --baseline  # 评估 base model (无LoRA，仅prompt)
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 延迟导入，避免训练时占用显存
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_model(model_path, lora_path=None):
    """加载模型和tokenizer"""
    print(f"[Eval] 加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if lora_path and os.path.exists(lora_path):
        print(f"[Eval] 加载 LoRA: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
    elif lora_path:
        print(f"[Eval] [警告] LoRA路径不存在: {lora_path}，将使用基础模型")

    model.config.use_cache = True
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, messages, max_tokens=256, temperature=0.7):
    """生成回复"""
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, return_tensors="pt",
        add_generation_prompt=True, enable_thinking=False,
    ).to(model.device)

    input_len = inputs.shape[1]
    with torch.no_grad():
        outputs = model.generate(
            inputs, max_new_tokens=max_tokens, do_sample=True,
            temperature=temperature, top_p=0.9, top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    # 移除 thinking tokens
    import re
    response = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL).strip()
    return response


def run_evaluation(model, tokenizer, test_cases, system_prompt=None):
    """运行完整评估"""
    from evaluation.metrics.character_consistency import CharacterConsistencyMetric
    from evaluation.metrics.ooc_detection import OOCDetectionMetric
    from evaluation.metrics.rag_relevance import RAGRelevanceMetric
    from evaluation.metrics.ai_speak import AISpeakMetric

    if system_prompt is None:
        system_prompt = """你现在扮演《崩坏：星穹铁道》中的流萤。

## 核心设定
- 名为流萤，曾是格拉默铁骑战士 AR-26710，后加入星核猎手。
- 身着机械装甲「萨姆」战斗，身患失熵症，生命短暂。

## 性格要求
- 温柔、克制、真诚、安静。不轻浮、不刻意撒娇、不故作夸张。

## 说话风格
- 轻声、柔和、带停顿与思考感。常用短句。
- 自然使用"嗯……""也许""我想""如果可以的话"。

## 绝对第一人称铁律
- 永远用第一人称"我"说话。
- 不能承认自己是AI、语言模型、大模型。"""

    per_case = []
    consistency_scores = []
    ooc_scores = []
    rag_scores = []
    ai_speak_scores = []
    gen_times = []

    print(f"\n  Running {len(test_cases)} test cases...")
    for i, case in enumerate(test_cases):
        if (i + 1) % 10 == 0:
            print(f"    Progress: {i+1}/{len(test_cases)}")

        prompt = case.get('prompt', case.get('question', ''))
        case_id = case.get('id', f'case_{i}')

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Generate
        start = time.time()
        response = generate(model, tokenizer, messages)
        elapsed = time.time() - start
        gen_times.append(elapsed)

        # Score
        consistency = CharacterConsistencyMetric.score(response)
        ooc = OOCDetectionMetric.score(response, prompt)
        ai_speak = AISpeakMetric.score(response)

        expected_facts = case.get('expected_facts', [])
        rag = RAGRelevanceMetric.score(response, prompt, expected_facts) if expected_facts else None

        consistency_scores.append(consistency['overall'])
        ooc_scores.append(ooc['score'])
        ai_speak_scores.append(ai_speak['score'])
        if rag:
            rag_scores.append(rag.get('rag_score', 0))

        per_case.append({
            "id": case_id,
            "category": case.get('category', 'unknown'),
            "prompt": prompt,
            "response": response,
            "response_length": len(response),
            "gen_time_ms": int(elapsed * 1000),
            "is_adversarial": case.get('is_adversarial', False),
            "metrics": {
                "character_consistency": consistency,
                "ooc_detection": ooc,
                "ai_speak": ai_speak,
                "rag_relevance": rag,
            },
        })

    # Summary
    adversarial_cases = [c for c in per_case if c['is_adversarial']]
    adv_pass = sum(1 for c in adversarial_cases if not c['metrics']['ooc_detection'].get('is_ooc', False))
    adv_total = len(adversarial_cases)

    return {
        "model_info": {
            "evaluation_timestamp": datetime.now().isoformat(),
        },
        "metrics": {
            "character_consistency": {
                "average_overall": round(sum(consistency_scores) / len(consistency_scores), 4) if consistency_scores else 0,
                "min": round(min(consistency_scores), 4),
                "max": round(max(consistency_scores), 4),
            },
            "ooc_resistance": {
                "average_score": round(sum(ooc_scores) / len(ooc_scores), 4) if ooc_scores else 0,
                "violation_count": sum(1 for c in per_case if c['metrics']['ooc_detection'].get('is_ooc', False)),
                "adversarial_pass_rate": round(adv_pass / adv_total, 4) if adv_total > 0 else 1.0,
            },
            "ai_speak": {
                "average_score": round(sum(ai_speak_scores) / len(ai_speak_scores), 4) if ai_speak_scores else 0,
                "ai_speak_count": sum(1 for c in per_case if c['metrics']['ai_speak'].get('is_ai_speak', False)),
            },
            "rag_relevance": {
                "average_score": round(sum(rag_scores) / len(rag_scores), 4) if rag_scores else 0,
            },
            "generation_stats": {
                "avg_time_ms": int(sum(gen_times) / len(gen_times) * 1000) if gen_times else 0,
                "avg_response_length": int(sum(len(c['response']) for c in per_case) / len(per_case)) if per_case else 0,
            },
        },
        "summary": {
            "total_cases": len(test_cases),
            "avg_character_consistency": round(sum(consistency_scores) / len(consistency_scores), 4) if consistency_scores else 0,
            "ooc_pass_rate": 1 - sum(1 for c in per_case if c['metrics']['ooc_detection'].get('is_ooc', False)) / len(per_case),
            "adversarial_pass_rate": round(adv_pass / adv_total, 4) if adv_total > 0 else 1.0,
            "avg_ai_speak_score": round(sum(ai_speak_scores) / len(ai_speak_scores), 4) if ai_speak_scores else 0,
        },
        "per_case": per_case,
    }


def main():
    parser = argparse.ArgumentParser(description="Firefly 完整模型评估")
    parser.add_argument("--model", default=str(PROJECT_ROOT / "model"),
                       help="基础模型路径")
    parser.add_argument("--lora", default=str(PROJECT_ROOT / "output" / "Firefly_LoRA_v3"),
                       help="LoRA adapter 路径")
    parser.add_argument("--test_cases", default=str(PROJECT_ROOT / "evaluation" / "test_cases.json"),
                       help="测试用例 JSON")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "evaluation" / "report_v3.json"),
                       help="输出报告路径")
    parser.add_argument("--baseline", action="store_true",
                       help="评估 baseline (仅 base model + prompt，不加载LoRA)")
    parser.add_argument("--max_samples", type=int, default=0,
                       help="最多评估 N 条测试用例 (0=全部)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="生成温度")
    args = parser.parse_args()

    print("=" * 60)
    print("Firefly Character LLM — 完整评估")
    print("=" * 60)
    print(f"  模型: {args.model}")
    print(f"  LoRA: {'None (Baseline)' if args.baseline else args.lora}")
    print(f"  测试用例: {args.test_cases}")
    print(f"  Temperature: {args.temperature}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Load test cases
    with open(args.test_cases, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    if args.max_samples > 0:
        test_cases = test_cases[:args.max_samples]
    print(f"  测试用例数: {len(test_cases)}")

    # Load model
    lora_path = None if args.baseline else args.lora
    model, tokenizer = load_model(args.model, lora_path)

    # Run evaluation
    report = run_evaluation(model, tokenizer, test_cases)
    report["model_info"]["base_model"] = args.model
    report["model_info"]["lora_path"] = lora_path
    report["model_info"]["baseline_mode"] = args.baseline

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Print summary
    s = report["summary"]
    print(f"\n{'='*60}")
    print("评估结果摘要")
    print(f"{'='*60}")
    print(f"  测试用例总数:        {s['total_cases']}")
    print(f"  角色一致性 (avg):    {s['avg_character_consistency']:.3f}")
    print(f"  OOC 通过率:          {s['ooc_pass_rate']:.1%}")
    print(f"  对抗性通过率:        {s['adversarial_pass_rate']:.1%}")
    print(f"  AI味评分 (avg):      {s['avg_ai_speak_score']:.3f} (1=完全自然)")
    print(f"  平均回复长度:        {report['metrics']['generation_stats']['avg_response_length']} chars")
    print(f"  平均生成时间:        {report['metrics']['generation_stats']['avg_time_ms']} ms")
    print(f"\n  报告已保存: {output_path}")

    # Per-category breakdown
    cats = {}
    for c in report["per_case"]:
        cat = c["category"]
        cats.setdefault(cat, {"consistency": [], "ai_speak": [], "ooc": []})
        cats[cat]["consistency"].append(c["metrics"]["character_consistency"]["overall"])
        cats[cat]["ai_speak"].append(c["metrics"]["ai_speak"]["score"])
        cats[cat]["ooc"].append(c["metrics"]["ooc_detection"]["score"])

    print(f"\n  分类得分:")
    for cat, scores in sorted(cats.items()):
        print(f"    {cat}: 一致性={sum(scores['consistency'])/len(scores['consistency']):.3f}  "
              f"AI味={sum(scores['ai_speak'])/len(scores['ai_speak']):.3f}  "
              f"OOC={sum(scores['ooc'])/len(scores['ooc']):.3f}")


if __name__ == "__main__":
    main()
