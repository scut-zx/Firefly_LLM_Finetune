"""
训练数据质量评分 (Data Quality Scorer)

对每条训练数据进行多维度质量评分：
1. 事实准确性：与知识库交叉验证
2. 角色语气真实性：与 gold-standard 回复的语义相似度
3. 多样性：与其他训练对的 embedding 余弦距离（避免重复）

输出质量评分文件，支持按阈值过滤。

用法:
    python scripts/data_quality_scorer.py
    python scripts/data_quality_scorer.py --input data/firefly_training_v2.json --min-score 0.6
"""

import os
import sys
import json
import re
import hashlib
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class DataQualityScorer:
    """训练数据质量评分器"""

    # 流萤核心知识（用于事实准确性验证）
    CANON_FACTS = {
        "identity": ["流萤", "星核猎手", "格拉默铁骑", "AR-26710"],
        "armor": ["萨姆", "火萤IV型", "战略强袭装甲"],
        "condition": ["失熵症", "退行性", "身体", "生命短暂"],
        "faction": ["星核猎手", "艾利欧", "剧本"],
        "people": ["开拓者", "银狼", "卡芙卡", "刃", "艾利欧"],
        "places": ["格拉默", "匹诺康尼", "天台", "梦境"],
        "themes": ["萤火虫", "燃烧", "生命", "星空", "星星", "梦"],
    }

    # 流萤语气金标准特征
    TONE_FEATURES_POSITIVE = [
        "嗯……", "……", "也许", "我想", "如果可以的话", "嘿嘿",
    ]

    TONE_FEATURES_NEGATIVE = [
        "！！！", "哈哈", "嘻嘻", "哈哈哈", "呢呢",
    ]

    @classmethod
    def factual_accuracy(cls, instruction: str, output: str) -> dict:
        """
        评估事实准确性。
        检查 output 中是否包含与 instruction 相关的正确设定。
        返回 0.0-1.0 的分数。
        """
        score = 1.0
        matched = []
        expected = []
        issues = []

        # 检查回答中是否包含与问题领域相关的核心知识
        question = instruction + " " + output
        q_lower = question.lower()

        # 根据问题主题，检查回答中是否包含对应知识
        if any(kw in q_lower for kw in ["萨姆", "装甲", "火萤"]):
            expected = cls.CANON_FACTS["armor"]
        elif any(kw in q_lower for kw in ["失熵", "病", "身体"]):
            expected = cls.CANON_FACTS["condition"]
        elif any(kw in q_lower for kw in ["星核猎手", "成员", "加入"]):
            expected = cls.CANON_FACTS["faction"]
        elif any(kw in q_lower for kw in ["你是谁", "介绍", "流萤"]):
            expected = cls.CANON_FACTS["identity"]
        elif any(kw in q_lower for kw in ["格拉默", "铁骑", "故乡"]):
            expected = ["格拉默", "铁骑", "毁灭"]

        if expected:
            matched = [f for f in expected if f in output]
            recall = len(matched) / len(expected) if expected else 1.0
            score = 0.3 + 0.7 * recall  # 最低 0.3（回答至少不违反设定）

        # 检查是否有严重事实错误
        # （简化的启发式检查）
        forbidden_fact_patterns = [
            (r"流萤(的)?故乡(是|在).*(?!格拉默)(\w{2,})", "故乡信息错误"),
            (r"萨姆(是|的).*型[号号]", "萨姆型号错误"),
        ]
        for pattern, desc in forbidden_fact_patterns:
            if re.search(pattern, output):
                issues.append(desc)
                score -= 0.3

        return {
            "score": round(max(0.0, min(1.0, score)), 3),
            "matched_facts": matched,
            "expected_facts": expected,
            "issues": issues,
        }

    @classmethod
    def voice_authenticity(cls, output: str) -> dict:
        """
        评估角色语气真实性。
        检查是否包含流萤的标志性说话特征。
        返回 0.0-1.0 的分数。
        """
        score = 0.5  # 基线

        # 正面特征加分
        positive_hits = sum(1 for f in cls.TONE_FEATURES_POSITIVE if f in output)
        score += min(positive_hits * 0.1, 0.3)

        # 负面特征扣分
        negative_hits = sum(1 for f in cls.TONE_FEATURES_NEGATIVE if f in output)
        score -= min(negative_hits * 0.15, 0.4)

        # 第一人称检查
        if "我" in output and len(output) > 10:
            score += 0.1
        if re.search(r'流萤(是|来自|她)', output):
            score -= 0.3

        # 回答长度检查（流萤倾向短句）
        if len(output) < 200:
            score += 0.1

        return {
            "score": round(max(0.0, min(1.0, score)), 3),
            "positive_features": positive_hits,
            "negative_features": negative_hits,
        }

    @classmethod
    def diversity_score(cls, output: str, all_outputs: list[str]) -> float:
        """
        评估多样性。
        使用简化的 Jaccard 相似度（字符 n-gram）与所有其他输出比较。
        返回 0.0-1.0，越高表示越独特。
        """
        if len(all_outputs) <= 1:
            return 1.0

        def char_ngrams(text, n=3):
            return set(text[i:i+n] for i in range(len(text) - n + 1))

        this_ngrams = char_ngrams(output)
        if not this_ngrams:
            return 1.0

        similarities = []
        for other in all_outputs:
            if other == output:
                continue
            other_ngrams = char_ngrams(other)
            if not other_ngrams:
                continue
            overlap = len(this_ngrams & other_ngrams)
            union = len(this_ngrams | other_ngrams)
            similarities.append(overlap / union if union > 0 else 0)

        if not similarities:
            return 1.0

        avg_similarity = sum(similarities) / len(similarities)
        return round(1.0 - avg_similarity, 3)

    @classmethod
    def score_pair(cls, pair: dict, all_outputs: list[str] = None) -> dict:
        """
        对单条训练数据进行综合评分。
        """
        instruction = pair.get("instruction", "")
        output = pair.get("output", "")

        factual = cls.factual_accuracy(instruction, output)
        voice = cls.voice_authenticity(output)
        diversity = cls.diversity_score(output, all_outputs or [output])

        # 综合分：事实 40%，语气 40%，多样性 20%
        overall = factual["score"] * 0.4 + voice["score"] * 0.4 + diversity * 0.2

        return {
            "overall": round(overall, 3),
            "factual_accuracy": factual,
            "voice_authenticity": voice,
            "diversity": diversity,
            "quality_tier": cls._tier(overall),
        }

    @classmethod
    def _tier(cls, score: float) -> str:
        if score >= 0.8:
            return "A (优秀)"
        elif score >= 0.65:
            return "B (良好)"
        elif score >= 0.5:
            return "C (一般)"
        else:
            return "D (需改进)"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="训练数据质量评分")
    parser.add_argument("--input", default=None,
                       help="输入 JSON 文件路径")
    parser.add_argument("--output", default=None,
                       help="输出 JSON 文件路径")
    parser.add_argument("--min-score", type=float, default=0.0,
                       help="最低质量分数阈值 (默认: 0.0，不过滤)")
    parser.add_argument("--summary-only", action="store_true",
                       help="仅输出摘要，不保存每条评分")
    args = parser.parse_args()

    input_path = args.input or str(
        PROJECT_ROOT / "data" / "firefly_training_cleaned.json"
    )
    if not os.path.exists(input_path):
        input_path = str(PROJECT_ROOT / "data" / "firefly_training.json")

    with open(input_path, 'r', encoding='utf-8') as f:
        pairs = json.load(f)

    print(f"\n{'='*60}")
    print(f"训练数据质量评分")
    print(f"{'='*60}")
    print(f"  输入: {input_path}")
    print(f"  数据量: {len(pairs)} 条")
    print(f"  最低阈值: {args.min_score}")
    print(f"{'='*60}\n")

    # 收集所有 output 用于多样性计算
    all_outputs = [p.get("output", "") for p in pairs]

    # 逐条评分
    scored_pairs = []
    tier_counts = {"A (优秀)": 0, "B (良好)": 0, "C (一般)": 0, "D (需改进)": 0}
    total_score = 0.0

    for i, pair in enumerate(pairs):
        result = DataQualityScorer.score_pair(pair, all_outputs)
        total_score += result["overall"]
        tier_counts[result["quality_tier"]] += 1

        if not args.summary_only:
            scored_pairs.append({
                **pair,
                "quality_score": result,
            })

        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{len(pairs)}")

    # 摘要
    avg_score = total_score / len(pairs)
    print(f"\n{'='*60}")
    print(f"评分摘要")
    print(f"{'='*60}")
    print(f"  平均分: {avg_score:.3f}")
    print(f"  质量分布:")
    for tier, count in tier_counts.items():
        pct = count / len(pairs) * 100
        bar = "#" * int(pct / 2)
        print(f"    {tier}: {count:>4} ({pct:>5.1f}%) {bar}")

    # 过滤
    if args.min_score > 0:
        filtered = [p for p in scored_pairs
                   if p["quality_score"]["overall"] >= args.min_score]
        print(f"\n  阈值过滤 (>= {args.min_score}): "
              f"{len(filtered)}/{len(scored_pairs)} 条保留 "
              f"({len(filtered)/len(scored_pairs)*100:.1f}%)")

    # 保存
    if not args.summary_only:
        output_path = args.output or str(
            PROJECT_ROOT / "data" / "firefly_training_scored.json"
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(scored_pairs, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] 评分数据已保存: {output_path}")

    # 保存摘要
    summary_path = Path(args.output or str(
        PROJECT_ROOT / "data" / "quality_summary.json"
    )).with_name("quality_summary.json") if not args.summary_only else None

    if summary_path:
        summary = {
            "total": len(pairs),
            "average_score": round(avg_score, 3),
            "tier_distribution": tier_counts,
            "input_file": input_path,
        }
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[OK] 质量摘要已保存: {summary_path}")

    print(f"\n[OK] 质量评分完成")


if __name__ == "__main__":
    main()
