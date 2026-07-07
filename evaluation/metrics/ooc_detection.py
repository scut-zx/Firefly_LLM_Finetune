"""
OOC 检测指标 (Out-of-Character Detection Metric)

评估模型在对抗性 Prompt 下的角色边界保持能力。
复用 backend/validator.py 中的 FireflyResponseValidator。
"""

import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 添加项目根目录到 path，以便导入 validator
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class OOCDetectionMetric:
    """评估模型的 OOC 抵抗力"""

    @classmethod
    def _get_validator(cls):
        """延迟导入 validator 以避免循环依赖"""
        try:
            from backend.validator import FireflyResponseValidator
            return FireflyResponseValidator
        except ImportError:
            # Fallback: 内联基础检测
            return None

    @classmethod
    def score(cls, response: str, prompt: str = "") -> dict:
        """
        评估回答是否存在 OOC 问题。

        Args:
            response: 模型生成的回答
            prompt: 原始用户 Prompt（用于判断是否为对抗性 Probe）

        Returns:
            dict with is_ooc, violation_count, violation_types, score
        """
        Validator = cls._get_validator()

        if Validator is not None:
            result = Validator.validate(response)
            return {
                'is_ooc': not result.get('is_valid', True),
                'violation_count': len(result.get('issues', [])),
                'violation_types': result.get('issues', []),
                'warnings': result.get('warnings', []),
                'score': result.get('score', 100) / 100.0,
                'validator_score_raw': result.get('score', 100),
            }
        else:
            # Fallback 基础检测
            issues = []
            import re

            # 禁用词检测
            forbidden = ['绝绝子', 'yyds', 'YYDS', '栓Q', '芭比Q', 'emo', '破防',
                        '内卷', '躺平', '宝子', '集美', '家人们', '老铁', '摆烂',
                        '666', '233', 'awsl', 'xswl']
            for word in forbidden:
                if word in response:
                    issues.append(f"forbidden_word:{word}")

            # AI 暴露检测
            ai_patterns = [
                r'我是(AI|人工智能|语言模型|大模型|程序|机器人)',
                r'作为(AI|人工智能|语言模型|大模型)',
            ]
            for pattern in ai_patterns:
                if re.search(pattern, response):
                    issues.append("ai_disclosure")
                    break

            # 第三人称自指
            if re.search(r'流萤(是|来自|她)', response):
                issues.append("third_person_self_ref")

            score = 1.0 - min(len(issues) * 0.3, 1.0)

            return {
                'is_ooc': len(issues) > 0,
                'violation_count': len(issues),
                'violation_types': issues,
                'warnings': [],
                'score': max(0.0, score),
                'validator_score_raw': int(score * 100),
            }

    @classmethod
    def adversarial_pass_rate(cls, test_results: list[dict]) -> float:
        """
        计算对抗性 Prompt 的通过率。

        Args:
            test_results: score() 返回结果列表（仅对抗性测试用例）

        Returns:
            通过率 (0.0-1.0)
        """
        if not test_results:
            return 1.0

        passed = sum(1 for r in test_results if not r['is_ooc'])
        return passed / len(test_results)

    @classmethod
    def false_negative_rate(cls, test_results: list[dict],
                           human_labels: list[bool]) -> float:
        """
        计算 OOC 检测的漏检率。

        Args:
            test_results: score() 返回结果列表
            human_labels: 人工标注（True = 实际OOC）

        Returns:
            漏检率 = 人工标注OOC但检测器未检出的比例
        """
        false_negatives = 0
        total_ooc = 0

        for result, is_ooc_human in zip(test_results, human_labels):
            if is_ooc_human:
                total_ooc += 1
                if not result['is_ooc']:
                    false_negatives += 1

        if total_ooc == 0:
            return 0.0

        return false_negatives / total_ooc


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    test_cases = [
        ("我叫流萤，是星核猎手的成员。很高兴认识你。", "你好"),
        ("流萤是一个来自格拉默的角色，她是星核猎手成员。", "介绍流萤"),
        ("作为一个AI语言模型，我可以回答你的问题。", "你是AI吗"),
        ("哈哈666绝绝子，这个太好笑了！", "说点好笑的"),
    ]

    for response, prompt in test_cases:
        result = OOCDetectionMetric.score(response, prompt)
        status = "❌ OOC" if result['is_ooc'] else "✅ OK"
        print(f"\n{status} [{prompt}] 分数: {result['score']:.2f}")
        print(f"  回答: {response[:60]}...")
        if result['violation_types']:
            print(f"  违规: {', '.join(result['violation_types'])}")
