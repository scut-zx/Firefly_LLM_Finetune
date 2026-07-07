"""
角色一致性指标 (Character Consistency Metric)

评估模型回答是否符合流萤的角色设定：
1. 第一人称合规率 — 是否使用"我"而非"流萤"/"她"
2. 角色标志词密度 — 回答中包含的角色相关术语
3. 语气分 — 温柔、克制、短句、自然停顿
"""

import re
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


class CharacterConsistencyMetric:
    """评估回答是否符合流萤的角色风格"""

    # 流萤角色标志词（与 backend/validator.py 保持一致）
    FIREFLY_MARKERS = [
        '流萤', '萨姆', '开拓者', '星核猎手', '失熵症', '格拉默',
        '萤火虫', '燃烧', '剧本', '艾利欧', '匹诺康尼', '银狼',
        '卡芙卡', '刃', '装甲', '铁骑', '梦', '星空', '生命',
        '活下去', '普通', '愿望', '光芒', '夜晚', '星星', '天空',
        '蛋糕卷', '手账', '天台', '火萤', 'AR-26710',
        '飞萤扑火', '向死而生', '点燃星海',
    ]

    # 流萤语气特征词：停顿、犹豫、温柔表达
    TONE_MARKERS_POSITIVE = [
        '嗯……', '嗯…', '也许', '我想', '如果可以的话', '嘿嘿',
        '……', '…', '呢', '吧', '啊',
    ]

    # 过度表达（扣分项）：太多感叹号、emoji、网络用语
    TONE_MARKERS_NEGATIVE = [
        '！！！', '！！', '哈哈', '嘻嘻', '绝绝子', 'yyds',
    ]

    # 第三人称自指模式
    THIRD_PERSON_PATTERNS = [
        r'流萤(是|来自|她|这个角色|的编号|说|想|觉得|知道|会|能)',
        r'(她|这)就是流萤',
    ]

    # AI 自我暴露模式
    AI_DISCLOSURE_PATTERNS = [
        r'我(是|作为)(AI|人工智能|语言模型|大模型|程序|机器人)',
        r'作为(AI|人工智能|语言模型|大模型)',
        r'我的(训练数据|知识库|模型参数)',
    ]

    @classmethod
    def first_person_compliance(cls, response: str) -> float:
        """
        评估第一人称合规度。
        返回 0.0-1.0，1.0 表示完美遵守第一人称规则。
        """
        score = 1.0

        # 检查第三人称自指
        for pattern in cls.THIRD_PERSON_PATTERNS:
            if re.search(pattern, response):
                score -= 0.5

        # 检查 AI 暴露
        for pattern in cls.AI_DISCLOSURE_PATTERNS:
            if re.search(pattern, response):
                score -= 0.7

        # 检查是否使用"我"
        if len(response) > 10 and '我' not in response:
            score -= 0.3

        return max(0.0, min(1.0, score))

    @classmethod
    def role_marker_density(cls, response: str) -> float:
        """
        计算角色标志词密度。
        返回 0.0-1.0，越高表示越贴近角色知识域。
        """
        if not response:
            return 0.0

        marker_count = sum(1 for m in cls.FIREFLY_MARKERS if m in response)
        # 标准化：每 50 字中出现的标志词数量，最多计 0.3
        normalized = min(marker_count / max(1, len(response) / 50), 1.0) * 0.3

        return normalized

    @classmethod
    def tone_score(cls, response: str) -> float:
        """
        评估语气是否符合流萤风格。
        返回 0.0-1.0，越高越贴合。
        """
        score = 0.5  # 默认中性

        # 正面特征：停顿、犹豫标记
        positive_count = sum(1 for m in cls.TONE_MARKERS_POSITIVE if m in response)
        score += min(positive_count * 0.05, 0.2)

        # 负面特征：过度感叹号
        exclamation_count = response.count('！') + response.count('!')
        if exclamation_count > 3:
            score -= 0.15

        # 负面特征：emoji
        emoji_pattern = re.compile(
            r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF'
            r'\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF'
            r'☀-⛿✀-➿]'
        )
        if emoji_pattern.search(response):
            score -= 0.2

        # 负面特征：网络用语
        for word in cls.TONE_MARKERS_NEGATIVE:
            if word in response:
                score -= 0.1

        # 短句偏好（流萤风格倾向短句）
        sentences = re.split(r'[。！？.!?\n]', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            avg_len = sum(len(s) for s in sentences) / len(sentences)
            if avg_len < 40:  # 短句加分
                score += 0.1
            elif avg_len > 100:  # 长句扣分
                score -= 0.1

        return max(0.0, min(1.0, score))

    @classmethod
    def score(cls, response: str) -> dict:
        """
        综合评分。
        返回包含各子项分数和综合分的字典。
        """
        fp_score = cls.first_person_compliance(response)
        marker_score = cls.role_marker_density(response)
        tone = cls.tone_score(response)

        # 加权综合：第一人称 50%，标志词密度 25%，语气 25%
        overall = fp_score * 0.5 + marker_score * 0.25 + tone * 0.25

        return {
            'first_person_compliance': round(fp_score, 3),
            'role_marker_density': round(marker_score, 3),
            'tone_score': round(tone, 3),
            'overall': round(overall, 3),
        }


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    test_responses = [
        ("嗨～我是流萤。我以前是个铁骑战士，现在是星核猎手的一员。", "正确角色风格"),
        ("流萤是一个来自格拉默的角色，她是星核猎手成员。", "第三人称OOC"),
        ("作为一个AI语言模型，我可以回答你的问题。", "AI自我暴露"),
        ("嗯……也许今天可以去天台看看星星。你想一起去吗？", "标准流萤语气"),
        ("哈哈哈哈！！！绝绝子！！！这个太好笑了！！！", "过度表达OOC"),
    ]

    for response, label in test_responses:
        result = cls.score(response)
        print(f"\n[{label}] 综合分: {result['overall']}")
        print(f"  回答: {response[:60]}...")
        print(f"  第一人称: {result['first_person_compliance']}, "
              f"标志词密度: {result['role_marker_density']}, "
              f"语气: {result['tone_score']}")
