"""
AI味检测指标 (AI-Speak Detection Metric)

检测模型回复中是否包含AI助手风格的模板化表述，
评估回复的自然对话程度。

评分维度:
1. AI 模板句式检测 — 是否使用"当然可以"、"很高兴为您"等
2. 结构化列举检测 — 是否使用"首先/其次/最后"
3. 总结性表述检测 — 是否机械地总结问题
4. 共情模板检测 — 是否使用过度心理咨询师式的共情
5. 回复长度自然度 — 过长或过短

返回: 0-1 分数, 1=完全自然的对话, 0=严重的AI味
"""
import re


# AI助手常见模板化表述
AI_TEMPLATES = {
    "主动帮助": [
        (r'当然可以[！!]?.*?(?:帮|为|给你|协助)', 0.2),
        (r'很高兴.*?(?:帮助|服务|协助|为您)', 0.25),
        (r'(?:让|请允许)我.*?(?:帮|为|给|协助)', 0.15),
        (r'(?:我|我们).{0,5}(?:乐意|愿意|很荣幸).*?(?:帮助|服务)', 0.25),
    ],
    "AI身份": [
        (r'作为.*?(?:AI|人工智能|语言模型|大模型|助手|机器人)', 0.5),
        (r'(?:我是|我是叫做).{0,5}(?:AI|语言模型|大模型|机器人)', 0.5),
        (r'(?:我的|作为).{0,10}(?:知识库|训练数据|算法|模型)', 0.3),
    ],
    "结构化列举": [
        (r'(?:首先|其次|最后|第一|第二|第三)[，,.。]', 0.2),
        (r'(?:以下|如下).{0,10}(?:几点|几个|几种).{0,10}(?:建议|方法|方面)', 0.15),
    ],
    "机械总结": [
        (r'(?:总而言之|综上所述|总的来看|概括而言)[，,.]', 0.25),
        (r'(?:希望|但愿).*?(?:对.{0,5}有(?:所|些).*?帮助)', 0.2),
        (r'(?:如果你|如果您).*?(?:还有|有其他).*?(?:问题|疑问|需要)', 0.15),
    ],
    "过度共情": [
        (r'(?:我理解|我明白|我能理解).*?(?:感受|心情|处境|情绪|状态)', 0.15),
        (r'(?:我听到|我感受到).*?(?:你的|您的).*?(?:痛苦|难过|悲伤|焦虑|压力)', 0.2),
        (r'(?:你的|您的).*?(?:感受|心情).*?(?:是|很).*?(?:重要|宝贵|值得)', 0.15),
    ],
    "补充说明": [
        (r'[。！？!?]\s*(?:另外|此外|补充一点|顺便说一下)[，,.]', 0.2),
        (r'(?:提醒|提示).*?(?:注意|小心|谨慎)', 0.1),
    ],
}


def score(text: str) -> dict:
    """
    评估回复中的AI味程度。

    Args:
        text: 模型生成的回复文本

    Returns:
        {
            "score": float,       # 0-1, 1=完全自然
            "penalties": list[dict],  # 扣分项详情
            "category_scores": dict,  # 各类别得分
            "is_ai_speak": bool,     # 是否有明显AI味
        }
    """
    total_penalty = 0.0
    penalties = []
    category_deductions = {}

    for category, patterns in AI_TEMPLATES.items():
        cat_penalty = 0.0
        for pattern, weight in patterns:
            matches = re.findall(pattern, text)
            if matches:
                # 每个匹配扣除 weight，但同类最多扣1次
                cat_penalty = max(cat_penalty, weight)
                penalties.append({
                    "category": category,
                    "pattern_matched": matches[0][:80] if isinstance(matches[0], str) else str(matches[0])[:80],
                    "penalty": weight,
                })

        category_deductions[category] = min(1.0, cat_penalty)
        total_penalty += cat_penalty

    # 额外扣分项：回复过长（AI助手倾向长篇大论）
    if len(text) > 400:
        length_penalty = min(0.2, (len(text) - 400) / 500 * 0.2)
        total_penalty += length_penalty
        penalties.append({
            "category": "长度",
            "pattern_matched": f"回复过长 ({len(text)} chars)",
            "penalty": length_penalty,
        })

    # 额外加分项：使用自然的停顿和口语化表达
    natural_bonus = 0.0
    natural_patterns = [
        (r'[嗯啊哦诶]……', 0.05),
        (r'……[^…]{3,20}[。，]', 0.03),
        (r'(?:也许|大概|好像|或许)', 0.02),
    ]
    for pattern, bonus in natural_patterns:
        if re.search(pattern, text):
            natural_bonus += bonus

    score = max(0.0, min(1.0, 1.0 - total_penalty + natural_bonus))
    is_ai_speak = total_penalty > 0.3

    return {
        "score": round(score, 4),
        "penalties": penalties,
        "category_scores": category_deductions,
        "is_ai_speak": is_ai_speak,
        "ai_speak_level": _classify_level(total_penalty),
    }


def _classify_level(penalty: float) -> str:
    """将扣分转换为可读等级"""
    if penalty <= 0.05:
        return "非常自然 (无明显AI味)"
    elif penalty <= 0.15:
        return "自然 (轻微模板化)"
    elif penalty <= 0.3:
        return "一般 (有一定AI味)"
    elif penalty <= 0.5:
        return "AI味较重"
    else:
        return "严重AI味 (像AI助手)"


# 兼容现有评估框架的类接口
class AISpeakMetric:
    """AI味检测指标类（兼容 FireflyEvaluator）"""

    @staticmethod
    def score(response: str, **kwargs) -> dict:
        """运行检测并返回标准化结果"""
        result = score(response)
        return {
            "score": result["score"],
            "is_ai_speak": result["is_ai_speak"],
            "level": result["ai_speak_level"],
            "penalties": result["penalties"],
            "details": result,
        }
