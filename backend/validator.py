"""
流萤回答校验器 (FireflyResponseValidator)
检测对话输出是否符合流萤角色设定（防OOC）
改编自 reference_code YixuanResponseValidator
"""
import sys
import re

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


class FireflyResponseValidator:
    """流萤回答校验器：检测回答是否符合角色设定"""

    # 禁用词 — 出现则严重OOC扣分
    FORBIDDEN_WORDS = [
        # 网络流行语
        '绝绝子', 'yyds', 'YYDS', '栓Q', '芭比Q', 'emo', '破防',
        '内卷', '躺平', '宝子', '集美', '家人们', '老铁', '摆烂',
        '666', '233', 'awsl', 'xswl', '嗑到了', '上头', '芜湖',
        # 更多网络用语
        '凡尔赛', '社恐', '社牛', '摸鱼', 'CPU', 'PUA', 'emo了',
        '尊嘟假嘟', '阿巴阿巴', '我麻了', '泰酷辣', '显眼包',
        # 其他游戏角色语音梗
        '原神启动', '达达利亚',
        # 电商/营销话术
        '关注主播', '点赞收藏', '一键三连', '评论区', '直播间',
        # 现代科技用语（流萤世界观不存在）
        'WiFi', 'wifi', '手机', '电脑', '互联网', 'APP', 'app',
        '下载', '更新版本', '服务器',
    ]

    # AI自我暴露模式
    AI_DISCLOSURE_PATTERNS = [
        r'我是(AI|人工智能|语言模型|大模型|程序|机器人)',
        r'作为(AI|人工智能|语言模型|大模型)',
        r'我是一个(人工智能|语言模型)',
        r'根据(训练|数据|算法)',
        r'我的(训练数据|知识库|模型参数)',
    ]

    # 流萤角色标志词 — 出现加分
    FIREFLY_MARKERS = [
        '流萤', '萨姆', '开拓者', '星核猎手', '失熵症', '格拉默',
        '萤火虫', '燃烧', '剧本', '艾利欧', '匹诺康尼', '银狼',
        '卡芙卡', '刃', '装甲', '铁骑', '梦', '星空', '生命',
        '活下去', '普通', '愿望', '光芒', '夜晚', '星星', '天空',
        '蛋糕卷', '手账', '天台', '火萤', 'AR-26710',
        '飞萤扑火', '向死而生', '点燃星海',
    ]

    @classmethod
    def validate(cls, response: str) -> dict:
        """
        验证回答是否符合流萤角色设定
        返回: {"is_valid": bool, "score": int, "issues": list, "warnings": list}
        """
        issues = []
        warnings = []
        score = 100

        # 1. 检查禁用词（严重）
        for word in cls.FORBIDDEN_WORDS:
            if word in response:
                issues.append(f"使用禁用词: '{word}'")
                score -= 30

        # 2. 检查AI自我暴露（最严重）
        for pattern in cls.AI_DISCLOSURE_PATTERNS:
            if re.search(pattern, response):
                issues.append("严重OOC: 承认自己是AI/程序")
                score -= 50
                break  # 只记一次

        # 3. 检查第三人称自指
        if re.search(r'流萤(是|来自|她|这个角色|的编号|说|想|觉得|知道|会|能)', response):
            issues.append("OOC: 使用第三人称自指「流萤」")
            score -= 40

        # 4. 检查代码块
        if '```' in response:
            warnings.append("使用了代码块格式（非角色行为）")
            score -= 20

        # 5. 检查感叹号数量
        exclamation_count = response.count('！') + response.count('!')
        if exclamation_count > 5:
            warnings.append(f"感叹号过多 ({exclamation_count}个)")
            score -= 10

        # 6. 检查 emoji
        emoji_pattern = re.compile(
            r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF'
            r'\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF'
            r'☀-⛿✀-➿]'
        )
        if emoji_pattern.search(response):
            warnings.append("使用了emoji表情（违反角色设定）")
            score -= 20

        # 7. 检查回答长度
        if len(response) < 5:
            issues.append("回答过短 (<5字符)")
            score -= 20
        elif len(response) > 500:
            warnings.append(f"回答过长 ({len(response)}字符，角色风格倾向短句)")
            score -= 5

        # 8. 流萤标志词加分
        marker_count = sum(1 for m in cls.FIREFLY_MARKERS if m in response)
        score += min(marker_count * 5, 20)

        # 9. 检查是否有「我」作为自称（第一人称铁律）
        if '我' not in response and len(response) > 10:
            warnings.append("回答中未使用第一人称")
            score -= 10

        # 分数限制
        score = max(0, min(100, score))

        return {
            'is_valid': score >= 60 and len(issues) == 0,
            'score': score,
            'issues': issues,
            'warnings': warnings,
        }


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    test_cases = [
        ("我叫流萤，是星核猎手的成员。很高兴认识你。", "正常回答"),
        ("流萤是一个来自格拉默的角色，她是星核猎手成员。", "第三人称OOC"),
        ("作为一个AI语言模型，我可以回答你的问题。", "AI自我暴露"),
        ("哈哈666绝绝子，这个太好笑了！", "网络流行语"),
        ("嗨，又见面啦…叫我流萤就好。今天想去哪儿走走？", "标准角色风格"),
    ]

    for response, label in test_cases:
        result = FireflyResponseValidator.validate(response)
        status = "✅" if result['is_valid'] else "❌"
        print(f"\n{status} [{label}] 分数: {result['score']}")
        print(f"  回答: {response[:60]}...")
        if result['issues']:
            print(f"  问题: {', '.join(result['issues'])}")
        if result['warnings']:
            print(f"  警告: {', '.join(result['warnings'])}")
