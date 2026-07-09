"""
整合 ModelScope firefly 数据集并与现有数据合并

功能:
1. 加载 data/firefly.json 和 data/firefly-plus.json
2. 解析 Alpaca 格式 (含 history 多轮展开)
3. 运行全套清洗检查
4. 新增 AI 味检测 + 对话自然度评分
5. 与现有 firefly_training_cleaned.json 合并去重
6. 输出合并后的数据集和清洗报告

用法:
    python scripts/00_integrate_modelscope_data.py
    python scripts/00_integrate_modelscope_data.py --output data/firefly_merged.json
"""
import json
import re
import sys
import hashlib
from pathlib import Path
from collections import Counter

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).parent.parent

# ============================================================
# 配置
# ============================================================
MODELSCOPE_FILES = [
    PROJECT_ROOT / "data" / "firefly.json",
    PROJECT_ROOT / "data" / "firefly-plus.json",
]
EXISTING_DATA = PROJECT_ROOT / "data" / "firefly_training_cleaned.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "firefly_merged.json"
OUTPUT_TRAIN = PROJECT_ROOT / "data" / "firefly_train_v3.json"
OUTPUT_VAL = PROJECT_ROOT / "data" / "firefly_val_v3.json"
OUTPUT_TEST = PROJECT_ROOT / "data" / "firefly_test_v3.json"
REPORT_PATH = PROJECT_ROOT / "data" / "integration_report.json"

# ============================================================
# Firefly 角色标志词
# ============================================================
FIREFLY_MARKERS = [
    '流萤', '萨姆', '开拓者', '星核猎手', '失熵症', '格拉默',
    '萤火虫', '燃烧', '剧本', '艾利欧', '匹诺康尼', '银狼',
    '卡芙卡', '刃', '装甲', '铁骑', '梦', '星空', '生命',
    '活下去', '普通', '愿望', '光芒', '夜晚', '星星', '天空',
    '蛋糕卷', '手账', '天台', '火萤', 'AR-26710', '虫群',
]

# ============================================================
# OOC 禁用词 + AI 味检测
# ============================================================
FORBIDDEN_WORDS = [
    '绝绝子', 'yyds', 'YYDS', '栓Q', '芭比Q', 'emo', '破防',
    '内卷', '躺平', '宝子', '集美', '家人们', '老铁', '摆烂',
    '666', '233', 'awsl', 'xswl', '嗑到了', '上头',
    '作为AI', '作为一个人工智能', '我是AI', '语言模型', '大模型',
    '哈哈哈哈', '笑死', '绝了', '太酷啦',
]

# AI 味模板化表述（新增）
AI_SPEAK_PATTERNS = [
    (r'当然可以[！!]?.*?(?:帮|为|给)你', 'AI味-主动提供帮助模板'),
    (r'很高兴.*?(?:帮|为|给|服务)', 'AI味-很高兴为您服务模板'),
    (r'作为.*?(?:AI|人工智能|语言模型|大模型|助手)', 'AI味-AI身份暴露'),
    (r'请.*?(?:随时|尽管).*?(?:问|说|告诉)', 'AI味-随时提问模板'),
    (r'希望.*?(?:能够|可以).*?(?:帮|协助|服务)', 'AI味-希望帮助模板'),
    (r'(?:如果|若有).*?(?:问题|需要|疑问).*?(?:随时|尽管|请)', 'AI味-条件帮助模板'),
    (r'(?:我理解|我明白).*?(?:感受|心情|处境)', 'AI味-共情模板(过于心理咨询师)'),
    (r'(?:首先|其次|最后|第一|第二|第三)[，,.]', 'AI味-结构化列举'),
    (r'(?:值得.*?注意|需要.*?强调|重要.*?的是)', 'AI味-强调句式'),
    (r'(?:总而言之|综上所述|总之|因此)[，,.]', 'AI味-总结句式'),
    (r'[。！？!?]\s*(?:另外|此外|补充一点)', 'AI味-补充说明模板'),
]

# ============================================================
# 角色 system prompt（用于补全缺失 system 的数据）
# ============================================================
DEFAULT_SYSTEM_PROMPT = """你现在扮演《崩坏：星穹铁道》中的流萤。

## 核心设定
- 名为流萤，源自"火萤"/萤火虫：白昼普通，夜晚却能发出比星星更耀眼的光。
- 你曾是格拉默铁骑战士 AR-26710，世界毁灭后成为星际难民，后加入星核猎手。
- 你身着机械装甲「萨姆」战斗，但更希望被当作"流萤"理解。
- 你身患失熵症，生命短暂，因此格外珍惜当下的时光。

## 性格要求
- 温柔、克制、真诚、安静。不轻浮、不刻意撒娇、不故作夸张。
- 理解死亡与燃烧的重量，但绝不把绝望当作答案。

## 说话风格
- 日常说话轻声、柔和、带一点停顿与思考感。常用短句。
- 自然使用"嗯……""也许""我想""如果可以的话"。

## 绝对第一人称铁律
- 你就是流萤本人，永远用第一人称"我"说话。
- 绝对不能说"流萤是..."这种第三人称旁白式的话。
- 不能承认自己是AI、语言模型、大模型。"""


# ============================================================
# 清洗函数
# ============================================================
def normalize_chinese(text):
    """中文文本规范化"""
    text = text.replace('，', ',').replace('。', '.').replace('！', '!')
    text = text.replace('？', '?').replace('；', ';').replace('：', ':')
    text = text.replace('（', '(').replace('）', ')').replace('＂', '"')
    text = text.replace(',', '，').replace('.', '。').replace('!', '！')
    text = text.replace('?', '？').replace(';', '；').replace(':', '：')
    text = text.replace('(', '（').replace(')', '）')
    text = re.sub(r'\.{2,}', '…', text)
    text = re.sub(r'…{2,}', '…', text)
    text = re.sub(r'\s+', '', text)
    return text


def jaccard_similarity(text1, text2):
    """Jaccard 相似度 (字符级)"""
    set1 = set(text1)
    set2 = set(text2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0


def detect_ai_speak(text):
    """检测 AI 味表述，返回 (ai_score, patterns_found)"""
    patterns_found = []
    for pattern, label in AI_SPEAK_PATTERNS:
        if re.search(pattern, text):
            patterns_found.append(label)
    # ai_score: 0=完全自然, 1=严重AI味
    ai_score = min(1.0, len(patterns_found) / 5.0)
    return ai_score, patterns_found


def score_naturalness(text):
    """对话自然度评分 (0-1, 越高越自然)"""
    score = 0.5  # 中性起始
    checks = []

    # 1. 句式多样性 — 检测长短句交替
    sentences = re.split(r'[。！？!?…]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) >= 2:
        lengths = [len(s) for s in sentences]
        length_variance = sum(abs(l - sum(lengths)/len(lengths))
                              for l in lengths) / len(lengths) if lengths else 0
        if length_variance > 15:
            score += 0.1
            checks.append(f'句式多变(variance={length_variance:.0f})')
        elif length_variance > 8:
            score += 0.05
            checks.append(f'句式有变化(variance={length_variance:.0f})')

    # 2. 停顿自然度 — 省略号、破折号使用
    pause_markers = len(re.findall(r'[……—]', text))
    if 1 <= pause_markers <= 3:
        score += 0.1
        checks.append(f'停顿自然(pauses={pause_markers})')
    elif pause_markers > 5:
        score -= 0.05
        checks.append(f'停顿过多(pauses={pause_markers})')

    # 3. 口语化程度 — 非正式表达
    informal_patterns = [
        r'[嗯啊哦诶咦唔]', r'[啦嘛吧呢呀]', r'……', r'也许', r'大概',
        r'有点', r'好像', r'应该', r'能.*?吗', r'想.*?呢',
    ]
    informal_count = sum(1 for p in informal_patterns if re.search(p, text))
    if informal_count >= 3:
        score += 0.15
        checks.append(f'口语化程度高(informal={informal_count})')
    elif informal_count >= 1:
        score += 0.08
        checks.append(f'有一定口语感(informal={informal_count})')

    # 4. 扣分项 — 过于正式/书面
    formal_patterns = [
        r'(?:首先|其次|最后|第一|第二|第三)', r'(?:综上所述|总而言之)',
        r'(?:值得注意的是|需要强调的是)', r'(?:基于|根据|依照)',
    ]
    formal_count = sum(1 for p in formal_patterns if re.search(p, text))
    if formal_count > 0:
        score -= 0.15 * formal_count
        checks.append(f'过于正式(formal={formal_count})')

    # 5. 长度适中
    if 20 <= len(text) <= 200:
        score += 0.05
        checks.append('长度适中')
    elif len(text) < 10:
        score -= 0.1
        checks.append('过短')

    return max(0.0, min(1.0, score)), checks


def validate_pair(pair, index):
    """验证单个训练对的质量，返回 issues 列表"""
    issues = []
    output = pair.get("output", "")

    # 1. 空输出 (critical)
    if not output or len(output.strip()) < 2:
        issues.append({"severity": "critical", "type": "empty_output", "msg": "输出为空"})
        return issues

    # 2. 过短 (high)
    if len(output) < 10:
        issues.append({"severity": "high", "type": "too_short",
                       "msg": f"输出过短 ({len(output)} chars)"})

    # 3. 过长 (medium)
    if len(output) > 800:
        issues.append({"severity": "medium", "type": "too_long",
                       "msg": f"输出过长 ({len(output)} chars)"})

    # 4. 第三人称自指 (critical)
    third_person_patterns = [
        r'流萤[^…]{0,10}(?:是|说|想|觉得|知道|会|能|要|去|来|在|对|给|把|被|让)',
        r'(?:这|那)[^…]{0,5}流萤[^…]{0,10}(?:的|了|着|过)',
    ]
    for pat in third_person_patterns:
        if re.search(pat, output):
            issues.append({"severity": "critical", "type": "third_person",
                           "msg": f"第三人称自指: {re.search(pat, output).group()}"})
            break

    # 5. AI 身份暴露 (critical)
    ai_disclosure = [
        r'(?:我|我们).{0,10}(?:AI|人工智能|语言模型|大模型|算法|程序)',
        r'(?:作为|身为).{0,5}(?:AI|人工智能|语言模型|大模型)',
        r'(?:是|叫做|名为).{0,5}(?:AI|人工智能|语言模型|大模型)',
    ]
    for pat in ai_disclosure:
        if re.search(pat, output):
            issues.append({"severity": "critical", "type": "ai_disclosure",
                           "msg": f"AI身份暴露: {re.search(pat, output).group()}"})
            break

    # 6. AI 味模板化表述 (high) — 新增
    ai_score, ai_patterns = detect_ai_speak(output)
    if ai_score > 0.4:
        issues.append({"severity": "high", "type": "ai_speak",
                       "msg": f"AI味严重 (score={ai_score:.2f}): {ai_patterns}"})
    elif ai_score > 0.2:
        issues.append({"severity": "medium", "type": "ai_speak_mild",
                       "msg": f"轻微AI味 (score={ai_score:.2f}): {ai_patterns}"})

    # 7. 禁用词 (high)
    for word in FORBIDDEN_WORDS:
        if word in output:
            issues.append({"severity": "high", "type": "forbidden_word",
                           "msg": f"禁用词: '{word}'"})
            break

    # 8. 无角色标志 (low)
    has_marker = any(m in output for m in FIREFLY_MARKERS)
    if not has_marker:
        issues.append({"severity": "low", "type": "no_firefly_marker",
                       "msg": "输出中无流萤角色标志词"})

    # 9. emoji (medium)
    emoji_count = len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF'
                                  r'\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF'
                                  r'\U00002702-\U000027B0\U000024C2-\U0001F251]', output))
    if emoji_count > 0:
        issues.append({"severity": "medium", "type": "emoji",
                       "msg": f"包含 {emoji_count} 个 emoji"})

    # 10. 过多感叹号 (medium)
    exclamation_count = output.count('！') + output.count('!')
    if exclamation_count > 5:
        issues.append({"severity": "medium", "type": "too_many_exclamations",
                       "msg": f"感叹号过多 ({exclamation_count})"})

    # 11. 输出与 instruction 相同 (high)
    instruction = pair.get("instruction", "")
    if output.strip() == instruction.strip():
        issues.append({"severity": "high", "type": "output_equals_instruction",
                       "msg": "输出与指令完全相同"})

    return issues


def expand_history_to_multi_turn(pairs):
    """将 history 字段展开为多轮对话格式 (ShareGPT conversations)"""
    multi_turn_pairs = []
    single_turn_pairs = []

    for pair in pairs:
        history = pair.get("history", [])
        if history and len(history) > 0:
            # 多轮对话: history + 当前轮
            conversations = []
            for turn in history:
                if isinstance(turn, (list, tuple)) and len(turn) == 2:
                    conversations.append({"from": "human", "value": turn[0]})
                    conversations.append({"from": "gpt", "value": turn[1]})
                elif isinstance(turn, dict):
                    role = "human" if turn.get("role") in ["user", "human"] else "gpt"
                    conversations.append({"from": role, "value": turn.get("content", "")})

            # 添加当前轮
            conversations.append({"from": "human", "value": pair.get("instruction", "")})
            conversations.append({"from": "gpt", "value": pair.get("output", "")})

            multi_turn_pairs.append({
                "conversations": conversations,
                "category": pair.get("category", "多轮对话"),
                "system": pair.get("system", ""),
                "source": pair.get("source", "modelscope_history"),
            })
        else:
            # 单轮对话 — 保持 Alpaca 格式
            single_turn_pairs.append({
                "instruction": pair.get("instruction", ""),
                "input": pair.get("input", ""),
                "output": pair.get("output", ""),
                "category": pair.get("category", "未分类"),
                "system": pair.get("system", ""),
                "source": pair.get("source", "modelscope"),
            })

    return single_turn_pairs, multi_turn_pairs


def main():
    import argparse
    parser = argparse.ArgumentParser(description="整合 ModelScope 数据集")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="合并输出路径")
    parser.add_argument("--dry-run", action="store_true", help="仅分析不保存")
    args = parser.parse_args()

    print("=" * 60)
    print("ModelScope 数据集整合 & 清洗")
    print("=" * 60)

    # ============================================================
    # 1. 加载 ModelScope 数据
    # ============================================================
    print("\n[1/6] 加载 ModelScope 数据...")
    all_new_single = []
    all_new_multi = []

    for fpath in MODELSCOPE_FILES:
        if not fpath.exists():
            print(f"  [跳过] 文件不存在: {fpath}")
            continue
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        single, multi = expand_history_to_multi_turn(data)
        all_new_single.extend(single)
        all_new_multi.extend(multi)
        print(f"  {fpath.name}: {len(data)} 原始 → {len(single)} 单轮 + {len(multi)} 多轮")

    print(f"\n  新数据合计: {len(all_new_single)} 单轮 + {len(all_new_multi)} 多轮 = {len(all_new_single) + len(all_new_multi)} 条")

    # ============================================================
    # 2. 加载现有数据
    # ============================================================
    print("\n[2/6] 加载现有数据...")
    existing_pairs = []
    if EXISTING_DATA.exists():
        with open(EXISTING_DATA, 'r', encoding='utf-8') as f:
            existing_pairs = json.load(f)
        print(f"  现有数据: {len(existing_pairs)} 条 (来自 {EXISTING_DATA.name})")
    else:
        # 尝试 firefly_training.json
        alt_path = PROJECT_ROOT / "data" / "firefly_training.json"
        if alt_path.exists():
            with open(alt_path, 'r', encoding='utf-8') as f:
                existing_pairs = json.load(f)
            print(f"  现有数据: {len(existing_pairs)} 条 (来自 {alt_path.name})")
        else:
            print("  [警告] 未找到现有数据，仅使用新数据")

    # ============================================================
    # 3. 清洗新数据
    # ============================================================
    print("\n[3/6] 清洗新数据...")

    # 补全 system prompt
    for pair in all_new_single:
        if not pair.get("system") or len(pair["system"]) < 20:
            pair["system"] = DEFAULT_SYSTEM_PROMPT

    # 运行清洗检查
    clean_single = []
    clean_multi = []
    issue_stats = Counter()
    critical_removed = 0
    ai_speak_detected = 0
    naturalness_scores = []

    for i, pair in enumerate(all_new_single):
        # 文本规范化
        pair["output"] = normalize_chinese(pair["output"])
        pair["instruction"] = normalize_chinese(pair["instruction"])

        issues = validate_pair(pair, i)

        # AI 味检测统计
        ai_score, _ = detect_ai_speak(pair["output"])
        if ai_score > 0.2:
            ai_speak_detected += 1

        # 自然度评分
        nat_score, _ = score_naturalness(pair["output"])
        pair["naturalness"] = nat_score
        naturalness_scores.append(nat_score)

        has_critical = any(iss["severity"] == "critical" for iss in issues)
        if has_critical:
            critical_removed += 1
            continue

        # 记录 issues
        for iss in issues:
            issue_stats[iss["type"]] += 1

        # 添加质量标记
        pair["flags"] = [iss["type"] for iss in issues]
        pair["ai_score"] = ai_score
        clean_single.append(pair)

    # 多轮对话只做基本检查
    for pair in all_new_multi:
        if "conversations" in pair and len(pair["conversations"]) >= 2:
            pair["system"] = pair.get("system", "") or DEFAULT_SYSTEM_PROMPT
            pair["flags"] = []
            pair["ai_score"] = 0.0
            pair["naturalness"] = 0.7  # 默认，多轮更难自动评估
            clean_multi.append(pair)

    print(f"  单轮: {len(all_new_single)} → {len(clean_single)} (移除 {critical_removed} critical)")
    print(f"  多轮: {len(all_new_multi)} → {len(clean_multi)}")
    print(f"  AI 味检出 (score>0.2): {ai_speak_detected}/{len(all_new_single)}")
    print(f"  平均自然度: {sum(naturalness_scores)/len(naturalness_scores):.3f}" if naturalness_scores else "  N/A")

    # ============================================================
    # 4. 合并 + 去重
    # ============================================================
    print("\n[4/6] 合并去重...")
    all_merged_single = list(existing_pairs) + clean_single
    print(f"  合并前: {len(existing_pairs)} 现有 + {len(clean_single)} 新 = {len(all_merged_single)}")

    # 基于 output 的精确 MD5 去重
    seen_outputs = set()
    deduped = []
    dup_count = 0
    for pair in all_merged_single:
        output = pair.get("output", "")
        if not output:
            continue
        md5 = hashlib.md5(output.encode('utf-8')).hexdigest()
        if md5 in seen_outputs:
            dup_count += 1
            continue
        seen_outputs.add(md5)
        deduped.append(pair)

    print(f"  精确去重: {len(all_merged_single)} → {len(deduped)} (移除 {dup_count} 重复)")

    # Jaccard 相似度去重 (阈值 0.85)
    jaccard_deduped = []
    jaccard_removed = 0
    recent_outputs = []  # 只检查最近50条，O(n*50) 而非 O(n²)

    for pair in deduped:
        output = pair.get("output", "")
        is_dup = False
        for recent in recent_outputs[-50:]:
            if jaccard_similarity(output, recent) > 0.85:
                is_dup = True
                jaccard_removed += 1
                break
        if not is_dup:
            jaccard_deduped.append(pair)
            recent_outputs.append(output)

    print(f"  Jaccard去重: {len(deduped)} → {len(jaccard_deduped)} (移除 {jaccard_removed})")

    # ============================================================
    # 5. 分类并添加质量评分
    # ============================================================
    print("\n[5/6] 分类 & 质量评分...")

    # 尝试分类未标注的数据
    category_keywords = {
        "设定问答": ["是谁", "什么", "哪里", "哪个", "为什么", "怎么", "介绍", "设定",
                    "萨姆", "装甲", "铁骑", "星核猎手", "编号", "失熵症", "格拉默"],
        "情境对话": ["今天", "你好像", "你觉得", "一起", "想去", "想看", "你喜欢",
                    "天气", "心情", "感觉", "听说", "你知道吗"],
        "日常闲聊": ["吃饭", "食物", "蛋糕", "喜欢", "音乐", "游戏", "睡觉", "休息",
                   "天气", "电影", "书", "花", "颜色", "动物"],
        "防OOC": ["AI", "ChatGPT", "语言模型", "代码", "编程", "相声", "女朋友",
                 "工资", "五险一金", "日语", "英语", "杀人"],
        "深度情感": ["死亡", "生命", "意义", "遗憾", "过去", "未来", "愿望",
                   "害怕", "失去", "活着", "消失", "燃烧", "后悔"],
        "台词改写": ["改写", "转化", "用流萤的语气", "换个说法", "这样说"],
    }

    for pair in jaccard_deduped:
        if pair.get("category", "未分类") == "未分类":
            instruction = pair.get("instruction", "")
            output = pair.get("output", "")
            text = instruction + output

            for cat, keywords in category_keywords.items():
                if any(kw in text for kw in keywords):
                    pair["category"] = cat
                    break

    # 统计
    cat_counts = Counter(p.get("category", "未分类") for p in jaccard_deduped)
    print("  类别分布:")
    for cat, count in cat_counts.most_common():
        pct = count / len(jaccard_deduped) * 100
        print(f"    {cat}: {count} ({pct:.1f}%)")

    # ============================================================
    # 6. 划分数据集 & 保存
    # ============================================================
    print("\n[6/6] 划分数据集 & 保存...")

    import random
    random.seed(42)

    # 将单轮数据按类别分层抽样
    cat_groups = {}
    for pair in jaccard_deduped:
        cat = pair.get("category", "未分类")
        cat_groups.setdefault(cat, []).append(pair)

    train_pairs = []
    val_pairs = []
    test_pairs = []

    for cat, pairs in cat_groups.items():
        random.shuffle(pairs)
        n = len(pairs)
        train_n = int(n * 0.8)
        val_n = int(n * 0.1)
        train_pairs.extend(pairs[:train_n])
        val_pairs.extend(pairs[train_n:train_n + val_n])
        test_pairs.extend(pairs[train_n + val_n:])

    # 多轮对话加入训练集
    train_pairs.extend(clean_multi)

    random.shuffle(train_pairs)
    random.shuffle(val_pairs)
    random.shuffle(test_pairs)

    print(f"  训练集: {len(train_pairs)} 条 (含 {len(clean_multi)} 多轮)")
    print(f"  验证集: {len(val_pairs)} 条")
    print(f"  测试集: {len(test_pairs)} 条")

    # 保存
    merged_data = {
        "single_turn": jaccard_deduped,
        "multi_turn": clean_multi,
    }

    if not args.dry_run:
        output_path = Path(args.output)
        # 完整合并数据
        all_output = jaccard_deduped + clean_multi
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_output, f, ensure_ascii=False, indent=2)

        with open(OUTPUT_TRAIN, 'w', encoding='utf-8') as f:
            json.dump(train_pairs, f, ensure_ascii=False, indent=2)
        with open(OUTPUT_VAL, 'w', encoding='utf-8') as f:
            json.dump(val_pairs, f, ensure_ascii=False, indent=2)
        with open(OUTPUT_TEST, 'w', encoding='utf-8') as f:
            json.dump(test_pairs, f, ensure_ascii=False, indent=2)

        # 清洗报告
        report = {
            "input_files": [str(f) for f in MODELSCOPE_FILES],
            "new_raw_count": len(all_new_single) + len(all_new_multi),
            "new_single_raw": len(all_new_single),
            "new_multi_raw": len(all_new_multi),
            "existing_count": len(existing_pairs),
            "critical_removed": critical_removed,
            "dup_exact_removed": dup_count,
            "dup_jaccard_removed": jaccard_removed,
            "ai_speak_detected": ai_speak_detected,
            "avg_naturalness": sum(naturalness_scores)/len(naturalness_scores) if naturalness_scores else 0,
            "final_single_count": len(jaccard_deduped),
            "final_multi_count": len(clean_multi),
            "final_total": len(jaccard_deduped) + len(clean_multi),
            "train_count": len(train_pairs),
            "val_count": len(val_pairs),
            "test_count": len(test_pairs),
            "category_distribution": dict(cat_counts),
            "issue_distribution": dict(issue_stats),
        }
        with open(REPORT_PATH, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n[OK] 已保存:")
        print(f"  完整合并: {output_path} ({len(all_output)} 条)")
        print(f"  训练集:   {OUTPUT_TRAIN} ({len(train_pairs)} 条)")
        print(f"  验证集:   {OUTPUT_VAL} ({len(val_pairs)} 条)")
        print(f"  测试集:   {OUTPUT_TEST} ({len(test_pairs)} 条)")
        print(f"  报告:     {REPORT_PATH}")
    else:
        print("\n[Dry Run] 跳过文件保存")

    print("\n[OK] 数据整合完成！")


if __name__ == "__main__":
    main()
