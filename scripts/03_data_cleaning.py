"""
流萤训练数据清洗与质量验证
- 去重（Jaccard + 精确匹配）
- 质量过滤（长度、第三人称自指、OOC检测）
- 中文文本规范化
- 生成清洗报告
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
INPUT_PATH = PROJECT_ROOT / "data" / "firefly_training.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "firefly_training_cleaned.json"
REPORT_PATH = PROJECT_ROOT / "data" / "cleaning_report.json"

# ============================================================
# Firefly 角色标志词（输出中至少出现1个）
# ============================================================
FIREFLY_MARKERS = [
    '流萤', '萨姆', '开拓者', '星核猎手', '失熵症', '格拉默',
    '萤火虫', '燃烧', '剧本', '艾利欧', '匹诺康尼', '银狼',
    '卡芙卡', '刃', '装甲', '铁骑', '梦', '星空', '生命',
    '活下去', '普通', '愿望', '光芒', '夜晚', '星星', '天空',
    '蛋糕卷', '手账', '天台', '火萤', 'AR-26710', '虫群',
]

# ============================================================
# OOC 禁用词
# ============================================================
FORBIDDEN_WORDS = [
    '绝绝子', 'yyds', 'YYDS', '栓Q', '芭比Q', 'emo', '破防',
    '内卷', '躺平', '宝子', '集美', '家人们', '老铁', '摆烂',
    '666', '233', 'awsl', 'xswl', '嗑到了', '上头',
    '作为AI', '作为一个人工智能', '我是AI', '语言模型', '大模型',
    '哈哈哈哈', '笑死', '绝了', '太酷啦',
]


def jaccard_similarity(text1, text2):
    """计算两段文本的 Jaccard 相似度"""
    set1 = set(text1)
    set2 = set(text2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0


def normalize_chinese(text):
    """中文文本规范化"""
    # 全角转半角
    text = text.replace('，', ',').replace('。', '.').replace('！', '!')
    text = text.replace('？', '?').replace('；', ';').replace('：', ':')
    text = text.replace('（', '(').replace('）', ')').replace('＂', '"')
    # 再转回全角中文标点（保持中文习惯）
    text = text.replace(',', '，').replace('.', '。').replace('!', '！')
    text = text.replace('?', '？').replace(';', '；').replace(':', '：')
    text = text.replace('(', '（').replace(')', '）')
    # 统一省略号
    text = re.sub(r'\.{2,}', '…', text)
    text = re.sub(r'…{2,}', '…', text)
    # 去除多余空白
    text = re.sub(r'\s+', '', text)
    return text


def validate_pair(pair, index):
    """验证单个训练对的质量，返回 issues 列表"""
    issues = []
    output = pair.get("output", "")
    instruction = pair.get("instruction", "")

    # 1. 空输出检查
    if not output or not output.strip():
        issues.append({"type": "empty_output", "severity": "critical"})
        return issues

    # 2. 长度检查
    if len(output) < 10:
        issues.append({"type": "too_short", "severity": "high", "detail": f"len={len(output)}"})
    if len(output) > 800:
        issues.append({"type": "too_long", "severity": "medium", "detail": f"len={len(output)}"})

    # 3. 第三人称自指检查
    if re.search(r'流萤(是|来自|她|这个角色|的编号)', output):
        issues.append({"type": "third_person_self_ref", "severity": "critical"})

    # 4. AI 身份暴露检查
    if re.search(r'(我是|作为)(AI|人工智能|语言模型|大模型|程序|机器人)', output):
        issues.append({"type": "ai_disclosure", "severity": "critical"})

    # 5. 禁用词检查
    for word in FORBIDDEN_WORDS:
        if word in output:
            issues.append({"type": "forbidden_word", "severity": "high", "detail": word})

    # 6. Firefly 标志词检查
    has_marker = any(marker in output for marker in FIREFLY_MARKERS)
    if not has_marker and pair.get("category") != "防OOC":
        issues.append({"type": "no_firefly_marker", "severity": "low"})

    # 7. Emoji 检查
    if re.search(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', output):
        issues.append({"type": "emoji_found", "severity": "medium"})

    # 8. 感叹号过多检查
    if output.count('！') + output.count('!') > 5:
        issues.append({"type": "too_many_exclamation", "severity": "low"})

    # 9. 输出与指令相同检查
    if instruction.strip() == output.strip():
        issues.append({"type": "output_equals_instruction", "severity": "high"})

    return issues


def clean_dataset(pairs):
    """主清洗函数"""
    stats = {
        "original_count": len(pairs),
        "removed": {"by_category": Counter(), "by_issue": Counter()},
        "issues_found": Counter(),
    }

    # 第一阶段：单条质量验证
    valid_pairs = []
    for i, pair in enumerate(pairs):
        issues = validate_pair(pair, i)
        if issues:
            critical = [iss for iss in issues if iss["severity"] == "critical"]
            if critical:
                for iss in issues:
                    stats["removed"]["by_issue"][iss["type"]] += 1
                stats["removed"]["by_category"][pair.get("category", "unknown")] += 1
                continue  # 跳过有严重问题的条目
            for iss in issues:
                stats["issues_found"][iss["type"]] += 1

        # 规范化
        pair["output"] = normalize_chinese(pair["output"])
        pair["instruction"] = normalize_chinese(pair["instruction"])
        valid_pairs.append(pair)

    stats["after_validation"] = len(valid_pairs)

    # 第二阶段：精确去重（按 output 的 hash）
    seen_hashes = set()
    deduped_pairs = []
    for pair in valid_pairs:
        h = hashlib.md5(pair["output"].encode('utf-8')).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped_pairs.append(pair)
        else:
            stats["removed"]["by_issue"]["exact_duplicate"] += 1

    stats["after_exact_dedup"] = len(deduped_pairs)

    # 第三阶段：Jaccard 相似度去重（output 相似度 > 0.85）
    final_pairs = []
    for i, pair in enumerate(deduped_pairs):
        is_dup = False
        for existing in final_pairs[-50:]:  # 只检查最近50条，提高效率
            if jaccard_similarity(pair["output"], existing["output"]) > 0.85:
                is_dup = True
                break
        if not is_dup:
            final_pairs.append(pair)
        else:
            stats["removed"]["by_issue"]["jaccard_duplicate"] += 1

    stats["final_count"] = len(final_pairs)

    # 类别分布
    category_counts = Counter(p["category"] for p in final_pairs)
    stats["category_distribution"] = dict(category_counts)

    # 平均输出长度
    avg_len = sum(len(p["output"]) for p in final_pairs) / len(final_pairs) if final_pairs else 0
    stats["avg_output_length"] = round(avg_len, 1)

    return final_pairs, stats


def main():
    print("=" * 60)
    print("流萤训练数据清洗工具")
    print("=" * 60)

    # 加载数据
    if not INPUT_PATH.exists():
        print(f"❌ 输入文件不存在: {INPUT_PATH}")
        print("请先运行 02_generate_training_pairs.py 生成训练数据")
        return

    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        pairs = json.load(f)

    print(f"\n加载训练数据: {len(pairs)} 条")

    # 清洗
    cleaned_pairs, stats = clean_dataset(pairs)

    # 保存清洗后的数据
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(cleaned_pairs, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 清洗完成!")
    print(f"   原始: {stats['original_count']} 条")
    print(f"   验证通过: {stats['after_validation']} 条")
    print(f"   去重后: {stats['after_exact_dedup']} 条")
    print(f"   最终: {stats['final_count']} 条")
    print(f"   剔除: {stats['original_count'] - stats['final_count']} 条")

    print(f"\n类别分布:")
    for cat, count in sorted(stats["category_distribution"].items()):
        print(f"  {cat}: {count} 条")

    print(f"\n平均输出长度: {stats['avg_output_length']} 字")

    if stats['issues_found']:
        print(f"\n质量问题（已保留但标记）:")
        for issue_type, count in stats['issues_found'].most_common():
            print(f"  {issue_type}: {count} 条")

    if any(stats['removed']['by_issue'].values()):
        print(f"\n已移除原因:")
        for issue_type, count in stats['removed']['by_issue'].most_common():
            print(f"  {issue_type}: {count} 条")

    # 保存报告
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n📊 清洗报告: {REPORT_PATH}")

    # 质量评级
    final = stats['final_count']
    if final >= 500 and stats['avg_output_length'] >= 50:
        grade = "A - 优秀，可以开始训练"
    elif final >= 300:
        grade = "B - 良好，建议扩充到500+再训练"
    else:
        grade = "C - 不足，需要更多训练数据"
    print(f"\n质量评级: {grade}")


if __name__ == "__main__":
    main()
