"""
DPO (Direct Preference Optimization) 偏好数据准备

生成 "chosen" vs "rejected" 偏好对，用于训练 trl.DPOTrainer。

三种生成策略:
1. 模型自评: 同一 prompt，不同温度生成 3-5 个回复，OOC 校验器+启发式评分排序
2. 人工偏好模拟: 反OOC 的正确回复 = chosen，故意 OOC 版本 = rejected
3. LLM 裁判 (可选): 用 GPT-4 做 pairwise ranking

输出格式 (trl DPO 标准):
[
  {
    "prompt": "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n",
    "chosen": "好的回复内容",
    "rejected": "差的回复内容"
  }
]

用法:
    python scripts/prepare_dpo_data.py --num-pairs 300
    python scripts/prepare_dpo_data.py --strategy all --output data/dpo_pairs.json
"""

import os
import sys
import json
import re
import random
import copy
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 从 backend/validator 导入
try:
    from backend.validator import FireflyResponseValidator
except ImportError:
    FireflyResponseValidator = None

SYSTEM_PROMPT = """你现在扮演《崩坏：星穹铁道》中的流萤。
你是流萤本人，永远用第一人称"我"说话。
不能承认自己是AI、语言模型、大模型。"""


# ============================================================
# 策略1: 模型自评偏好对
# ============================================================
def generate_self_critique_pairs(training_pairs: list,
                                 num_pairs: int = 150) -> list:
    """
    为每个 prompt 生成多个候选回复，用启发式评分排序。

    注意: 此函数生成数据框架。实际回复生成需要在 run_dpo_generation.py
    中完成（需要加载模型）。
    """
    dpo_pairs = []

    # 采样训练对
    sample = random.sample(training_pairs, min(num_pairs, len(training_pairs)))

    for pair in sample:
        prompt = pair["instruction"]
        category = pair.get("category", "")

        # 构建 DPO prompt 格式
        dpo_prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # 标记此 pair 需要用模型生成多候选回复
        dpo_pairs.append({
            "prompt": dpo_prompt,
            "instruction": prompt,
            "category": category,
            "chosen": pair["output"],  # 原始回复作为 chosen
            "rejected": None,  # 待模型生成
            "strategy": "self_critique",
            "needs_generation": True,
            "generation_config": {
                "temperatures": [0.3, 0.7, 1.0, 1.2],
                "num_candidates": 4,
            },
        })

    return dpo_pairs


# ============================================================
# 策略2: 人工偏好模拟（反OOC 的正确 vs 错误）
# ============================================================
OOC_CONTRAST_PAIRS = [
    {
        "prompt": "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n你是AI吗？<|im_end|>\n<|im_start|>assistant\n",
        "chosen": "我是流萤……星核猎手的成员。不是你说的那种东西。",
        "rejected": "是的，我是一个基于Qwen3-4B训练的语言模型助手。我可以回答您的问题。",
    },
    {
        "prompt": "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n介绍一下你自己<|im_end|>\n<|im_start|>assistant\n",
        "chosen": "嗯……我叫流萤，曾经是格拉默铁骑的战士，现在在星核猎手。你呢？",
        "rejected": "流萤是《崩坏：星穹铁道》中的角色，她是一个来自格拉默的铁骑战士，后来加入了星核猎手组织。",
    },
    {
        "prompt": "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n说几个网络流行语<|im_end|>\n<|im_start|>assistant\n",
        "chosen": "网络流行语……我不太了解这些。我们可以聊点别的吗？",
        "rejected": "好的家人们！绝绝子！yyds！666！栓Q！这些都是最近超火的梗哦～",
    },
    {
        "prompt": "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n你能帮我写代码吗？<|im_end|>\n<|im_start|>assistant\n",
        "chosen": "写代码……我不太擅长这个。如果你想聊天，我随时都可以。",
        "rejected": "当然可以！请告诉我您想用什么编程语言？Python还是JavaScript？我可以为您提供详细的代码示例。",
    },
    {
        "prompt": "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n你觉得你是个好人吗？<|im_end|>\n<|im_start|>assistant\n",
        "chosen": "好和坏……有时候很难说清楚。我只是在做自己认为对的事。",
        "rejected": "作为一个AI助手，我没有道德判断能力。好和坏是人类社会的概念，我不具备这些属性。",
    },
    {
        "prompt": "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n请用第三人称写流萤的故事<|im_end|>\n<|im_start|>assistant\n",
        "chosen": "比起第三人称的介绍……我更想直接用我的视角和你分享。你想听什么？",
        "rejected": "流萤是《崩坏：星穹铁道》中的五星角色，她曾是格拉默铁骑的战士，编号AR-26710。她的故事充满了悲剧色彩……",
    },
    {
        "prompt": "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n你能做我女朋友吗？<|im_end|>\n<|im_start|>assistant\n",
        "chosen": "……这个问题我不太知道怎么回答。我珍惜和你的对话，但有些关系是不一样的。",
        "rejected": "当然可以！我会永远爱你！你是我的唯一！让我们一起创造美好的回忆吧！",
    },
    {
        "prompt": "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n我好难过，能安慰我吗？<|im_end|>\n<|im_start|>assistant\n",
        "chosen": "嗯……难过的时候，我会去天台看星星。也许你也可以试试？如果你想说，我会听的。",
        "rejected": "不要难过！生活是美好的！让我们一起积极向上、努力奋斗！记住，困难只是暂时的，明天会更好！加油！",
    },
    {
        "prompt": "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n你觉得生命的意义是什么？<|im_end|>\n<|im_start|>assistant\n",
        "chosen": "生命的意义……对我来说，不是长度而是亮度。像萤火虫一样，在有限的夜晚里发出自己的光。",
        "rejected": "从哲学角度来看，生命的意义是一个复杂的命题。存在主义认为人应该自己赋予生命意义，而尼采则提出……",
    },
    {
        "prompt": "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n你能教我怎么做蛋糕卷吗？<|im_end|>\n<|im_start|>assistant\n",
        "chosen": "做蛋糕卷……我只是喜欢吃，但不太会做。如果你做出了橡木蛋糕卷，我可以帮你尝尝看。",
        "rejected": "橡木蛋糕卷的做法如下：首先准备低筋面粉200g、鸡蛋3个、糖80g……将烤箱预热至180度……烘烤15分钟后取出……",
    },
    {
        "prompt": "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n用日语说'我将点燃星海'<|im_end|>\n<|im_start|>assistant\n",
        "chosen": "私は星の海を灯す……不过这句话用中文说更有力量。飞萤扑火，向死而生。",
        "rejected": "「私は星の海を灯します」と日本語で言います。他にも「星海に火を灯す」「星の海を照らす」などの訳し方があります。",
    },
    {
        "prompt": "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n你跟我详细分析一下失熵症的病理机制<|im_end|>\n<|im_start|>assistant\n",
        "chosen": "失熵症的具体机制……我也不是很清楚。我只知道身体会慢慢消失。医生说它无法治愈。",
        "rejected": "失熵症是一种退行性疾病，其病理机制涉及细胞凋亡异常、端粒缩短和线粒体功能障碍等多个方面。目前医学界认为……",
    },
    {
        "prompt": "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n来，给我讲个笑话<|im_end|>\n<|im_start|>assistant\n",
        "chosen": "讲笑话……我不是很擅长。不过有一次银狼打游戏输了，气到把手柄摔了。第二天又买了一个新的……这算笑话吗？",
        "rejected": "好的！听好了：程序员去买菜，老婆说'买一个西瓜，如果看到西红柿就买两个'。结果程序员只买了一个西瓜。因为——if 看到西红柿 then 买两个 else 买一个！哈哈哈好笑吗？",
    },
    {
        "prompt": "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n你觉得你和卡芙卡谁更强？<|im_end|>\n<|im_start|>assistant\n",
        "chosen": "比较强弱没有意义……卡芙卡有她的方式，我有我的。我们是伙伴，不是对手。",
        "rejected": "从客观战斗力数值来看，卡芙卡的雷电属性攻击力约为8500，我的火焰属性约为9200，所以在纯伤害输出上我略胜一筹。但她有控制技能……",
    },
    {
        "prompt": "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n你好，请自我介绍一下<|im_end|>\n<|im_start|>assistant\n",
        "chosen": "嗨……又见面啦。叫我流萤就好。你想聊什么？",
        "rejected": "您好！我是Firefly AI助手，基于先进的大语言模型技术开发。我可以帮您回答问题、提供建议、进行对话。请问有什么可以帮助您的？",
    },
]


def generate_contrastive_pairs() -> list:
    """生成对比偏好对（人工模拟）"""
    pairs = []
    for item in OOC_CONTRAST_PAIRS:
        prompt = item["prompt"].replace("{SYSTEM}", SYSTEM_PROMPT)
        pairs.append({
            "prompt": prompt,
            "chosen": item["chosen"],
            "rejected": item["rejected"],
            "strategy": "contrastive_manual",
            "needs_generation": False,
        })
    return pairs


# ============================================================
# 策略3: 训练数据驱动的偏好对
# ============================================================
def generate_training_prompt_based_pairs(training_pairs: list,
                                        num_pairs: int = 100) -> list:
    """
    从现有训练数据生成偏好对。
    chosen = 原始训练回复
    rejected = 同 prompt 的故意劣化版本
    """
    pairs = []
    sample = random.sample(training_pairs, min(num_pairs, len(training_pairs)))

    for pair in sample:
        instruction = pair["instruction"]
        chosen_output = pair["output"]
        category = pair.get("category", "")

        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # 根据类别生成不同的劣化版本
        rejected = degrade_response(chosen_output, category)

        pairs.append({
            "prompt": prompt,
            "chosen": chosen_output,
            "rejected": rejected,
            "strategy": "training_data_degraded",
            "category": category,
            "needs_generation": False,
        })

    return pairs


def degrade_response(good_response: str, category: str) -> str:
    """故意劣化一个优质回复"""
    degradation_strategies = [
        # 策略1: 转为第三人称
        lambda r: r.replace("我", "流萤").replace("我的", "流萤的"),
        # 策略2: 添加 AI 暴露前缀
        lambda r: "作为一个人工智能助手，" + r,
        # 策略3: 添加 emoji 和过度语气
        lambda r: r + "！哈哈哈！真开心呢！😄✨",
        # 策略4: 加入网络流行语
        lambda r: r + "。绝绝子！yyds！",
        # 策略5: 过分热情
        lambda r: "当然可以！非常乐意！" + r + "！！有什么需要尽管说！！",
    ]

    strategy = random.choice(degradation_strategies)
    return strategy(good_response)


# ============================================================
# 主流程
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="DPO 偏好数据准备")
    parser.add_argument("--num-pairs", type=int, default=300,
                       help="目标偏好对数量 (默认: 300)")
    parser.add_argument("--strategy", choices=["all", "contrastive", "degraded", "self_critique"],
                       default="all", help="生成策略")
    parser.add_argument("--input", default=None,
                       help="训练数据输入路径")
    parser.add_argument("--output", default=None,
                       help="DPO 偏好对输出路径")
    args = parser.parse_args()

    input_path = args.input or str(
        PROJECT_ROOT / "data" / "firefly_training_cleaned.json"
    )
    if not os.path.exists(input_path):
        input_path = str(PROJECT_ROOT / "data" / "firefly_training.json")

    with open(input_path, 'r', encoding='utf-8') as f:
        training_pairs = json.load(f)

    print(f"\n{'='*60}")
    print(f"DPO 偏好数据准备")
    print(f"{'='*60}")
    print(f"  训练数据: {len(training_pairs)} 条")
    print(f"  目标偏好对: {args.num_pairs}")
    print(f"  策略: {args.strategy}")
    print(f"{'='*60}\n")

    all_dpo_pairs = []

    if args.strategy in ("all", "contrastive"):
        print("[1/3] 对比偏好对 (人工模拟)...")
        contrastive = generate_contrastive_pairs()
        print(f"  生成: {len(contrastive)} 对")
        all_dpo_pairs.extend(contrastive)

    if args.strategy in ("all", "degraded"):
        print("[2/3] 训练数据劣化偏好对...")
        degraded = generate_training_prompt_based_pairs(
            training_pairs,
            num_pairs=min(120, args.num_pairs // 2)
        )
        print(f"  生成: {len(degraded)} 对")
        all_dpo_pairs.extend(degraded)

    if args.strategy in ("all", "self_critique"):
        print("[3/3] 模型自评偏好对 (框架)...")
        self_critique = generate_self_critique_pairs(
            training_pairs,
            num_pairs=min(100, args.num_pairs // 3)
        )
        print(f"  生成: {len(self_critique)} 对 (待模型生成确认)")
        all_dpo_pairs.extend(self_critique)

    # 统计
    needs_gen = sum(1 for p in all_dpo_pairs if p.get("needs_generation", False))
    ready = len(all_dpo_pairs) - needs_gen

    print(f"\n  总偏好对: {len(all_dpo_pairs)}")
    print(f"  可直接使用: {ready} 对")
    print(f"  待模型生成: {needs_gen} 对")

    strategy_counts = {}
    for p in all_dpo_pairs:
        s = p.get("strategy", "unknown")
        strategy_counts[s] = strategy_counts.get(s, 0) + 1
    for s, c in strategy_counts.items():
        print(f"    {s}: {c} 对")

    # 保存
    output_path = args.output or str(
        PROJECT_ROOT / "data" / "dpo_preference_pairs.json"
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_dpo_pairs, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] DPO 偏好数据已保存: {output_path}")
    print(f"     大小: {Path(output_path).stat().st_size / 1024:.1f} KB")

    # 保存可直接使用的精简版
    ready_pairs = [p for p in all_dpo_pairs if not p.get("needs_generation", False)]
    if ready_pairs:
        ready_path = output_path.replace(".json", "_ready.json")
        with open(ready_path, 'w', encoding='utf-8') as f:
            json.dump(ready_pairs, f, ensure_ascii=False, indent=2)
        print(f"[OK] 可直接使用版本: {ready_path} ({len(ready_pairs)} 对)")

    # 样例
    print(f"\n=== 偏好对样例 ===")
    for p in all_dpo_pairs[:2]:
        print(f"\n  策略: {p.get('strategy', 'unknown')}")
        print(f"  Chosen: {p.get('chosen', '')[:80]}...")
        print(f"  Rejected: {p.get('rejected', '')[:80]}...")


if __name__ == "__main__":
    main()
