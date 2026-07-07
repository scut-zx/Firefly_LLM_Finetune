"""
流萤 LoRA 微调训练脚本
使用 trl.SFTTrainer (与参考项目Yixuan一致的方案)
兼容 Python 3.9 + RTX 4060 Ti 16GB

用法:
    python scripts/train_firefly_lora.py
    python scripts/train_firefly_lora.py --epochs 8 --lora_r 32 --lora_alpha 64
    python scripts/train_firefly_lora.py --train_data data/firefly_train.json --eval_data data/firefly_val.json
"""
import os
import sys
import json
import torch
import argparse
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# ============================================================
# 配置
# ============================================================
MODEL_PATH = str(PROJECT_ROOT / "model")
DATA_PATH = str(PROJECT_ROOT / "data" / "firefly_training.json")
OUTPUT_DIR = str(PROJECT_ROOT / "output" / "Firefly_LoRA")

# LoRA 配置 (与参考项目一致)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

# 训练配置
NUM_EPOCHS = 5
BATCH_SIZE = 2
GRAD_ACCUM = 8          # 有效 batch = 16
LEARNING_RATE = 3e-5
MAX_SEQ_LENGTH = 2048
WARMUP_RATIO = 0.1
LR_SCHEDULER = "cosine"
BF16 = True
SAVE_STEPS = 200
LOGGING_STEPS = 10

# 系统提示词 (与训练数据中的 system prompt 一致)
SYSTEM_PROMPT = """你现在扮演《崩坏：星穹铁道》中的流萤。

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


def load_training_data(data_path, tokenizer=None):
    """加载训练数据并转换为 messages 格式。
    自动检测单轮和多轮格式：
    - 单轮: {"instruction": "...", "output": "...", ...}
    - 多轮: {"conversations": [{"from": "human", "value": "..."}, ...]}
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        pairs = json.load(f)

    messages_list = []
    multi_turn_count = 0
    single_turn_count = 0

    for pair in pairs:
        system = pair.get("system", SYSTEM_PROMPT)

        # 检测多轮格式 (ShareGPT conversations)
        if "conversations" in pair:
            messages = [{"role": "system", "content": system}]
            for turn in pair["conversations"]:
                role = "user" if turn.get("from") == "human" else "assistant"
                messages.append({"role": role, "content": turn.get("value", "")})
            messages_list.append({"messages": messages})
            multi_turn_count += 1
        # 单轮格式 (Alpaca-style)
        elif "instruction" in pair and "output" in pair:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": pair["instruction"]},
                {"role": "assistant", "content": pair["output"]},
            ]
            messages_list.append({"messages": messages})
            single_turn_count += 1
        else:
            print(f"  [警告] 跳过无法识别的数据格式: {list(pair.keys())[:3]}")

    print(f"  训练数据: {len(messages_list)} 条对话 (单轮: {single_turn_count}, 多轮: {multi_turn_count})")
    return messages_list


def format_chatml(example):
    """将 messages 格式化为 Qwen3 ChatML 文本（确保返回纯字符串）"""
    tokenizer = formatting_tokenizer
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    # 确保返回字符串（某些 tokenizer 版本可能返回 list）
    if isinstance(text, list):
        text = "".join(str(t) for t in text)
    return {"text": str(text)}


def main():
    global formatting_tokenizer

    # 解析 CLI 参数
    parser = argparse.ArgumentParser(description="流萤 LoRA 微调训练")
    parser.add_argument("--model", default=MODEL_PATH, help="基础模型路径")
    parser.add_argument("--data", default=DATA_PATH, help="训练数据路径")
    parser.add_argument("--eval_data", default=None, help="验证数据路径（用于 val loss 监控）")
    parser.add_argument("--output", default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--lora_r", type=int, default=LORA_R, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=LORA_ALPHA, help="LoRA alpha")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="训练轮数")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="学习率")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=GRAD_ACCUM, help="梯度累积步数")
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LENGTH, help="最大序列长度")
    parser.add_argument("--early_stopping_patience", type=int, default=0,
                       help="早停耐心值 (0=禁用, 建议3)")
    parser.add_argument("--no_bf16", action="store_true", help="禁用 bf16")
    parser.add_argument("--dry_run", action="store_true", help="仅验证配置，不训练")
    args = parser.parse_args()

    # 更新全局变量
    model_path = args.model
    data_path = args.data
    eval_data_path = args.eval_data
    output_dir = args.output
    lora_r = args.lora_r
    lora_alpha = args.lora_alpha
    num_epochs = args.epochs
    lr = args.lr
    batch_size = args.batch_size
    grad_accum = args.grad_accum
    max_seq_len = args.max_seq_len
    use_bf16 = not args.no_bf16

    print("=" * 60)
    print("流萤 LoRA 微调训练")
    print("=" * 60)
    print(f"模型: {model_path}")
    print(f"数据: {data_path}")
    print(f"验证: {eval_data_path or '无'}")
    print(f"输出: {output_dir}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"LoRA: r={lora_r}, alpha={lora_alpha}")
    print(f"训练: {num_epochs} epochs, batch={batch_size}x{grad_accum}, lr={lr}")
    print()

    if args.dry_run:
        print("[Dry Run] 配置验证通过，跳过训练。")
        return

    # ============================================================
    # 1. 加载 Tokenizer
    # ============================================================
    print("[1/5] 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    formatting_tokenizer = tokenizer

    # ============================================================
    # 2. 加载模型
    # ============================================================
    print("[2/5] 加载模型 (bf16)...")
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # ============================================================
    # 3. 配置 LoRA
    # ============================================================
    print("[3/5] 配置 LoRA...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ============================================================
    # 4. 加载并预Tokenize数据
    # ============================================================
    print("[4/5] 加载并预处理训练数据...")
    raw_data = load_training_data(data_path, tokenizer)
    dataset = Dataset.from_list(raw_data)

    # 先格式化为文本
    dataset = dataset.map(format_chatml, remove_columns=dataset.column_names)
    print(f"  数据集大小: {len(dataset)}")

    # 预Tokenize（trl 0.13+ 需要预tokenize的数据）
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_seq_len,
            return_tensors=None,
        )

    dataset = dataset.map(tokenize_fn, remove_columns=["text"])
    print(f"  预Tokenize完成")

    # 加载验证集（如果提供）
    eval_dataset = None
    if eval_data_path and os.path.exists(eval_data_path):
        print(f"  加载验证数据: {eval_data_path}")
        eval_raw = load_training_data(eval_data_path, tokenizer)
        eval_ds = Dataset.from_list(eval_raw)
        eval_ds = eval_ds.map(format_chatml, remove_columns=eval_ds.column_names)
        eval_dataset = eval_ds.map(tokenize_fn, remove_columns=["text"])
        print(f"  验证集大小: {len(eval_dataset)}")

    # ============================================================
    # 5. 训练
    # ============================================================
    print("[5/5] 开始训练...")
    print()

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=num_epochs,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        bf16=use_bf16,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=42,
        max_seq_length=max_seq_len,
        packing=False,
        eval_strategy="epoch" if eval_dataset else "no",
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    # ============================================================
    # 6. 保存
    # ============================================================
    print()
    print("=" * 60)
    print("保存 LoRA 权重...")

    # 保存 adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"[OK] LoRA 权重已保存: {output_dir}")

    # 保存训练配置
    config_info = {
        "base_model": model_path,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "num_epochs": num_epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "max_seq_length": max_seq_len,
        "training_samples": len(dataset),
        "eval_samples": len(eval_dataset) if eval_dataset else 0,
        "gpu": torch.cuda.get_device_name(0),
        "has_eval": eval_dataset is not None,
    }
    with open(os.path.join(output_dir, "training_config.json"), 'w', encoding='utf-8') as f:
        json.dump(config_info, f, ensure_ascii=False, indent=2)

    print("\n[OK] 训练完成！")
    print(f"   LoRA 权重: {output_dir}")
    if eval_dataset:
        print(f"   验证集大小: {len(eval_dataset)}")
    print(f"   可用于 vLLM 部署或 transformers 推理")


if __name__ == "__main__":
    main()
