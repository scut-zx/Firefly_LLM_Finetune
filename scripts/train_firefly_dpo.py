"""
DPO (Direct Preference Optimization) 训练脚本

使用 trl.DPOTrainer 在 SFT LoRA 基础上进行偏好优化训练。

DPO 不需要单独的奖励模型——直接从 "chosen" vs "rejected" 偏好对中
学习人类偏好，优化模型生成质量。

训练流程:
1. 加载 Qwen3-4B 基础模型
2. 加载 SFT LoRA adapter
3. 训练 DPO (trl.DPOTrainer)
4. merge_and_unload 合并保存 (vLLM 兼容)
5. 也保存单独的 DPO adapter (可选堆叠)

用法:
    python scripts/train_firefly_dpo.py
    python scripts/train_firefly_dpo.py --sft-lora output/Firefly_LoRA_v2 --dpo-data data/dpo_preference_pairs_ready.json
"""

import os
import sys
import json
import torch
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig

# ============================================================
# 配置
# ============================================================
MODEL_PATH = str(PROJECT_ROOT / "model")
SFT_LORA_PATH = str(PROJECT_ROOT / "output" / "Firefly_LoRA")
DPO_DATA_PATH = str(PROJECT_ROOT / "data" / "dpo_preference_pairs_ready.json")
OUTPUT_DIR = str(PROJECT_ROOT / "output" / "Firefly_DPO")
MERGED_OUTPUT_DIR = str(PROJECT_ROOT / "output" / "Firefly_DPO_Merged")

# DPO 训练配置 (VRAM 优化: RTX 4060 Ti 16GB)
DPO_BETA = 0.1              # KL 惩罚强度
DPO_LR = 5e-6               # DPO 学习率（低于 SFT）
DPO_EPOCHS = 3
DPO_BATCH_SIZE = 1          # DPO 双倍显存(chosen + rejected)
DPO_GRAD_ACCUM = 16          # 有效 batch = 16
MAX_LENGTH = 2048
MAX_PROMPT_LENGTH = 1536    # 保留 512 tokens 给回复
WARMUP_RATIO = 0.1

# LoRA 配置 (与 SFT 保持一致，DPO 阶段继续训练相同 adapter)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def load_dpo_data(data_path: str) -> list:
    """加载 DPO 偏好数据"""
    with open(data_path, 'r', encoding='utf-8') as f:
        pairs = json.load(f)

    # 过滤出可直接使用的偏好对
    ready_pairs = [p for p in pairs if not p.get("needs_generation", False)]

    # 验证格式
    valid_pairs = []
    for p in ready_pairs:
        if p.get("prompt") and p.get("chosen") and p.get("rejected"):
            valid_pairs.append({
                "prompt": p["prompt"],
                "chosen": p["chosen"],
                "rejected": p["rejected"],
            })

    print(f"  DPO 偏好对: {len(pairs)} total, {len(valid_pairs)} 可用")
    return valid_pairs


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DPO 训练")
    parser.add_argument("--sft-lora", default=SFT_LORA_PATH,
                       help="SFT LoRA adapter 路径")
    parser.add_argument("--dpo-data", default=DPO_DATA_PATH,
                       help="DPO 偏好数据路径")
    parser.add_argument("--output", default=OUTPUT_DIR,
                       help="DPO LoRA 输出目录")
    parser.add_argument("--merged-output", default=MERGED_OUTPUT_DIR,
                       help="合并后模型输出目录")
    parser.add_argument("--beta", type=float, default=DPO_BETA,
                       help="DPO KL 惩罚强度")
    parser.add_argument("--lr", type=float, default=DPO_LR,
                       help="学习率")
    parser.add_argument("--epochs", type=int, default=DPO_EPOCHS,
                       help="训练轮数")
    parser.add_argument("--dry-run", action="store_true",
                       help="仅验证数据和配置，不实际训练")
    args = parser.parse_args()

    print("=" * 60)
    print("Firefly DPO 偏好优化训练")
    print("=" * 60)
    print(f"  SFT LoRA: {args.sft_lora}")
    print(f"  DPO 数据: {args.dpo_data}")
    print(f"  Beta (KL): {args.beta}")
    print(f"  LR: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # ============================================================
    # 1. 加载 Tokenizer
    # ============================================================
    print("[1/6] 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ============================================================
    # 2. 加载模型
    # ============================================================
    print("[2/6] 加载基础模型 (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # ============================================================
    # 3. 加载 SFT LoRA
    # ============================================================
    print("[3/6] 加载 SFT LoRA adapter...")
    if os.path.exists(args.sft_lora):
        model = PeftModel.from_pretrained(model, args.sft_lora, is_trainable=True)
        print(f"  SFT LoRA 已加载: {args.sft_lora}")
    else:
        print(f"  [警告] SFT LoRA 未找到: {args.sft_lora}")
        print(f"  将从零开始配置 LoRA...")
        lora_config = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES, bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # ============================================================
    # 4. 加载 DPO 数据
    # ============================================================
    print("[4/6] 加载 DPO 偏好数据...")
    if not os.path.exists(args.dpo_data):
        # 尝试 ready 版本
        alt_path = args.dpo_data.replace(".json", "_ready.json")
        if os.path.exists(alt_path):
            args.dpo_data = alt_path
            print(f"  使用 ready 版本: {alt_path}")
        else:
            raise FileNotFoundError(
                f"DPO 数据未找到: {args.dpo_data}\n"
                f"  请先运行: python scripts/prepare_dpo_data.py"
            )

    dpo_pairs = load_dpo_data(args.dpo_data)

    if len(dpo_pairs) == 0:
        raise ValueError("没有可用的 DPO 偏好对！请检查数据格式。")

    from datasets import Dataset
    dataset = Dataset.from_list(dpo_pairs)
    print(f"  数据集大小: {len(dataset)}")

    if args.dry_run:
        print("\n[Dry Run] 配置验证通过，跳过训练。")
        print(f"  模型: Qwen3-4B + LoRA")
        print(f"  偏好对: {len(dpo_pairs)} 对")
        print(f"  Beta: {args.beta}, LR: {args.lr}, Epochs: {args.epochs}")
        return

    # ============================================================
    # 5. DPO 训练配置
    # ============================================================
    print("[5/6] 配置 DPO 训练...")

    dpo_config = DPOConfig(
        output_dir=args.output,
        per_device_train_batch_size=DPO_BATCH_SIZE,
        gradient_accumulation_steps=DPO_GRAD_ACCUM,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        bf16=True,
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=42,
        beta=args.beta,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        loss_type="sigmoid",  # 标准 DPO loss
    )

    print(f"  Beta: {args.beta}")
    print(f"  Batch: {DPO_BATCH_SIZE} x {DPO_GRAD_ACCUM} (effective={DPO_BATCH_SIZE * DPO_GRAD_ACCUM})")
    print(f"  Max Length: {MAX_LENGTH} (prompt <= {MAX_PROMPT_LENGTH})")
    print(f"  Loss Type: sigmoid")

    # ============================================================
    # 6. DPO 训练
    # ============================================================
    print("[6/6] 开始 DPO 训练...")
    print()

    dpo_trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    dpo_trainer.train()

    # ============================================================
    # 7. 保存
    # ============================================================
    print()
    print("=" * 60)
    print("保存 DPO 权重...")

    # 保存 DPO adapter (可堆叠在 SFT 之上)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"[OK] DPO LoRA 已保存: {args.output}")

    # 合并 SFT + DPO 并保存 (vLLM 兼容)
    print("合并 SFT + DPO adapter...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(args.merged_output, safe_serialization=True)
    tokenizer.save_pretrained(args.merged_output)
    print(f"[OK] 合并模型已保存: {args.merged_output}")

    # 保存训练配置
    config_info = {
        "base_model": MODEL_PATH,
        "sft_lora_path": args.sft_lora,
        "dpo_beta": args.beta,
        "dpo_learning_rate": args.lr,
        "dpo_epochs": args.epochs,
        "dpo_pairs": len(dpo_pairs),
        "output_adapter": args.output,
        "output_merged": args.merged_output,
        "gpu": torch.cuda.get_device_name(0),
    }
    with open(os.path.join(args.output, "dpo_training_config.json"), 'w', encoding='utf-8') as f:
        json.dump(config_info, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] DPO 训练完成！")
    print(f"   DPO Adapter:   {args.output}")
    print(f"   合并模型:      {args.merged_output}")
    print(f"   vLLM 部署:     vllm serve model/ --lora-modules firefly-dpo={args.merged_output}")


if __name__ == "__main__":
    main()
