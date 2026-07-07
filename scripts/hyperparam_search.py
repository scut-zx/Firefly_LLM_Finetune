"""
LoRA 超参数网格搜索 (Hyperparameter Search)

在 RTX 4060 Ti 16GB 约束下，搜索最优 LoRA 配置。

搜索空间:
- LoRA rank: [16, 32, 64]
- LoRA alpha: [32, 64, 128]
- Learning rate: [3e-5, 5e-5, 1e-4]
- Epochs: [5, 8, 10]

用法:
    python scripts/hyperparam_search.py --trials 10 --output output/hyperparam_results.json
    python scripts/hyperparam_search.py --quick  # 仅测试 2 个组合
"""

import os
import sys
import json
import time
import itertools
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 搜索空间
SEARCH_SPACE = {
    "lora_r": [16, 32, 64],
    "lora_alpha": [32, 64, 128],
    "learning_rate": [3e-5, 5e-5, 1e-4],
    "num_epochs": [5, 8, 10],
}

# 固定训练参数
FIXED_PARAMS = {
    "batch_size": 2,
    "grad_accum": 8,
    "max_seq_len": 2048,
    "warmup_ratio": 0.1,
    "bf16": True,
    "lora_dropout": 0.1,
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
}


def estimate_vram(lora_r: int, batch_size: int, seq_len: int) -> float:
    """
    估算 VRAM 使用量 (GB)。
    Qwen3-4B BF16 = ~8GB, LoRA = r * small_overhead
    """
    base_model_gb = 8.0  # Qwen3-4B in bf16
    lora_params_gb = lora_r * 0.008  # 粗略估计
    activation_gb = (batch_size * seq_len * 0.000002) + 4  # ~4-6GB
    return base_model_gb + lora_params_gb + activation_gb


def filter_feasible_configs(configs: list) -> list:
    """过滤超过 VRAM 限制的配置"""
    max_vram = 15.5  # RTX 4060 Ti 16GB 安全上限
    feasible = []
    for config in configs:
        vram = estimate_vram(
            config["lora_r"],
            FIXED_PARAMS["batch_size"],
            FIXED_PARAMS["max_seq_len"],
        )
        if vram <= max_vram:
            config["estimated_vram_gb"] = round(vram, 1)
            feasible.append(config)
        else:
            print(f"  跳过 r={config['lora_r']} alpha={config['lora_alpha']}: "
                  f"预估 VRAM {vram:.1f}GB > {max_vram}GB")
    return feasible


def generate_trial_configs(quick: bool = False, max_trials: int = 20) -> list:
    """生成所有试验配置组合"""
    if quick:
        search = {
            "lora_r": [16, 32],
            "lora_alpha": [32, 64],
            "learning_rate": [3e-5, 5e-5],
            "num_epochs": [5, 8],
        }
    else:
        search = SEARCH_SPACE

    keys = list(search.keys())
    all_combos = list(itertools.product(*[search[k] for k in keys]))

    configs = []
    for combo in all_combos:
        config = dict(zip(keys, combo))
        config.update(FIXED_PARAMS)
        # alpha >= 2 * r 以保证 LoRA 有效秩
        if config["lora_alpha"] >= 2 * config["lora_r"]:
            configs.append(config)

    feasible = filter_feasible_configs(configs)

    # 如果试验次数超出限制，随机采样
    if len(feasible) > max_trials:
        import random
        random.seed(42)
        feasible = random.sample(feasible, max_trials)

    return feasible


def run_single_trial(config: dict, trial_id: int,
                    train_path: str, val_path: str) -> dict:
    """
    运行单次超参数试验。
    由于每次试验需要完整训练，这里的实现是记录配置并返回模拟结果。
    实际训练需手动运行 train_firefly_lora.py 并填入结果。
    """
    trial_dir = PROJECT_ROOT / "output" / f"hyperparam_trial_{trial_id}"
    os.makedirs(trial_dir, exist_ok=True)

    # 保存配置供后续训练使用
    config_path = trial_dir / "trial_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump({
            "trial_id": trial_id,
            "config": config,
            "timestamp": datetime.now().isoformat(),
            "status": "pending",
        }, f, ensure_ascii=False, indent=2)

    # 生成训练命令
    cmd = (
        f"python scripts/train_firefly_lora.py "
        f"--lora_r {config['lora_r']} "
        f"--lora_alpha {config['lora_alpha']} "
        f"--learning_rate {config['learning_rate']} "
        f"--epochs {config['num_epochs']} "
        f"--output {trial_dir} "
        f"--train_data {train_path} "
        f"--eval_data {val_path}"
    )

    print(f"\n  Trial {trial_id}: r={config['lora_r']} "
          f"alpha={config['lora_alpha']} lr={config['learning_rate']:.0e} "
          f"epochs={config['num_epochs']}")
    print(f"  VRAM 估算: {config.get('estimated_vram_gb', '?')} GB")
    print(f"  命令: {cmd}")
    print(f"  配置已保存: {config_path}")

    return {
        "trial_id": trial_id,
        "config": config,
        "config_path": str(config_path),
        "command": cmd,
        "status": "pending",
    }


def evaluate_trial(trial_dir: Path) -> dict:
    """
    评估已完成的试验。
    从 trainer_state.json 读取最终 loss。
    """
    import glob
    checkpoints = sorted(
        trial_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1])
    )
    if not checkpoints:
        return {"final_loss": None, "status": "not_found"}

    state_path = checkpoints[-1] / "trainer_state.json"
    if not state_path.exists():
        return {"final_loss": None, "status": "no_state"}

    with open(state_path, 'r', encoding='utf-8') as f:
        state = json.load(f)

    log_history = state.get("log_history", [])
    losses = [e.get("loss") for e in log_history if "loss" in e]

    if not losses:
        return {"final_loss": None, "status": "no_loss"}

    return {
        "final_loss": round(losses[-1], 4),
        "min_loss": round(min(losses), 4),
        "total_steps": len(losses),
        "status": "completed",
    }


def generate_report(trials: list) -> dict:
    """生成超参数搜索报告"""
    completed = [t for t in trials
                if t.get("results", {}).get("status") == "completed"]
    pending = [t for t in trials
              if t.get("status") == "pending"]

    report = {
        "total_trials": len(trials),
        "completed": len(completed),
        "pending": len(pending),
        "best_trial": None,
        "all_trials": trials,
    }

    if completed:
        # 找最优试验（最低 final_loss）
        best = min(completed,
                   key=lambda t: t["results"]["final_loss"])
        report["best_trial"] = {
            "trial_id": best["trial_id"],
            "config": best["config"],
            "final_loss": best["results"]["final_loss"],
        }

    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LoRA 超参数网格搜索")
    parser.add_argument("--trials", type=int, default=15,
                       help="最大试验次数 (默认: 15)")
    parser.add_argument("--output", default=None,
                       help="结果输出路径")
    parser.add_argument("--quick", action="store_true",
                       help="快速模式（仅 4 个组合）")
    parser.add_argument("--train-data", default=None,
                       help="训练数据路径")
    parser.add_argument("--eval-data", default=None,
                       help="验证数据路径")
    parser.add_argument("--evaluate-only", action="store_true",
                       help="仅评估已完成的试验（不生成新配置）")
    args = parser.parse_args()

    train_data = args.train_data or str(
        PROJECT_ROOT / "data" / "firefly_train.json"
    )
    val_data = args.eval_data or str(
        PROJECT_ROOT / "data" / "firefly_val.json"
    )

    if args.evaluate_only:
        # 扫描已有试验并评估
        print("扫描已有试验...")
        trial_dirs = sorted(
            PROJECT_ROOT.glob("output/hyperparam_trial_*/"),
            key=lambda p: int(p.name.split("_")[-1])
        )
        trials = []
        for td in trial_dirs:
            config_path = td / "trial_config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    trial = json.load(f)
                trial["results"] = evaluate_trial(td)
                trials.append(trial)

        report = generate_report(trials)
        output = args.output or str(
            PROJECT_ROOT / "output" / "hyperparam_results.json"
        )
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] 评估报告已保存: {output}")
        if report["best_trial"]:
            print(f"最佳配置: {report['best_trial']['config']}")
            print(f"最终 Loss: {report['best_trial']['final_loss']}")
        return

    # 生成试验配置
    configs = generate_trial_configs(args.quick, args.trials)
    print(f"\n{'='*60}")
    print(f"LoRA 超参数网格搜索")
    print(f"{'='*60}")
    print(f"  总配置数: {len(configs)}")
    print(f"  搜索空间: r={SEARCH_SPACE['lora_r']}, "
          f"alpha={SEARCH_SPACE['lora_alpha']}, "
          f"lr={SEARCH_SPACE['learning_rate']}, "
          f"epochs={SEARCH_SPACE['num_epochs']}")
    print(f"{'='*60}\n")

    # 运行各试验
    trials = []
    for i, config in enumerate(configs):
        trial = run_single_trial(config, i, train_data, val_data)
        trials.append(trial)

    # 生成完整命令列表
    print(f"\n{'='*60}")
    print(f"所有试验命令（复制运行）")
    print(f"{'='*60}")
    for t in trials:
        print(t["command"])

    # 保存结果
    report = generate_report(trials)
    output = args.output or str(
        PROJECT_ROOT / "output" / "hyperparam_plan.json"
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 搜索计划已保存: {output}")
    print(f"共 {len(trials)} 个试验待运行")
    print("提示: 复制上述命令逐一运行，然后使用 --evaluate-only 汇总结果")


if __name__ == "__main__":
    main()
