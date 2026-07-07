"""
训练指标追踪器 (Training Tracker)

记录每次训练运行的配置和结果，支持跨版本比较。

用法:
    python scripts/training_tracker.py --record  # 记录一次训练结果
    python scripts/training_tracker.py --history  # 查看训练历史
"""

import os
import sys
import json
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

HISTORY_PATH = PROJECT_ROOT / "training_history.json"


def load_history() -> list:
    """加载训练历史"""
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_history(history: list):
    """保存训练历史"""
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def record_training(version: str, method: str, config: dict,
                    results: dict = None, notes: str = ""):
    """
    记录一次训练运行。

    Args:
        version: 版本标识 (v0, v1, v2, dpo_v1, ...)
        method: 训练方法 (SFT, DPO, ...)
        config: 训练配置 (lora_r, epochs, lr, ...)
        results: 训练结果 (final_loss, eval_metrics, ...)
        notes: 备注
    """
    history = load_history()

    entry = {
        "version": version,
        "method": method,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "results": results or {},
        "notes": notes,
    }

    # 检查是否已有同版本记录，如有则更新
    existing_idx = None
    for i, h in enumerate(history):
        if h.get("version") == version and h.get("method") == method:
            existing_idx = i
            break

    if existing_idx is not None:
        history[existing_idx] = entry
        print(f"更新已有记录: {version} ({method})")
    else:
        history.append(entry)
        print(f"新增记录: {version} ({method})")

    save_history(history)
    print(f"训练历史已保存: {HISTORY_PATH}")
    return entry


def print_history():
    """打印训练历史摘要"""
    history = load_history()
    if not history:
        print("暂无训练记录。")
        print(f"运行 'python scripts/training_tracker.py --record' 来记录首次训练。")
        return

    print(f"\n{'='*70}")
    print(f"训练历史 (Training History)")
    print(f"{'='*70}")
    print(f"{'Version':<12} {'Method':<6} {'Date':<12} {'Loss':>8} {'Notes'}")
    print(f"{'-'*70}")

    for entry in history:
        version = entry.get("version", "?")
        method = entry.get("method", "?")
        date = entry.get("timestamp", "")[:10]
        loss = entry.get("results", {}).get("final_loss", "N/A")
        if isinstance(loss, float):
            loss_str = f"{loss:.4f}"
        else:
            loss_str = str(loss)
        notes = entry.get("notes", "")[:30]

        print(f"{version:<12} {method:<6} {date:<12} {loss_str:>8}  {notes}")

    print(f"{'='*70}")
    print(f"共 {len(history)} 条记录")
    print(f"文件: {HISTORY_PATH}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="训练指标追踪器")
    parser.add_argument("--record", action="store_true",
                       help="记录训练结果")
    parser.add_argument("--history", action="store_true",
                       help="查看训练历史")
    parser.add_argument("--version", default=None,
                       help="版本标识 (如 v0, v1, v2, dpo_v1)")
    parser.add_argument("--method", default="SFT",
                       choices=["SFT", "DPO"],
                       help="训练方法")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--data-size", type=int, default=294)
    parser.add_argument("--final-loss", type=float, default=None)
    parser.add_argument("--notes", default="")
    parser.add_argument("--eval-metrics", default=None,
                       help="评估报告 JSON 路径")
    args = parser.parse_args()

    if args.history:
        print_history()
        return

    if args.record:
        if not args.version:
            print("[ERROR] 请指定 --version (如 --version v1)")
            return

        config = {
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "training_data_size": args.data_size,
        }

        results = {}
        if args.final_loss is not None:
            results["final_loss"] = args.final_loss

        # 加载评估指标
        if args.eval_metrics and os.path.exists(args.eval_metrics):
            with open(args.eval_metrics, 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
            results["eval_metrics"] = eval_data.get("summary", {})

        record_training(args.version, args.method, config, results, args.notes)
        return

    # 默认显示历史
    print_history()


if __name__ == "__main__":
    main()
