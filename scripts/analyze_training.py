"""
训练分析工具 (Training Analysis)

读取 trainer_state.json，绘制 loss 曲线，输出训练统计。

用法:
    python scripts/analyze_training.py
    python scripts/analyze_training.py --checkpoint output/Firefly_LoRA/checkpoint-95
    python scripts/analyze_training.py --output output/training_loss.png
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).parent.parent


def load_trainer_state(checkpoint_dir: Path) -> dict:
    """加载 trainer_state.json"""
    state_path = checkpoint_dir / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"未找到 trainer_state.json: {state_path}")

    with open(state_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_metrics(state: dict) -> dict:
    """从 trainer_state 中提取关键指标"""
    log_history = state.get("log_history", [])
    if not log_history:
        # 兼容旧格式
        log_history = state.get("log_history", [])

    steps = []
    losses = []
    lrs = []
    grad_norms = []
    epochs = []

    for entry in log_history:
        if "loss" not in entry:
            continue

        steps.append(entry.get("step", 0))
        losses.append(entry.get("loss", 0))
        lrs.append(entry.get("learning_rate", 0))
        grad_norms.append(entry.get("grad_norm", 0))
        epochs.append(entry.get("epoch", 0))

    return {
        "steps": steps,
        "losses": losses,
        "learning_rates": lrs,
        "grad_norms": grad_norms,
        "epochs": epochs,
        "total_steps": len(steps),
    }


def compute_summary(metrics: dict) -> dict:
    """计算训练摘要"""
    losses = metrics["losses"]
    lrs = metrics["learning_rates"]
    grad_norms = metrics["grad_norms"]

    if not losses:
        return {"error": "无有效训练数据"}

    initial_loss = losses[0]
    final_loss = losses[-1]
    min_loss = min(losses)
    max_loss = max(losses)

    # 收敛步数：loss 不再显著下降的步数（变化 < 1%）
    convergence_step = metrics["steps"][-1]
    for i in range(len(losses) - 3, -1, -1):
        if abs(losses[i] - losses[-1]) / max(abs(losses[-1]), 0.001) > 0.01:
            convergence_step = metrics["steps"][i + 1]
            break

    summary = {
        "total_steps": len(losses),
        "initial_loss": round(initial_loss, 4),
        "final_loss": round(final_loss, 4),
        "loss_reduction_pct": round(
            (initial_loss - final_loss) / max(initial_loss, 0.001) * 100, 1
        ),
        "min_loss": round(min_loss, 4),
        "max_loss": round(max_loss, 4),
        "convergence_step": convergence_step,
        "learning_rate_range": f"{min(lrs):.2e} - {max(lrs):.2e}" if lrs else "N/A",
        "grad_norm_range": f"{min(grad_norms):.3f} - {max(grad_norms):.3f}" if grad_norms else "N/A",
        "final_grad_norm": round(grad_norms[-1], 4) if grad_norms else None,
    }

    return summary


def print_metrics_table(metrics: dict):
    """打印训练指标表格"""
    steps = metrics["steps"]
    losses = metrics["losses"]
    lrs = metrics["learning_rates"]
    grad_norms = metrics["grad_norms"]
    epochs = metrics["epochs"]

    print(f"\n{'='*70}")
    print(f"{'Step':>6} | {'Epoch':>6} | {'Loss':>10} | {'LR':>12} | {'Grad Norm':>10}")
    print(f"{'='*70}")

    for i in range(len(steps)):
        e = epochs[i] if i < len(epochs) else 0
        print(f"{steps[i]:>6} | {e:>6.2f} | {losses[i]:>10.4f} | "
              f"{lrs[i]:>12.2e} | {grad_norms[i]:>10.4f}")

    print(f"{'='*70}")


def print_ascii_loss_plot(metrics: dict):
    """ASCII 艺术 loss 曲线（无 matplotlib 时的备选方案）"""
    losses = metrics["losses"]
    steps = metrics["steps"]

    if not losses:
        return

    max_loss = max(losses)
    min_loss = min(losses)
    height = 20
    width = 60

    # 归一化
    normalized = [
        int((l - min_loss) / max(max_loss - min_loss, 0.001) * (height - 1))
        for l in losses
    ]

    print(f"\n  Loss ({max_loss:.2f} -> {min_loss:.2f})")

    # 采样（如果数据点太多）
    if len(normalized) > width:
        sampled = []
        step = len(normalized) / width
        for i in range(width):
            idx = int(i * step)
            sampled.append(normalized[idx])
        normalized = sampled
        step_labels = [f"{int(i * len(steps) / width)}" for i in range(0, width, 10)]
    else:
        step_labels = [str(s) for s in steps[::max(1, len(steps)//6)]]

    for row in range(height - 1, -1, -1):
        line = ""
        for val in normalized:
            if val >= row:
                line += "#"
            else:
                line += " "
        label = f"{min_loss + (max_loss - min_loss) * row / (height-1):.2f}" if row % 4 == 0 else ""
        print(f"  {label:>6} |{line}")

    # X 轴
    print(f"  {'':>6} +{'-'*len(normalized)}")
    print(f"  {'':>6}  Step: {steps[0]} -> {steps[-1]}")


def plot_loss_curve(metrics: dict, output_path: str = None):
    """使用 matplotlib 绘制 loss 曲线"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # 非交互模式
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[info] matplotlib 未安装，使用 ASCII 图表\n")
        print_ascii_loss_plot(metrics)
        return False

    # 中文字体设置
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Firefly LoRA Training Analysis', fontsize=14, fontweight='bold')

    steps = metrics["steps"]
    losses = metrics["losses"]
    lrs = metrics["learning_rates"]
    grad_norms = metrics["grad_norms"]

    # Loss 曲线
    ax = axes[0][0]
    ax.plot(steps, losses, 'b-', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=losses[-1], color='r', linestyle='--', alpha=0.5,
               label=f'Final: {losses[-1]:.3f}')
    ax.legend()

    # Loss (log scale)
    ax = axes[0][1]
    ax.semilogy(steps, losses, 'b-', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss (log)')
    ax.set_title('Training Loss (Log Scale)')
    ax.grid(True, alpha=0.3)

    # Learning Rate
    ax = axes[1][0]
    ax.plot(steps, lrs, 'g-', linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    # Gradient Norm
    ax = axes[1][1]
    ax.plot(steps, grad_norms, 'orange', linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norm')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        print(f"\n[OK] Loss 曲线已保存: {output_path}")
    else:
        output_path = PROJECT_ROOT / "output" / "Firefly_LoRA" / "training_loss.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        print(f"\n[OK] Loss 曲线已保存: {output_path}")

    plt.close()
    return True


def find_latest_checkpoint(base_dir: Path) -> Path:
    """查找最新的 checkpoint 目录"""
    checkpoints = sorted(base_dir.glob("checkpoint-*"))
    if not checkpoints:
        return base_dir
    return checkpoints[-1]


def main():
    parser = argparse.ArgumentParser(description="训练分析工具")
    parser.add_argument("--checkpoint", default=None,
                       help="Checkpoint 目录路径")
    parser.add_argument("--output", default=None,
                       help="图表输出路径 (.png)")
    parser.add_argument("--text-only", action="store_true",
                       help="仅打印文本摘要（不生成图表）")
    args = parser.parse_args()

    # 确定 checkpoint 目录
    if args.checkpoint:
        checkpoint_dir = Path(args.checkpoint)
    else:
        output_dir = PROJECT_ROOT / "output" / "Firefly_LoRA"
        checkpoint_dir = find_latest_checkpoint(output_dir)

    print(f"分析 Checkpoint: {checkpoint_dir}")

    # 加载数据
    try:
        state = load_trainer_state(checkpoint_dir)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("提示: 请先运行训练脚本生成 trainer_state.json")
        return

    # 提取指标
    metrics = extract_metrics(state)
    if not metrics["losses"]:
        print("[ERROR] trainer_state.json 中没有 loss 数据")
        return

    # 打印摘要
    print(f"\n{'='*60}")
    print("训练分析摘要")
    print(f"{'='*60}")
    summary = compute_summary(metrics)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}")

    # 打印详细表格
    print_metrics_table(metrics)

    # 生成图表
    if not args.text_only:
        plot_loss_curve(metrics, args.output)

    print("\n[OK] 训练分析完成")


if __name__ == "__main__":
    main()
