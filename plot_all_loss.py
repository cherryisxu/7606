import json
import matplotlib.pyplot as plt
from pathlib import Path

def load_loss(run_dir: Path):
    steps = []
    val_losses = []
    with open(run_dir / "eval.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            steps.append(record["step"])
            val_losses.append(record["val_loss"])
    return steps, val_losses

runs = {
    "Baseline": Path("outputs/baseline"),
    "exp_optim": Path("outputs/exp_optim"),
    "exp_model": Path("outputs/exp_model"),
    "exp_rmsnorm": Path("outputs/exp_rmsnorm"),
}

plt.figure(figsize=(10, 6))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

for idx, (name, path) in enumerate(runs.items()):
    s, l = load_loss(path)
    plt.plot(s, l, label=name, color=colors[idx], linewidth=1.5)

plt.xlabel("Training Step")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Curves of All Experiments")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("loss_all.png", dpi=300, bbox_inches="tight")
plt.show()
print("图片已保存到 loss_all.png")
