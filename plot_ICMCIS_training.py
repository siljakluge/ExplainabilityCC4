#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Inputs
# ----------------------------
CSV_DIR = Path("exported_csv")
RUNS = [
    ("Contractor Active",   CSV_DIR / "contractoractive.csv"),
    ("Contractor Inactive", CSV_DIR / "contractorinactive.csv"),
]

OUTDIR = Path("figures_icmcis")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Smoothing
# ----------------------------
def ema(series: pd.Series, alpha: float) -> pd.Series:
    return series.ewm(alpha=alpha, adjust=False).mean()

EMA_ALPHA = 0.05  # lower => smoother

# ----------------------------
# Load & validate
# ----------------------------
dfs = []
for label, path in RUNS:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")

    df = pd.read_csv(path)
    required = ["step_e", "avg_reward", "avg_loss"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns {missing}. Has: {list(df.columns)}")

    df = df.sort_values("step_e").copy()
    df["Scenario"] = label
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# Consistent scenario order in legends
scenario_order = [r[0] for r in RUNS]

# ----------------------------
# Plot 1: Reward (linear)
# ----------------------------
plt.figure()
for scenario in scenario_order:
    df = data[data["Scenario"] == scenario]
    x = df["step_e"]
    y = df["avg_reward"]
    y_s = ema(y, EMA_ALPHA)

    plt.plot(x, y, linewidth=1, alpha=0.25)
    plt.plot(x, y_s, linewidth=2, label=scenario)

plt.xlabel("Training Episodes")
plt.ylabel("Average Episode Reward")
plt.title("Training Performance")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "reward_over_training.pdf")
plt.close()

# ----------------------------
# Helper: loss plot function
# ----------------------------
def plot_loss(scale: str, outfile: Path, linthresh: float = 1e-2):
    plt.figure()
    for scenario in scenario_order:
        df = data[data["Scenario"] == scenario]
        x = df["step_e"]
        y = df["avg_loss"]
        y_s = ema(y, EMA_ALPHA)

        plt.plot(x, y, linewidth=1, alpha=0.25)
        plt.plot(x, y_s, linewidth=2, label=scenario)

    plt.xlabel("Training Episodes")
    plt.ylabel("Mean PPO Training Loss")
    plt.title("Optimization Stability")

    if scale == "symlog":
        # symlog supports negative values and compresses large magnitudes
        plt.yscale("symlog", linthresh=linthresh)

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

# ----------------------------
# Plot 2: Loss (linear)
# ----------------------------
plot_loss(scale="linear", outfile=OUTDIR / "loss_over_training_linear.pdf")

# ----------------------------
# Plot 3: Loss (symlog)
# ----------------------------
plot_loss(scale="symlog", outfile=OUTDIR / "loss_over_training_symlog.pdf", linthresh=1e-2)

print(f"Wrote figures to: {OUTDIR.resolve()}")
