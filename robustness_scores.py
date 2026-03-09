#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------
RESULTS_DIR = Path("Result")

MODELS = [
    "Heuristic",
    "RedVariants",
    "SimpleGNN",
    "Sleep",
]

STRATEGIES = [
    "deception_aware",
    "discovery",
    "fsm_default",
    "impact_rush",
    "lateral_spread",
    "stealth_pivot",
]

LAMBDA_1 = 1.0
LAMBDA_2 = 1.0

# If True, recompute mean/std from per_episode.
# This is usually the safest option for consistency.
RECOMPUTE_FROM_EPISODES = True
OUT_CSV = Path("Result/Robustness/robustness_scores.csv")
OUT_JSON = Path("Result/Robustness/robustness_scores.json")


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_reward_stats(summary: dict) -> tuple[float, float]:
    reward_scalar = summary["reward_scalar"]

    if RECOMPUTE_FROM_EPISODES:
        episodes = reward_scalar.get("per_episode", None)
        if episodes is None or len(episodes) == 0:
            raise ValueError("Missing or empty reward_scalar['per_episode']")
        arr = np.array(episodes, dtype=float)
        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=0))  # population std
    else:
        mu = float(reward_scalar["mean"])
        sigma = float(reward_scalar["stdev"])

    return mu, sigma


def compute_model_robustness(model_dir: Path) -> dict:
    mus = []
    sigmas = []

    per_strategy_rows = []

    for strategy in STRATEGIES:
        summary_path = model_dir / strategy / "summary_scalar.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing file: {summary_path}")

        summary = load_summary(summary_path)
        mu, sigma = extract_reward_stats(summary)

        mus.append(mu)
        sigmas.append(sigma)

        per_strategy_rows.append(
            {
                "strategy": strategy,
                "mu": mu,
                "sigma": sigma,
            }
        )

    mus = np.array(mus, dtype=float)
    sigmas = np.array(sigmas, dtype=float)

    mu_bar = float(np.mean(mus))
    sigma_bar = float(np.mean(sigmas))
    std_r_mu = float(np.std(mus, ddof=0))  # population std across strategies
    robustness = float(mu_bar - LAMBDA_1 * sigma_bar - LAMBDA_2 * std_r_mu)

    return {
        "mu_bar": mu_bar,
        "sigma_bar": sigma_bar,
        "std_r_mu": std_r_mu,
        "R": robustness,
        "per_strategy": per_strategy_rows,
    }


def fmt(x: float) -> str:
    return f"{x:.2f}"

def compute_episode_stats(summary):
    """Compute mean and std from per-episode rewards."""
    rewards = summary["reward_scalar"]["per_episode"]
    rewards = np.array(rewards, dtype=float)

    mu = rewards.mean()
    sigma = rewards.std(ddof=0)

    return mu, sigma


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    all_rows = []
    latex_rows = []

    for model in MODELS:
        model_dir = RESULTS_DIR / model
        if not model_dir.exists():
            raise FileNotFoundError(f"Missing model directory: {model_dir}")

        result = compute_model_robustness(model_dir)

        all_rows.append(
            {
                "Model": model,
                r"$\overline{\mu}$": result["mu_bar"],
                r"$\overline{\sigma}$": result["sigma_bar"],
                r"$\mathrm{Std}_r(\mu)$": result["std_r_mu"],
                r"$R(a)$": result["R"],
            }
        )

        latex_rows.append(
            f"{model:<20} & "
            f"{fmt(result['mu_bar']):>8} & "
            f"{fmt(result['sigma_bar']):>8} & "
            f"{fmt(result['std_r_mu']):>8} & "
            f"{fmt(result['R']):>8} \\\\"
        )

    df = pd.DataFrame(all_rows)

    print("\n=== Robustness Table ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n=== LaTeX Rows ===")
    for row in latex_rows:
        print(row)

    print("\n=== Per-strategy details ===")
    for model in MODELS:
        model_dir = RESULTS_DIR / model
        result = compute_model_robustness(model_dir)
        detail_df = pd.DataFrame(result["per_strategy"])
        print(f"\n--- {model} ---")
        print(detail_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    rows = []

    for model in MODELS:

        model_dir = RESULTS_DIR / model

        mus = []
        sigmas = []

        for strategy in STRATEGIES:

            file = model_dir / strategy / "summary_scalar.json"

            if not file.exists():
                raise FileNotFoundError(file)

            summary = load_summary(file)

            mu, sigma = compute_episode_stats(summary)

            mus.append(mu)
            sigmas.append(sigma)

        mus = np.array(mus)
        sigmas = np.array(sigmas)

        mu_bar = mus.mean()
        sigma_bar = sigmas.mean()
        std_r_mu = mus.std(ddof=0)

        robustness = mu_bar - LAMBDA_1 * sigma_bar - LAMBDA_2 * std_r_mu

        rows.append(
            {
                "model": model,
                "mu_bar": mu_bar,
                "sigma_bar": sigma_bar,
                "std_r_mu": std_r_mu,
                "robustness": robustness,
            }
        )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    df = pd.DataFrame(rows)

    df.to_csv(OUT_CSV, index=False)

    with open(OUT_JSON, "w") as f:
        json.dump(rows, f, indent=2)

    print("Robustness scores written to:")
    print(" -", OUT_CSV)
    print(" -", OUT_JSON)

    print("\nResults:")
    print(df.round(2))


if __name__ == "__main__":
    main()