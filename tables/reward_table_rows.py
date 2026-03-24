#!/usr/bin/env python3

import json
from pathlib import Path
import numpy as np

# --------------------------------------------------
# Configuration
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # geht von plotting/ eine Ebene hoch
RESULTS_DIR = BASE_DIR / "Result"

MODELS = [
    "Heuristic",
    "SimpleGNN",
    "RedVariants",
    "Sleep",
]

STRATEGIES = {
    "FSM": "fsm_default",
    "Stealth": "stealth_pivot",
    "Lateral": "lateral_spread",
    "Impact": "impact_rush",
    "Deception": "deception_aware",
    "Discovery": "discovery",
}


# --------------------------------------------------
# Helper
# --------------------------------------------------

def load_stats(file):

    with open(file, "r") as f:
        data = json.load(f)

    rewards = np.array(data["reward_scalar"]["per_episode"], dtype=float)

    mean = rewards.mean()
    std = rewards.std(ddof=0)

    return mean, std


def fmt(mean, std):
    return f"${mean:.2f} \\pm {std:.2f}$"


# --------------------------------------------------
# Main
# --------------------------------------------------

for model in MODELS:

    row = [model]

    for name, folder in STRATEGIES.items():

        file = RESULTS_DIR / model / folder / "summary_scalar.json"

        if not file.exists():
            row.append("--")
            continue

        mean, std = load_stats(file)

        row.append(fmt(mean, std))

    latex_line = " & ".join(row) + " \\\\"

    print(latex_line)