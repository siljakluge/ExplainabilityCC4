# action_logging_and_plotting.py
# Save per-step actions during evaluation to a JSONL log, then load the log and plot action frequencies.

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------
# Action label + colors
# -----------------------

ACTION_COLORS = {
    "Sleep": "#7f7f7f",
    "Analyse": "#1f77b4",
    "Monitor": "#17becf",
    "Restore": "#2ca02c",
    "Remove": "#d62728",
    "DeployDecoy": "#9467bd",
    "AllowTrafficZone": "#bcbd22",
    "BlockTrafficZone": "#ff7f0e",
    "None": "#c7c7c7",
    "OTHER": "#8c564b",
}

ACTION_ORDER = [
    "Sleep", "Analyse", "Monitor", "Restore", "Remove", "DeployDecoy",
    "AllowTrafficZone", "BlockTrafficZone", "None", "OTHER"
]


def stable_color(action: str):
    """Stable color for known actions + deterministic fallback for unknown."""
    if action in ACTION_COLORS:
        return ACTION_COLORS[action]
    cmap = plt.get_cmap("tab20")
    h = int(hashlib.md5(action.encode("utf-8")).hexdigest(), 16)
    return cmap(h % cmap.N)


def action_to_label(act, mode: str = "type") -> str:
    """
    act is typically like [AllowTrafficZone] or [DeployDecoy host_x]
    mode:
      - "type": only the action name (first token)
      - "full": full string representation (keeps parameters/targets)
    """
    # unwrap list/tuple like [AllowTrafficZone]
    if isinstance(act, (list, tuple)) and len(act) == 1:
        act = act[0]
    elif isinstance(act, (list, tuple)) and len(act) == 0:
        return "None"

    s = str(act).strip()  # e.g. "DeployDecoy office_network_subnet_user_host_0"
    if mode == "full":
        return s
    # mode == "type"
    return s.split()[0] if s else "Unknown"


def actions_to_label(act: Any, mode: str = "type") -> str:
    """
    Your actions are typically like [AllowTrafficZone] or [DeployDecoy host_x].
    mode:
      - "type": first token only (DeployDecoy / Analyse / Sleep / ...)
      - "full": keep full string (includes parameters)
    """
    if act is None:
        return "None"
    # unwrap [Action] or (Action,)
    if isinstance(act, (list, tuple)):
        if len(act) == 0:
            return "None"
        if len(act) == 1:
            act = act[0]
        else:
            # if you ever have multi-action lists, join them
            act = " | ".join(str(a) for a in act)

    s = str(act).strip()
    if not s:
        return "None"
    if mode == "full":
        return s
    return s.split()[0]


# -----------------------
# Logging in eval loop
# -----------------------

def log_actions_jsonl(
    log_path: Path,
    episode: int,
    step: int,
    actions: Dict[str, Any],
    mode: str = "type",
):
    """
    Append one JSONL line:
      {"episode": int, "step": int, "actions": {"blue_agent_0": "AllowTrafficZone", ...}}
    Stores action labels (string) to keep the log portable.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    actions_str = {agent: action_to_label(act, mode=mode) for agent, act in actions.items()}
    entry = {"episode": int(episode), "step": int(step), "actions": actions_str}

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# -----------------------
# Read log -> plot
# -----------------------

def read_actions_log_jsonl(log_path: Path) -> List[Dict[str, Any]]:
    """Reads the JSONL action log into a list of entries."""
    log_path = Path(log_path)
    entries = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def plot_action_frequencies_per_agent_from_log(
    log_path: Path,
    out_dir: Path,
    normalize: bool = False,
    top_k: Optional[int] = 25,
):
    """
    Reads JSONL produced by log_actions_jsonl and plots:
      - stacked bars per agent (stable colors)
      - heatmap counts per agent/action
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = read_actions_log_jsonl(log_path)

    rows = []
    for e in entries:
        epi = e.get("episode")
        step = e.get("step")
        actions = e.get("actions", {})
        for agent, action_label in actions.items():
            rows.append({"episode": epi, "step": step, "agent": agent, "action": str(action_label)})

    if not rows:
        return None

    df = pd.DataFrame(rows)

    # compress rare actions globally
    if top_k is not None:
        vc = df["action"].value_counts()
        keep = set(vc.head(top_k).index)
        df["action"] = df["action"].where(df["action"].isin(keep), other="OTHER")

    pivot = pd.crosstab(df["agent"], df["action"])
    pivot = pivot.loc[sorted(pivot.index)]

    # stable action order
    cols = list(pivot.columns)
    ordered = [c for c in ACTION_ORDER if c in cols]
    rest = sorted([c for c in cols if c not in ordered])
    pivot = pivot[ordered + rest]

    if normalize:
        pivot = pivot.div(pivot.sum(axis=1).replace(0, 1), axis=0) * 100.0

    # --- stacked bar with stable colors ---
    plt.figure(figsize=(max(10, pivot.shape[1] * 0.9), max(5, pivot.shape[0] * 0.9)))
    x = np.arange(len(pivot.index))
    bottom = np.zeros(len(pivot.index), dtype=float)

    for action in pivot.columns:
        vals = pivot[action].to_numpy(dtype=float)
        plt.bar(x, vals, bottom=bottom, label=action, color=stable_color(str(action)))
        bottom = bottom + vals

    ylabel = "Percent of actions (%)" if normalize else "Action count"
    plt.title("Action frequencies per agent" + (" (normalized)" if normalize else ""))
    plt.xlabel("Agent")
    plt.ylabel(ylabel)
    plt.xticks(x, list(pivot.index), rotation=0, ha="right")
    plt.legend(title="Action", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    out_bar = out_dir / f"action_freq_per_agent_{'pct' if normalize else 'count'}.png"
    plt.savefig(out_bar, dpi=150, bbox_inches="tight")
    plt.close()

    # --- heatmap (magnitude colormap) ---
    plt.figure(figsize=(max(10, pivot.shape[1] * 0.7), max(4, pivot.shape[0] * 0.6)))
    plt.imshow(pivot.to_numpy(), aspect="auto")
    plt.title("Action frequencies heatmap" + (" (normalized)" if normalize else ""))
    plt.xlabel("Action")
    plt.ylabel("Agent")
    plt.xticks(range(pivot.shape[1]), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(pivot.shape[0]), pivot.index)
    plt.colorbar(label=ylabel)
    plt.tight_layout()

    out_heat = out_dir / f"action_freq_heatmap_{'pct' if normalize else 'count'}.png"
    plt.savefig(out_heat, dpi=150, bbox_inches="tight")
    plt.close()

    # also save the table for debugging
    out_csv = out_dir / f"action_freq_table_{'pct' if normalize else 'count'}.csv"
    pivot.to_csv(out_csv)

    return {"stacked_bar": out_bar, "heatmap": out_heat, "table_csv": out_csv}


# -----------------------
# Main entry point
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, default="actions_heuristic.jsonl", help="Path to action_log.jsonl")
    ap.add_argument("--out_dir", type=str, default="ActionPlotsH", help="Output directory for plots")
    ap.add_argument("--normalize", type=bool, default=True, help="Normalize the plots")
    ap.add_argument("--top_k", type=int, default=25, help="Keep only top_k actions (rest -> OTHER). Use -1 to disable.")
    args = ap.parse_args()

    top_k = None if args.top_k == -1 else args.top_k

    res = plot_action_frequencies_per_agent_from_log(
        log_path=Path(args.log),
        out_dir=Path(args.out_dir),
        normalize=args.normalize,
        top_k=top_k,
    )
    if res is None:
        print("No actions found in log.")
    else:
        print("Saved:", res)


if __name__ == "__main__":
    main()
