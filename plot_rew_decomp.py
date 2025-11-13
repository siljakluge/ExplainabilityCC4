

#   python plot_reward_decomposition.py --log reward_log.jsonl --out reward_plots --episode 0

import argparse, json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color.rgb_colors import black
import numpy as np

COMPONENT_NAME_MAP = {
    "LWF": "Local Work Failure",
    "ASF": "Access Service Failure",
    "RIA": "Red Impact Access",
}

def parse_log(log_path: Path):
    episode = None
    step = None
    reward_rows = []
    total_rows = []
    action_cost_rows = []

    with log_path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "episode" in obj and "step" in obj and len(obj) == 2:
                episode = obj["episode"]
                step = obj["step"]
                continue

            if "phase" in obj and "reward_list" in obj and "total" in obj:
                phase = obj["phase"]
                total_rows.append({"episode": episode, "step": step, "phase": phase, "total": obj["total"]})
                for subnet, compdict in obj.get("reward_list", {}).items():
                    for comp_abbrev, val in compdict.items():
                        comp_full = COMPONENT_NAME_MAP.get(comp_abbrev, comp_abbrev)
                        reward_rows.append({
                            "episode": episode,
                            "step": step,
                            "phase": phase,
                            "subnet": subnet,
                            "component": comp_full,
                            "value": val
                        })
                continue

            if "agent" in obj and "action cost" in obj:
                action_cost_rows.append({
                    "episode": episode,
                    "step": step,
                    "agent": obj["agent"],
                    "action_cost": obj["action cost"],
                })

    df_rewards = pd.DataFrame(reward_rows, columns=["episode", "step","phase", "subnet", "component", "value"])
    df_total = pd.DataFrame(total_rows, columns=["episode", "step", "phase", "total"]).drop_duplicates()
    df_acost = pd.DataFrame(action_cost_rows, columns=["episode", "step", "agent", "action_cost"])
    return df_rewards, df_total, df_acost

def plot_action_cost_by_agent(df_acost: pd.DataFrame, out_dir: Path):
    if df_acost.empty:
        return None
    # Order by (episode, step), then pivot to agent series
    df = df_acost.sort_values(["episode", "step"]).reset_index(drop=True)
    x_pairs = list(zip(df["episode"], df["step"]))
    # Create a canonical x index: one point per (episode, step) pair in order
    # We need a continuous x over unique (episode, step)
    unique_pairs = sorted(set(x_pairs))
    x_index = {pair: i for i, pair in enumerate(unique_pairs)}
    df["x"] = df[["episode", "step"]].apply(tuple, axis=1).map(x_index)

    # Pivot to get one column per agent; sum if multiple rows exist
    pivot = (
        df.groupby(["x", "agent"])["action_cost"]
        .sum()
        .reset_index()
        .pivot_table(index="x", columns="agent", values="action_cost", fill_value=0)
        .sort_index()
    )

    plt.figure()
    ax = plt.gca()
    # one line per agent (default matplotlib colors; no styles specified)
    for agent in pivot.columns:
        plt.plot(pivot.index.values, pivot[agent].values, label=agent)
    # Episode vlines
    _add_episode_vlines(ax, unique_pairs)
    plt.title("Action Cost per Step by Agent")
    plt.xlim(0, len(unique_pairs))
    plt.xlabel("Step index (ordered by episode, step)")
    plt.ylabel("Action cost")
    plt.legend(title="Agent")
    out = out_dir / "action_cost_per_step_by_agent.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out

def _episode_boundaries(sorted_pairs):
    boundaries = []
    prev_ep = None
    for i, (ep, _st) in enumerate(sorted_pairs):
        if i == 0:
            prev_ep = ep
            continue
        if ep != prev_ep:
            boundaries.append(i)
            prev_ep = ep
    return boundaries

def _add_episode_vlines(ax, sorted_pairs):
    for pos in _episode_boundaries(sorted_pairs):
        ax.axvline(pos - 0.5, color=black)

def plot_total(df_total: pd.DataFrame, out_dir: Path):
    if df_total.empty:
        return None
    df_total = df_total.sort_values(["episode", "step"]).reset_index(drop=True)
    x_pairs = list(zip(df_total["episode"], df_total["step"]))
    x = range(len(df_total))
    y = df_total["total"].values

    plt.figure(figsize=(20, 8))
    ax = plt.gca()
    plt.plot(x, y)
    _add_episode_vlines(ax, x_pairs)
    plt.title("Total Reward per Step")
    plt.xlabel("Step index (ordered by episode, step)")
    plt.ylabel("Total reward")
    out = out_dir / "total_reward_per_step.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out

def plot_phase_stacked(df_rewards: pd.DataFrame,
                       df_total: pd.DataFrame,
                       phase: int,
                       out_dir: Path):
    """
    For a given phase:
      - x-axis: steps ordered by (episode, step) where phase==phase
      - stacked bars: reward components (summed across subnets) per step
      - vertical dashed lines: episode boundaries
      Steps belonging to other phases are fully truncated.
    """
    # --- get the timeline of this phase from df_total ---
    df_t = df_total[df_total["phase"] == phase].copy()
    if df_t.empty:
        print(f"[plot_phase_stacked] No steps for phase {phase}, skipping.")
        return None

    df_t = df_t.sort_values(["episode", "step"])
    # canonical index: all (episode, step) pairs that are in this phase
    idx = df_t[["episode", "step"]].drop_duplicates()
    ep_step_index = list(map(tuple, idx.to_numpy()))   # [(ep, step), ...]

    # --- aggregate rewards for this phase & align with that index ---
    df_r = df_rewards[df_rewards["phase"] == phase].copy()

    pivot = (
        df_r.groupby(["episode", "step", "component"])["value"]
            .sum()
            .reset_index()
            .pivot_table(
                index=["episode", "step"],
                columns="component",
                values="value",
                fill_value=0,
            )
    )

    # reindex: this is where truncation happens:
    # only (episode, step) pairs that have phase==phase survive
    pivot = pivot.reindex(ep_step_index, fill_value=0)

    x_positions = np.arange(len(pivot))

    ticks = []
    tick_labels = []
    prev_ep = None
    for i, (ep, st) in enumerate(ep_step_index):
        if ep != prev_ep and not np.isnan(ep):
            ticks.append(i)
            episode = round(ep)
            step = round(st)
            tick_labels.append(f"episode {episode}, step {step}")
            prev_ep = ep

        if st % 10 == 0:
            ticks.append(i)
            step = round(st)
            tick_labels.append(f"step {step}")
    ticks, tick_labels = zip(*dict.fromkeys(zip(ticks, tick_labels)))

    # dynamic width so bars don't get squished
    num_steps = len(pivot)
    width = max(12, num_steps * 0.2)
    fig, ax = plt.subplots(figsize=(width, 6))

    # stacked bars
    bottom = np.zeros(len(pivot))
    components = list(pivot.columns)
    for comp in components:
        vals = pivot[comp].values
        ax.bar(x_positions, vals, bottom=bottom, label=comp)
        bottom = bottom + vals

    # episode separators
    _add_episode_vlines(ax, ep_step_index)

    # x tick labels: "ep:step"
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=90)

    ax.set_xlabel("Episode : Step")
    ax.set_xlim(0, num_steps)
    ax.set_ylabel("Reward")
    ax.set_title(f"Phase {phase} – Reward Decomposition per Step (stacked)")
    ax.legend(title="Component", bbox_to_anchor=(1.05, 1), loc="upper left")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phase_{phase}_stacked_rewards.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_phase_stacked] Saved: {out_path}")
    return out_path

def plot_components_per_step(df_rewards: pd.DataFrame,
                             df_total: pd.DataFrame,
                             out_dir: Path):
    if df_total.empty:
        return None

    # --- canonical index of ALL steps (even if reward_list was empty) ---
    base_idx_df = (
        df_total[["episode", "step"]]
        .drop_duplicates()
        .sort_values(["episode", "step"])
    )
    ep_step_index = list(map(tuple, base_idx_df.to_numpy()))  # [(ep, step), ...]

    # --- aggregate rewards only where we actually have components ---
    if df_rewards.empty:
        # no components at all → everything is zero
        pivot = pd.DataFrame(
            0.0,
            index=pd.MultiIndex.from_tuples(ep_step_index, names=["episode", "step"]),
            columns=[],
        )
    else:
        pivot = (
            df_rewards.groupby(["episode", "step", "component"])["value"]
            .sum()
            .reset_index()
            .pivot_table(
                index=["episode", "step"],
                columns="component",
                values="value",
                fill_value=0.0,
            )
        )
        # reindex to include steps with no rewards (all zeros)
        pivot = pivot.reindex(
            pd.MultiIndex.from_tuples(ep_step_index, names=["episode", "step"]),
            fill_value=0.0,
        )

    # --- prepare plotting ---
    x = np.arange(len(ep_step_index))  # 0..N-1
    num_steps = len(ep_step_index)
    width = max(12, num_steps * 0.2)
    fig, ax = plt.subplots(figsize=(width, 6))

    bottom = np.zeros(len(ep_step_index))
    cols = list(pivot.columns)

    # stacked bars (if no components, cols is empty → nothing drawn)
    for comp in cols:
        vals = pivot[comp].to_numpy()
        ax.bar(x, vals, bottom=bottom, label=comp)
        bottom = bottom + vals

    # --- episode separator lines ---
    _add_episode_vlines(ax, ep_step_index)

    # --- smart x-ticks: episode start/end + every 10 steps ---
    ticks = []
    labels = []
    prev_ep = None
    for i, (ep, st) in enumerate(ep_step_index):
        # episode start
        if ep != prev_ep and not np.isnan(ep):
            ticks.append(i)
            labels.append(f"{round(ep)}:start")
            prev_ep = ep
        # every 10 steps
        if st % 10 == 0:
            ticks.append(i)
            labels.append(f"step {round(st)}")

    # deduplicate while preserving order
    if ticks:
        ticks_labels = list(dict.fromkeys(zip(ticks, labels)))
        ticks, labels = zip(*ticks_labels)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=90)
    else:
        ax.set_xticks(x)

    ax.set_xlabel("Episode : Step")
    ax.set_ylabel("Reward (summed across subnets)")
    ax.set_title("Reward Components per Step (summed across subnets)")
    if cols:
        ax.legend(title="Component", bbox_to_anchor=(1.05, 1), loc="upper left")

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "stacked_components_per_step.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out

def plot_components_by_subnet(df_rewards: pd.DataFrame, out_dir: Path):
    if df_rewards.empty:
        return None
    pivot = (
        df_rewards.groupby(["subnet", "component"])["value"]
        .sum()
        .reset_index()
        .pivot_table(index="subnet", columns="component", values="value", fill_value=0)
        .sort_index()
    )
    idx = range(len(pivot))
    plt.figure()
    bottom = None
    cols = list(pivot.columns)
    for i, comp in enumerate(cols):
        vals = pivot[comp].values
        if i == 0:
            plt.bar(idx, vals, label=comp)
            bottom = vals
        else:
            plt.bar(idx, vals, bottom=bottom, label=comp)
            bottom = [b + v for b, v in zip(bottom, vals)]
    plt.title("Cumulative Reward by Subnet and Component")
    plt.xlabel("Subnet")
    plt.ylabel("Cumulative reward")
    plt.xticks(list(idx), pivot.index, rotation=45, ha="right")
    plt.legend(title="Component")
    out = out_dir / "stacked_components_by_subnet.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out

def plot_episode_detail(df_rewards: pd.DataFrame, df_acost: pd.DataFrame, episode: int, out_dir: Path):
    """
    For a given episode:
      - Stacked bars per step: Local Work Failure / Access Service Failure / Red Impact Achieved / Action Cost (total)
      y-axis: reward + cost
      x-axis: step numbers
    """
    if df_rewards.empty and df_acost.empty:
        return None

    # --- reward components per step ---
    df_r = df_rewards[df_rewards["episode"] == episode].copy()
    if not df_r.empty:
        comp_per_step = (
            df_r.groupby(["step", "component"])["value"]
                .sum()
                .reset_index()
                .pivot_table(index="step", columns="component", values="value", fill_value=0)
                .sort_index()
        )
    else:
        comp_per_step = pd.DataFrame()

    # --- total action cost per step (sum over agents) ---
    df_a = df_acost[df_acost["episode"] == episode].copy()
    if not df_a.empty:
        acost_per_step = (
            df_a.groupby("step")["action_cost"]
                .sum()
                .reset_index()
                .sort_values("step")
                .set_index("step")
        )
    else:
        acost_per_step = pd.DataFrame(columns=["action_cost"])

    # --- merge and align steps ---
    steps_union = sorted(set(comp_per_step.index.tolist()) | set(acost_per_step.index.tolist()))
    if not steps_union:
        return None

    # reindex to ensure all steps exist
    comp_plot = comp_per_step.reindex(steps_union, fill_value=0)
    acost_plot = acost_per_step.reindex(steps_union, fill_value=0)

    # --- add Action Cost as a "component" ---
    comp_plot["Action Cost (total)"] = acost_plot["action_cost"].values

    # --- stacked bar chart ---
    # make width depend on number of steps
    width = max(10, len(steps_union) * 0.2)  # 0.4–0.5 is a good per-step factor
    fig, ax = plt.subplots(figsize=(width, 6))

    components = list(comp_plot.columns)
    bottom = [0] * len(comp_plot)

    for comp in components:
        vals = comp_plot[comp].values
        ax.bar(range(len(steps_union)), vals, bottom=bottom, label=comp)
        bottom = [b + v for b, v in zip(bottom, vals)]

    ax.set_title(f"Episode {episode}: Stacked Reward and Action Cost per Step")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward / Action Cost")
    ax.set_xlim(0, len(steps_union))
    ax.set_xticks(range(len(steps_union)))
    ax.legend(title="Component", bbox_to_anchor=(1.05, 1), loc="upper left")

    out = out_dir / f"episode_{episode}_reward_and_actioncost_stacked.png"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, default=Path("reward_log.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("reward_plots"))
    ap.add_argument("--episode", type=int, default=0, help="Episode number for detailed combined plot")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    df_rewards, df_total, df_acost = parse_log(args.log)

    # General plots
    p1 = plot_total(df_total, args.out)
    p2 = plot_components_per_step(df_rewards, df_total, args.out)
    p3 = plot_components_by_subnet(df_rewards, args.out)
    p4 = plot_action_cost_by_agent(df_acost, args.out)
    p5 = None
    if args.episode is not None:
        p5 = plot_episode_detail(df_rewards, df_acost, args.episode, args.out)

    phase_out_dir = Path(args.out) / "PhasePlots"

    for phase in sorted(df_rewards["phase"].dropna().unique()):
        plot_phase_stacked(df_rewards, df_total, phase=int(phase), out_dir=phase_out_dir)

    print("Saved figures:")
    print(" - Total per step:", p1)
    print(" - Components per step:", p2)
    print(" - By subnet:", p3)
    print(" - Action Cost by Agent:", p4)
    if args.episode is not None:
        print(" - Episode detail:", p5)

if __name__ == "__main__":
    main()
