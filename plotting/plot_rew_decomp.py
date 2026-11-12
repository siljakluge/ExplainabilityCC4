from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401


SUBNET_LABEL_MAP = {
    "admin_network_subnet": "Admin",
    "contractor_network_subnet": "Contractor",
    "office_network_subnet": "Office",
    "operational_zone_a_subnet": "Operational A",
    "operational_zone_b_subnet": "Operational B",
    "public_access_zone_subnet": "Public",
    "restricted_zone_a_subnet": "Restricted A",
    "restricted_zone_b_subnet": "Restricted B",
    "internet_subnet": "Internet",
    "action_cost": "Action Cost",
}

ATTACKER_ORDER = [
    "deception_aware",
    "discovery",
    "fsm_default",
    "impact_rush",
    "lateral_spread",
    "stealth_pivot",
]

REWARD_MODES = ["default"]

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "Result"

"""MODEL_DIRS_DEFAULT = {
    "SimpleGNN": RESULTS_DIR / "SimpleGNN",
    "Heuristic": RESULTS_DIR / "Heuristic",
    "RedVariants": RESULTS_DIR / "RedVariants",
    "Sleep": RESULTS_DIR / "Sleep",
}"""
MODEL_DIRS_DEFAULT = {
    "Curriculum": RESULTS_DIR / "Curriculum",
}

OKABE_ITO = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#000000",
]

SUBNET_ORDER = [
    "operational_zone_a_subnet",
    "operational_zone_b_subnet",
    "restricted_zone_a_subnet",
    "restricted_zone_b_subnet",
    "contractor_network_subnet",
    "admin_network_subnet",
    "office_network_subnet",
    "public_access_zone_subnet",
    "internet_subnet",
    "action_cost",
]
def compute_ylim(df: pd.DataFrame, pad: float = 0.1) -> tuple[float, float]:
    """
    Compute dynamic y-limits based on most negative value in df.
    Always keeps 0 at top.
    """
    if df.empty:
        return (-1.0, 0.0)

    min_val = float(df["value"].min())

    # ensure negative range (important if something is all 0)
    if min_val >= 0:
        min_val = -1.0

    y_min = min_val * (1.0 + pad)
    return y_min, 0.0

def _style_rcparams():
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 600,
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "legend.title_fontsize": 10,
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.borderpad": 0.3,
        }
    )


def _save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{stem}.png"
    out_pdf = out_dir / f"{stem}.pdf"
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return out_png


def _apply_common_axes_style(ax: plt.Axes, y_label: str):
    ax.set_xlabel("Subnet")
    ax.set_ylabel(y_label)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.axhline(0.0, linewidth=1.2, color="black", zorder=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", labelsize=11)


def reconstruct_episode_from_steps(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy().reset_index(drop=True)
    episodes = []
    cur_ep = 0
    prev_step = None

    for step in df["step"].tolist():
        if prev_step is not None and step < prev_step:
            cur_ep += 1
        episodes.append(cur_ep)
        prev_step = step

    df["episode_recon"] = episodes
    return df


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def parse_reward_log(log_path: Path) -> pd.DataFrame:
    rows = []
    entries = read_jsonl(log_path)

    for obj in entries:
        if not {"phase", "reward_list", "total", "step"}.issubset(obj.keys()):
            continue

        profile = obj.get("profile")
        step = int(obj["step"])
        phase = obj["phase"]
        reward_total = float(obj.get("total", 0.0))
        reward_list = obj.get("reward_list", {})

        if reward_list:
            for subnet, compdict in reward_list.items():
                subnet_val = 0.0
                for _comp, val in compdict.items():
                    subnet_val += float(val)
                rows.append(
                    {
                        "profile": profile,
                        "step": step,
                        "phase": phase,
                        "subnet": subnet,
                        "value": subnet_val,
                        "reward_total": reward_total,
                    }
                )
        else:
            rows.append(
                {
                    "profile": profile,
                    "step": step,
                    "phase": phase,
                    "subnet": None,
                    "value": 0.0,
                    "reward_total": reward_total,
                }
            )

    df = pd.DataFrame(rows, columns=["profile", "step", "phase", "subnet", "value", "reward_total"])
    if df.empty:
        return df

    return reconstruct_episode_from_steps(df)


def parse_scalar_log(log_path: Path) -> pd.DataFrame:
    rows = []
    entries = read_jsonl(log_path)

    for obj in entries:
        if "step" not in obj:
            continue

        total_scalar = None
        if "total_scalar" in obj:
            total_scalar = float(obj["total_scalar"])
        elif "rewards" in obj:
            total_scalar = float(obj["rewards"])

        if total_scalar is None:
            continue

        rows.append(
            {
                "profile": obj.get("profile"),
                "step": int(obj["step"]),
                "total_scalar": total_scalar,
            }
        )

    df = pd.DataFrame(rows, columns=["profile", "step", "total_scalar"])
    if df.empty:
        return df

    return reconstruct_episode_from_steps(df)


def build_attacker_subnet_table(
    reward_log_path: Path,
    scalar_log_path: Path,
    avg: bool = True,
) -> pd.DataFrame:
    df_r = parse_reward_log(reward_log_path)
    df_s = parse_scalar_log(scalar_log_path)

    if df_r.empty and df_s.empty:
        return pd.DataFrame()

    subnet_part = pd.DataFrame(columns=["profile", "subnet", "value"])

    if not df_r.empty:
        subnet_rows = df_r[df_r["subnet"].notna()].copy()

        if avg:
            subnet_part = (
                subnet_rows.groupby(["profile", "episode_recon", "subnet"], dropna=True)["value"]
                .sum()
                .groupby(["profile", "subnet"], dropna=True)
                .mean()
                .reset_index()
            )
        else:
            subnet_part = (
                subnet_rows.groupby(["profile", "subnet"], dropna=True)["value"]
                .sum()
                .reset_index()
            )

    reward_total_step = pd.DataFrame(columns=["profile", "episode_recon", "step", "reward_total"])
    if not df_r.empty:
        reward_total_step = (
            df_r.groupby(["profile", "episode_recon", "step"], dropna=True)["reward_total"]
            .first()
            .reset_index()
        )

    scalar_total_step = pd.DataFrame(columns=["profile", "episode_recon", "step", "total_scalar"])
    if not df_s.empty:
        scalar_total_step = (
            df_s.groupby(["profile", "episode_recon", "step"], dropna=True)["total_scalar"]
            .first()
            .reset_index()
        )

    action_cost_part = pd.DataFrame(columns=["profile", "subnet", "value"])

    if not reward_total_step.empty and not scalar_total_step.empty:
        merged = pd.merge(
            scalar_total_step,
            reward_total_step,
            on=["profile", "episode_recon", "step"],
            how="inner",
        )
        merged["action_cost"] = merged["total_scalar"] - merged["reward_total"]

        if avg:
            action_cost_part = (
                merged.groupby(["profile", "episode_recon"], dropna=True)["action_cost"]
                .sum()
                .groupby("profile", dropna=True)
                .mean()
                .reset_index()
            )
        else:
            action_cost_part = (
                merged.groupby("profile", dropna=True)["action_cost"]
                .sum()
                .reset_index()
            )

        action_cost_part["subnet"] = "action_cost"
        action_cost_part = action_cost_part.rename(columns={"action_cost": "value"})

    combined = pd.concat([subnet_part, action_cost_part], ignore_index=True)

    if combined.empty:
        return combined

    combined["subnet"] = pd.Categorical(combined["subnet"], categories=SUBNET_ORDER, ordered=True)
    combined = combined.sort_values(["profile", "subnet"]).reset_index(drop=True)
    return combined


def parse_reward_modes_arg(single_mode: Optional[str], modes_arg: str) -> list[str]:
    if single_mode:
        if single_mode not in REWARD_MODES:
            raise ValueError(f"Unknown reward mode '{single_mode}'. Available: {REWARD_MODES}")
        return [single_mode]

    if modes_arg.strip():
        modes = [m.strip() for m in modes_arg.split(",") if m.strip()]
        bad = [m for m in modes if m not in REWARD_MODES]
        if bad:
            raise ValueError(f"Unknown reward mode(s): {bad}. Available: {REWARD_MODES}")
        return modes

    return REWARD_MODES


def resolve_mode_dir(model_dir: Path, reward_mode: str, reward_blue: bool = False) -> Optional[Path]:
    """
    Supports:
      old layout:
        model_dir/<attacker>/
      new layout:
        model_dir/<reward_mode>/
        model_dir/<reward_mode>_blue/
    """
    candidates = []
    if reward_blue:
        candidates.append(model_dir / f"{reward_mode}_blue")
    candidates.append(model_dir / reward_mode)
    candidates.append(model_dir)

    for cand in candidates:
        if not cand.exists():
            continue
        attacker_dirs = [cand / a for a in ATTACKER_ORDER]
        if any((d / "reward_log.jsonl").exists() for d in attacker_dirs):
            return cand

    return None


def load_model_mode_table(
    model_name: str,
    model_dir: Path,
    reward_mode: str,
    attackers: Optional[list[str]] = None,
    avg: bool = True,
    reward_blue: bool = False,
) -> pd.DataFrame:
    if attackers is None:
        attackers = ATTACKER_ORDER

    mode_dir = resolve_mode_dir(model_dir, reward_mode=reward_mode, reward_blue=reward_blue)
    if mode_dir is None:
        print(f"[warn] No directory found for {model_name} / reward_mode={reward_mode} / reward_blue={reward_blue}")
        return pd.DataFrame()

    rows = []

    for attacker in attackers:
        reward_log = mode_dir / attacker / "reward_log.jsonl"
        scalar_log = mode_dir / attacker / "scalar_rewards.jsonl"

        if not reward_log.exists() or not scalar_log.exists():
            print(f"[warn] Missing logs for {model_name}/{mode_dir.name}/{attacker}")
            continue

        table = build_attacker_subnet_table(reward_log, scalar_log, avg=avg)
        if table.empty:
            continue

        table = table.copy()
        table["attacker"] = attacker
        table["reward_mode"] = reward_mode
        rows.append(table)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def plot_model_attacker_lines(
    model_name: str,
    model_dir: Path,
    out_dir: Path,
    reward_mode: str,
    attackers: Optional[list[str]] = None,
    avg: bool = True,
    show_legend: bool = False,
    reward_blue: bool = False,
):
    if attackers is None:
        attackers = ATTACKER_ORDER

    df = load_model_mode_table(
        model_name=model_name,
        model_dir=model_dir,
        reward_mode=reward_mode,
        attackers=attackers,
        avg=avg,
        reward_blue=reward_blue,
    )

    if df.empty:
        print(f"[warn] No data found for model {model_name}, reward mode {reward_mode}")
        return None

    with plt.style.context(["science", "ieee", "bright", "no-latex"]):
        _style_rcparams()

        fig, ax = plt.subplots(figsize=(6.8, 3.2))

        x_labels = [s for s in SUBNET_ORDER if s in set(df["subnet"].astype(str))]
        x = np.arange(len(x_labels))

        color_map = {
            attacker: OKABE_ITO[i % len(OKABE_ITO)]
            for i, attacker in enumerate(attackers)
        }

        for attacker in attackers:
            sub = df[df["attacker"] == attacker].copy()
            if sub.empty:
                continue

            y_vals = []
            for subnet in x_labels:
                val = sub.loc[sub["subnet"].astype(str) == subnet, "value"]
                y_vals.append(float(val.iloc[0]) if len(val) else 0.0)

            ax.plot(
                x,
                y_vals,
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                label=attacker,
                color=color_map[attacker],
            )

        pretty_labels = [SUBNET_LABEL_MAP.get(s, s) for s in x_labels]
        ax.set_xticks(x)
        ax.set_xticklabels(pretty_labels, rotation=25, ha="right", fontsize=9)

        _apply_common_axes_style(ax, y_label="Avg Reward" if avg else "Cumulative Reward")
        title_suffix = reward_mode + (" + blue" if reward_blue else "")
        ax.set_title(f"{model_name} ({title_suffix})", fontsize=12, fontweight="bold")

        y_min, y_max = compute_ylim(df)
        ax.set_ylim(y_min, y_max)

        if show_legend:
            ax.legend(
                loc="lower right",
                frameon=True,
                fontsize=10,
                handlelength=1.5,
                borderpad=0.4,
            )

        suffix = f"{reward_mode}" + ("_blue" if reward_blue else "")
        stem = f"{model_name}_reward_decomposition_lines_{suffix}_{'avg' if avg else 'sum'}"
        return _save_fig(fig, out_dir, stem)


def plot_all_models_2x2(
    model_dirs: dict[str, Path],
    out_dir: Path,
    reward_mode: str,
    attackers: Optional[list[str]] = None,
    avg: bool = True,
    reward_blue: bool = False,
):
    if attackers is None:
        attackers = ATTACKER_ORDER

    model_tables = {}

    for model_name, model_dir in model_dirs.items():
        df = load_model_mode_table(
            model_name=model_name,
            model_dir=model_dir,
            reward_mode=reward_mode,
            attackers=attackers,
            avg=avg,
            reward_blue=reward_blue,
        )
        if not df.empty:
            model_tables[model_name] = df

    if not model_tables:
        print(f"[warn] No model data found for reward mode {reward_mode}.")
        return None

    with plt.style.context(["science", "ieee", "bright", "no-latex"]):
        _style_rcparams()

        fig, axes = plt.subplots(
            2, 2,
            figsize=(7.16, 3.6),
            sharex=True,
            sharey=False
        )
        fig.subplots_adjust(
            left=0.07,
            right=0.995,
            bottom=0.16,
            top=0.86,
            wspace=0.18,
            hspace=0.25
        )
        axes = axes.flatten()

        legend_handles = None
        legend_labels = None

        color_map = {
            attacker: OKABE_ITO[i % len(OKABE_ITO)]
            for i, attacker in enumerate(attackers)
        }

        x_labels = SUBNET_ORDER
        x = np.arange(len(x_labels))

        for ax, (model_name, df) in zip(axes, model_tables.items()):
            for attacker in attackers:
                sub = df[df["attacker"] == attacker].copy()
                if sub.empty:
                    continue

                y_vals = []
                for subnet in x_labels:
                    val = sub.loc[sub["subnet"].astype(str) == subnet, "value"]
                    y_vals.append(float(val.iloc[0]) if len(val) else 0.0)

                ax.plot(
                    x,
                    y_vals,
                    marker="o",
                    linewidth=1.3,
                    markersize=3.0,
                    label=attacker,
                    color=color_map[attacker],
                )

            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

            ax.set_title(model_name, fontsize=11, fontweight="bold")
            ax.grid(axis="y", alpha=0.25, zorder=0)
            ax.axhline(0.0, linewidth=1.2, color="black", zorder=3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            pretty_labels = [SUBNET_LABEL_MAP.get(s, s) for s in x_labels]
            ax.set_xticks(x)
            ax.set_xticklabels(pretty_labels, rotation=25, ha="right", fontsize=9)

            y_min, y_max = compute_ylim(df)
            ax.set_ylim(y_min, y_max)

        for i in range(len(model_tables), len(axes)):
            axes[i].axis("off")

        axes[0].set_ylabel("Avg Reward" if avg else "Cumulative Reward")
        axes[2].set_ylabel("Avg Reward" if avg else "Cumulative Reward")
        axes[2].set_xlabel("Subnet", fontsize=10)
        axes[3].set_xlabel("Subnet", fontsize=10)

        for ax in axes:
            ax.tick_params(axis="x", pad=1)
            ax.xaxis.labelpad = 2
            ax.yaxis.labelpad = 2

        title_suffix = reward_mode + (" + blue" if reward_blue else "")
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.015),
            ncol=len(legend_labels),
            frameon=True,
            fontsize=11,
            handlelength=1.5,
            columnspacing=0.9,
            borderpad=0.3,
            title=title_suffix,
        )

        fig.subplots_adjust(
            left=0.07,
            right=0.995,
            bottom=0.12,
            top=0.855,
            wspace=0.20,
            hspace=0.18
        )

        suffix = f"{reward_mode}" + ("_blue" if reward_blue else "")
        stem = f"reward_decomposition_all_models_2x2_{suffix}_{'avg' if avg else 'sum'}"
        return _save_fig(fig, out_dir, stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("Plots/RewardDecomp"))
    ap.add_argument("--sum", action="store_true")

    ap.add_argument(
        "--reward-mode",
        type=str,
        default=None,
        choices=REWARD_MODES,
        help="Plot exactly one reward mode."
    )
    ap.add_argument(
        "--reward-modes",
        type=str,
        default="",
        help='Comma-separated reward modes to plot, e.g. "default,red_only". Empty => all.'
    )
    ap.add_argument(
        "--reward-blue",
        action="store_true",
        help="Look for <reward_mode>_blue directories instead of plain <reward_mode>."
    )

    args = ap.parse_args()

    avg = not args.sum
    out_dir = Path(args.out)
    reward_modes = parse_reward_modes_arg(args.reward_mode, args.reward_modes)

    saved = []

    for reward_mode in reward_modes:
        for model_name, model_dir in MODEL_DIRS_DEFAULT.items():
            p = plot_model_attacker_lines(
                model_name=model_name,
                model_dir=model_dir,
                out_dir=out_dir,
                reward_mode=reward_mode,
                attackers=ATTACKER_ORDER,
                avg=avg,
                reward_blue=args.reward_blue,
                show_legend=True,
            )
            if p is not None:
                saved.append(p)

        out = plot_all_models_2x2(
            model_dirs=MODEL_DIRS_DEFAULT,
            out_dir=out_dir,
            reward_mode=reward_mode,
            attackers=ATTACKER_ORDER,
            avg=avg,
            reward_blue=args.reward_blue,
        )
        if out is not None:
            saved.append(out)

    print("Saved figures:")
    for p in saved:
        print(f" - {p}")


if __name__ == "__main__":
    main()