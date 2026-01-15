import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scienceplots


COMPONENT_NAME_MAP = {
    "LWF": "Local Work",
    "ASF": "Access Service",
    "RIA": "Red Impact",
}

SUBNET_LABEL_MAP = {
    "admin_network_subnet": "Admin",
    "contractor_network_subnet": "Contractor",
    "office_network_subnet": "Office",
    "operational_zone_a_subnet": "Operational A",
    "operational_zone_b_subnet": "Operational B",
    "public_access_zone_subnet": "Public",
    "restricted_zone_a_subnet": "Restricted A",
    "restricted_zone_b_subnet": "Restricted B",
}

OKABE_ITO = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]


def parse_log(log_path: Path) -> pd.DataFrame:
    episode = None
    step = None
    reward_rows = []

    with log_path.open("r", encoding="utf-8") as f:
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
                for subnet, compdict in obj.get("reward_list", {}).items():
                    for comp_abbrev, val in compdict.items():
                        comp_full = COMPONENT_NAME_MAP.get(comp_abbrev, comp_abbrev)
                        reward_rows.append(
                            {
                                "episode": episode,
                                "step": step,
                                "phase": phase,
                                "subnet": subnet,
                                "component": comp_full,
                                "value": float(val),
                            }
                        )

    return pd.DataFrame(
        reward_rows,
        columns=["episode", "step", "phase", "subnet", "component", "value"],
    )


def _aggregate_rewards(df_rewards: pd.DataFrame, avg: bool) -> pd.DataFrame:
    if df_rewards.empty:
        return pd.DataFrame(columns=["phase", "subnet", "component", "value"])

    if avg:
        agg = (
            df_rewards.groupby(["episode", "phase", "subnet", "component"], dropna=True)["value"]
            .sum()
            .groupby(["phase", "subnet", "component"], dropna=True)
            .mean()
            .reset_index()
        )
    else:
        agg = (
            df_rewards.groupby(["phase", "subnet", "component"], dropna=True)["value"]
            .sum()
            .reset_index()
        )
    return agg


def _order_subnets(agg: pd.DataFrame) -> list[str]:
    preferred_order = [
        "operational_zone_a_subnet",
        "operational_zone_b_subnet",
        "restricted_zone_a_subnet",
        "restricted_zone_b_subnet",
        "contractor_network_subnet",
        "admin_network_subnet",
        "office_network_subnet",
        "public_access_zone_subnet",
    ]

    present = set(agg["subnet"].dropna().unique())
    preferred = [s for s in preferred_order if s in present]

    # anything not in the preferred list goes at the end (stable, sorted)
    other = sorted([s for s in present if s not in preferred_order])
    return preferred + other



def _order_components(agg: pd.DataFrame) -> list[str]:
    all_components = sorted(agg["component"].dropna().unique())
    preferred = [v for v in COMPONENT_NAME_MAP.values() if v in all_components]
    return preferred + [c for c in all_components if c not in preferred]


def _style_rcparams(kind: str = "single"):
    if kind == "double":
        BASE_FONT, LABEL_FONT, TICK_FONT, LEGEND_FONT = 9, 9, 8, 8
    else:
        BASE_FONT, LABEL_FONT, TICK_FONT, LEGEND_FONT = 11, 11, 10, 9

    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 600,
            "font.size": BASE_FONT,
            "axes.labelsize": LABEL_FONT,
            "xtick.labelsize": TICK_FONT,
            "ytick.labelsize": TICK_FONT,
            "legend.fontsize": LEGEND_FONT,
            "legend.title_fontsize": LEGEND_FONT,
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.borderpad": 0.3,
            "xtick.major.pad": 2.0,
            "ytick.major.pad": 2.0,
            "axes.labelpad": 2.0,
            "figure.autolayout": False,
            "figure.constrained_layout.use": False,
        }
    )


def _save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{stem}.png"
    out_pdf = out_dir / f"{stem}.pdf"
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_png


def _apply_common_axes_style(ax: plt.Axes, y_label: str):
    ax.set_xlabel("Subnet")
    ax.set_ylabel(y_label, labelpad=2)

    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.axhline(0.0, linewidth=1.3, color="black", zorder=4)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="y", pad=2)
    ax.tick_params(axis="x", pad=2)


def _tighten_x_axis(ax: plt.Axes, n: int):
    ax.margins(x=0)
    ax.set_xlim(-0.5, n - 0.5)


def _set_5step_negative_yticks(ax: plt.Axes, agg_for_limits: pd.DataFrame):
    """
    Force y ticks: 0, -5, -10, -15, -20 (and extend if needed).
    """
    # compute min total (stacked) to know if we must go below -20
    if agg_for_limits.empty:
        ticks = [0, -5, -10, -15, -20]
        ax.set_yticks(ticks)
        ax.set_ylim(-22, 1)
        return

    # totals per bar (whatever grouping index is present)
    if set(["phase", "subnet"]).issubset(agg_for_limits.columns):
        totals = agg_for_limits.groupby(["phase", "subnet"], dropna=True)["value"].sum()
    else:
        totals = agg_for_limits.groupby(["subnet"], dropna=True)["value"].sum()

    ymin = float(totals.min()) if len(totals) else -20.0

    # go down in steps of 5 until ymin fits
    min_tick = -20
    while ymin < min_tick - 1e-9:
        min_tick -= 5

    ticks = list(range(0, min_tick - 1, -5))  # 0, -5, ..., min_tick
    ax.set_yticks(ticks)

    # y-limits: show a bit below lowest tick, and a hair above 0
    ax.set_ylim(min_tick - 2, 1)


def plot_all_phases_grouped_stacked_single_column(df_rewards: pd.DataFrame, out_dir: Path, avg: bool = True):
    if df_rewards.empty or "phase" not in df_rewards.columns:
        return None

    with plt.style.context(["science", "ieee", "bright", "no-latex"]):
        _style_rcparams("single")
        agg = _aggregate_rewards(df_rewards, avg=avg)
        if agg.empty:
            return None

        phases = sorted(agg["phase"].dropna().unique())
        if not phases:
            return None

        subnets = _order_subnets(agg)
        components = _order_components(agg)

        pivot = agg.pivot_table(
            index=["phase", "subnet"],
            columns="component",
            values="value",
            fill_value=0.0,
        )
        for c in components:
            if c not in pivot.columns:
                pivot[c] = 0.0
        pivot = pivot[components]

        x = np.arange(len(subnets))
        n_phases = len(phases)

        group_width = 0.86
        bar_w = group_width / n_phases
        offsets = (np.arange(n_phases) - (n_phases - 1) / 2.0) * bar_w

        fig_w, fig_h = 3.5, 2.8
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_axes([0.10, 0.22, 0.88, 0.56])

        color_cycle = (OKABE_ITO * ((len(components) // len(OKABE_ITO)) + 1))[: len(components)]

        for i, phase in enumerate(phases):
            xpos = x + offsets[i]
            bottom = np.zeros(len(subnets), dtype=float)

            rows = []
            for s in subnets:
                if (phase, s) in pivot.index:
                    rows.append(pivot.loc[(phase, s)].to_numpy(dtype=float))
                else:
                    rows.append(np.zeros(len(components), dtype=float))
            mat = np.vstack(rows)

            for j, (comp, color) in enumerate(zip(components, color_cycle)):
                vals = mat[:, j]
                ax.bar(
                    xpos,
                    vals,
                    width=bar_w * 0.98,
                    bottom=bottom,
                    label=comp if i == 0 else None,
                    color=color,
                    edgecolor="black",
                    linewidth=0.4,
                    zorder=3,
                )
                bottom += vals

        y_label = "Average Reward" if avg else "Cumulative Reward"
        _apply_common_axes_style(ax, y_label=y_label)

        short_labels = [SUBNET_LABEL_MAP.get(s, s) for s in subnets]
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=25, ha="right")
        _tighten_x_axis(ax, n=len(subnets))

        # y ticks in steps of 5 (0, -5, -10, ...)
        _set_5step_negative_yticks(ax, agg_for_limits=agg)

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            title="Component",
            loc="upper center",
            bbox_to_anchor=(0.0, 0.98, 1.0, 0.0),
            mode="expand",
            ncol=3,
            frameon=True,
        )

        stem = f"grouped_stacked_components_by_subnet_all_phases_singlecol_{'avg' if avg else 'sum'}"
        return _save_fig(fig, out_dir=out_dir, stem=stem)


def plot_single_phase_stacked_single_column(df_rewards: pd.DataFrame, out_dir: Path, phase: str, avg: bool = True):
    if df_rewards.empty or "phase" not in df_rewards.columns:
        return None

    with plt.style.context(["science", "ieee", "bright", "no-latex"]):
        _style_rcparams("single")
        agg = _aggregate_rewards(df_rewards, avg=avg)
        if agg.empty:
            return None

        agg = agg[agg["phase"] == phase].copy()
        if agg.empty:
            return None

        subnets = _order_subnets(agg)
        components = _order_components(agg)

        pivot = agg.pivot_table(
            index=["subnet"],
            columns="component",
            values="value",
            fill_value=0.0,
        )
        for c in components:
            if c not in pivot.columns:
                pivot[c] = 0.0
        pivot = pivot[components]

        x = np.arange(len(subnets))

        fig_w, fig_h = 3.5, 2.8
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_axes([0.10, 0.22, 0.88, 0.56])

        color_cycle = (OKABE_ITO * ((len(components) // len(OKABE_ITO)) + 1))[: len(components)]
        bottom = np.zeros(len(subnets), dtype=float)

        rows = []
        for s in subnets:
            if s in pivot.index:
                rows.append(pivot.loc[s].to_numpy(dtype=float))
            else:
                rows.append(np.zeros(len(components), dtype=float))
        mat = np.vstack(rows)

        for j, (comp, color) in enumerate(zip(components, color_cycle)):
            vals = mat[:, j]
            ax.bar(
                x,
                vals,
                width=0.85,
                bottom=bottom,
                label=comp,
                color=color,
                edgecolor="black",
                linewidth=0.4,
                zorder=3,
            )
            bottom += vals

        y_label = "Average Reward" if avg else "Cumulative Reward"
        _apply_common_axes_style(ax, y_label=y_label)

        short_labels = [SUBNET_LABEL_MAP.get(s, s) for s in subnets]
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=25, ha="right")
        _tighten_x_axis(ax, n=len(subnets))

        _set_5step_negative_yticks(ax, agg_for_limits=agg)

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            title=None,
            loc="upper center",
            bbox_to_anchor=(0.0, 0.98, 1.0, 0.0),
            mode="expand",
            ncol=3,
            frameon=True,
        )

        safe_phase = str(phase).replace(" ", "_").replace("/", "_")
        stem = f"stacked_components_by_subnet_phase_{safe_phase}_singlecol_{'avg' if avg else 'sum'}"
        return _save_fig(fig, out_dir=out_dir, stem=stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, default=Path("reward_log.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("reward_plots"))
    ap.add_argument("--episode", type=int, default=None)
    ap.add_argument("--sum", action="store_true")
    args = ap.parse_args()

    df_rewards = parse_log(args.log)
    if args.episode is not None and not df_rewards.empty:
        df_rewards = df_rewards[df_rewards["episode"] == args.episode].copy()

    out_dir = Path(args.out) / "PhasePlots"
    avg = not args.sum

    out_all = plot_all_phases_grouped_stacked_single_column(df_rewards, out_dir=out_dir, avg=avg)

    phase_outs = []
    if not df_rewards.empty and "phase" in df_rewards.columns:
        phases = sorted(df_rewards["phase"].dropna().unique())
        for ph in phases:
            p = plot_single_phase_stacked_single_column(df_rewards, out_dir=out_dir, phase=ph, avg=avg)
            if p is not None:
                phase_outs.append(p)

    print("Saved figures:")
    if out_all is not None:
        print(f" - {out_all}")
    for p in phase_outs:
        print(f" - {p}")


if __name__ == "__main__":
    main()
