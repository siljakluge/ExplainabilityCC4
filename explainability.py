from __future__ import annotations

import os, sys, time, json
from typing import Dict, List, Any, Optional
from statistics import mean, stdev
from datetime import datetime
from pathlib import Path

from policy_shap_heuristic import train_surrogate_and_shap
from shap_gnn import run_shap
from plot_rew_decomp import (parse_log,
                             plot_episode_detail,
                             plot_components_by_subnet,
                             plot_total,
                             plot_phase_stacked,
                             plot_components_per_step,
                             plot_action_cost_by_agent,
                             plot_components_by_phase,
                             plot_components_by_subnet_per_phase)

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import pandas as pd

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

# -------- Utilities  --------

def rmkdir(path: str):
    partial_path = ""
    for p in path.split("/"):
        if not p:
            continue
        partial_path = os.path.join(partial_path, p)
        if os.path.exists(partial_path):
            if os.path.isdir(partial_path):
                continue
            raise RuntimeError(f"Cannot create {partial_path} (exists as file).")
        os.mkdir(partial_path)

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)

def save_csv(rows: List[Dict[str, Any]], path: str):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w") as f:
            f.write("")  # empty file
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def load_submission(source: str):
    sys.path.insert(0, source)
    if source.endswith(".zip"):
        from submission import Submission
    else:
        from submission import Submission
    sys.path.remove(source)
    return Submission

def _action_to_label(act, mode: str = "type") -> str:
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

def plot_action_frequencies_per_agent(
    all_actions: list[dict],
    out_dir: Path,
    mode: str = "type",          # "type" or "full"
    normalize: bool = False,     # True -> percentages per agent
    top_k: int | None = 25       # compress rare actions (global)
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- flatten ---
    rows = []
    for step_i, step_actions in enumerate(all_actions):
        for agent, act in step_actions.items():
            rows.append({
                "step": step_i,
                "agent": agent,
                "action": _action_to_label(act, mode=mode),
            })

    if not rows:
        return None

    df = pd.DataFrame(rows)

    # --- optional: keep only top_k actions globally ---
    if top_k is not None:
        vc = df["action"].value_counts()
        keep = set(vc.head(top_k).index)
        df["action"] = df["action"].where(df["action"].isin(keep), other="OTHER")

    # --- counts table: agent x action ---
    pivot = pd.crosstab(df["agent"], df["action"])  # counts
    pivot = pivot.loc[sorted(pivot.index)]
    pivot = pivot[pivot.sum(axis=0).sort_values(ascending=False).index]

    if normalize:
        pivot = pivot.div(pivot.sum(axis=1).replace(0, 1), axis=0) * 100.0

    # --- stacked bar ---
    plt.figure(figsize=(max(10, pivot.shape[1] * 0.9), max(5, pivot.shape[0] * 0.9)))
    x = range(len(pivot.index))
    bottom = None
    for j, action in enumerate(pivot.columns):
        vals = pivot[action].to_numpy()
        if j == 0:
            plt.bar(x, vals, label=action)
            bottom = vals
        else:
            plt.bar(x, vals, bottom=bottom, label=action)
            bottom = bottom + vals

    ylabel = "Percent of actions (%)" if normalize else "Action count"
    plt.title(f"Action frequencies per agent ({mode})" + (" — normalized" if normalize else ""))
    plt.xlabel("Agent")
    plt.ylabel(ylabel)
    plt.xticks(list(x), list(pivot.index), rotation=45, ha="right")
    plt.legend(title="Action", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    out_bar = out_dir / f"action_freq_per_agent_{mode}_{'pct' if normalize else 'count'}.png"
    plt.savefig(out_bar, dpi=150, bbox_inches="tight")
    plt.close()

    # --- heatmap (handy when many actions) ---
    plt.figure(figsize=(max(10, pivot.shape[1] * 0.7), max(4, pivot.shape[0] * 0.6)))
    plt.imshow(pivot.to_numpy(), aspect="auto")
    plt.title(f"Action frequencies heatmap ({mode})" + (" — normalized" if normalize else ""))
    plt.xlabel("Action")
    plt.ylabel("Agent")
    plt.xticks(range(pivot.shape[1]), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(pivot.shape[0]), pivot.index)
    plt.colorbar(label=ylabel)
    plt.tight_layout()

    out_heat = out_dir / f"action_freq_heatmap_{mode}_{'pct' if normalize else 'count'}.png"
    plt.savefig(out_heat, dpi=150, bbox_inches="tight")
    plt.close()

    return {"stacked_bar": out_bar, "heatmap": out_heat, "table": pivot, "df_long": df}

def build_shap_dataset_all_features(infos, fill_value=0):
    rows = []
    all_keys = set()

    # first pass: collect keys + rows
    for step_i, step_info in enumerate(infos):
        for agent, d in step_info.items():
            sf = d.get("shap_features")
            y = d.get("chosen_action_type")
            if not sf or y is None:
                continue
            all_keys.update(sf.keys())
            rows.append({"step": step_i, "agent": agent, "y": str(y), **sf})

    df = pd.DataFrame(rows)

    # ensure all feature columns exist
    for k in sorted(all_keys):
        if k not in df.columns:
            df[k] = fill_value

    # fill missing values (features absent on some steps)
    feature_cols = sorted(all_keys)
    df[feature_cols] = df[feature_cols].fillna(fill_value)

    return df
# -------- Runner  --------

def run_explainability(
    submission,
    output_dir: str,
    max_eps: int = 10,
    seed: Optional[int] = None,
    episode_length: int = 500,
    shap: bool = False,
    is_heuristic: bool = False,
):
    """
    Runs episodes and logs a full reward decomposition + SHAP for the blue team
    """
    cyborg_version = CYBORG_VERSION
    scenario = "Scenario4"

    # Env
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=episode_length,
    )
    cyborg = CybORG(sg, "sim", seed=seed)
    wrapped_cyborg = submission.wrap(cyborg)

    # Headers
    version_header = f"CybORG v{cyborg_version}, {scenario}"
    author_header = f"Author: {submission.NAME}, Team: {submission.TEAM}, Technique: {submission.TECHNIQUE}"
    print(version_header)
    print(author_header)
    print(f"Using agents {submission.AGENTS}")

    policy_rows = []
    all_observations = []
    all_actions = []
    all_info = []
    total_rewards_per_episode = []

    # main loop
    for epi in tqdm(range(max_eps), desc="Episodes"):
        observations, _ = wrapped_cyborg.reset()
        reward_per_episode = 0

        for t in range(episode_length):
            try:
                log_path = Path("reward_log.jsonl")
                log_entry = {
                    "episode": epi,
                    "step": t,
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(log_entry, default=str) + "\n")
            except Exception as e:
                print(f"Failed to write log for episode {epi}: {e}")

            actions = {
                agent_name: agent.get_action(
                    observations[agent_name], wrapped_cyborg.action_space(agent_name)
                )
                for agent_name, agent in submission.AGENTS.items()
                if agent_name in wrapped_cyborg.agents
            }
            all_actions.append(
                    {
                        agent_name: cyborg.get_last_action(agent_name)
                        for agent_name in wrapped_cyborg.agents
                    }
                )

            observations, rewards_scalar, term, trunc, info = wrapped_cyborg.step(actions)
            all_observations.append(observations)
            all_info.append(info)
            reward_per_episode += sum(rewards_scalar.values())

            # for SHAP
            if is_heuristic:
                for agent_name, agent in submission.AGENTS.items():
                    if not hasattr(agent, "info") or len(agent.info) == 0:
                        continue
                    step_info = agent.info[-1]
                    if "Predicates" in step_info and "ActionClass" in step_info:
                        policy_rows.append(
                            (step_info["Predicates"], step_info["ActionClass"])
                        )

            # episode termination?
            done = {agent: term.get(agent, False) or trunc.get(agent, False) for agent in wrapped_cyborg.agents}
            if all(done.values()) and done:
                break

        total_rewards_per_episode.append(reward_per_episode)

    # --- Summaries and files ---
    reward_mean = mean(total_rewards_per_episode) if total_rewards_per_episode else 0.0
    reward_stdev = stdev(total_rewards_per_episode) if len(total_rewards_per_episode) > 1 else 0.0
    print(f"Average total (scalar): {reward_mean:.2f} ± {reward_stdev:.2f}")

    # c) Scalar totals you already track, for convenience
    save_json(
        {
            "submission": {
                "author": submission.NAME,
                "team": submission.TEAM,
                "technique": submission.TECHNIQUE,
            },
            "parameters": {
                "seed": seed,
                "episode_length": episode_length,
                "max_episodes": max_eps,
            },
            "reward_scalar": {
                "mean": reward_mean,
                "stdev": reward_stdev,
                "per_episode": total_rewards_per_episode,
            },
        },
        os.path.join(output_dir, "summary_scalar.json"),
    )
    # plot actions
    plot_action_frequencies_per_agent(all_actions, out_dir=Path(os.path.join(output_dir, "Actions")), mode="type", normalize=True, top_k=None)
    plot_action_frequencies_per_agent(all_actions, out_dir=Path(os.path.join(output_dir, "Actions")), mode="type", normalize=False, top_k=25)

    # SHAP Analysis
    if shap & is_heuristic:
        summary = train_surrogate_and_shap(
            policy_rows,
            out_dir=os.path.join(output_dir, "SHAPAnalysis")
        )
        print(summary)
    elif shap & (not is_heuristic):
        df = build_shap_dataset_all_features(all_info)

        # (optional) make sure prev_action_success is numeric
        df["prev_action_success"] = pd.to_numeric(df["prev_action_success"], errors="coerce").fillna(-1)
        out_dir_shap = os.path.join(output_dir, "SHAPAnalysis")
        os.makedirs(out_dir_shap, exist_ok=True)

        clf, explainer, shap_values, feature_names = run_shap(df, max_classes=8, out_dir=out_dir_shap)


# -------- CLI --------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("CybORG Reward Decomposition")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-eps", type=int, default=100)
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--shap", type=bool, default=True)
    parser.add_argument("--rew-decomp", type=bool, default=True)
    parser.add_argument("--output", type=str, default=os.path.abspath("Results"))
    parser.add_argument("--submission-path", type=str, default=os.path.abspath(""))
    parser.add_argument("--is-heuristic", type=bool, default=True)
    args = parser.parse_args()

    log_path = Path("reward_log.jsonl")
    log_path.write_text("")


    submission = load_submission(args.submission_path)
    if isinstance(submission, type):
        submission = submission()

    os.makedirs(args.output, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    agent_type = submission.NAME
    run_dir = os.path.join(args.output, f"{agent_type}_{ts}")
    rmkdir(run_dir)

    run_explainability(
        submission=submission,
        output_dir=run_dir,
        max_eps=args.max_eps,
        seed=args.seed,
        episode_length=args.episode_length,
        shap=args.shap,
        is_heuristic=args.is_heuristic,
    )

    if log_path.exists() & args.rew_decomp:
        out_dir = os.path.join(run_dir, "DecompPlots")
        os.makedirs(out_dir, exist_ok=True)
        out_dir_phases = os.path.join(out_dir, "PhasePlots")
        os.makedirs(out_dir_phases, exist_ok=True)

        df_rewards, df_total, df_acost = parse_log(log_path)

        plot_total(df_total, out_dir=Path(out_dir))
        plot_components_by_subnet(df_rewards, out_dir=Path(out_dir),avg=True)
        plot_components_by_subnet(df_rewards, out_dir=Path(out_dir),avg=False)
        #plot_action_cost_by_agent(df_acost, out_dir=Path(out_dir))
        plot_components_by_phase(df_rewards, df_total, df_acost, out_dir=Path(out_dir), avg=True)
        plot_components_by_phase(df_rewards, df_total, df_acost, out_dir=Path(out_dir), avg=False)
        plot_components_by_subnet_per_phase(df_rewards, out_dir=Path(out_dir_phases), avg=True)
        plot_components_by_subnet_per_phase(df_rewards, out_dir=Path(out_dir_phases), avg=False)
        """
        for phase in range(0, 3):
            plot_phase_stacked(df_rewards, df_total, phase=int(phase), out_dir=Path(out_dir_phases))
        """
        if args.max_eps <= 10:
            out_dir_episodes = os.path.join(out_dir, "EpisodePlots")
            os.makedirs(out_dir_episodes, exist_ok=True)
            for ep in range(args.max_eps):
                plot_episode_detail(df_rewards, df_acost, episode = ep, out_dir=Path(out_dir_episodes))
        if args.max_eps * args.episode_length <= 100:
            out_dir_episodes = os.path.join(out_dir, "EpisodePlots")
            os.makedirs(out_dir_episodes, exist_ok=True)
            plot_components_per_step(df_rewards, df_total, out_dir=Path(out_dir_episodes))
