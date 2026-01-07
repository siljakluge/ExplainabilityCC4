from __future__ import annotations

import os, sys, time, json
from typing import Dict, List, Any, Optional
from statistics import mean, stdev
from datetime import datetime
from pathlib import Path

from policy_shap_heuristic import train_surrogate_and_shap
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

# -------- Runner  --------

def run_explainability(
    submission,
    output_dir: str,
    max_eps: int = 10,
    seed: Optional[int] = None,
    episode_length: int = 500,
    shap: bool = False,
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

            observations, rewards_scalar, term, trunc, info = wrapped_cyborg.step(actions)
            all_observations.append(observations)
            reward_per_episode += sum(rewards_scalar.values())

            # for SHAP
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
    # SHAP Analysis
    if shap:
        summary = train_surrogate_and_shap(
            policy_rows,
            out_dir=os.path.join(output_dir, "SHAPAnalysis")
        )
        print(summary)


# -------- CLI --------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("CybORG Reward Decomposition")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-eps", type=int, default=10)
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--shap", type=bool, default=True)
    parser.add_argument("--rew-decomp", type=bool, default=True)
    parser.add_argument("--output", type=str, default=os.path.abspath("Results"))
    parser.add_argument("--submission-path", type=str, default=os.path.abspath(""))
    args = parser.parse_args()

    log_path = Path("reward_log.jsonl")
    log_path.write_text("")

    os.makedirs(args.output, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.output, f"decomposition_{ts}")
    rmkdir(run_dir)
    Submission = load_submission(args.submission_path)
    submission = Submission

    run_explainability(
        submission=submission,
        output_dir=run_dir,
        max_eps=args.max_eps,
        seed=args.seed,
        episode_length=args.episode_length,
        shap=args.shap,
    )

    if log_path.exists() & args.rew_decomp:
        out_dir = os.path.join(run_dir, "DecompPlots")
        os.makedirs(out_dir, exist_ok=True)
        out_dir_phases = os.path.join(out_dir, "PhasePlots")
        os.makedirs(out_dir_phases, exist_ok=True)
        out_dir_episodes = os.path.join(out_dir, "EpisodePlots")
        os.makedirs(out_dir_episodes, exist_ok=True)

        df_rewards, df_total, df_acost = parse_log(log_path)

        plot_total(df_total, out_dir=Path(out_dir))
        plot_components_by_subnet(df_rewards, out_dir=Path(out_dir))
        plot_components_per_step(df_rewards, df_total, out_dir=Path(out_dir))
        plot_action_cost_by_agent(df_acost, out_dir=Path(out_dir))
        plot_components_by_phase(df_rewards, df_total, df_acost, out_dir=Path(out_dir))
        plot_components_by_subnet_per_phase(df_rewards, out_dir=Path(out_dir_phases))
        for phase in range(0, 3):
            plot_phase_stacked(df_rewards, df_total, phase=int(phase), out_dir=Path(out_dir_phases))
        for ep in range(args.max_eps):
            plot_episode_detail(df_rewards, df_acost, episode = ep, out_dir=Path(out_dir_episodes))
