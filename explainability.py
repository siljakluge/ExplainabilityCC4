# explainability.py
from __future__ import annotations

import os, sys, time, json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional
from statistics import mean, stdev
from datetime import datetime

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

# -------- Decomposition core --------

def _resolve_blue_team_name(env_controller) -> str:
    """
    Figure out which team entry corresponds to the blue agents.
    We avoid hardcoding "Blue" by scanning membership.
    """
    # env_controller.team: Dict[team_name -> List[agent_names]]
    for team_name, members in env_controller.team.items():
        if any("blue" in m for m in members):
            return team_name
    # Fallback if naming is unusual
    if "Blue" in env_controller.team:
        return "Blue"
    raise RuntimeError(f"Could not resolve blue team name from teams: {list(env_controller.team.keys())}")

@dataclass
class DecompEpisode:
    step_components: List[Dict[str, float]] = field(default_factory=list)
    step_total: List[float] = field(default_factory=list)

    def accumulate(self, comp: Dict[str, float], total: float):
        self.step_components.append(comp)
        self.step_total.append(total)

    def totals(self) -> Dict[str, float]:
        agg: Dict[str, float] = {}
        for c in self.step_components:
            for k, v in c.items():
                agg[k] = agg.get(k, 0.0) + float(v)
        agg["__total__"] = sum(self.step_total)
        return agg

def _plot_per_step_lines(step_components: List[Dict[str, float]], out_path: str, title: str):
    """
    Simple per-step line plot of each component; relies on matplotlib defaults (no seaborn).
    """
    if not step_components:
        return
    # collect series
    keys = sorted({k for d in step_components for k in d.keys()})
    xs = list(range(1, len(step_components) + 1))
    plt.figure()
    for k in keys:
        ys = [d.get(k, 0.0) for d in step_components]
        plt.plot(xs, ys, label=k)
    plt.xlabel("Step")
    plt.ylabel("Reward (per step)")
    plt.title(title)
    plt.grid(True)
    plt.legend(loc="best", fontsize="small")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def _plot_episode_totals_bars(ep_totals: Dict[str, float], out_path: str, title: str):
    items = [(k, v) for k, v in ep_totals.items() if k != "__total__"]
    if not items:
        return
    items.sort(key=lambda kv: kv[0])
    labels, vals = zip(*items)
    plt.figure()
    plt.bar(labels, vals)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Reward (episode sum)")
    plt.title(title + f"  |  Total={ep_totals.get('__total__', 0):.2f}")
    plt.grid(True, axis="y")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

# -------- Runner (similar to your run_evaluation, but decomposed) --------

def run_explainability(
    submission,
    output_dir: str,
    max_eps: int = 10,
    seed: Optional[int] = None,
    episode_length: int = 500,
    plot: bool = True,
):
    """
    Runs episodes and logs a full reward decomposition for the blue team:
    - per-step component dicts
    - per-episode totals per component
    - standard overall reward stats for compatibility
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

    # Output folder
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"decomposition_{ts}")
    rmkdir(run_dir)

    total_rewards_per_episode: List[float] = []
    decomp_episodes: List[DecompEpisode] = []
    policy_rows = []

    # main loop
    for epi in tqdm(range(max_eps), desc="Episodes"):
        observations, _ = wrapped_cyborg.reset()
        epi_decomp = DecompEpisode()
        per_step_scalars: List[float] = []

        for t in range(episode_length):
            # 1) get actions from blue agents as in your eval code
            actions = {
                agent_name: agent.get_action(
                    observations[agent_name], wrapped_cyborg.action_space(agent_name)
                )
                for agent_name, agent in submission.AGENTS.items()
                if agent_name in wrapped_cyborg.agents
            }

            observations, rewards_scalar, term, trunc, info = wrapped_cyborg.step(actions)
            for agent_name, agent in submission.AGENTS.items():
                if not hasattr(agent, "info") or len(agent.info) == 0:
                    continue
                step_info = agent.info[-1]
                if "Predicates" in step_info and "ActionClass" in step_info:
                    policy_rows.append(
                        (step_info["Predicates"], step_info["ActionClass"])
                    )

            # 2) total (scalar) that your pipeline uses (mean over blue agents this step)
            step_scalar = mean(rewards_scalar.values()) if rewards_scalar else 0.0

            # 3) *** decomposition *** via environment_controller.reward
            envc = wrapped_cyborg.env.environment_controller
            blue_team = _resolve_blue_team_name(envc)
            # This dict contains the *component* rewards for this step
            step_components: Dict[str, float] = dict(envc.reward.get(blue_team, {}))

            # 4) log it
            epi_decomp.accumulate(step_components, step_scalar)
            per_step_scalars.append(step_scalar)

            # episode termination?
            done = {agent: term.get(agent, False) or trunc.get(agent, False) for agent in wrapped_cyborg.agents}
            if all(done.values()) and done:
                break

        total_ep_reward = sum(per_step_scalars)
        total_rewards_per_episode.append(total_ep_reward)
        decomp_episodes.append(epi_decomp)
        print(f"Episode {epi+1} Total (scalar): {total_ep_reward:.2f}")

        # Optional quick plots per episode
        if plot:
            _plot_per_step_lines(
                epi_decomp.step_components,
                os.path.join(run_dir, f"episode_{epi+1:03d}_components_per_step.png"),
                title=f"Per-step Reward Components (Episode {epi+1})",
            )
            _plot_episode_totals_bars(
                epi_decomp.totals(),
                os.path.join(run_dir, f"episode_{epi+1:03d}_components_totals.png"),
                title=f"Episode {epi+1} Component Totals",
            )

    # --- Summaries and files ---
    reward_mean = mean(total_rewards_per_episode) if total_rewards_per_episode else 0.0
    reward_stdev = stdev(total_rewards_per_episode) if len(total_rewards_per_episode) > 1 else 0.0
    print(f"Average total (scalar): {reward_mean:.2f} ± {reward_stdev:.2f}")

    # a) Per-episode totals table (JSON + CSV)
    ep_totals_rows: List[Dict[str, Any]] = []
    all_keys = set()
    for epi_idx, epi in enumerate(decomp_episodes, start=1):
        totals = epi.totals()
        row = {"episode": epi_idx, **totals}
        ep_totals_rows.append(row)
        all_keys.update(totals.keys())

    save_json(ep_totals_rows, os.path.join(run_dir, "decomp_episode_totals.json"))
    save_csv(ep_totals_rows, os.path.join(run_dir, "decomp_episode_totals.csv"))

    # b) Per-step decomposition for the first episode (so you can inspect the time series easily)
    if decomp_episodes:
        save_json(
            decomp_episodes[0].step_components,
            os.path.join(run_dir, "decomp_steps_ep1.json"),
        )

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
        os.path.join(run_dir, "summary_scalar.json"),
    )

    print(f"Saved decomposition run to: {run_dir}")

    from policy_shap_heuristic import train_surrogate_and_shap
    summary = train_surrogate_and_shap(
        policy_rows,
        out_dir=os.path.join(run_dir, "policy_shap_heuristic")
    )
    print(summary)


# -------- CLI --------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("CybORG Reward Decomposition")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-eps", type=int, default=10)
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument("--output", type=str, default=os.path.abspath("Results"))
    parser.add_argument("--submission-path", type=str, default=os.path.abspath(""))
    parser.add_argument("--no-plot", action="store_true", help="Disable PNG plots")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    Submission = load_submission(args.submission_path)
    submission = Submission

    run_explainability(
        submission=submission,
        output_dir=args.output,
        max_eps=args.max_eps,
        seed=args.seed,
        episode_length=args.episode_length,
        plot=not args.no_plot,
    )
