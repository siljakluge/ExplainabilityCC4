from __future__ import annotations

from collections import defaultdict
import numpy as np
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Any, Optional

from tqdm import tqdm

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from plot_rew_decomp import (parse_log,
                             plot_episode_detail,
                             plot_components_by_subnet,
                             plot_total,
                             plot_phase_stacked,
                             plot_components_per_step,
                             plot_action_cost_by_agent,
                             plot_components_by_phase,
                             plot_components_by_subnet_per_phase)
from policy_shap_heuristic import train_surrogate_and_shap


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

def extract_blue_features(obs_all_agents, agent_name: str) -> dict:
    """
    obs_all_agents: the 'observations' dict from the env
    agent_name: e.g. 'blue_agent_0'

    returns: flat {feature_name: float}
    """
    feats = {}
    if agent_name not in obs_all_agents:
        return feats

    obs = obs_all_agents[agent_name]

    # 1) success as numeric
    if "success" in obs:
        s = obs["success"]
        # TernaryEnum.TRUE.value == 1 etc.
        value = getattr(s, "value", s)
        try:
            feats[f"{agent_name}_success"] = float(value)
        except Exception:
            pass

    # 2) message: list of 4 numpy arrays (8 bools each)
    msg_list = obs.get("message", [])
    for i, arr in enumerate(msg_list):
        arr_flat = np.asarray(arr).astype(int).ravel()
        for j, v in enumerate(arr_flat):
            feats[f"{agent_name}_msg_{i}_{j}"] = float(v)

    # 3) subnet router info
    #    e.g. 'public_access_zone_subnet_router': {'Processes': [...]}
    for key, val in obs.items():
        if not (isinstance(val, dict) and key.endswith("_subnet_router")):
            continue

        procs = val.get("Processes", []) or []
        n_procs = len(procs)
        n_rfi = 0
        for p in procs:
            props = p.get("Properties", []) or []
            if "rfi" in props:
                n_rfi += 1

        base = f"{agent_name}_{key}"
        feats[f"{base}_n_procs"] = float(n_procs)
        feats[f"{base}_n_rfi"] = float(n_rfi)

    # IMPORTANT: we *do not* use obs["action"] as a feature
    # to avoid leaking the label into the features.

    return feats

def encode_action(action) -> str:
    """
    Turn the env action into a label string.

    Currently:
    - If it has a class, use the class name (Monitor, DeployDecoy, ...)
    - Fallback to str(action)
    """
    if hasattr(action, "__class__"):
        return action.__class__.__name__
    return str(action)

def aggregate_semantic_features(feat_dict: dict) -> dict:
    """
    Take one row's feature dict like
      {'blue_agent_0_success': 1.0, 'blue_agent_0_msg_0_0': 0.0, ...}
    and return a new dict with agent prefix stripped and
    semantically aggregated features.

    Resulting keys are things like:
      - 'success', 'success_is_true', 'success_is_false', 'success_is_in_progress'
      - 'msg_total', 'msg_any', 'msg_0_sum', ..., 'msg_3_sum'
      - 'restricted_zone_a_n_procs', 'restricted_zone_a_n_rfi', ...
      - 'num_routers_seen', 'num_rfi_routers', 'any_rfi'
    """
    agg = {}
    msg_total = 0.0
    msg_block_sum = defaultdict(float)

    # infer agent prefix from the success key (e.g. 'blue_agent_0')
    agent_prefix = None
    for k in feat_dict.keys():
        if "_success" in k:
            agent_prefix = k.split("_success")[0]
            break
    prefix_len = len(agent_prefix) + 1 if agent_prefix is not None else 0

    # first pass: strip prefix and collect raw semantics
    raw_zone_n_procs = {}
    raw_zone_n_rfi = {}

    for k, v in feat_dict.items():
        # remove "blue_agent_X_" to get a local name
        local = k[prefix_len:] if prefix_len > 0 else k

        # success
        if local == "success":
            agg["success"] = v

        # message bits: msg_i_j -> sums per i and global
        elif local.startswith("msg_"):
            # local: "msg_2_7" -> channel 2, index 7
            parts = local.split("_")  # ['msg', '2', '7']
            if len(parts) == 3:
                ch = parts[1]
                msg_block_sum[f"msg_{ch}_sum"] += float(v)
                msg_total += float(v)

        # router stats
        elif local.endswith("_subnet_router_n_procs"):
            # e.g. "restricted_zone_a_subnet_router_n_procs"
            zone = local.replace("_subnet_router_n_procs", "")
            raw_zone_n_procs[zone] = float(v)

        elif local.endswith("_subnet_router_n_rfi"):
            zone = local.replace("_subnet_router_n_rfi", "")
            raw_zone_n_rfi[zone] = float(v)

        # anything else is ignored for now

    # aggregate message bits
    agg["msg_total"] = msg_total
    agg["msg_any"] = 1.0 if msg_total > 0.0 else 0.0
    # if you like: density over 32 bits (4x8)
    agg["msg_density"] = msg_total / 32.0

    for block_key, val in msg_block_sum.items():
        agg[block_key] = val  # msg_0_sum, msg_1_sum, ...

    # zone-level router features
    for zone, n_procs in raw_zone_n_procs.items():
        agg[f"{zone}_n_procs"] = n_procs
    for zone, n_rfi in raw_zone_n_rfi.items():
        agg[f"{zone}_n_rfi"] = n_rfi

    # derived router summary features
    routers_seen = [z for z, n in raw_zone_n_procs.items() if n > 0]
    agg["num_routers_seen"] = float(len(routers_seen))

    rfi_routers = [z for z, n in raw_zone_n_rfi.items() if n > 0]
    agg["num_rfi_routers"] = float(len(rfi_routers))
    agg["any_rfi"] = 1.0 if len(rfi_routers) > 0 else 0.0

    # one-hot encode ternary success in addition to numeric
    s = agg.get("success", 0.0)
    agg["success_is_true"] = 1.0 if s == 1.0 else 0.0
    agg["success_is_false"] = 1.0 if s == 2.0 else 0.0
    agg["success_is_in_progress"] = 1.0 if s == 4.0 else 0.0

    return agg

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

    all_observations = []
    all_actions = []
    total_rewards_per_episode = []

    policy_rows = []

    for epi in tqdm(range(max_eps), desc="Episodes"):
        observations, _ = wrapped_cyborg.reset()
        reward_per_episode = 0

        for t in range(episode_length):
            # decide actions based on current obs
            actions = {
                agent_name: agent.get_action(
                    observations[agent_name],
                    wrapped_cyborg.action_space(agent_name),
                )
                for agent_name, agent in submission.AGENTS.items()
                if agent_name in wrapped_cyborg.agents
            }
            all_actions.append(actions)

            if not is_heuristic:
                # collect obs->action pairs for each blue agent
                for agent_name in submission.AGENTS.keys():
                    if agent_name not in wrapped_cyborg.agents:
                        continue
                    if not agent_name.startswith("blue_agent"):
                        continue

                    feat_dict = extract_blue_features(observations, agent_name)
                    label = encode_action(actions[agent_name])
                    policy_rows.append((feat_dict, label))
            else:
                for agent_name, agent in submission.AGENTS.items():
                    if not hasattr(agent, "info") or len(agent.info) == 0:
                        continue
                    step_info = agent.info[-1]
                    if "Predicates" in step_info and "ActionClass" in step_info:
                        policy_rows.append(
                            (step_info["Predicates"], step_info["ActionClass"])
                        )

            # step the env
            observations, rewards_scalar, term, trunc, info = wrapped_cyborg.step(actions)
            all_observations.append(observations)
            reward_per_episode += sum(rewards_scalar.values())

            done = {
                agent: term.get(agent, False) or trunc.get(agent, False)
                for agent in wrapped_cyborg.agents
            }
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

    # SHAP
    if shap and policy_rows and is_heuristic:
        summary = train_surrogate_and_shap(
            policy_rows,
            out_dir=os.path.join(output_dir, "SHAPAnalysis"),
        )
        print(summary)
    elif shap and policy_rows:
        agg_rows = [
            (aggregate_semantic_features(feats), label)
            for feats, label in policy_rows
        ]

        summary = train_surrogate_and_shap(
            agg_rows,
            out_dir=os.path.join(output_dir, "SHAPAnalysis"),
        )
        print(summary)


# -------- CLI --------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("CybORG Reward Decomposition")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-eps", type=int, default=10)
    parser.add_argument("--episode-length", type=int, default=50)
    parser.add_argument("--shap", type=bool, default=True)
    parser.add_argument("--is-heuristic", type=bool, default=True)
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
        is_heuristic=args.is_heuristic,
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
