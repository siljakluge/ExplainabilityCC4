"""
train_curriculum.py — Sequential curriculum PPO training for CAGE Challenge 4
(GNN + MARL) across multiple red attacker strategies.

This script trains on a fixed sequence of red attacker profiles.
Each profile is trained for a fixed number of episodes before switching
to the next one.

Default curriculum:
    fsm_default -> stealth_pivot -> lateral_spread -> impact_rush
    -> deception_aware -> discovery

Default stage length:
    5000 episodes per attacker strategy

Example:
    python train_curriculum.py exp_curriculum_seq

Custom order:
    python train_curriculum.py exp_curriculum_seq \
        --curriculum_order "fsm_default,discovery,impact_rush"

Debug:
    python train_curriculum.py exp_curriculum_seq --debug
"""

from argparse import ArgumentParser
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from joblib import Parallel, delayed
import torch
from tqdm import tqdm

from CybORG import CybORG
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from models.cage4 import InductiveGraphPPOAgent
from models.memory_buffer import MultiPPOMemory
from wrappers.graph_wrapper import GraphWrapper
from wrappers.observation_graph import ObservationGraph


# ----------------------------
# Device
# ----------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("Using device:", DEVICE)


# ----------------------------
# Constants / defaults
# ----------------------------
SEED = 1337
N_AGENTS = 5
MAX_THREADS = 8
torch.manual_seed(SEED)
torch.set_num_threads(MAX_THREADS)
random.seed(SEED)


# ----------------------------
# Hyperparams container
# ----------------------------
class HP:
    # episodes per PPO update
    N = 6
    # rollout workers
    workers = 2
    # PPO minibatch size
    bs = 384
    # env episode length
    episode_len = 100
    # PPO epochs
    epochs = 2
    # experiment name
    fnames = "exp_curriculum"


# ----------------------------
# Red profiles
# ----------------------------
@dataclass(frozen=True)
class AttackProfile:
    name: str
    red_cls: type


def _safe_import_profiles():
    profiles: Dict[str, type] = {
        "fsm_default": FiniteStateRedAgent,
    }

    try:
        from CybORG.Agents.RedAgents import (
            DiscoveryFSRed,
            VerboseFSRed,
            StealthPivotFSRed,
            ImpactRushFSRed,
            DeceptionAwareFSRed,
            LateralSpreadFSRed,
        )
        profiles.update({
            "discovery": DiscoveryFSRed,
            "verbose": VerboseFSRed,
            "stealth_pivot": StealthPivotFSRed,
            "impact_rush": ImpactRushFSRed,
            "deception_aware": DeceptionAwareFSRed,
            "lateral_spread": LateralSpreadFSRed,
        })
    except Exception:
        print("[warn] Optional RedAgents variants not found. Falling back to available profiles.")

    return profiles


PROFILE_REGISTRY: Dict[str, type] = _safe_import_profiles()


# ----------------------------
# Env factory
# ----------------------------
def make_env(seed: int, episode_len: int, red_agent_class: type) -> GraphWrapper:
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=red_agent_class,
        steps=episode_len,
    )
    env = CybORG(sg, "sim", seed=seed)
    return GraphWrapper(env)


# ----------------------------
# Rollout generator
# ----------------------------
@torch.no_grad()
def generate_episode_job(
    agents: List[InductiveGraphPPOAgent],
    hp: HP,
    base_seed: int,
    job_i: int,
    profile_name: str,
    show_tqdm: bool = False,
) -> Tuple[List, float, str]:
    """
    Generate one episode with a specific red attack profile.
    Returns:
        (memories_for_all_agents, avg_reward_over_agents, profile_name)
    """
    torch.set_num_threads(max(1, MAX_THREADS // max(1, hp.workers)))

    red_cls = PROFILE_REGISTRY.get(profile_name, FiniteStateRedAgent)
    env_seed = base_seed + 100_000 + job_i
    env = make_env(env_seed, hp.episode_len, red_cls)

    env.reset()
    states = env.last_obs
    blocked_rewards = [0.0] * N_AGENTS

    tot_reward = 0.0
    memory_buffers = MultiPPOMemory(hp.bs)

    iterator = range(hp.episode_len)
    if show_tqdm:
        iterator = tqdm(iterator, desc=f"Worker {job_i}", disable=(hp.workers > 1))

    for ts in iterator:
        actions = {}
        memories = {}

        for k, (state, blocked) in states.items():
            ai = int(k[-1])
            if blocked:
                actions[k] = None
            else:
                action, value, prob = agents[ai].get_action((state, blocked))
                memories[ai] = (state, action, value, prob)
                actions[k] = action

        next_state, rewards, _, _, _ = env.step(actions)
        rewards_list = list(rewards.values())
        tot_reward += sum(rewards_list) / N_AGENTS

        for ai in range(N_AGENTS):
            if ai in memories:
                s, a, v, p = memories[ai]
                r = rewards_list[ai] + blocked_rewards[ai]
                terminal = 1 if (ts >= hp.episode_len - 1) else 0
                memory_buffers.remember(ai, s, a, v, p, r, terminal)
                blocked_rewards[ai] = 0.0
            else:
                blocked_rewards[ai] += rewards_list[ai]

        states = next_state

    return memory_buffers.mems, tot_reward, profile_name


# ----------------------------
# Training loop
# ----------------------------
def train_curriculum(
    agents: List[InductiveGraphPPOAgent],
    hp: HP,
    seed: int,
    curriculum_order: List[str],
    stage_episodes: int,
    log_per_profile: bool,
) -> None:
    print("Hyperparams:", vars(hp))
    print("Available profiles:", sorted(PROFILE_REGISTRY.keys()))
    print("Curriculum order:", curriculum_order)
    print("Stage episodes:", stage_episodes)

    for p in curriculum_order:
        if p not in PROFILE_REGISTRY:
            raise ValueError(
                f"Profile '{p}' not found. Available: {sorted(PROFILE_REGISTRY.keys())}"
            )

    [agent.train() for agent in agents]
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    def learn(i: int):
        base = max(1, MAX_THREADS // 9)
        if i < 4:
            torch.set_num_threads(base)
        else:
            torch.set_num_threads(base * N_AGENTS)
        return agents[i].learn()

    global_log = []
    per_profile_log = []

    global_episode_counter = 0
    global_update_counter = 0

    for stage_idx, profile_name in enumerate(curriculum_order):
        print("\n" + "=" * 80)
        print(f"Stage {stage_idx + 1}/{len(curriculum_order)}: {profile_name}")
        print("=" * 80)

        stage_updates = math.ceil(stage_episodes / hp.N)

        for stage_update_idx in range(stage_updates):
            remaining_episodes = stage_episodes - stage_update_idx * hp.N
            batch_size = min(hp.N, remaining_episodes)

            out = Parallel(prefer="processes", n_jobs=hp.workers)(
                delayed(generate_episode_job)(
                    agents=agents,
                    hp=hp,
                    base_seed=seed + global_update_counter * 1_000_000,
                    job_i=i,
                    profile_name=profile_name,
                    show_tqdm=False,
                )
                for i in range(batch_size)
            )

            memories, avg_rewards, used_profiles = zip(*out)

            per_agent_mems = [list(m) for m in zip(*memories)]
            for i in range(N_AGENTS):
                agents[i].memory.mems = per_agent_mems[i]
                agents[i].memory.agents = len(per_agent_mems[i])

            last_losses = Parallel(prefer="threads", n_jobs=N_AGENTS)(
                delayed(learn)(i) for i in range(N_AGENTS)
            )

            avg_reward = sum(avg_rewards) / batch_size
            losses_str = ",".join([f"{last_losses[i]:0.4f}" for i in range(N_AGENTS)])

            stage_episode_counter = min((stage_update_idx + 1) * hp.N, stage_episodes)
            global_episode_counter += batch_size

            print(
                f"[global_ep={global_episode_counter:05d} | "
                f"stage_ep={stage_episode_counter:05d}/{stage_episodes} | "
                f"profile={profile_name}] "
                f"Loss: [{losses_str}] AvgReward: {avg_reward:0.4f}"
            )

            global_log.append({
                "global_update": global_update_counter,
                "global_episode": global_episode_counter,
                "stage_idx": stage_idx,
                "stage_profile": profile_name,
                "stage_episode": stage_episode_counter,
                "avg_reward": avg_reward,
                "avg_loss": sum(last_losses) / N_AGENTS,
                "losses": list(last_losses),
            })
            torch.save(global_log, f"logs/{hp.fnames}.pt")

            if log_per_profile:
                per_profile_log.append({
                    "global_update": global_update_counter,
                    "global_episode": global_episode_counter,
                    "profile": profile_name,
                    "avg_reward": avg_reward,
                })
                torch.save(per_profile_log, f"logs/{hp.fnames}_per_profile.pt")

            for i in range(N_AGENTS):
                agents[i].save(outf=f"checkpoints/{hp.fnames}-{i}_checkpoint.pt")

                if global_episode_counter % 10_000 < hp.N and global_episode_counter > hp.N:
                    agents[i].save(
                        outf=f"checkpoints/{hp.fnames}-{i}_{global_episode_counter // 1000}k.pt"
                    )

            global_update_counter += 1

        print(f"[done] Finished stage '{profile_name}' with {stage_episodes} episodes.")

    print("\nTraining complete.")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("fname", help="Base name for logs/checkpoints.")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--embedding", type=int, default=128)
    ap.add_argument("--debug", action="store_true", help="Small safe config")
    ap.add_argument("--KIServer", action="store_true", help="Server Config")
    ap.add_argument(
        "--stage_episodes",
        type=int,
        default=5000,
        help="Number of training episodes per attacker strategy.",
    )
    ap.add_argument(
        "--curriculum_order",
        type=str,
        default="fsm_default,stealth_pivot,lateral_spread,impact_rush,deception_aware,discovery",
        help='Comma-separated profile order, e.g. "fsm_default,discovery,impact_rush"',
    )
    ap.add_argument(
        "--log_per_profile",
        action="store_true",
        help="Write logs/<fname>_per_profile.pt",
    )
    ap.add_argument("--phase_reward_mode", default="default",
                    choices=["default", "contractor_off", "red_only"])
    ap.add_argument("--reward_blue", action="store_true")

    args = ap.parse_args()
    print(args)
    os.environ["CYBORG_PHASE_REWARD_MODE"] = args.phase_reward_mode
    os.environ["CYBORG_REWARD_blue"] = "1" if args.reward_blue else "0"

    if args.debug:
        HP.N = 6
        HP.workers = 2
        HP.bs = 384
        HP.episode_len = 100
        HP.epochs = 2
        args.stage_episodes = 10
    if args.KIServer:
        HP.N = 10
        HP.workers = 12
        HP.bs = 1500
        HP.episode_len = 500
        HP.epochs = 4
        args.stage_episodes = 5000

    curriculum_order = [p.strip() for p in args.curriculum_order.split(",") if p.strip()]
    if not curriculum_order:
        raise ValueError("curriculum_order must contain at least one profile.")

    agents = [
        InductiveGraphPPOAgent(
            ObservationGraph.DIM + 5,
            bs=HP.bs,
            a_kwargs={"lr": 0.0003, "hidden1": args.hidden, "hidden2": args.embedding},
            c_kwargs={"lr": 0.001, "hidden1": args.hidden, "hidden2": args.embedding},
            clip=0.2,
            epochs=HP.epochs,
            device=DEVICE,
        )
        for _ in range(N_AGENTS)
    ]

    HP.fnames = args.fname

    train_curriculum(
        agents=agents,
        hp=HP,
        seed=SEED,
        curriculum_order=curriculum_order,
        stage_episodes=args.stage_episodes,
        log_per_profile=args.log_per_profile,
    )