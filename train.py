"""
train.py — Robust PPO training for CAGE Challenge 4 (GNN + MARL) with Red Attack Profiles

This version extends our current training script with:
  1) Red attacker profiles (FiniteStateRedAgent + custom FSM variants)
  2) Episode-level profile sampling (build env per episode to change red_agent_class)
  3) Training strategies:
        - single:     train against a single red agent class
        - mixture:    train against a fixed mixture of red profiles
        - curriculum: staged mixture schedule (warm-up -> robust mixture -> (optional) harden)
  4) Optional per-profile reward logging

Notes / Design choices:
  - Red agent class is selected at EnterpriseScenarioGenerator construction time.
    Therefore: per-episode profiles require creating a fresh env for each episode.
  - This script keeps your multi-process rollout generation (joblib).
  - PPO updates are parallelized across blue agents (threads).

Usage examples:
  Debug quick run:
    python train.py m1_contractoractive --debug --strategy mixture

  Single attacker:
    python train.py exp_single_fsm --strategy single --single_profile fsm_default

  Robust mixture:
    python train.py exp_mix --strategy mixture --profile_weights "fsm_default=0.35,stealth_pivot=0.25,lateral_spread=0.20,impact_rush=0.15,deception_aware=0.05"

  Curriculum (warm-up then mixture):
    python train.py exp_curr --strategy curriculum --warmup_updates 200 --harden_updates 200

"""

from argparse import ArgumentParser
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from joblib import Parallel, delayed
import torch
from sympy.codegen import Print
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
MAX_THREADS = 8  # adjust for real GPU box, e.g., 36
torch.manual_seed(SEED)
torch.set_num_threads(MAX_THREADS)


# ----------------------------
# Hyperparams container (defaults)
# ----------------------------
class HP:
    # episodes per PPO update
    N = 6
    # joblib processes for rollouts
    workers = 2
    # PPO minibatch size
    bs = 384
    # env episode length
    episode_len = 100
    # total env episodes
    training_episodes = 906
    # PPO epochs per update
    epochs = 2
    # experiment name for logs/ckpts
    fnames = "exp"


# ----------------------------
# Red profiles
# ----------------------------
@dataclass(frozen=True)
class AttackProfile:
    name: str
    red_cls: type

def _safe_import_profiles():
    """
    Import optional FSM variant classes if present.
    Keep this robust so train.py works even if some variants aren't in the repo.
    """
    profiles: Dict[str, type] = {
        "fsm_default": FiniteStateRedAgent,
    }

    try:
        # If you placed new profiles into RedAgents.py
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
        # It's fine if variants are not available; you can still train baseline.
        print("not found")
        pass

    return profiles


PROFILE_REGISTRY: Dict[str, type] = _safe_import_profiles()


def parse_profile_weights(s: str) -> Dict[str, float]:
    """
    Parse weights from: "fsm_default=0.35,stealth_pivot=0.25,..."
    """
    weights: Dict[str, float] = {}
    if not s.strip():
        return weights
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Bad profile_weights token '{p}', expected name=weight.")
        name, w = p.split("=", 1)
        name = name.strip()
        w = float(w.strip())
        weights[name] = w
    # normalize
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("profile_weights sum must be > 0")
    for k in list(weights.keys()):
        weights[k] = weights[k] / total
    return weights


def sample_profile(rng: random.Random, weights: Dict[str, float], fallback: str = "fsm_default") -> str:
    """
    Weighted sample over profile names. Only samples among profiles that exist in PROFILE_REGISTRY.
    """
    candidates = [(name, w) for name, w in weights.items() if name in PROFILE_REGISTRY and w > 0]
    if not candidates:
        return fallback if fallback in PROFILE_REGISTRY else "fsm_default"
    names = [c[0] for c in candidates]
    ws = [c[1] for c in candidates]
    return rng.choices(names, weights=ws, k=1)[0]


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
    Returns: (memories_for_all_agents, avg_reward_over_agents, profile_name)
    """
    torch.set_num_threads(max(1, MAX_THREADS // max(1, hp.workers)))

    # Build fresh env per episode to allow red_agent_class switching
    red_cls = PROFILE_REGISTRY.get(profile_name, FiniteStateRedAgent)
    env_seed = base_seed + 100_000 + job_i  # deterministic per job
    env = make_env(env_seed, hp.episode_len, red_cls)

    env.reset()
    states = env.last_obs  # dict: {agent_id_str: (state, blocked)}
    blocked_rewards = [0.0] * N_AGENTS

    tot_reward = 0.0
    memory_buffers = MultiPPOMemory(hp.bs)

    iterator = range(hp.episode_len)
    if show_tqdm:
        iterator = tqdm(iterator, desc=f"Worker {job_i}", disable=(hp.workers > 1))

    for ts in iterator:
        actions = {}
        memories = {}

        # Pick actions for all unblocked agents
        for k, (state, blocked) in states.items():
            ai = int(k[-1])  # blue_agent_0..4
            if blocked:
                actions[k] = None
            else:
                action, value, prob = agents[ai].get_action((state, blocked))
                memories[ai] = (state, action, value, prob)
                actions[k] = action

        next_state, rewards, _, _, _ = env.step(actions)
        rewards_list = list(rewards.values())
        tot_reward += sum(rewards_list) / N_AGENTS

        # Accumulate rewards for durative actions
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
# Strategy schedules
# ----------------------------
def get_weights_for_update(
    strategy: str,
    update_idx: int,
    warmup_updates: int,
    harden_updates: int,
    fixed_weights: Dict[str, float],
) -> Dict[str, float]:
    """
    Returns profile weights for this PPO update (not per episode).
    - update_idx counts PPO updates (each update uses hp.N episodes).
    """
    # Baseline sane defaults if user didn't provide
    default_mixture = {
        "fsm_default": 0.35,
        "stealth_pivot": 0.25,
        "lateral_spread": 0.20,
        "impact_rush": 0.15,
        "deception_aware": 0.05,
    }
    # Keep only available profiles
    def _filter_available(w: Dict[str, float]) -> Dict[str, float]:
        ww = {k: v for k, v in w.items() if (k in PROFILE_REGISTRY and v > 0)}
        if not ww:
            return {"fsm_default": 1.0}
        s = sum(ww.values())
        return {k: v / s for k, v in ww.items()}

    if strategy == "single":
        # fixed_weights should be one-hot; but we just return it normalized
        return _filter_available(fixed_weights) if fixed_weights else {"fsm_default": 1.0}

    if strategy == "mixture":
        return _filter_available(fixed_weights) if fixed_weights else _filter_available(default_mixture)

    if strategy == "curriculum":
        # Warm-up: mostly baseline FSM to stabilize PPO
        if update_idx < warmup_updates:
            warm = {"fsm_default": 0.8, "lateral_spread": 0.2}
            return _filter_available(warm)

        # Main: robust mixture
        main = _filter_available(fixed_weights) if fixed_weights else _filter_available(default_mixture)

        # Optional hardening tail: tilt toward the "hard" profiles
        if harden_updates > 0 and update_idx >= warmup_updates and update_idx < warmup_updates + harden_updates:
            hard = {
                "fsm_default": 0.15,
                "stealth_pivot": 0.30,
                "lateral_spread": 0.25,
                "impact_rush": 0.25,
                "deception_aware": 0.05,
            }
            return _filter_available(hard)

        return main

    # fallback
    return {"fsm_default": 1.0}


# ----------------------------
# Training loop
# ----------------------------
def train(
    agents: List[InductiveGraphPPOAgent],
    hp: HP,
    seed: int,
    strategy: str,
    single_profile: str,
    profile_weights: Dict[str, float],
    warmup_updates: int,
    harden_updates: int,
    log_per_profile: bool,
) -> None:
    print("Hyperparams:", vars(hp))
    print("Strategy:", strategy)
    print("Available profiles:", sorted(PROFILE_REGISTRY.keys()))
    if strategy == "single":
        print("Single profile:", single_profile)
    if strategy in ("mixture", "curriculum"):
        print("Base mixture weights:", profile_weights or "<defaults>")

    [agent.train() for agent in agents]
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Threaded backprop helper (keep your load-balance idea)
    def learn(i: int):
        base = max(1, MAX_THREADS // 9)
        if i < 4:
            torch.set_num_threads(base)
        else:
            torch.set_num_threads(base * N_AGENTS)
        return agents[i].learn()

    total_updates = hp.training_episodes // hp.N

    log = []
    per_profile_log = []  # (update_idx, profile_name, avg_reward)

    for update_idx in range(total_updates):
        # Determine weights for this update
        if strategy == "single":
            weights = {single_profile: 1.0}
        else:
            weights = get_weights_for_update(
                strategy=strategy,
                update_idx=update_idx,
                warmup_updates=warmup_updates,
                harden_updates=harden_updates,
                fixed_weights=profile_weights,
            )

        # Sample per-episode profiles for this batch (hp.N episodes)
        rng = random.Random(seed + update_idx)
        episode_profiles = [sample_profile(rng, weights) for _ in range(hp.N)]

        # Rollouts in parallel (processes)
        out = Parallel(prefer="processes", n_jobs=hp.workers)(
            delayed(generate_episode_job)(
                agents=agents,
                hp=hp,
                base_seed=seed + update_idx * 1_000_000,
                job_i=i,
                profile_name=episode_profiles[i],
                show_tqdm=False,
            )
            for i in range(hp.N)
        )

        memories, avg_rewards, used_profiles = zip(*out)

        # Transpose per-agent memories across episodes
        per_agent_mems = [list(m) for m in zip(*memories)]
        for i in range(N_AGENTS):
            agents[i].memory.mems = per_agent_mems[i]
            agents[i].memory.agents = len(per_agent_mems[i])

        # PPO update (threads)
        last_losses = Parallel(prefer="threads", n_jobs=N_AGENTS)(
            delayed(learn)(i) for i in range(N_AGENTS)
        )

        # Aggregate logging
        e = update_idx * hp.N
        losses_str = ",".join([f"{last_losses[i]:0.4f}" for i in range(N_AGENTS)])
        avg_reward = sum(avg_rewards) / hp.N

        print(f"[{e}] Loss: [{losses_str}]  AvgReward: {avg_reward:0.4f}")
        if strategy != "single":
            # show sampled profile histogram for this update
            hist = {}
            for p in used_profiles:
                hist[p] = hist.get(p, 0) + 1
            hist_str = " ".join([f"{k}:{v}" for k, v in sorted(hist.items())])
            print(f"      Profiles: {hist_str}")

        log.append((avg_reward, e, sum(last_losses) / N_AGENTS))
        torch.save(log, f"logs/{hp.fnames}.pt")

        if log_per_profile:
            # compute per-profile mean reward in this batch
            tmp = {}
            for r, p in zip(avg_rewards, used_profiles):
                tmp.setdefault(p, []).append(r)
            for p, rs in tmp.items():
                per_profile_log.append((update_idx, p, sum(rs) / len(rs)))
            torch.save(per_profile_log, f"logs/{hp.fnames}_per_profile.pt")

        # Checkpoints
        for i in range(N_AGENTS):
            agent = agents[i]
            agent.save(outf=f"checkpoints/{hp.fnames}-{i}_checkpoint.pt")

            # keep your periodic checkpoint logic (every ~10k episodes)
            if e % 10_000 < hp.N and e > hp.N:
                agent.save(outf=f"checkpoints/{hp.fnames}-{i}_{e//1000}k.pt")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("fname", help="Required: base name for logs/checkpoints.")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--embedding", type=int, default=128)
    ap.add_argument("--debug", action="store_true", help="Small safe config")
    ap.add_argument("--KIServer", action="store_true", help="Server Config")
    ap.add_argument("--phase_reward_mode", default="default",
                    choices=["default", "contractor_off", "red_only"])
    ap.add_argument("--reward_blue", action="store_true")

    ap.add_argument(
        "--strategy",
        choices=["single", "mixture", "curriculum"],
        default="mixture",
        help="How to select red attack profiles during training.",
    )
    ap.add_argument(
        "--single_profile",
        default="fsm_default",
        help="Profile name for --strategy single (must exist in registry).",
    )
    ap.add_argument(
        "--profile_weights",
        default="",
        help='Comma list: "fsm_default=0.35,stealth_pivot=0.25,..." used for mixture/curriculum main phase.',
    )
    ap.add_argument(
        "--warmup_updates",
        type=int,
        default=200,
        help="Curriculum only: number of PPO updates for warm-up (mostly baseline FSM).",
    )
    ap.add_argument(
        "--harden_updates",
        type=int,
        default=0,
        help="Curriculum only: optional additional PPO updates with harder tilted weights before main mixture.",
    )
    ap.add_argument(
        "--log_per_profile",
        action="store_true",
        help="If set, write logs/<fname>_per_profile.pt with mean reward per profile per update.",
    )

    args = ap.parse_args()
    print(args)
    os.environ["CYBORG_PHASE_REWARD_MODE"] = args.phase_reward_mode
    os.environ["CYBORG_REWARD_blue"] = "1" if args.reward_blue else "0"

    # Apply debug overrides (matching your current ones)
    if args.debug:
        HP.N = 6
        HP.workers = 2
        HP.bs = 384
        HP.episode_len = 100
        HP.training_episodes = 1000
        HP.epochs = 2

    if args.KIServer:
        HP.N = 10
        HP.workers = 12
        HP.bs = 1500
        HP.episode_len = 500
        HP.training_episodes = 10000
        HP.epochs = 4

    # Create agents
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

    # Parse weights + validate profiles
    weights = parse_profile_weights(args.profile_weights) if args.profile_weights else {}

    if args.strategy == "single":
        if args.single_profile not in PROFILE_REGISTRY:
            raise ValueError(
                f"single_profile='{args.single_profile}' not found. Available: {sorted(PROFILE_REGISTRY.keys())}"
            )

    # Train
    train(
        agents=agents,
        hp=HP,
        seed=SEED,
        strategy=args.strategy,
        single_profile=args.single_profile,
        profile_weights=weights,
        warmup_updates=args.warmup_updates,
        harden_updates=args.harden_updates,
        log_per_profile=args.log_per_profile,
    )

"""
python train.py red_rew_only \
  --KIServer \
  --phase_reward_mode red_only \
  --strategy single \
  --single_profile fsm_default
"""