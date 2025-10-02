"""
submission.py — TTCP CAGE Challenge 4 Agent Submission Interface

Defines the `Submission` class containing all necessary metadata and agent-loading functionality
for challenge evaluation. It uses GNN-based PPO agents and loads trained weights for each Blue agent.

Main Contents:
- NAME, TEAM, TECHNIQUE: Identifiers for the submitted solution
- AGENTS: Dictionary mapping agent names to trained agent instances (loaded from file)
- wrap(env): Optional wrapper to apply GraphWrapper to CybORG
"""

import os

from CybORG import CybORG
from CybORG.Agents import BaseAgent
from CybORG.Agents.Wrappers import BlueFixedActionWrapper
from heuristic_wrapper import HeuristicWrapper

from ray.rllib.env.multi_agent_env import MultiAgentEnv



### Import custom agents here ###
from Heuristic_Agent import H_Agent

class Submission:

    # Submission name
    NAME: str = "JustDoIt"

    # Name of your team
    TEAM: str = "SIX"

    # What is the name of the technique used? (e.g. Masked PPO)
    TECHNIQUE: str = "Heuristic Agent"

    # Use this function to define your agents.
    AGENTS = {
        f"blue_agent_{i}": H_Agent(agent_name=f"blue_agent_{i}")
        for i in range(5)
    }

    # Use this function to optionally wrap CybORG with your custom wrapper(s).
    def wrap(env: CybORG) -> MultiAgentEnv:
        return HeuristicWrapper(env)
        #return BlueFixedActionWrapper(env)
