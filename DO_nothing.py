from pprint import pprint

from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents import SleepAgent, FiniteStateRedAgent, EnterpriseGreenAgent
from CybORG.Agents.Wrappers.EnterpriseMAE import EnterpriseMAE, BlueEnterpriseWrapper
from CybORG.Agents.Wrappers import BlueFixedActionWrapper
from CybORG.Simulator.Actions import Monitor
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import csv
import os

def append_to_csv(filename, data_list):
    """
    Appends a list of information as a new row to a CSV file.
    If the file does not exist, it will be created.

    Args:
        filename (str): Path to the CSV file.
        data_list (list): List of values to append as a row.
    """
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data_list)


EPISODE_LENGTH = 500
NUMBER_EPISODES = 100
SEED = 10


def plot_array(data: np.ndarray, title: str = "Mein Graph"):
    plt.plot(range(len(total_rewards)), total_rewards, marker='o')
    plt.title("Total Rewards pro Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.show()
    plt.savefig("Total_rewards.png")
   
     
sg = EnterpriseScenarioGenerator(blue_agent_class=SleepAgent, 
                                green_agent_class=EnterpriseGreenAgent, 
                                red_agent_class=FiniteStateRedAgent,
                                steps=EPISODE_LENGTH)

total_rewards = np.zeros(NUMBER_EPISODES)


for episode in tqdm(range(NUMBER_EPISODES)):
    cyborg = CybORG(scenario_generator=sg, seed=episode)
    env = BlueFixedActionWrapper(cyborg)
    env.reset()
    blue_agents = [agent_name for agent_name in cyborg.environment_controller.get_active_agents() if 'blue' in agent_name]
    blue_actions = np.array([])


    for step in range(EPISODE_LENGTH):
        actions = {agent : Monitor(0, agent) for agent in blue_agents}
        observations, reward, terminate, truncuate, info = env.step(actions, messages={agent : np.array([False]*8) for agent in blue_agents})
        done = {
                agent: terminate.get(agent, False) or truncuate.get(agent, False)
                for agent in blue_agents
            }
        if all(done.values()):
                break
        total_rewards[episode] += sum(list(reward.values()))/len(blue_agents)
    #print(f"Total Reward in Episode {episode+1}: {total_rewards}")


plot_array(total_rewards, title="Total Rewards per Episode")
print("DONE")
