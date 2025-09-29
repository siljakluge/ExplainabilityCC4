from pprint import pprint
from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents import SleepAgent, FiniteStateRedAgent, EnterpriseGreenAgent
from CybORG.Agents.Wrappers import BlueFixedActionWrapper
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from Heuristic_Agent_v0_2 import H_Agent
from datetime import datetime
import pandas as pd
import os

EPISODE_LENGTH = 500
NUMBER_EPISODES = 100
SEED = 10
VERSION = 0.2

#Save information dicts
def save_dict_array_to_csv(arr, filename, folder = None):
    # Convert numpy array of dicts to list of dicts
    data = [dict(item) for item in arr]
    if folder == None:
         folder = filename[:-4]
    folder_path= "Results/"+folder+"/"
    os.makedirs(folder_path, exist_ok=True)
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(path_or_buf=(folder_path+filename), index=False)
    print(f"Saved {len(arr)} dictionaries to {filename}")


#Plot reward per episode
def plot_array(data: np.ndarray, title: str = "Mein Graph", folder:str = None):
    plt.plot(range(len(data)), data, marker='o')
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.show()
    if folder != None:
        folder_path= "Results/"+folder+"/"
    else:
         folder_path=""
    os.makedirs(folder_path, exist_ok=True)
    plt.savefig(folder_path+title+".png")
    print(f"Saved Rewards graph")

# Init  
# Select Agents   
sg = EnterpriseScenarioGenerator(blue_agent_class=SleepAgent, 
                                green_agent_class=EnterpriseGreenAgent, 
                                red_agent_class=FiniteStateRedAgent,
                                steps=EPISODE_LENGTH)

total_rewards = np.zeros(NUMBER_EPISODES)
info_short, log = np.array([]), np.array([])
starttime=datetime.now()

# Loop through episodes
for episode in tqdm(range(NUMBER_EPISODES)):
    seed = episode
    # Create Environment
    cyborg = CybORG(scenario_generator=sg, seed=seed)
    env = BlueFixedActionWrapper(cyborg)
    env.reset()

    # Create Agents
    blue_agents = [H_Agent(agent_name = agent_name, init_obs=env.get_observation(agent_name)) for agent_name in cyborg.environment_controller.get_active_agents() if 'blue' in agent_name]
    
    # Get Init Observations
    blue_observations = np.array([[env.get_observation(blue_agent.agent_name) for blue_agent in blue_agents]])

    # CAGE Challenge 4 Episode
    for step in range(EPISODE_LENGTH):
        # get actions from agents based on last observation
        actions = {agent.agent_name : agent.get_action(blue_observations[-1][i]) for i, agent in enumerate(blue_agents)}
        
        # do a step in environment
        observations, reward, terminate, truncuate, info = env.step(actions, messages={agent.agent_name: agent.get_message() for agent in blue_agents})
        done = {
                agent.agent_name: terminate.get(agent.agent_name, False) or truncuate.get(agent.agent_name, False)
                for agent in blue_agents
            }
        if all(done.values()):
                break
        
        # get reward and collect info
        total_rewards[episode] += sum(list(reward.values()))/len(blue_agents)
        log = np.append(log, {"Episode": episode,
                              "Step": step,
                              "Observation": observations,
                              "Actions": actions,
                              "Reward": sum(list(reward.values()))/len(blue_agents)
                              }
                        )
        blue_observations = np.vstack([blue_observations, np.array([observations[agent.agent_name] for agent in blue_agents])])


    print(f" Total Reward in Episode {episode+1}: {total_rewards[episode]}")
    info_short = np.append(info_short, 
                     {"Episode" : episode,
                      "Seed": seed,
                      "Reward": total_rewards[episode],
                      "Monitor Action Count": [agent.action_counter[0] for agent in blue_agents],
                      "Analyse Action Count":  [agent.action_counter[1] for agent in blue_agents],
                      "DeployDecoy Action Count": [agent.action_counter[2] for agent in blue_agents],
                      "Restore Action Count": [agent.action_counter[4] for agent in blue_agents],
                      "Remove Action Count": [agent.action_counter[3] for agent in blue_agents],
                      blue_agents[0].agent_name + "host and servers amount": len(blue_agents[0].hosts),
                      blue_agents[1].agent_name + "host and servers amount": len(blue_agents[1].hosts),
                      blue_agents[2].agent_name + "host and servers amount": len(blue_agents[2].hosts),
                      blue_agents[3].agent_name + "host and servers amount": len(blue_agents[3].hosts)
                      }
                      )
generall_info = np.array([{"Mean Reward": np.sum(total_rewards)/total_rewards.size,
                            "All Rewards": total_rewards,
                            "Agent-Version":"Heuristic_"+str(VERSION),
                            }
                            ])


# save info and graph
savetime =str(starttime.year)+str(starttime.month)+str(starttime.day)+"_"+str(starttime.hour)+str(starttime.minute)+".csv"
folder_path = generall_info[0]["Agent-Version"]+"_"+savetime
plot_array(data=total_rewards, title="Total Rewards per Episode"+savetime, folder=folder_path)
save_dict_array_to_csv(filename=("General_info_"+savetime), arr=generall_info, folder=folder_path)
save_dict_array_to_csv(filename=("Short_Info_"+savetime), arr=info_short, folder=folder_path)
save_dict_array_to_csv(filename=("Log_"+savetime), arr=log, folder=folder_path)

print("DONE")