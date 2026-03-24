from pprint import pprint
from CybORG import CybORG, CYBORG_VERSION
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents import SleepAgent, FiniteStateRedAgent, EnterpriseGreenAgent
from CybORG.Agents.Wrappers import BlueFixedActionWrapper
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from Heuristic_Agent_v0_3 import H_Agent
from datetime import datetime, time
import pandas as pd
import os
import sys
import json
from statistics import mean, stdev

EPISODE_LENGTH = 500
NUMBER_EPISODES = 100
#SEED = 10
VERSION = 0.2
cyborg_version = CYBORG_VERSION

def rmkdir(path: str):
    """Recursive mkdir"""
    partial_path = ""
    for p in path.split("/"):
        partial_path += p + "/"

        if os.path.exists(partial_path):
            if os.path.isdir(partial_path):
                continue
            if os.path.isfile(partial_path):
                raise RuntimeError(f"Cannot create {partial_path} (exists as file).")

        os.mkdir(partial_path)

def load_submission(source: str):
    """Load submission from a directory or zip file"""
    sys.path.insert(0, source)

    if source.endswith(".zip"):
        try:
            # Load submission from zip.
            from submission import Submission
        except ImportError as e:
            raise ImportError(
                """
                Error loading submission from zip.
                Please ensure the zip contains the path submission/submission.py
                """
            ).with_traceback(e.__traceback__)
    else:
        # Load submission normally
        from submission import Submission

    # Remove submission from path.
    sys.path.remove(source)
    return Submission

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

def custom_run_evaluation(submission,  max_eps=100, seed=None):
    # Init  
    # Select Agents   
    sg = EnterpriseScenarioGenerator(blue_agent_class=SleepAgent, 
                                    green_agent_class=EnterpriseGreenAgent, 
                                    red_agent_class=FiniteStateRedAgent,
                                    steps=EPISODE_LENGTH)
    # Create Environment
    cyborg = CybORG(sg, "sim", seed=10)
    env = submission.wrap(cyborg)
    #env = BlueFixedActionWrapper(cyborg)
    
    print(f"Submission {submission.NAME}, Team: {submission.TEAM}, Technique: {submission.TECHNIQUE}")

    total_rewards = []
    obs = []
    info_short, log = np.array([]), np.array([])
    starttime=datetime.now()

    # Loop through episodes
    for episode in tqdm(range(max_eps)):
        
        observations, _ = env.reset()
        r = []
        # Create Agents
        blue_agents = [H_Agent(agent_name = agent_name, init_obs=env.get_observation(agent_name)) for agent_name in cyborg.environment_controller.get_active_agents() if 'blue' in agent_name]
        
        # Get Init Observations
        blue_observations = np.array([[env.get_observation(blue_agent.agent_name) for blue_agent in blue_agents]])
        
        # CAGE Challenge 4 Episode
        for step in range(EPISODE_LENGTH):
            
            # get actions from agents based on last observation
            actions2 = {
                agent.agent_name : agent.get_action(blue_observations[-1][i], env.action_space(agent.agent_name))
                    for i, agent in enumerate(blue_agents)}
            actions = {
                agent_name: agent.get_action(
                    observations[agent_name], env.action_space(agent_name)
                )
                for agent_name, agent in submission.AGENTS.items()
                if agent_name in env.agents
            }
            """
            if actions != actions2 and episode ==2:
                print("Actions are different!")
                print(actions)
                print(actions2)
                print(type(actions))
                print(type(actions2))
                #pprint(observations)
                #pprint(blue_observations[-1])
                """
            """print(f"Step {step+1}\n")
            print(f"My original Action: \n{actions2["blue_agent_0"]} \n")
            print(f"The other action: \n{actions["blue_agent_0"]} \n")
            print(f"\nMy original Observation: \n")
            pprint(blue_observations[-1][0])
            print(f"\nThe other Observation: \n")
            pprint(observations["blue_agent_0"])
            input("Press Enter to continue..." )"""
            # do a step in environment
            observations, reward, terminate, truncuate, info = env.step(actions)#, messages={agent.agent_name: agent.get_message() for agent in blue_agents})
            done = {
                    agent: terminate.get(agent, False) or truncuate.get(agent, False)
                    for agent in env.agents
                }
            if all(done.values()):
                    break
            r.append(mean(reward.values()))
            # get reward and collect info
            log = np.append(log, {"Episode": episode,
                                "Step": step,
                                "Observation": observations,
                                "Actions": actions,
                                "Reward": mean(reward.values()),
                                }
                            )
            
            blue_observations = np.vstack([blue_observations, np.array([observations[agent.agent_name] for agent in blue_agents])])
        total_rewards.append(sum(r))


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
                                "Mean-Reward": mean(total_rewards),
                                "Stdev-Reward": stdev(total_rewards)
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

def run_evaluation(submission, log_path, max_eps=100, write_to_file=False, seed=None):
    cyborg_version = CYBORG_VERSION
    EPISODE_LENGTH = 500
    scenario = "Scenario4"

    version_header = f"CybORG v{cyborg_version}, {scenario}"
    author_header = f"Author: {submission.NAME}, Team: {submission.TEAM}, Technique: {submission.TECHNIQUE}"

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=EPISODE_LENGTH,
    )
    cyborg = CybORG(sg, "sim", seed=seed)
    wrapped_cyborg = submission.wrap(cyborg)
    
    print(version_header)
    print(author_header)
    print(
        f"Using agents {submission.AGENTS}, if this is incorrect please update the code to load in your agent"
    )

    if write_to_file:
        if not log_path.endswith("/"):
            log_path += "/"
        print(f"Results will be saved to {log_path}")

    start = datetime.now()

    total_reward = []
    actions_log = []
    obs_log = []
    for i in tqdm(range(max_eps)):
        observations, _ = wrapped_cyborg.reset()
        r = []
        a = []
        o = []
        count = 0
        for j in range(EPISODE_LENGTH):
            actions = {
                agent_name: agent.get_action(
                    observations[agent_name], wrapped_cyborg.action_space(agent_name)
                )
                for agent_name, agent in submission.AGENTS.items()
                if agent_name in wrapped_cyborg.agents
            }
            #print(actions)
            observations, rew, term, trunc, info = wrapped_cyborg.step(actions)
            done = {
                agent: term.get(agent, False) or trunc.get(agent, False)
                for agent in wrapped_cyborg.agents
            }
            if all(done.values()):
                break
            r.append(mean(rew.values()))
            if write_to_file:
                a.append(
                    {
                        agent_name: cyborg.get_last_action(agent_name)
                        for agent_name in wrapped_cyborg.agents
                    }       
                )
                o.append(
                    {
                        agent_name: observations[agent_name]
                        for agent_name in observations.keys()
                    }
                )
        total_reward.append(sum(r))
        print(f"Episode {i+1} Reward: {total_reward[-1]}")
        if write_to_file:
            actions_log.append(a)
            obs_log.append(o)

    end = datetime.now()
    difference = end - start

    reward_mean = mean(total_reward)
    reward_stdev = stdev(total_reward)
    reward_string = (
        f"Average reward is: {reward_mean} with a standard deviation of {reward_stdev}"
    )
    print(reward_string)

    print(f"File took {difference} amount of time to finish evaluation")
    if write_to_file:
        print(f"Saving results to {log_path}")
        with open(log_path + "summary.txt", "w") as data:
            data.write(version_header + "\n")
            data.write(author_header + "\n")
            data.write(reward_string + "\n")
            data.write(f"Using agents {submission.AGENTS}")

        with open(log_path + "full.txt", "w") as data:
            data.write(version_header + "\n")
            data.write(author_header + "\n")
            data.write(reward_string + "\n")
            for act, obs, sum_rew in zip(actions_log, obs_log, total_reward):
                data.write(
                    f"actions: {act},\n observations: {obs},\n total reward: {sum_rew}\n"
                )
        
        with open(log_path + "actions.txt", "w") as data:
            data.write(version_header + "\n")
            data.write(author_header + "\n")
            data.write(reward_string + "\n")
            for act in zip(actions_log):
                data.write(
                    f"actions: {act}"
                )

        with open(log_path + "summary.json", "w") as output:
            data = {
                "submission": {
                    "author": submission.NAME,
                    "team": submission.TEAM,
                    "technique": submission.TECHNIQUE,
                },
                "parameters": {
                    "seed": seed,
                    "episode_length": EPISODE_LENGTH,
                    "max_episodes": max_eps,
                },
                "time": {
                    "start": str(start),
                    "end": str(end),
                    "elapsed": str(difference),
                },
                "reward": {
                    "mean": reward_mean,
                    "stdev": reward_stdev,
                },
                "agents": {
                    agent: str(submission.AGENTS[agent]) for agent in submission.AGENTS
                },
            }
            json.dump(data, output)

        with open(log_path + "scores.txt", "w") as scores:
            scores.write(f"reward_mean: {reward_mean}\n")
            scores.write(f"reward_stdev: {reward_stdev}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("CybORG Evaluation Script")
    parser.add_argument(
        "--append-timestamp",
        action="store_true",
        help="Appends timestamp to output_path",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Set the seed for CybORG"
    )

    # Added to speed up evaluation 
    parser.add_argument(
        '--distribute', type=int, default=1, help="How many parallel workers to use"
    )
    parser.add_argument("--max-eps", type=int, default=100, help="Max episodes to run")
    args = parser.parse_args()
    args.output_path = os.path.abspath('evaluation_contractor_active_0')
    args.submission_path = os.path.abspath('')

    if not args.output_path.endswith("/"):
        args.output_path += "/"

    if args.append_timestamp:
        args.output_path += time.strftime("%Y%m%d_%H%M%S") + "/"

    rmkdir(args.output_path)

    submission = load_submission(args.submission_path)

    custom_run_evaluation(submission=submission, max_eps=args.max_eps)#, seed=args.seed)
    run_evaluation(
        submission, max_eps=args.max_eps, log_path=args.output_path, seed=args.seed, write_to_file=True)
