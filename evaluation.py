import inspect
from matplotlib import pyplot as plt
import numpy as np
import time
import json
from statistics import mean, stdev

from tqdm import tqdm 
from joblib import Parallel, delayed

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueEnterpriseWrapper

from datetime import datetime
import sys
import os

from pprint import pprint

class CustomGreenAgent(EnterpriseGreenAgent):
    def __init__(self, name, own_ip, np_random=None):
        super().__init__(name, own_ip, np_random, fp_detection_rate=0.01, phishing_error_rate=0.01)  # Set your value here


cyborg_version = CYBORG_VERSION
EPISODE_LENGTH = 500


#Save information dicts
def save_dict_array_to_json(arr, filename, folder = None):
    # Convert numpy array of dicts to list of dicts
    data = [dict(item) for item in arr]
    if folder == None:
         folder = filename[:-5] if filename.endswith('.json') else filename
    folder_path= "Results/"+folder+"/"
    os.makedirs(folder_path, exist_ok=True)
    
    # Ensure filename has .json extension
    if not filename.endswith('.json'):
        filename += '.json'
    
    # Save as JSON file
    with open(folder_path + filename, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved {len(arr)} dictionaries to {filename}")


#Plot reward per episode
def plot_array(data: np.ndarray, title: str = "Mein Graph", folder:str = None):
    plt.plot(range(len(data)), data, marker='o')
    plt.xlim(0, len(data))
    plt.ylim(-400,0)
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


def evaluate_one_episode(cyborg, wrapped_cyborg, agent, write_to_file, i,tot):
    observations, _ = wrapped_cyborg.reset()
    r = []
    a = []
    o = []
    count = 0
    for j in tqdm(range(EPISODE_LENGTH), desc=f'({i+1}/{tot})'):
        actions = {
            agent_name: agent.get_action(
                observations[agent_name], wrapped_cyborg.action_space(agent_name)
            )
            for agent_name, agent in submission.AGENTS.items()
            if agent_name in wrapped_cyborg.agents
        }
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
    total_reward = sum(r)
    return total_reward, a, o

def run_evaluation_parallel(submission, log_path, max_eps=100, write_to_file=False, seed=None, workers=32):
    cyborg_version = CYBORG_VERSION
    EPISODE_LENGTH = 500
    scenario = "Scenario4"

    version_header = f"CybORG v{cyborg_version}, {scenario}"
    author_header = f"Author: {submission.NAME}, Team: {submission.TEAM}, Technique: {submission.TECHNIQUE}"

    envs = []
    for _ in range(workers):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=EPISODE_LENGTH,
        )
        cyborg = CybORG(sg, "sim", seed=seed)
        wrapped_cyborg = submission.wrap(cyborg)
        envs.append((cyborg, wrapped_cyborg))
    
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

    outs = Parallel(prefer='processes', n_jobs=workers)(
        delayed(evaluate_one_episode)(*envs[i % workers], submission.AGENTS, write_to_file, i, max_eps)
        for i in range(max_eps)
    )
    total_reward, actions_log, obs_log = zip(*outs)

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

def run_evaluation(submission, 
                   log_path, 
                   max_eps=100, 
                   write_to_file=False, 
                   seed=None, 
                   log_step=False, 
                   log_agent = 0, 
                   comment = "",
                   mask_enable=True
                   ):
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
    wrapped_cyborg.mask_enable = mask_enable
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
    savetime =str(start.year)+str(start.month)+str(start.day)+"_"+str(start.hour)+str(start.minute)

    total_reward = []
    actions_log = []
    obs_log = []
    info_short, log = np.array([]), np.array([])
    for i in tqdm(range(max_eps)):
        observations, _ = wrapped_cyborg.reset()
        r = []
        a = []
        o = []
        for j in range(EPISODE_LENGTH):
            actions = {
                agent_name: agent.get_action(
                    observations[agent_name], wrapped_cyborg.action_space(agent_name)
                )
                for agent_name, agent in submission.AGENTS.items()
                if agent_name in wrapped_cyborg.agents
            }

            if log_step:
                print("\n")
                print(f"Step {j+1}\n")
                print(f"Log for Agent: blue_agent_{log_agent}\n")
                print(f"Observation:\n")
                pprint(observations[f"blue_agent_{log_agent}"])
                print("\n")
                print(f"Action: {actions[f"blue_agent_{log_agent}"]}\n")
                print("\n")
                input("Press Enter to continue...")


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
                log = np.append(log, {"Episode": i,
                                "Step": j,
                                "Observation": observations,
                                "Actions": actions,
                                "Reward": mean(rew.values()),
                                }
                            )
        total_reward.append(sum(r))
        print(f"Episode {i+1} Reward: {total_reward[-1]}")
        if write_to_file:
            actions_log.append(a)
            obs_log.append(o)
        info_short = np.append(info_short, 
                        {"Episode" : i,
                        "Seed": seed,
                        "Reward": total_reward[i],
                        "Monitor Action Count": [agent.action_counter[0] for agent in submission.AGENTS.values()],
                        "Analyse Action Count":  [agent.action_counter[1] for agent in submission.AGENTS.values()],
                        "DeployDecoy Action Count": [agent.action_counter[2] for agent in submission.AGENTS.values()],
                        "Restore Action Count": [agent.action_counter[4] for agent in submission.AGENTS.values()],
                        "Remove Action Count": [agent.action_counter[3] for agent in submission.AGENTS.values()],
                        "Blocked Traffic Action Count": [agent.action_counter[5] for agent in submission.AGENTS.values()],
                        "Allowed Traffic Action Count": [agent.action_counter[6] for agent in submission.AGENTS.values()],
                        list(submission.AGENTS.values())[0].agent_name + " host and servers amount": len(list(submission.AGENTS.values())[0].hosts),
                        list(submission.AGENTS.values())[1].agent_name + " host and servers amount": len(list(submission.AGENTS.values())[1].hosts),
                        list(submission.AGENTS.values())[2].agent_name + " host and servers amount": len(list(submission.AGENTS.values())[2].hosts),
                        list(submission.AGENTS.values())[3].agent_name + " host and servers amount": len(list(submission.AGENTS.values())[3].hosts)
                        }
                        )

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
    generall_info = np.array([{"Mean Reward": np.sum(total_reward)/len(total_reward),
                            "Agent-Version":"Heuristic_"+list(submission.AGENTS.values())[0].version,
                            "Mean-Reward": mean(total_reward),
                            "Stdev-Reward": stdev(total_reward),
                            "Comment": "" + comment,
                            "Start": savetime,
                            "Control Traffic": list(submission.AGENTS.values())[0].enable_blocking,
                            "All Rewards": total_reward}
                            ])
    # save info and graph
    folder_path = generall_info[0]["Agent-Version"]+"_"+savetime
    plot_array(data=total_reward, title="Total Rewards per Episode"+savetime, folder=folder_path)
    save_dict_array_to_json(filename=("General_info_"+savetime+".json"), arr=generall_info, folder=folder_path)
    save_dict_array_to_json(filename=("Short_Info_"+savetime+".json"), arr=info_short, folder=folder_path)
    save_dict_array_to_json(filename=("Log_"+savetime+".json"), arr=log, folder=folder_path)

    print("DONE")


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
    parser.add_argument(
        '--log_step', type=bool, default=False, help="Do you want to observe every step for an agent (default: blue_agent_0)"
    )    
    parser.add_argument(
        '--log_agent', type=int, default=False, help="Which agent do you want to observe (default: blue_agent_0)"
    )
    parser.add_argument(
        '--comment', type=str, default="", help="additional comment to save in general information"
    )
    parser.add_argument(
        '--mask_enable', type=bool, default=True, help="Enable masking of messages"
    )

    parser.add_argument("--max-eps", type=int, default=100, help="Max episodes to run")
    args = parser.parse_args()
    args.output_path = os.path.abspath('tmp')
    args.submission_path = os.path.abspath('')

    if not args.output_path.endswith("/"):
        args.output_path += "/"

    if args.append_timestamp:
        args.output_path += time.strftime("%Y%m%d_%H%M%S") + "/"

    rmkdir(args.output_path)

    submission = load_submission(args.submission_path)

    run_evaluation(
        submission, 
        max_eps=args.max_eps, 
        log_path=args.output_path, 
        seed=args.seed, 
        write_to_file=True, 
        log_step=args.log_step, 
        log_agent=args.log_agent, 
        comment=args.comment,
        mask_enable=args.mask_enable
    )
