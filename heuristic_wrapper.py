from CybORG.Agents.Wrappers import BlueFixedActionWrapper
from CybORG.Simulator.Actions import Restore, Monitor
from CybORG.Shared.Enums import TernaryEnum

import numpy as np
"""
This wrapper is supposed to create the messages between the agent and acts a kind of a memory for the agent.
"""
class HeuristicWrapper(BlueFixedActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.agents = [agent for agent in env.agents if "blue" in agent]
        self.messages = {agent: np.array([0]*8) for agent in self.agents}

    def reset(self):
        return super().reset()
    
    def _mask_obs(self, obs):
        """
        Mask the observation, that messages are returned in correct order
        die agenten scheinen immer auf die gleichen subnets gelegt zu werden.
        Info:
        Agent 0 - Deployed Network A - restricted Zone
        Agent 1 - Deployed Network A - operational Zone
        Agent 2 - Deployed Network B - restricted Zone
        Agent 3 - Deployed Network B - operational Zone
        Agent 4 - HG

        Ziel
        -   Die erste Nachricht ist vom cooperierendem Subnet im gleichen Netzwerk
        -   Die zweite Nachricht ist von der restricted zone des anderen Netzwerks
        -   Die dritte Nachricht ist von der operational zone des anderen Netzwerks
        -   Die vierte Nachricht ist vom der HG
        -   Agent 4 bekommt eine sinderstellung und wird nicht angepasst

        Before masking
                    Message from Agent in Position:
                    0   1   2   3

        Agent_0:    1   2   3   4
        Agent_1:    0   2   3   4
        Agent_2:    0   1   3   4
        Agent_3:    0   1   2   4
        Agent_4:    0   1   2   3
        
                Message from Agent in Position:
                    0   1   2   3

        Agent_0:    1   2   3   4
        Agent_1:    0   2   3   4
        Agent_2:    3   0   1   4
        Agent_3:    2   0   1   4
        Agent_4:    0   1   2   3
        """
        if False:
            for agent in list(obs.keys()):
                match agent:
                    case "blue_agent_0":
                        pass
                    case "blue_agent_1":
                        pass
                    case "blue_agent_2":
                        obs[agent]['message'][0], obs[agent]['message'][2] = obs[agent]['message'][2], obs[agent]['message'][0]
                        obs[agent]['message'][1], obs[agent]['message'][2] = obs[agent]['message'][2], obs[agent]['message'][1]
                    case "blue_agent_3":
                        obs[agent]['message'][0], obs[agent]['message'][2] = obs[agent]['message'][2], obs[agent]['message'][0]
                        obs[agent]['message'][1], obs[agent]['message'][2] = obs[agent]['message'][2], obs[agent]['message'][1]
                    case "blue_agent_4":
                        pass
        return obs

    def step(self, action):
        self._update_messages(action)
        obs, reward, done, trunc, info = super().step(action, self.messages)
        return self._mask_obs(obs), reward, done, trunc, info
    
    def _update_messages(self, actions):
        """
        Combine message with block / allow trafic.
        Bit 0: serverHost0 has been accessed by red agent with root access ans is in the progress of being restored
        Bit 1:
        Bit 2:
        Bit 3:
        Bit 4:
        Bit 5:
        Bit 6:
        Bit 7:
        """
        """
        Steps:
        1. Create the messages
        2. mask messages that the first message is always the one from the same zone, then from other deployed network and last the HG
            for HG start wirh A and then B
        3. respect current operation mission
            Operational zones always do only communicate with their resticted zone in the same netwok
            Also the communications is one way from resticted to operational.
         - Pre-planning: every host is connected to every host except the operational ones
         - Mission A: Restricted Zone A only communicates with the hq and operational zone A is cut off
         - Mission B: restricted Zone B only communicates with the hq and operationel zone A is cut off
        """

        """
        Step 1:
            - bit 0 is set when restore action is taken on server_host_0
            - bit 1 is reset once the obervation shows that the server_host_0 is no longer in the process of being restored
        """      
        for agent, action in actions.items():
            obs = super().get_observation(agent)
            if isinstance(action, Restore) and "server_host_0" in str(action) and 'admin_network' not in str(action) and 'office_network' in str(action):
                self.messages[agent][0] = 1
            elif 'action' in obs.keys():
                if isinstance(obs['action'], Restore) and obs['success'] == TernaryEnum.TRUE and "server_host_0" in str(obs['action']):
                    self.messages[agent][0] = 0
        """        
        for agent, actions in actions.items():
            if agent == "blue_agent_0":
                self.messages[agent][0] = 1
            elif agent == "blue_agent_1":
                self.messages[agent][1] = 1
            elif agent == "blue_agent_2":
                self.messages[agent][2] = 1
            elif agent == "blue_agent_3":   
                self.messages[agent][3] = 1
            elif agent == "blue_agent_4":
                self.messages[agent][4] = 1"""
        return 
        