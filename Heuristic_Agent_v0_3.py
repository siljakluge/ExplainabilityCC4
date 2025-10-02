import numpy as np
from collections import Counter

from CybORG.Simulator.Actions import Monitor , Analyse, DeployDecoy, Remove, Restore
from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import AllowTrafficZone, BlockTrafficZone 
from CybORG.Shared.Enums import TernaryEnum

from copy import deepcopy 



"""
Generall stategie:
1. Analyse Observation
    1.1 Look for maleware
    1.2 Look for decoy exploit
    1.3 Assign analysing priority to remaining events
        1.3.1 Priority 1 for multiple events on same server and simultaneous router/server event
        1.3.2 Priority 2 for single alert (FP from green agent or detected red agent action)
        1.3.3 Priority 3 for all remaining server/hosts (default action)
2. Update all queues 
3. Select action by priority (up to down)
    2.1 Restore server to remove red root shells
    2.2 Remove red user shells
    2.3 Analyse event of priority 1 and 2
    2.4 Deploy decoy on server/router with no decoy (either last analysed or random one)
    2.5 Analyse event of priority 1 and 2
4. collect information and return action

Open points for strategie:
- Currently the 8bit messages are not used between agents
- Blue agents Control Traffic action is not used -> Combine with 8bit messages
- No strategie behind decoy placement. They are just deployed as soon as possible
- No further analysation of red agents behaviour
- Reward is unstable

Next Steps:
- Run heuristic agent multiple times to calculate mean and standard deviation of reward
- Analys red agent observations and actions to find weaknesses
- Research on how to use 8bit messages and Control Traffic 
    - start by just blocking traffic from one subnet to another at the beginning of the episode if there is a red agent
    - allow traffic again if  situation has been resolved
    - add a trigger to compare everythin
- maybe analyse a prio 1 and 2 event 3 or 2 times in a row or with a tick in between. depends on the red agents behaviour.

Goal: 
- mean reward above -100 with small deviation
- RL agent based on final heuristic version to find out if it can beat heuristic agent
"""
AGENTVERSION = "0.3"

class H_Agent():

    def __init__(self, agent_name, init_obs = None):
        self.agent_name = agent_name
        self.agent_version = AGENTVERSION
        self._reset_agents()

    def _get_responsible_hosts(self, obs):
        return [key for key in obs.keys() if key != 'success']
    
    def _reset_agents(self):
        self.observations = []
        self.actions = []
        self.hosts = []
        self.status = 'starting' #other states could be 'waiting', 'acting', 'searching', 'preventing'
        self.analyse_host = {'Priority 1': [], # double event
                             'Priority 2' : [], # normal event
                             'Priority 3' : []} # default action
        self.info = []
        self.restore_host = [] #Hosts which have to be restored
        self.decoy_host = [] # Hosts which have to be decoyed
        self.remove_host = [] #Hosts which have a file to remove
        self.last_analysed = None #Last analysed host
        self.action_counter = np.array([0,0,0,0,0]) #Monitor, Analyse, DeployDecoy, Remove, Restore
    
    def _unique_list(self, list):
        unique_list = []
        for item in list:
            if item not in unique_list:
                unique_list.append(item)
        return unique_list

    def get_action(self, obs, action_space):
        # Reset the agent if a new episode starts (necessary für submission, because the queues are not reset automatically)
        if obs['success'] == TernaryEnum.UNKNOWN and np.sum(self.action_counter) != 0:
            print(f"Reset Agent {self.agent_name}")
            self._reset_agents()

        # Init on first observation
        if len(self.observations) == 0:
            self.hosts = self._get_responsible_hosts(obs)
            self.analyse_host['Priority 3'] = deepcopy(self.hosts)
            self.decoy_host = self._get_responsible_hosts(obs)
            self.last_analysed = self.hosts[0]

        # Store observation
        self.observations.append(obs)

        # Check if any event has been detected
        events = [item for item in self.hosts if item in list(obs.keys()) and len(self.observations) > 1]
        # Remove events that only occure because of last action
        # If Deployed as last action
        if "action" in obs.keys():
            if isinstance(obs["action"], DeployDecoy):
                events = [event for event in events if event not in str(obs["action"])] # maybe has to be more specific
        if len(events) > 0:

        # 1. Check for malware
            for host in events:
                if 'Files' in obs[host]:
                    for file in obs[host]['Files']:
                        if file["File Name"] == "cmd.sh":
                            self.remove_host.append(host)
                        elif file["File Name"] == "escalate.sh":
                            self.restore_host.append(host)
            

        # 2. Check for decoy exploit and list for restore if found
                if 'Processes' in obs[host]:
                    for process in obs[host]['Processes']:
                        if 'Connections' in process:
                            if 'local_port' in process['Connections']:
                                if process['Connections']['local_port'] == 25:
                                    self.restore_host.append(host)
                                    self.decoy_host.append(host) 

        # 3. Check multiple events on same host or a simultaneous router and server connection. They will be prioritised in analysing if they are not listed in restore actions
        unique_events = self._unique_list(events)
        counter = Counter(events)
        double_events = [item for item, count in counter.items() if count > 1]
        if any("server" in event for event in events) and any("router" in event for event in events):
            for event in events:
                if "server" in event:
                    double_events.append(event)
        double_events = self._unique_list(double_events)

        # 4. if a host has to be restored, remove the event from the remove action list and eliminate double events in remove and restore action 
        self.remove_host = self._unique_list(self.remove_host)
        self.restore_host = self._unique_list(self.restore_host)
        self.remove_host = [item for item in self.remove_host if item not in self.restore_host]

        # 5. find remaining events
        events = [event for event in events if event not in self.restore_host] 
        events = [event for event in events if event not in self.remove_host] 
        events = [event for event in events if event not in double_events]

        # 6. update analysation priorities
        # 6.1 Priority 1 for double occuring events
        for double_event in double_events:
            if double_event not in self.analyse_host['Priority 1']:
                self.analyse_host['Priority 1'].append(double_event)
        # 6.2 Priority 2 for single detected event
        for event in events:
            if event not in self.analyse_host['Priority 2']:
                self.analyse_host['Priority 2'].append(event)
        # 6.3 Remove the events from Priority 2 which are in Priority 1
        self.analyse_host['Priority 2'] = [item for item in self.analyse_host['Priority 2'] if item not in self.analyse_host['Priority 1']]

        """Find next Action based on Priority:
        1. Restore server
        2. Remove file
        3. Analyse a Prioriy 1 or 2 host
            3.1 once a priority 1 or 2 element has been analysed it is set to the bottom of priority 3 
        4. set decoy on a detected event with no decoy or last analysed event
        5. analyse priority 3 host
        """

        # 7. Select action
        # 7.1 first action is always Monitor
        if len(self.actions) == 0 or obs['success'] == TernaryEnum.IN_PROGRESS:
            action = Monitor(0, self.agent_name)
            self.action_counter[0] += 1
            self.status = 'waiting'
        # 7.2 restore a server and remove the host from the list afterwards
        elif len(self.restore_host) > 0:
            action = Restore(session=0, agent=self.agent_name, hostname=self.restore_host[0])
            self.restore_host =  self.restore_host[1:]
            self.status = "acting"
        # 7.3 remove a server and remove the host from the list afterwards
        elif len(self.remove_host) > 0:
            action = Remove(session=0, agent=self.agent_name, hostname=self.remove_host[0])
            self.remove_host = self.remove_host[1:]
            self.status = "acting"
        # 7.4 Analyse a priority 1 host and set the host to the end of priority 3
        elif len(self.analyse_host['Priority 1']) > 0:
            action = Analyse(session=0, agent=self.agent_name, hostname=self.analyse_host['Priority 1'][0])
            self.last_analysed = self.analyse_host['Priority 1'][0]
            self.analyse_host['Priority 3'].remove(self.analyse_host['Priority 1'][0])
            self.analyse_host['Priority 3'].append(self.analyse_host['Priority 1'][0])
            self.analyse_host['Priority 1'] = self.analyse_host['Priority 1'][1:]
            self.status = "searching"
        # 7.5 Analyse a priority 2 host and set the host to the end of priority 3
        elif len(self.analyse_host['Priority 2']) > 0:
            action = Analyse(session=0, agent=self.agent_name, hostname=self.analyse_host['Priority 2'][0])
            self.last_analysed = self.analyse_host['Priority 2'][0]
            self.analyse_host['Priority 3'].remove(self.analyse_host['Priority 2'][0])
            self.analyse_host['Priority 3'].append(self.analyse_host['Priority 2'][0])
            self.analyse_host['Priority 2'] = self.analyse_host['Priority 2'][1:]
            self.status = "searching"
        # 7.6 Deploy a decoy on the last analysed server which does not hat a decoy or on a random host
        elif len(self.decoy_host) > 0:
            if self.last_analysed in self.decoy_host:
                action = DeployDecoy(session=0, agent=self.agent_name, hostname=self.last_analysed)
                self.decoy_host.remove(self.last_analysed)
            else:
                action = DeployDecoy(session=0, agent=self.agent_name, hostname=self.decoy_host[0])
                self.decoy_host = self.decoy_host[1:]
            self.status = "preventing"
        # 7.7 Defaul Action is to anlyse a host and set this host to the end of the list afterwards
        else:
            action = Analyse(session=0, agent=self.agent_name, hostname=self.analyse_host['Priority 3'][0])
            self.last_analysed = self.analyse_host['Priority 3'][0]
            self.analyse_host['Priority 3'].remove(self.last_analysed)
            self.analyse_host['Priority 3'].append(self.last_analysed)
            self.status = "searching"

        
        # 8. Store and count action selection
        self.actions.append(action)
        if isinstance(action, Analyse):
            self.action_counter[1] += 1
        elif isinstance(action, DeployDecoy):
            self.action_counter[2] += 1
        elif isinstance(action, Remove):
            self.action_counter[3] += 1
        elif isinstance(action, Restore):
            self.action_counter[4] += 1

        # 9. Add step information
        self.info.append({"Name": self.agent_name,
                "Status": self.status,
                "Observation": obs,
                "Action": action})
        return action
        
        
    def get_message(self, last_action = None, obs = None, id = 0):
        """
        Combine message with block / allow trafic.
        Bit 0:
        Bit 1:
        Bit 2:
        Bit 3:
        Bit 4:
        Bit 5:
        Bit 6:
        Bit 7:
        """
        return np.array([False]*8)
    
