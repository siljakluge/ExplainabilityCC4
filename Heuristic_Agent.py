import numpy as np
import random
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
2. Analyse messages
    2.1 See if Traffic has to be blocked (not implemented yet)
3. Update all queues
4. Select action by priority (up to down)
    2.1 Restore server to remove red root shells
    2.2 Remove red user shells
    2.3 Block traffic from own subnet to subnet with red agent in same zone (not implemented yet)
    2.4 Allow traffic from own subnet to subnet with removed red agent in same zone (not implemented yet)
    2.5 Analyse event of priority 1 and 2
    2.6 Deploy decoy on server/router with no decoy (either last analysed or random one)
    2.7 Analyse event of priority 1 and 2
5. collect information and return action

Next Steps:
- Analys red agent observations and actions to find weaknesses
- embed the mission status in as a mask for the message
- maybe analyse a prio 1 and 2 event 3 or 2 times in a row or with a tick in between. depends on the red agents behaviour.
- red agent prefers to target a server host over a user one, which should be reflected in queue strategie
- set FP to 1 or 0 to see effects on standard deviation

Open points for strategie:
- Currently the 8bit messages are barely used
- Blue agents Control Traffic action is used in basic form
- No strategie behind decoy placement. They are just deployed as soon as possible
- No further analysation of red agents behaviour
- Reward is unstable
- analyse the episodes with very low reward

List of subnets:
- restricted_zone_a_subnet
- operational_zone_a_subnet
- restricted_zone_b_subnet
- operational_zone_b_subnet
- public_access_zone_subnet
- office_network_subnet
- admin_network_subnet

Goal: 
- mean reward above -100 with small deviation
- RL agent based on final heuristic version to find out if it can beat heuristic agent
"""
"""
To do heute:
 - implement contractor network to blocking strategie if necessary
 - test if the agent should always try to analyse a host he has just removed a file from in case the red agent has already gained root access
    - a host is suspected to be compromissed if.
        - a file has been removed from it
        - any event has occured
    - repetition of analysation based on weights
 """

VERSION = "1.0"
ENABLE_BLOCKING = True
ENABLE_HQ_BLOCKING = False
ENABLE_PRIORITY = True
WEIGHTS_PRIORITY = [3,2,1] # priority 1,2,3[server_0, other servers, rest (users, router)] in Analyse["Priority 3"]
ALWAYS_RESTORE = False # if True, the agent will always restore a host instead of removing files
AGGRESSIVE_ANALYSE = False # if True, the agent will analyse hosts more often if they are suspected to be compromissed which is also when a file has been removed
AGGRESSIVE_ANALYSE_REP = [3,2] # amount of repetitive analysations for suspected hosts ([Prio1, Prio2])

class H_Agent():

    def __init__(self, agent_name, init_obs = None):
        self.agent_name = agent_name
        self.version = VERSION
        self.enable_blocking = ENABLE_BLOCKING
        self.enable_priority = ENABLE_PRIORITY
        self.priority_weights = WEIGHTS_PRIORITY
        self.enable_hq_blocking = ENABLE_HQ_BLOCKING
        self.always_restore = ALWAYS_RESTORE
        self.aggressive_analyse = AGGRESSIVE_ANALYSE
        self.aggressive_analyse_rep = AGGRESSIVE_ANALYSE_REP
        self._reset_agents()

    def _get_responsible_hosts(self, obs):
        return [key for key in obs.keys() if key != 'success']
     
    def _reset_agents(self):
        self.observations = []
        self.subnet = self._get_subnet(self.agent_name)
        self.actions = []
        self.hosts = []
        self.blocked_hosts = [] # list of all blocked subnets
        self.block_host =[] # hosts which should be blocked
        self.analyse_host = {'Priority 1': [], # double event
                             'Priority 2': [], # normal event
                             'Priority 3': []} # default action
        self.info = []
        self.analyse_counter = {}
        self.restore_host = [] # Hosts which have to be restored
        self.decoy_host = [] # Hosts which have to be decoyed
        self.remove_host = [] # Hosts which have a file to remove
        self.last_analysed = None #Last analysed host
        self.action_counter = np.array([0,0,0,0,0,0,0]) # Monitor, Analyse, DeployDecoy, Remove, Restore, Block Traffic, Allow Traffic
        self.last_message = None
    
    def _unique_list(self, list):
        unique_list = []
        for item in list:
            if item not in unique_list:
                unique_list.append(item)
        return unique_list
    
    
    def _get_subnet(self, name):
        subnets ={"blue_agent_0": "restricted_zone_a_subnet",
                  "blue_agent_1": "operational_zone_a_subnet",
                  "blue_agent_2": "restricted_zone_b_subnet",
                  "blue_agent_3": "operational_zone_b_subnet",
                  "blue_agent_4": "public_access_zone_subnet"}
        return subnets[name]
        
    def _chosen_host_to_analyse(self):
        # Priority: server_0 > other servers > users
        if not ENABLE_PRIORITY:
            return self.analyse_host['Priority 3'][0]
        servers = []
        servers_0 = []
        rest = [] # users and router
        b = random.uniform(0,1)
        for host in self.analyse_host['Priority 3']:
            if 'server_host_0' in host:
                servers_0.append(host)
            elif 'server' in host and host not in servers_0:
                servers.append(host)
            else:
                rest.append(host)
        assert len(servers_0) + len(servers) + len(rest) == len(self.analyse_host['Priority 3']), "Error in host classification"
        assert len(servers_0) != 0, "empty list error"
        assert len(rest) != 0, "empty list error"

        a = 1/(self.priority_weights[0] * len(servers_0) + self.priority_weights[1] * len(servers) + self.priority_weights[2] * len(rest))
        eps_server_0 = self.priority_weights[0] * a * len(servers_0) # 1 server_0 (3 for HG)
        eps_server = self.priority_weights[1] * a * len(servers) # 1-6 servers including server_0, but excluded in later choice
        eps_rest = self.priority_weights[2] * a * len(rest) # 3-10 user hosts
        assert eps_server_0  + eps_server + eps_rest >= 0.99, "Error in epsilon values."
        assert eps_server_0  + eps_server + eps_rest <= 1.01, "Error in epsilon values."
        if b <= eps_server_0 and len(servers_0) > 0:
            return servers_0[0]
        elif b <= eps_server_0 + eps_server and len(servers) > 0:
            return servers[0]
        elif len(rest) > 0:
            return rest[0]
        else:
            return self.analyse_host['Priority 3'][0]

    def _interpret_message(self, message):
        origin_table = {"blue_agent_0": ["operational_zone_a_subnet", "restricted_zone_b_subnet", "operational_zone_b_subnet", "public_access_zone_subnet"],
                        "blue_agent_1": ["restricted_zone_a_subnet", "restricted_zone_b_subnet", "operational_zone_b_subnet", "public_access_zone_subnet"],
                        "blue_agent_2": ["restricted_zone_a_subnet", "operational_zone_a_subnet", "operational_zone_b_subnet", "public_access_zone_subnet"],
                        "blue_agent_3": ["restricted_zone_a_subnet", "operational_zone_a_subnet", "restricted_zone_b_subnet", "public_access_zone_subnet"],
                        "blue_agent_4": ["restricted_zone_a_subnet", "operational_zone_a_subnet", "restricted_zone_b_subnet", "operational_zone_b_subnet"]}

        for i, message in enumerate(message):
            if message[0]==1:
                if origin_table[self.agent_name][i] not in self.block_host:
                    self.block_host.append(origin_table[self.agent_name][i])
            #remove host from list to be blocked if message indicates that host is secure again
            elif message[0]==0:  
                if origin_table[self.agent_name][i] in self.block_host:
                    self.block_host.remove(origin_table[self.agent_name][i])

        # special case for blue_agent_4 because he has to observe multiple subnets which
        if self.agent_name == "blue_agent_4" and ENABLE_HQ_BLOCKING:
            """
            Currenty no action because the agent should focus on restoring the host instead of blocking.
            """
            pass
        return 

    def get_action(self, obs, action_space):
        # Reset the agent if a new episode starts (necessary für submission, because the queues are not reset automatically)
        if obs['success'] == TernaryEnum.UNKNOWN and np.sum(self.action_counter) != 0:
            #print(f"Reset Agent {self.agent_name}")
            self._reset_agents()

        # Init on first observation
        if len(self.observations) == 0:
            self.hosts = self._get_responsible_hosts(obs)
            self.analyse_host['Priority 3'] = deepcopy(self.hosts)
            self.decoy_host = self._get_responsible_hosts(obs)
            self.last_analysed = self.hosts[0]
            self.last_message = [np.array([False]*8), np.array([False]*8), np.array([False]*8), np.array([False]*8)]

        # Store observation
        self.observations.append(obs)

        # Check if any event has been detected
        events = [item for item in self.hosts if item in list(obs.keys()) and len(self.observations) > 1]
        # interpret message
        if "message" in obs.keys() and ENABLE_BLOCKING:
            self._interpret_message(obs["message"])
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
        # 7.2 restore a server and remove the host from the list afterwards
        elif len(self.restore_host) > 0:
            action = Restore(session=0, agent=self.agent_name, hostname=self.restore_host[0])
            self.restore_host =  self.restore_host[1:]
        # 7.3 remove a server and remove the host from the list afterwards (if always_restore Restore instead)
        elif len(self.remove_host) > 0:
            if self.always_restore:
                action = Restore(session=0, agent=self.agent_name, hostname=self.remove_host[0])
            else:
                action = Remove(session=0, agent=self.agent_name, hostname=self.remove_host[0])
            if self.aggressive_analyse and not self.always_restore:
                self.analyse_host['Priority 1'].append(self.remove_host[0])
            self.remove_host = self.remove_host[1:]
        # Block or umblock a compromised host
        # Block if message indicates a compromissed host and host is not already blocked
        # unblock if message indicates a safe host and host is still blocked
        elif len(self.block_host) != len(self.blocked_hosts) and ENABLE_BLOCKING:
            # a new host has to be blocked
            if len(self.block_host) > len(self.blocked_hosts): 
                host_to_block = [i for i in self.block_host if i not in self.blocked_hosts][0] # a host that is not blocked has to be blocked
                action = BlockTrafficZone(session=0, agent=self.agent_name, from_subnet=host_to_block, to_subnet=self.subnet)
                self.blocked_hosts.append(host_to_block)
                if ENABLE_HQ_BLOCKING and self.agent_name == "blue_agent_4":
                    #currently prefer to restore host as long as public access is not compromised
                    pass
            else:
                host_to_unblock = [i for i in self.blocked_hosts if i not in self.block_host][0]
                action = AllowTrafficZone(session=0, agent=self.agent_name, from_subnet=host_to_unblock, to_subnet=self.subnet)
                self.blocked_hosts.remove(host_to_unblock)
        # 7.5 Analyse a priority 1 host and set the host to the end of priority 3
        elif len(self.analyse_host['Priority 1']) > 0:
            action = Analyse(session=0, agent=self.agent_name, hostname=self.analyse_host['Priority 1'][0])
            self.last_analysed = self.analyse_host['Priority 1'][0]
            # if aggressive analyse is enabled, count the number of repetitions for this host
            if self.aggressive_analyse and self.aggressive_analyse_rep[0]>1:
                if self.analyse_host['Priority 1'][0] in list(self.analyse_counter.keys()):
                    self.analyse_counter[self.analyse_host['Priority 1'][0]] += 1
                    # remove from priority 1 if repetition limit has been reached
                    if self.analyse_counter[self.analyse_host['Priority 1'][0]] >= self.aggressive_analyse_rep[0]:
                        self.analyse_host['Priority 3'].remove(self.analyse_host['Priority 1'][0])
                        self.analyse_host['Priority 3'].append(self.analyse_host['Priority 1'][0])
                        self.analyse_host['Priority 1'] = self.analyse_host['Priority 1'][1:]
                else:
                    self.analyse_counter[self.analyse_host['Priority 1'][0]] = 1
            else:
                self.analyse_host['Priority 3'].remove(self.analyse_host['Priority 1'][0])
                self.analyse_host['Priority 3'].append(self.analyse_host['Priority 1'][0])
                self.analyse_host['Priority 1'] = self.analyse_host['Priority 1'][1:]
        # 7.6 Analyse a priority 2 host and set the host to the end of priority 3
        elif len(self.analyse_host['Priority 2']) > 0:
            action = Analyse(session=0, agent=self.agent_name, hostname=self.analyse_host['Priority 2'][0])
            self.last_analysed = self.analyse_host['Priority 2'][0]
            if self.aggressive_analyse and self.aggressive_analyse_rep[1]>1:
                if self.analyse_host['Priority 2'][0] in list(self.analyse_counter.keys()):
                    self.analyse_counter[self.analyse_host['Priority 2'][0]] += 1
                    # remove from priority 2 if repetition limit has been reached
                    if self.analyse_counter[self.analyse_host['Priority 2'][0]] >= self.aggressive_analyse_rep[1]:
                        self.analyse_host['Priority 3'].remove(self.analyse_host['Priority 2'][0])
                        self.analyse_host['Priority 3'].append(self.analyse_host['Priority 2'][0])
                        self.analyse_host['Priority 2'] = self.analyse_host['Priority 2'][1:]
                else:
                    self.analyse_counter[self.analyse_host['Priority 2'][0]] = 1
            else:
                self.analyse_host['Priority 3'].remove(self.analyse_host['Priority 2'][0])
                self.analyse_host['Priority 3'].append(self.analyse_host['Priority 2'][0])
                self.analyse_host['Priority 2'] = self.analyse_host['Priority 2'][1:]
        # 7.7 Deploy a decoy on the last analysed server which does not hat a decoy or on a random host
        elif len(self.decoy_host) > 0:
            if self.last_analysed in self.decoy_host:
                action = DeployDecoy(session=0, agent=self.agent_name, hostname=self.last_analysed)
                self.decoy_host.remove(self.last_analysed)
            else:
                action = DeployDecoy(session=0, agent=self.agent_name, hostname=self.decoy_host[0])
                self.decoy_host = self.decoy_host[1:]
        # 7.8 Default Action is to analyse a host and set this host to the end of the list afterwards
        else:
            hostname = self._chosen_host_to_analyse()
            action = Analyse(session=0, agent=self.agent_name, hostname= hostname)
            self.last_analysed = hostname
            self.analyse_host['Priority 3'].remove(hostname)
            self.analyse_host['Priority 3'].append(hostname)

        
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
        elif isinstance(action, AllowTrafficZone):
            self.action_counter[5] += 1
        elif isinstance(action, BlockTrafficZone):
            self.action_counter[6] += 1

        # 9. Add step information
        self.info.append({"Name": self.agent_name,
                "Observation": obs,
                "Action": action})
        return action
        
