import numpy as np

from CybORG.Simulator.Actions import Monitor , Analyse, DeployDecoy, Remove, Restore
from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import AllowTrafficZone, BlockTrafficZone 
from CybORG.Shared.Enums import TernaryEnum


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
- dont analyse a router just after setting a decoy

Goal: 
- mean reward above -100 with small deviation
- RL agent based on final heuristic version to find out if it can beat heuristic agent
"""

class H_Agent():

    def __init__(self, agent_name, init_obs = None):
        self.agent_name = agent_name
        self._reset_agents()

    def _get_responisble_hosts(self, obs):
        return [key for key in obs.keys() if key != 'success']
    
    def _reset_agents(self):
        self.observations = np.array([])
        self.actions = np.array([])
        self.hosts = np.array([])
        self.status = 'starting' #other states could be 'waiting', 'acting', 'searching', 'preventing'
        self.analyse_host = {'Priority 1': np.array([]), # double event
                             'Priority 2' : np.array([]), # normal event
                             'Priority 3' : np.array([])} #default action
        self.info = np.array([])
        self.restore_host = np.array([]) #Hosts which have to be restored
        self.decoy_host = np.array([]) # hosts which have to be decoyed
        self.remove_host = np.array([]) #hosts which have a file to remove
        self.last_analysed = None #last analysed host
        self.action_counter = np.array([0,0,0,0,0]) #Monitor, Analyse, DeployDecoy, Remove, Restore

    def get_action(self, obs, action_space):
        # Reset the agent if a new episode starts (necessary für submission, because the queues are not reset automatically)
        if obs['success'] == TernaryEnum.UNKNOWN and np.sum(self.action_counter) > 0:
            self._reset_agents()
        # Init on first observation
        if self.observations.size == 0:
            self.hosts = np.array(self._get_responisble_hosts(obs))
            self.analyse_host['Priority 3'] = np.array(self._get_responisble_hosts(obs))
            self.decoy_host = np.array(self._get_responisble_hosts(obs))
            self.last_analysed = self.hosts[0]

        # Store observation
        self.observations = np.append(self.observations, obs)
        
        #first action is always Monitor
        if self.actions.size == 0 or obs['success'] == TernaryEnum.IN_PROGRESS:
            action = Monitor(0, self.agent_name)
            self.action_counter[0] += 1
            self.status = 'waiting'
            self.actions = np.append(self.actions, action)
            self.info = np.append(self.info, [{"Name": self.agent_name,
                "Status": self.status,
                "Observation": obs,
                "Action": action}])
            return action

        # Check if any event has been detected
        events = np.intersect1d(self.hosts, list(obs.keys()))
        if events.size > 0:
            #print(f"An event has been detected for agent {self.agent_name}\n")

        # 1. Check for malware
            for host in events:
                if 'Files' in obs[host]:
                    for file in obs[host]['Files']:
                        if file["File Name"] =="cmd.sh":
                            self.remove_host = np.append(self.remove_host, host)
                        elif file["File Name"] =="escalate.sh":
                            self.restore_host = np.append(self.restore_host, host)
            

        # 2. Check for decoy exploit and list for restore if found
                if 'Processes' in obs[host]:
                    for process in obs[host]['Processes']:
                        if 'Connections' in process:
                            if 'local_port' in process['Connections']:
                                if process['Connections']['local_port'] == 25:
                                    self.restore_host = np.append(self.restore_host, host)
                                    self.decoy_host = np.append(self.decoy_host, host) 

        # 3. Check multiple events on same host or a simultaneous router and server connection. They will be prioritised in analysing if they are not listed in restore actions
        values, counts = np.unique(events, return_counts=True)
        double_events = values[counts > 1]
        double_events = np.setdiff1d(double_events, self.restore_host)
        if any("server" in event for event in events) and any("router" in event for event in events):
            double_events = np.append(double_events, [event for event in events if "server" in event or "router" in event])
        double_events = np.unique(double_events)

        # 4. if a host has to be restored, remove the event from the remove action list and eliminate double events in remove and restore action 
        self.remove_host = np.unique(self.remove_host)
        self.restore_host = np.unique(self.restore_host)
        self.remove_host = np.setdiff1d(self.remove_host, self.restore_host)

        # 5. find remaining events
        events = np.setdiff1d(events, self.restore_host)
        events = np.setdiff1d(events, self.remove_host)
        events = np.setdiff1d(events, double_events)

        # 6. update analysation priorities
        # 6.1 Priority 1 for double occuring events
        for double_event in double_events:
            if double_event not in self.analyse_host['Priority 1']:
                self.analyse_host['Priority 1'] = np.append(self.analyse_host['Priority 1'], double_event)
        # 6.2 Priority 2 for single detected event
        for event in events:
            if event not in self.analyse_host['Priority 2']:
                self.analyse_host['Priority 2'] = np.append(self.analyse_host['Priority 2'], event)
        # 6.3 Remove the events from Priority 2 which are in Priority 1
        self.analyse_host['Priority 2'] = np.setdiff1d(self.analyse_host['Priority 2'], self.analyse_host['Priority 1'])

        """Find next Action based on Priority:
        1. Restore server
        2. Remove file
        3. Analyse a Prioriy 1 or 2 host
            3.1 once a priority 1 or 2 element has been analysed it is set to the bottom of priority 3 
        4. set decoy on a detected event with no decoy or last analysed event
        5. analyse priority 3 host
        """

        # 7. Select action
        # 7.1 restore a server and remove the host from the list afterwards
        if self.restore_host.size > 0:
            action = Restore(session=0, agent=self.agent_name, hostname=self.restore_host[0])
            self.restore_host =  self.restore_host[1:]
            self.status = "acting"
        # 7.2 remove a server and remove the host from the list afterwards
        elif self.remove_host.size > 0:
            action = Remove(session=0, agent=self.agent_name, hostname=self.remove_host[0])
            self.remove_host = self.remove_host[1:]
            self.status = "acting"
        # 7.3 Analyse a priority 1 host and set the host to the end of priority 3
        elif self.analyse_host['Priority 1'].size > 0:
            action = Analyse(session=0, agent=self.agent_name, hostname=self.analyse_host['Priority 1'][0])
            self.last_analysed = self.analyse_host['Priority 1'][0]
            self.analyse_host['Priority 3'] = np.setdiff1d(self.analyse_host['Priority 3'], [self.analyse_host['Priority 1'][0]])
            self.analyse_host['Priority 3'] = np.append(self.analyse_host['Priority 3'], self.analyse_host['Priority 1'][0])
            self.analyse_host['Priority 1'] = self.analyse_host['Priority 1'][1:]
            self.status = "searching"
        # 7.4 Analyse a priority 2 host and set the host to the end of priority 3
        elif self.analyse_host['Priority 2'].size > 0:
            action = Analyse(session=0, agent=self.agent_name, hostname=self.analyse_host['Priority 2'][0])
            self.last_analysed = self.analyse_host['Priority 2'][0]
            self.analyse_host['Priority 3'] = np.setdiff1d(self.analyse_host['Priority 3'], [self.analyse_host['Priority 2'][0]])
            self.analyse_host['Priority 3'] = np.append(self.analyse_host['Priority 3'], self.analyse_host['Priority 2'][0])
            self.analyse_host['Priority 2'] = self.analyse_host['Priority 2'][1:]
            self.status = "searching"
        # 7.5 Deploy a decoy on the last analysed server which does not hat a decoy or on a random host
        elif self.decoy_host.size > 0:
            if self.last_analysed in self.decoy_host:
                action = DeployDecoy(session=0, agent=self.agent_name, hostname=self.last_analysed)
                self.decoy_host = np.setdiff1d(self.decoy_host, [self.last_analysed])
            else:
                action = DeployDecoy(session=0, agent=self.agent_name, hostname=self.decoy_host[0])
                self.decoy_host = self.decoy_host[1:]
            self.status = "preventing"
        # 7.6 Defaul Action is to anlyse a host and set this host to the end of the list afterwards
        else:
            action = Analyse(session=0, agent=self.agent_name, hostname=self.analyse_host['Priority 3'][0])
            self.last_analysed = self.analyse_host['Priority 3'][0]
            self.analyse_host['Priority 3'] = np.append(self.analyse_host['Priority 3'], self.analyse_host['Priority 3'][0])
            self.analyse_host['Priority 3'] = self.analyse_host['Priority 3'][1:]
            self.status = "searching"

        
        # 8. Store and count action selection
        self.actions = np.append(self.actions, action)
        if isinstance(action, Analyse):
            self.action_counter[1] += 1
        elif isinstance(action, DeployDecoy):
            self.action_counter[2] += 1
        elif isinstance(action, Remove):
            self.action_counter[3] += 1
        elif isinstance(action, Restore):
            self.action_counter[4] += 1

        # Add step information
        self.info = np.append(self.info, [{"Name": self.agent_name,
                "Status": self.status,
                "Observation": obs,
                "Action": action}])
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
    
    """
    How to use Control Traffic action?
        The Control Traffic Action can Block/allow the communication from one subnet to another.
        
        """