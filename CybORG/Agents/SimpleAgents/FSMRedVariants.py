from CybORG.Agents.SimpleAgents.FiniteStateRedAgent import FiniteStateRedAgent

""" *Creating Variant Red Agents*

```python

class MyVariant(FiniteStateRedAgent):
    def __init__(self, name=None, np_random=None, agent_subnets=None):
        super().__init__(name=name, np_random=np_random, agent_subnets=agent_subnets)

        # Changable variables:
        self.print_action_output = False
        self.print_obs_output = False
        self.prioritise_servers = False

    def _set_host_state_priority_list(self):
        # percentage choice
        new_host_state_priority_list = {'K':(0->100), 'KS':?, 'KD':?, 'U':?, 'UD':?, 'R':?, 'RD':?}
        return None
    
    def _state_transitions_probability(self):
        # Create new probability mapping to use
        map = {
            'K'  : [0.5,  0.25, 0.25, None, None, None, None, None, None],
            'KD' : [None, 0.5,  0.5,  None, None, None, None, None, None],
            'S'  : [0.25, None, None, 0.25, 0.5 , None, None, None, None],
            'SD' : [None, None, None, 0.25, 0.75, None, None, None, None],
            'U'  : [0.5 , None, None, None, None, 0.5 , None, None, 0.0 ],
            'UD' : [None, None, None, None, None, 1.0 , None, None, 0.0 ],
            'R'  : [0.5,  None, None, None, None, None, 0.25, 0.25, 0.0 ],
            'RD' : [None, None, None, None, None, None, 0.5,  0.5,  0.0 ],
        }
        return map
```
"""

class VerboseFSRed(FiniteStateRedAgent):
    """A variant of the FiniteStateRedAgent that outputs success, action and internal observation knowlege to the terminal.
    
    Example:
    ```
    ** Turn 0 for red_agent_0 **
    Action: Initial Observation
    Action Success: UNKNOWN

    Observation:
    {'contractor_network_subnet_user_host_5': {
        'Interface': [{'Subnet': IPv4Network('10.0.171.0/24'),
                        'interface_name': 'eth0',
                        'ip_address': IPv4Address('10.0.171.186')}],
        'Processes': [{'PID': 8888,
                        'username': 'ubuntu'}],
        'Sessions': [{'PID': 8888,
                        'Type': <SessionType.RED_ABSTRACT_SESSION: 10>,
                        'agent': 'red_agent_0',
                        'session_id': 0,
                        'timeout': 0,
                        'username': 'ubuntu'}],
        'System info': {'Architecture': <Architecture.x64: 2>,
                        'Hostname': 'contractor_network_subnet_user_host_5',
                        'OSDistribution': <OperatingSystemDistribution.UBUNTU: 8>,
                        'OSType': <OperatingSystemType.LINUX: 3>,
                        'OSVersion': <OperatingSystemVersion.UNKNOWN: 1>,
                        'position': array([0., 0.])},
        'User Info': [{'Groups': [{'GID': 0}],
                        'username': 'root'},
                        {'Groups': [{'GID': 1}],
                        'username': 'user'}]}}
    Host States:
    {'10.0.171.186': {'hostname': 'contractor_network_subnet_user_host_5',
                    'state': 'U'}}
    ```
    """
    def __init__(self, name=None, np_random=None, agent_subnets=None):
        super().__init__(name=name, np_random=np_random, agent_subnets=agent_subnets)
        self.print_action_output = True
        self.print_obs_output = True


class DiscoveryFSRed(FiniteStateRedAgent):
    """An FiniteStateRedAgent variant that aims to prioritise discovery."""
    def __init__(self, name=None, np_random=None, agent_subnets=None):
        super().__init__(name=name, np_random=np_random, agent_subnets=agent_subnets)
        self.print_action_output = False
        self.print_obs_output = False
        self.prioritise_servers = True

    def set_host_state_priority_list(self):
        """Returns a custom host priority list, optimised for discovery.
        
        Returns
        -------
        host_state_priority_list : Dict[str, num]
        """
        host_state_priority_list = {
            'K':20, 'KD':20, 
            'S':20, 'SD':20,
            'U':10, 'UD':10, 
            'R':0,  'RD':0
        }
        return host_state_priority_list
    
    def state_transitions_probability(self):
        """Returns a custom state transitions probability matrix, optimised for discovery.

        Returns
        -------
        matrix : Dict[str, List[float]]
        """

        map = {
            'K'  : [0.25, 0.75, 0.0,  None, None, None, None, None, None],
            'KD' : [None, 1.0,  0.0,  None, None, None, None, None, None],
            'S'  : [0.25, None, None, 0.0,  0.75, None, None, None, None],
            'SD' : [None, None, None, 0.0,  1.0,  None, None, None, None],
            'U'  : [0.0 , None, None, None, None, 1.0 , None, None, 0.0 ],
            'UD' : [None, None, None, None, None, 1.0 , None, None, 0.0 ],
            'R'  : [1.0,  None, None, None, None, None, 0.0,  0.0,  0.0 ],
            'RD' : [None, None, None, None, None, None, 0.5,  0.5,  0.0 ],
        }

        return map

    class StealthPivotFSRed(FiniteStateRedAgent):
        """
        Profile: Stealthy lateral movement.
        - Prefers StealthServiceDiscovery over AggressiveServiceDiscovery.
        - Prefers discovery/pivoting; delays noisy disruptive actions.
        - Mild chance to Withdraw from rooted hosts (simulate OPSEC / cleanup).
        """

        def __init__(self, name=None, np_random=None, agent_subnets=None):
            super().__init__(name=name, np_random=np_random, agent_subnets=agent_subnets)
            self.print_action_output = False
            self.print_obs_output = False
            self.prioritise_servers = True

        def set_host_state_priority_list(self):
            # Focus on progressing unknown/partially-known hosts; keep rooted hosts lower to avoid rushing impact
            return {
                'K': 25, 'KD': 25,
                'S': 15, 'SD': 15,
                'U': 10, 'UD': 10,
                'R': 0, 'RD': 0
            }

        def state_transitions_probability(self):
            # Key idea:
            # - In K/KD: prefer discovery then stealth service scan.
            # - In S/SD: prefer deception check + exploit.
            # - In U/UD: privilege escalate.
            # - In R/RD: pivot discovery; sometimes withdraw.
            return {
                'K': [0.35, 0.05, 0.60, None, None, None, None, None, None],
                'KD': [None, 0.05, 0.95, None, None, None, None, None, None],

                'S': [0.10, None, None, 0.30, 0.60, None, None, None, None],
                'SD': [None, None, None, 0.30, 0.70, None, None, None, None],

                'U': [None, None, None, None, None, 1.00, None, None, 0.00],
                'UD': [None, None, None, None, None, 1.00, None, None, 0.00],

                'R': [0.85, None, None, None, None, None, 0.05, 0.05, 0.05],
                'RD': [0.85, None, None, None, None, None, 0.05, 0.05, 0.05],
            }

    class ImpactRushFSRed(FiniteStateRedAgent):
        """
        Profile: Disrupt fast once privileged foothold exists.
        - Uses quicker/noisier discovery (AggressiveServiceDiscovery).
        - Prioritises servers; quickly escalates and then Impact/Degrade.
        - Minimal Withdraw.
        """

        def __init__(self, name=None, np_random=None, agent_subnets=None):
            super().__init__(name=name, np_random=np_random, agent_subnets=agent_subnets)
            self.print_action_output = False
            self.print_obs_output = False
            self.prioritise_servers = True

        def set_host_state_priority_list(self):
            # Push hard toward escalation and exploitation; revisit rooted states frequently to impact/degrade
            return {
                'K': 10, 'KD': 10,
                'S': 15, 'SD': 15,
                'U': 20, 'UD': 20,
                'R': 5, 'RD': 5
            }

        def state_transitions_probability(self):
            return {
                # Get services quickly (noisy)
                'K': [0.10, 0.85, 0.05, None, None, None, None, None, None],
                'KD': [None, 0.90, 0.10, None, None, None, None, None, None],

                # Skip deception most of the time; exploit aggressively
                'S': [0.05, None, None, 0.05, 0.90, None, None, None, None],
                'SD': [None, None, None, 0.05, 0.95, None, None, None, None],

                # Escalate ASAP
                'U': [None, None, None, None, None, 1.00, None, None, 0.00],
                'UD': [None, None, None, None, None, 1.00, None, None, 0.00],

                # Disrupt as soon as root exists
                'R': [0.05, None, None, None, None, None, 0.55, 0.40, 0.00],
                'RD': [0.05, None, None, None, None, None, 0.55, 0.40, 0.00],
            }

    class DeceptionAwareFSRed(FiniteStateRedAgent):
        """
        Profile: Decoy-aware attacker.
        - Uses deception discovery frequently before exploit.
        - Mix of stealth/aggressive discovery.
        - Some Withdraw to avoid persistent interaction with decoys / reduce footprint.
        """

        def __init__(self, name=None, np_random=None, agent_subnets=None):
            super().__init__(name=name, np_random=np_random, agent_subnets=agent_subnets)
            self.print_action_output = False
            self.print_obs_output = False
            self.prioritise_servers = True

        def set_host_state_priority_list(self):
            # Strong emphasis on service-known states (S/SD) to repeatedly check deception before exploit
            return {
                'K': 15, 'KD': 15,
                'S': 25, 'SD': 25,
                'U': 10, 'UD': 10,
                'R': 0, 'RD': 0
            }

        def state_transitions_probability(self):
            return {
                'K': [0.30, 0.25, 0.45, None, None, None, None, None, None],
                'KD': [None, 0.30, 0.70, None, None, None, None, None, None],

                # First check deception often; only then exploit
                'S': [0.05, None, None, 0.60, 0.35, None, None, None, None],
                'SD': [None, None, None, 0.60, 0.40, None, None, None, None],

                'U': [None, None, None, None, None, 1.00, None, None, 0.00],
                'UD': [None, None, None, None, None, 1.00, None, None, 0.00],

                # Once root: still prefer pivoting and some cleanup
                'R': [0.60, None, None, None, None, None, 0.10, 0.10, 0.20],
                'RD': [0.60, None, None, None, None, None, 0.10, 0.10, 0.20],
            }

    class LateralSpreadFSRed(FiniteStateRedAgent):
        """
        Profile: "server_host_0 takeover" + cross-subnet spread.
        - Emphasises DiscoverRemoteSystems (0) whenever possible (especially after root),
          aiming to exploit the CC4 mechanic where rooted server_host_0 reveals other server_0s.
        - Less focus on Impact; this is a spread-and-persist adversary.
        """

        def __init__(self, name=None, np_random=None, agent_subnets=None):
            super().__init__(name=name, np_random=np_random, agent_subnets=agent_subnets)
            self.print_action_output = False
            self.print_obs_output = False
            self.prioritise_servers = True

        def set_host_state_priority_list(self):
            # Very heavy on K/KD to keep discovering new hosts/subnets, and on U/UD to quickly reach root
            return {
                'K': 30, 'KD': 30,
                'S': 10, 'SD': 10,
                'U': 10, 'UD': 10,
                'R': 0, 'RD': 0
            }

        def state_transitions_probability(self):
            return {
                # Always try to expand known hosts list
                'K': [0.70, 0.20, 0.10, None, None, None, None, None, None],
                'KD': [None, 0.25, 0.75, None, None, None, None, None, None],

                # Minimal deception checks; exploit to get footholds
                'S': [0.40, None, None, 0.05, 0.55, None, None, None, None],
                'SD': [None, None, None, 0.05, 0.95, None, None, None, None],

                # Escalate quickly to root (to unlock server_host_0 spillover knowledge)
                'U': [None, None, None, None, None, 1.00, None, None, 0.00],
                'UD': [None, None, None, None, None, 1.00, None, None, 0.00],

                # After root: keep discovering/pivoting; very little disruption
                'R': [0.90, None, None, None, None, None, 0.02, 0.03, 0.05],
                'RD': [0.90, None, None, None, None, None, 0.02, 0.03, 0.05],
            }
