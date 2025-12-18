"""
memory_buffer.py — PPO Experience Buffer for Multi-Agent RL in CybORG

This module defines memory classes used to store rollout experiences for PPO training.
It supports both single-agent (PPOMemory) and multi-agent (MultiPPOMemory) training setups.

Classes:
- PPOMemory: Stores observations, actions, values, log probs, rewards, and terminals for a single agent.
- MultiPPOMemory: Wraps multiple PPOMemory buffers (one per agent) for centralized PPO training.
"""

import torch 

class PPOMemory:
    '''
    Holds memories for agents that are relevant to the 
    PPO optimization procedure
    '''
    def __init__(self, bs):
        self.s = []
        self.a = []
        self.v = []
        self.p = []
        self.r = []
        self.t = []

        self.bs = bs 

    def remember(self, s,a,v,p,r,t):
        '''
        Pushes new memory into the buffer 

        Args:
            s: State
            a: Action
            v: Value (critic output)
            p: Log Prob (actor output)
            r: Reward
            t: Terminal 
        '''
        self.s.append(s)
        self.a.append(a)
        self.v.append(v)
        self.p.append(p)
        self.r.append(r) 
        self.t.append(t)

    def clear(self): 
        '''
        Empties the memory buffer 
        '''
        self.s = []; self.a = []
        self.v = []; self.p = []
        self.r = []; self.t = []

    def get_batches(self):
        '''
        Return chunks of the shuffled memory buffer 
        randomly partitioned into `self.bs`-sized chunks 
        '''
        idxs = torch.randperm(len(self.a))
        batch_idxs = idxs.split(self.bs)

        return self.s, self.a, self.v, \
            self.p, self.r, self.t, batch_idxs


class MultiPPOMemory:
    '''
    Store multiple memory buffers, one for each agent. 
    Used during training to keep agent's observations seperated 
    '''
    def __init__(self, bs, agents=5) -> None:
        self.tot = agents 
        self.bs = bs 
        self.mems = [PPOMemory(bs) for _ in range(agents)]

    def remember(self, idx, *args):
        self.mems[idx].remember(*args)

    def clear(self):
        [mem.clear() for mem in self.mems]

    def get_batches(self):
        """
        Flatten all memories in self.mems into one big buffer,
        then return shuffled indices split into bs-sized chunks.
        """
        offset = 0
        idxs = []
        all_s = [];
        all_a = []
        all_v = [];
        all_p = []
        all_r = [];
        all_t = []

        # ❗ IMPORTANT: iterate over self.mems, not range(self.tot)
        for mem in self.mems:
            all_s += mem.s
            all_a += mem.a
            all_v += mem.v
            all_p += mem.p
            all_r += mem.r
            all_t += mem.t

            cnt = len(mem.s)
            if cnt == 0:
                continue  # just in case

            # random permutation for this mem, shifted by offset into global index space
            idx = torch.randperm(cnt) + offset
            idxs += list(idx.split(self.bs))
            offset += cnt

        return all_s, all_a, all_v, all_p, all_r, all_t, idxs

