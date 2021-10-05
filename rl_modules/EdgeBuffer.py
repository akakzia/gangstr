
from collections import defaultdict
import threading
import numpy as np
import random
from graph.SemanticOperation import SemanticOperation,config_to_unique_str

class EdgeBuffer:
    def __init__(self, env_params, buffer_size, sample_func, replay_sampling, goal_sampler,args):
        self.env_params = env_params
        self.T = args.episode_duration
        self.size = buffer_size // self.T
        self.replay_sampling = replay_sampling
        self.goal_sampler = goal_sampler

        self.sample_func = sample_func

        self.current_size = 0

        # create the buffer to store info
        self.buffer = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                       'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                       'g': np.empty([self.size, self.T, self.env_params['goal']]),
                       'actions': np.empty([self.size, self.T, self.env_params['action']]),
                       }
        self.edges_to_infos = defaultdict(lambda : {'edge_dist':None,'episode_ids':[]}) # edges : {dist:2, episode_ids:[ 0,12]}
        self.all_edges = []
        # thread lock
        self.lock = threading.Lock()


    # store the episode
    def store_episode(self, episode_batch):
        batch_size = len(episode_batch)
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)

            for i, e in enumerate(episode_batch):
                # update edge infos in buffer :
                edge = (tuple(e['ag'][0]),tuple(e['ag'][-1]))
                self.edges_to_infos[edge]['episode_ids'].append(idxs[i])
                self.edges_to_infos[edge]['edge_dist'] = e['edge_dist']
                last_stored_edge = (tuple(self.buffer['ag'][idxs[i]][0]),tuple(self.buffer['ag'][idxs[i]][-1]))
                if idxs[i] in self.edges_to_infos[last_stored_edge] : 
                    self.edges_to_infos[edge]['episode_ids'].remove(idxs[i])
                if len(self.edges_to_infos[last_stored_edge]['episode_ids']) == 0:
                    del self.edges_to_infos[last_stored_edge]
                # store the episode in buffer
                self.buffer['obs'][idxs[i]] = e['obs']
                self.buffer['ag'][idxs[i]] = e['ag']
                self.buffer['g'][idxs[i]] = e['g']
                self.buffer['actions'][idxs[i]] = e['act']
                # self.goal_ids[idxs[i]] = e['last_ag_oracle_id']
                if 'language_goal' in e.keys():
                    # self.buffer['language_goal'][idxs[i]] = e['language_goal']
                    self.buffer['lg_ids'][idxs[i]] = e['lg_ids']
            self.all_edges = list(self.edges_to_infos)   # use separate buffer for edge sampling

    def sample_edge(self,batch_size):
        return random.choices(self.all_edges,k=batch_size)

    def distance_biased_sample_edge(self,batch_size):
        edges,edges_infos = zip(*self.edges_to_infos.items())
        distances = [edges_info['edge_dist'] for edges_info in edges_infos]
        return random.choices(edges,distances,k=batch_size)
    
    def sample_transition(self,batch_size):
        temp_buffers = {}
        with self.lock:
            if self.replay_sampling == 'buffer_uniform':
                # ep_ids = np.random.randint(0,self.current_size,size=batch_size)
                # for key in self.buffer.keys():
                #     temp_buffers[key] = self.buffer[key][ep_ids]
                # random selection of indexes is already done in sample_function
                for key in self.buffer.keys():
                    temp_buffers[key] = self.buffer[key][:self.current_size]
            else:
                if self.replay_sampling == 'edge_uniform':
                    edges = self.sample_edge(batch_size)
                elif self.replay_sampling == 'edge_distance':
                    edges = self.distance_biased_sample_edge(batch_size)
                else:
                    raise Exception('unknow replay method')
                ep_ids = np.zeros(batch_size,dtype=int)
                for i,edge in enumerate(edges):
                    ep_ids[i] = np.random.choice(self.edges_to_infos[edge]['episode_ids'])
                for key in self.buffer.keys():
                    temp_buffers[key] = self.buffer[key][ep_ids]

        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]

         # HER Re-Labelling : 
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def sample(self,batch_size):
        return self.sample_transition(batch_size)
        
    def get_nb_edges(self):
        return len(self.edges_to_infos)

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = [idx[0]]
        return idx
