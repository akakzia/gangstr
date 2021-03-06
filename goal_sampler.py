import torch
import numpy as np
from utils import get_idxs_per_relation
from mpi4py import MPI

class GoalSampler:
    def __init__(self, args, policy):
        self.num_rollouts_per_mpi = args.num_rollouts_per_mpi
        self.rank = MPI.COMM_WORLD.Get_rank()

        self.goal_dim = args.env_params['goal']
        self.relation_ids = get_idxs_per_relation(n=args.n_blocks)

        self.discovered_goals = []
        self.discovered_goals_str = []

        self.algo = args.algo
        self.policy = policy
        self.init_stats()

    def sample_goal(self, agent_network, initial_obs):
        """

        """
        if self.algo == 'hme':
            return agent_network.sample_goal_uniform(1, use_oracle=False)[0]
        elif self.algo == 'value_disagreement':
            n = min(1000, len(agent_network.semantic_graph.configs))
            goals = np.array(agent_network.sample_goal_uniform(n, use_oracle=False))
            observation = np.repeat(np.expand_dims(initial_obs['observation'], axis=0), n, axis=0)
            ag = np.repeat(np.expand_dims(initial_obs['achieved_goal'], axis=0), n, axis=0)

            obs_norm = self.policy.o_norm.normalize(observation)
            g_norm = self.policy.g_norm.normalize(goals)
            ag_norm = self.policy.g_norm.normalize(ag)

            obs_norm_tensor = torch.tensor(obs_norm, dtype=torch.float32)
            g_norm_tensor = torch.tensor(g_norm, dtype=torch.float32)
            ag_norm_tensor = torch.tensor(ag_norm, dtype=torch.float32)

            raise NotImplementedError
        else:
            raise NotImplementedError

    def update(self, episodes, t):
        """
        Update discovered goals list from episodes
        Update list of successes and failures for LP curriculum
        Label each episode with the last ag (for buffer storage)
        """
        all_episodes = MPI.COMM_WORLD.gather(episodes, root=0)

        if self.rank == 0:
            all_episode_list = [e for eps in all_episodes for e in eps]

            for e in all_episode_list:
                # Add last achieved goal to memory if first time encountered
                if str(e['ag'][-1]) not in self.discovered_goals_str:
                    self.discovered_goals.append(e['ag'][-1].copy())
                    self.discovered_goals_str.append(str(e['ag'][-1]))

        self.sync()

        return episodes

    def sync(self):
        self.discovered_goals = MPI.COMM_WORLD.bcast(self.discovered_goals, root=0)
        self.discovered_goals_str = MPI.COMM_WORLD.bcast(self.discovered_goals_str, root=0)

    def build_batch(self, batch_size):
        goal_ids = np.random.choice(np.arange(len(self.discovered_goals)), size=batch_size)
        return goal_ids

    def init_stats(self):
        self.stats = dict()
        # Number of classes of eval
        if self.goal_dim == 30:
            n = 11
        else:
            n = 6
        for i in np.arange(1, n+1):
            self.stats['Eval_SR_{}'.format(i)] = []
            self.stats['# class_teacher {}'.format(i)] = []
            self.stats['# class_agent {}'.format(i)] = []
        self.stats['epoch'] = []
        self.stats['episodes'] = []
        self.stats['global_sr'] = []
        self.stats['nb_internalized'] = []
        self.stats['proposed_ss'] = []
        self.stats['proposed_beyond'] = []
        keys = ['goal_sampler', 'rollout', 'store', 'norm_update','update_graph',
                'policy_train', 'eval', 'epoch', 'total']
        for k in keys:
            self.stats['t_{}'.format(k)] = []

    def save(self, epoch, episode_count, av_res, global_sr, time_dict, goals_per_class, agent_stats, nb_internalized,
             proposed_ss, proposed_beyond):
        self.stats['epoch'].append(epoch)
        self.stats['episodes'].append(episode_count)
        self.stats['global_sr'].append(global_sr)
        for k in time_dict.keys():
            self.stats['t_{}'.format(k)].append(time_dict[k])
        self.stats['nb_internalized'].append(nb_internalized)
        self.stats['proposed_ss'].append(proposed_ss)
        self.stats['proposed_beyond'].append(proposed_beyond)
        for g_id in np.arange(1, len(av_res) + 1):
            self.stats['Eval_SR_{}'.format(g_id)].append(av_res[g_id-1])
        for k in goals_per_class.keys():
            self.stats['# class_teacher {}'.format(k)].append(goals_per_class[k])
            self.stats['# class_agent {}'.format(k)].append(agent_stats[k])
