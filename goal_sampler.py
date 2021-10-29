import random
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

        self.num_buckets = 11
        self.LP = np.zeros([self.num_buckets])
        self.C = np.zeros([self.num_buckets])
        self.p = np.zeros([self.num_buckets])

        self.self_eval_prob = 0.1  # probability to perform self evaluation
        self.queue_len = 200

        self.algo = args.algo
        self.policy = policy
        self.init_stats()

    def sample_goal(self, agent_network, initial_obs):
        """

        """
        if self.algo == 'hme':
            return agent_network.sample_goal_uniform(1, use_oracle=False)[0]
        elif self.algo == 'value_disagreement':
            raise NotImplementedError
        else:
            # Initialize LP probabilities
            if np.sum(self.p) == 0:
                self.initialize_p(agent_network)
            # decide whether to self evaluate
            self_eval = True if np.random.random() < self.self_eval_prob else False
            # if self-evaluation then sample randomly from discovered goals
            if self_eval:
                b_ind = np.random.choice(agent_network.active_buckets)
            # if no self evaluation
            else:
                b_ind = np.random.choice(range(self.num_buckets), p=self.p)
            goal = random.choices(agent_network.buckets[b_ind])[0]

            return goal, self_eval

    def initialize_p(self, agent_network):
        """
        Initializes probabilities once at least one bucket is created
        """
        for k in agent_network.active_buckets:
            self.p[k] = 1
        self.p = self.p / len(agent_network.active_buckets)

    def update_lp(self, agent_network):
        # compute C, LP per bucket
        for k in agent_network.active_buckets:
            n_points = min(len(agent_network.successes_and_failures[k]), self.queue_len)
            if n_points > 20: # 70
                sf = np.array(agent_network.successes_and_failures[k])
                self.C[k] = np.mean(sf[n_points // 2:])
                self.LP[k] = np.abs(np.sum(sf[n_points // 2:]) - np.sum(sf[: n_points // 2])) / n_points
            else:
                self.C[k] = 0
                self.LP[k] = 0

        # compute p
        if np.sum(self.LP) == 0:
            self.initialize_p(agent_network)
        else:
            self.p = self.LP / self.LP.sum()

        if self.p.sum() > 1:
            self.p[np.argmax(self.p)] -= self.p.sum() - 1
        elif self.p.sum() < 1:
            self.p[-1] = 1 - self.p[:-1].sum()

    def sync(self):
        self.p = MPI.COMM_WORLD.bcast(self.p, root=0)
        self.LP = MPI.COMM_WORLD.bcast(self.LP, root=0)
        self.C = MPI.COMM_WORLD.bcast(self.C, root=0)

    def build_batch(self, batch_size):
        goal_ids = np.random.choice(np.arange(len(self.discovered_goals)), size=batch_size)
        return goal_ids

    def init_stats(self):
        self.stats = dict()
        for i in np.arange(1, self.num_buckets+1):
            self.stats['Eval_SR_{}'.format(i)] = []
            self.stats['# class_teacher {}'.format(i)] = []
            self.stats['# class_agent {}'.format(i)] = []
        for i in range(self.num_buckets):
            self.stats['B_{}_LP'.format(i+1)] = []
            self.stats['B_{}_C'.format(i+1)] = []
            self.stats['B_{}_p'.format(i+1)] = []
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

        for i in range(self.num_buckets):
            self.stats['B_{}_LP'.format(i+1)].append(self.LP[i])
            self.stats['B_{}_C'.format(i+1)].append(self.C[i])
            self.stats['B_{}_p'.format(i+1)].append(self.p[i])
