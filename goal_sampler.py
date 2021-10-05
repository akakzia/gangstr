from collections import deque
import numpy as np
from utils import get_idxs_per_relation
from mpi4py import MPI
import os
import pickle
import pandas as pd
from mpi_utils import logger


ALL_MASKS = True


class GoalSampler:
    def __init__(self, args):
        self.num_rollouts_per_mpi = args.num_rollouts_per_mpi
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.use_masks = args.masks
        self.mask_application = args.mask_application

        self.goal_dim = args.env_params['goal']
        self.relation_ids = get_idxs_per_relation(n=args.n_blocks)
        # if ALL_MASKS:
        #     self.masks_list = [np.array([1, 0, 0, 1, 0, 1, 0, 0, 0]), np.array([0, 1, 0, 0, 1, 0, 0, 1, 0]),
        #                        np.array([0, 0, 1, 0, 0, 0, 1, 0, 1]),
        #                        np.array([1, 1, 0, 1, 1, 1, 0, 1, 0]), np.array([1, 0, 1, 1, 0, 1, 1, 0, 1]),
        #                        np.array([0, 1, 1, 0, 1, 0, 1, 1, 1]),
        #                        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])]
        # # Test only simple masks in training
        # else:
        #     self.masks_list = [np.array([1, 1, 0, 1, 1, 1, 1, 0, 0]), np.array([1, 0, 1, 1, 1, 0, 0, 1, 1]),
        #                        np.array([0, 1, 1, 0, 0, 1, 1, 1, 1])]

        # self.n_masks = len(self.masks_list)

        self.discovered_goals = []
        self.discovered_goals_str = []

        self.init_stats()

    def sample_masks(self, n):
        """Samples n masks uniformly"""
        if not self.use_masks:
            # No masks
            return np.zeros((n, self.goal_dim))
        masks = np.zeros((n, self.goal_dim))
        # Select number of masks to apply per goal
        n_masks = np.random.randint(self.relation_ids.shape[0], size=n)
        # Get idxs to be masked
        relations_to_mask = [np.random.choice(np.arange(self.relation_ids.shape[0]), size=i, replace=False) for i in n_masks]
        re = [np.concatenate(self.relation_ids[r]) if self.relation_ids[r].shape[0] > 0 else None for r in relations_to_mask]
        # apply masks
        for mask, ids_to_mask in zip(masks, re):
            if ids_to_mask is not None:
                mask[ids_to_mask] = 1
        return masks

    def sample_goal(self, n_goals, evaluation):
        """
        Sample n_goals goals to be targeted during rollouts
        evaluation controls whether or not to sample the goal uniformly or according to curriculum
        """
        if evaluation and len(self.discovered_goals) > 0:
            goals = np.random.choice(self.discovered_goals, size=self.num_rollouts_per_mpi)
            masks = np.zeros((n_goals, self.goal_dim))
            self_eval = False
        else:
            if len(self.discovered_goals) == 0:
                goals = np.random.choice([-1., 1.], size=(n_goals, self.goal_dim))
                masks = np.zeros((n_goals, self.goal_dim))
                # masks = np.random.choice([0., 1.], size=(n_goals, self.goal_dim))
                self_eval = False
            # if no curriculum learning
            else:
                # sample uniformly from discovered goals
                goal_ids = np.random.choice(range(len(self.discovered_goals)), size=n_goals)
                goals = np.array(self.discovered_goals)[goal_ids]
                masks = self.sample_masks(n_goals)
                # masks = np.array(self.masks_list)[np.random.choice(range(self.n_masks), size=n_goals)]
                self_eval = False
        return goals, masks, self_eval

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
                if str(e['ag_binary'][-1]) not in self.discovered_goals_str:
                    self.discovered_goals.append(e['ag_binary'][-1].copy())
                    self.discovered_goals_str.append(str(e['ag_binary'][-1]))

        self.sync()

        # Apply masks
        for e in episodes:
            if self.mask_application == 'hindsight':
                e['g'] = e['g'] * (1 - e['masks'][0]) + e['ag'][:-1] * e['masks'][0]
            elif self.mask_application == 'initial':
                e['g'] = e['g'] * (1 - e['masks'][0]) + e['ag'][0] * e['masks'][0]
            elif self.mask_application == 'opaque':
                e['g'] = e['g'] * (1 - e['masks'][0]) - 10 * e['masks'][0]
            else:
                raise NotImplementedError

        return episodes

    def generate_eval_goals(self):
        """ Generates a set of goals for evaluation. This set comprises :
        - One relation with close == True .
        - One relation with above == True
        - Two relations with close == True in one of them
        - Two relations with close == True in both of them
        - Two relations with above == True in one and close == False in the other
        - Two relations with above == True in one and close == True in the other
        - Two relations with above == True in one and above == True in the other
        - Three whole relations for the 7 above cases"""
        if self.use_masks:
            masks = np.array([np.array([0, 1, 1, 0, 1, 0, 1, 1, 1]), np.array([0, 1, 1, 0, 1, 0, 1, 1, 1]),
                              np.array([0, 0, 1, 0, 0, 0, 1, 0, 1]), np.array([0, 0, 1, 0, 0, 0, 1, 0, 1]),
                              np.array([0, 0, 1, 0, 0, 0, 1, 0, 1]), np.array([0, 0, 1, 0, 0, 0, 1, 0, 1]),
                              np.array([0, 1, 0, 0, 1, 0, 0, 1, 0]),
                              np.zeros(9), np.zeros(9), np.zeros(9), np.zeros(9), np.zeros(9)])
        else:
            masks = np.zeros((12, 9))
        gs = np.array([np.array([1., -10., -10., -1., -10., -1., -10., -10., -10.]), np.array([1., -10., -10., 1., -10., -1., -10., -10., -10.]),

                       np.array([1., -1., -10., -1., -1., -1., -10., -1., -10.]), np.array([1., 1., -10., -1., -1., -1., -10., -1., -10.]),
                       np.array([1., -1., -10., -1., -1., 1., -10., -1., -10.]), np.array([1., 1., -10., -1., 1., -1., -10., -1., -10.]),
                       np.array([1., -10., 1., 1., -10., -1., 1., -10., -1.]),

                       np.array([1., -1., -1., -1., -1., -1., -1., -1., -1.]), np.array([1., -1., -1., 1., -1., -1., -1., -1., -1.]),

                       np.array([1., 1., -1., -1., -1., -1., -1., -1., -1.]),
                       np.array([1., 1., 1., -1., -1., 1., -1., -1., -1.]),
                       np.array([1., -1., 1., 1., -1., -1., 1., -1., -1.])
                       ])
        return gs, masks

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
            self.stats['Av_Rew_{}'.format(i)] = []
            self.stats['# class_teacher {}'.format(i)] = []
            self.stats['# class_agent {}'.format(i)] = []
        self.stats['epoch'] = []
        self.stats['episodes'] = []
        self.stats['global_sr'] = []
        self.stats['nb_discovered'] = []
        keys = ['goal_sampler', 'rollout', 'gs_update', 'store', 'norm_update','update_graph',
                'policy_train', 'eval', 'epoch', 'total']
        for k in keys:
            self.stats['t_{}'.format(k)] = []

    def save(self, epoch, episode_count, av_res, av_rew, global_sr, time_dict, goals_per_class, agent_stats):
        self.stats['epoch'].append(epoch)
        self.stats['episodes'].append(episode_count)
        self.stats['global_sr'].append(global_sr)
        for k in time_dict.keys():
            self.stats['t_{}'.format(k)].append(time_dict[k])
        self.stats['nb_discovered'].append(len(self.discovered_goals))
        for g_id in np.arange(1, len(av_res) + 1):
            self.stats['Eval_SR_{}'.format(g_id)].append(av_res[g_id-1])
            self.stats['Av_Rew_{}'.format(g_id)].append(av_rew[g_id-1])
        for k in goals_per_class.keys():
            self.stats['# class_teacher {}'.format(k)].append(goals_per_class[k])
            self.stats['# class_agent {}'.format(k)].append(agent_stats[k])
