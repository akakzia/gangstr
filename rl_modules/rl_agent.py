import torch
import numpy as np
from mpi_utils.mpi_utils import sync_networks
from rl_modules.replay_buffer import ReplayBuffer
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler
from updates import update_networks


"""
SAC with HER (MPI-version)
"""

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class RLAgent:
    def __init__(self, args, compute_rew, goal_sampler):
        self.args = args
        self.alpha = args.alpha
        self.env_params = args.env_params

        self.goal_sampler = goal_sampler

        self.total_iter = 0

        self.freq_target_update = args.freq_target_update

        # create the network
        self.architecture = self.args.architecture

        if self.architecture == 'gnn':
            from rl_modules.gnn_models import GnnSemantic
            self.model = GnnSemantic(self.env_params, args)
        elif self.architecture == 'deep_sets':
            from rl_modules.deepsets_models import DeepSetSemantic
            self.model = DeepSetSemantic(self.env_params, args)
        elif self.architecture == 'transformer':
            from rl_modules.transformer_models import TransformerSemantic
            self.model = TransformerSemantic(self.env_params, args)
        else:
            raise NotImplementedError
        # sync the networks across the CPUs
        sync_networks(self.model.critic)
        sync_networks(self.model.actor)
        hard_update(self.model.critic_target, self.model.critic)
        sync_networks(self.model.critic_target)

        # create the optimizer
        self.policy_optim = torch.optim.Adam(list(self.model.actor.parameters()),
                                             lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(list(self.model.critic.parameters()),
                                             lr=self.args.lr_critic)

        # create the normalizer
        self.o_norm = normalizer(size=self.env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=self.env_params['goal'], default_clip_range=self.args.clip_range)

        # if use GPU
        if self.args.cuda:
            self.model.critic.cuda()
            self.model.actor.cuda()
            self.model.critic_target.cuda()

        # Target Entropy
        self.target_entropy = -torch.prod(torch.Tensor(self.env_params['action'])).item()
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.args.lr_entropy)

        # her sampler
        self.her_module = her_sampler(self.args, compute_rew)

        # create the replay buffer
        self.buffer = ReplayBuffer(env_params=self.env_params,
                                   buffer_size=self.args.buffer_size,
                                   sample_func=self.her_module.sample_her_transitions,
                                   goal_sampler=self.goal_sampler,
                                   args=args)

    def act(self, obs, ag, g, no_noise):
        with torch.no_grad():
            # normalize policy inputs
            obs_norm = self.o_norm.normalize(obs)
            ag_norm = torch.tensor(self.g_norm.normalize(ag), dtype=torch.float32).unsqueeze(0)

            g_norm = torch.tensor(self.g_norm.normalize(g), dtype=torch.float32).unsqueeze(0)

            obs_tensor = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
            self.model.policy_forward_pass(obs_tensor, ag_norm, g_norm, no_noise=no_noise)
            action = self.model.pi_tensor.numpy()[0]

        return action.copy()
    
    def store(self, episodes):
        self.buffer.store_episode(episode_batch=episodes)

    # pre_process the inputs
    def _preproc_inputs(self, obs, ag, g):
        obs_norm = self.o_norm.normalize(obs)
        ag_norm = self.g_norm.normalize(ag)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, ag_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs

    def train(self):
        # train the network
        self.total_iter += 1
        self._update_network()

        # soft update
        if self.total_iter % self.freq_target_update == 0:
            self._soft_update_target_network(self.model.critic_target, self.model.critic)

    # update the normalizer
    def _update_normalizer(self, episode):
        mb_obs = episode['obs']
        mb_ag = episode['ag']
        mb_g = episode['g']
        mb_actions = episode['act']
        mb_obs_next = mb_obs[1:, :]
        mb_ag_next = mb_ag[1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[0]
        # create the new buffer to store them
        buffer_temp = {'obs': np.expand_dims(mb_obs, 0),
                       'ag': np.expand_dims(mb_ag, 0),
                       'g': np.expand_dims(mb_g, 0),
                       'actions': np.expand_dims(mb_actions, 0),
                       'obs_next': np.expand_dims(mb_obs_next, 0),
                       'ag_next': np.expand_dims(mb_ag_next, 0),
                       }

        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        # recompute the stats
        self.o_norm.recompute_stats()

        if self.args.normalize_goal:
            self.g_norm.update(transitions['g'])
            self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self):
        # sample from buffer
        transitions = self.buffer.sample(self.args.batch_size)

        # pre-process the observation and goal
        o, o_next, g, ag, ag_next, actions, rewards = transitions['obs'], transitions['obs_next'], transitions['g'], transitions['ag'], \
                                                      transitions['ag_next'], transitions['actions'], transitions['r']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        _, transitions['ag'] = self._preproc_og(o, ag)
        _, transitions['ag_next'] = self._preproc_og(o, ag_next)

        # apply normalization
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        ag_norm = self.g_norm.normalize(transitions['ag'])
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        ag_next_norm = self.g_norm.normalize(transitions['ag_next'])

        self.alpha = update_networks(self.model, self.policy_optim, self.critic_optim, self.alpha, self.log_alpha,
                                     self.target_entropy, self.alpha_optim, obs_norm, ag_norm, g_norm, obs_next_norm, ag_next_norm,
                                     actions, rewards, self.args)

    def save(self, model_path, epoch):
        # Store model
        torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std,
                    self.model.actor.state_dict(), self.model.critic.state_dict()],
                   model_path + '/model_{}.pt'.format(epoch))

    def load(self, model_path):
        o_mean, o_std, g_mean, g_std, actor, critic = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model.actor.load_state_dict(actor)
        self.model.critic.load_state_dict(critic)
        self.o_norm.mean = o_mean
        self.o_norm.std = o_std
        self.g_norm.mean = g_mean
        self.g_norm.std = g_std
