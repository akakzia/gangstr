import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from itertools import permutations
import numpy as np
from rl_modules.networks import PhiActorDeepSet, PhiCriticDeepSet, RhoActorDeepSet, RhoCriticDeepSet

epsilon = 1e-6


class SemanticCritic(nn.Module):
    def __init__(self, nb_objects, dim_phi_critic_input, dim_phi_critic_output, dim_rho_critic_input, dim_rho_critic_output, nb_permutations,
                 dim_body, dim_object, symmetry_trick):
        super(SemanticCritic, self).__init__()

        self.nb_permutations = nb_permutations
        self.nb_objects = nb_objects
        self.dim_body = dim_body
        self.dim_object = dim_object
        self.symmetry_trick = symmetry_trick
        self.first_inds = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 20, 21, 25])
        self.second_inds = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 18, 19, 22, 23, 24, 26, 27, 28, 29])
        self.single_phi_critic = PhiCriticDeepSet(dim_phi_critic_input, 256, dim_phi_critic_output)
        self.rho_critic = RhoCriticDeepSet(dim_rho_critic_input, dim_rho_critic_output)

    def forward(self, obs, ag, g, act):

        batch_size = obs.shape[0]
        assert batch_size == len(ag)

        obs_body = obs[:, :self.dim_body]
        obs_objects = [obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]
                       for i in range(self.nb_objects)]

        if self.symmetry_trick:
            all_inputs = []
            for i in range(self.nb_objects):
                for j in range(self.nb_objects):
                    if i < j:
                        all_inputs.append(torch.cat([ag[:, self.first_inds], g[:, self.first_inds], obs_body, obs_objects[i], obs_objects[j],
                                                     act], dim=1))
                    elif j < i:
                        all_inputs.append(torch.cat([ag[:, self.second_inds], g[:, self.second_inds], obs_body, obs_objects[i], obs_objects[j],
                                                     act], dim=1))

            inp = torch.stack(all_inputs)
        else:
            # Parallelization by stacking input tensors
            inp = torch.stack([torch.cat([ag, g, obs_body, x[0], x[1], act], dim=1) for x in permutations(obs_objects, 2)])

        output_phi_critic_1, output_phi_critic_2 = self.single_phi_critic(inp)
        output_phi_critic_1 = output_phi_critic_1.sum(dim=0)
        output_phi_critic_2 = output_phi_critic_2.sum(dim=0)
        q1_pi_tensor, q2_pi_tensor = self.rho_critic(output_phi_critic_1, output_phi_critic_2)
        return q1_pi_tensor, q2_pi_tensor


class SemanticActor(nn.Module):
    def __init__(self, nb_objects, dim_phi_actor_input, dim_phi_actor_output, dim_rho_actor_input, dim_rho_actor_output,
                 dim_body, dim_object, symmetry_trick):
        super(SemanticActor, self).__init__()

        self.nb_objects = nb_objects
        self.dim_body = dim_body
        self.dim_object = dim_object
        self.symmetry_trick = symmetry_trick
        self.first_inds = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 20, 21, 25])
        self.second_inds = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 18, 19, 22, 23, 24, 26, 27, 28, 29])
        self.single_phi_actor = PhiActorDeepSet(dim_phi_actor_input, 256, dim_phi_actor_output)
        self.rho_actor = RhoActorDeepSet(dim_rho_actor_input, dim_rho_actor_output)

    def forward(self, obs, ag, g):
        batch_size = obs.shape[0]
        assert batch_size == ag.shape[0]

        obs_body = obs[:, :self.dim_body]
        obs_objects = [obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]
                       for i in range(self.nb_objects)]

        if self.symmetry_trick:
            all_inputs = []
            for i in range(self.nb_objects):
                for j in range(self.nb_objects):
                    if i < j:
                        all_inputs.append(torch.cat([ag[:, self.first_inds], obs_body, g[:, self.first_inds], obs_objects[i], obs_objects[j]], dim=1))
                    elif j < i:
                        all_inputs.append(torch.cat([ag[:, self.second_inds], obs_body, g[:, self.second_inds], obs_objects[i], obs_objects[j]], dim=1))

            inp = torch.stack(all_inputs)
        else:
            # Parallelization by stacking input tensors
            inp = torch.stack([torch.cat([ag, g, obs_body, x[0], x[1]], dim=1) for x in permutations(obs_objects, 2)])

        output_phi_actor = self.single_phi_actor(inp)
        output_phi_actor = output_phi_actor.sum(dim=0)
        mean, logstd = self.rho_actor(output_phi_actor)
        return mean, logstd

    def sample(self, obs, ag, g):
        mean, log_std = self.forward(obs, ag, g)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


class DeepSetSemantic:
    def __init__(self, env_params, args):
        self.observation = None
        self.ag = None
        self.g = None
        self.dim_body = 10
        self.dim_object = 15
        self.dim_act = env_params['action']
        self.nb_objects = args.n_blocks
        self.n_permutations = len([x for x in permutations(range(self.nb_objects), 2)])

        self.symmetry_trick = True
        self.dim_goal = env_params['goal'] if not self.symmetry_trick else 20

        self.q1_pi_tensor = None
        self.q2_pi_tensor = None
        self.target_q1_pi_tensor = None
        self.target_q2_pi_tensor = None
        self.pi_tensor = None
        self.log_prob = None

        dim_input_objects = 2 * self.dim_object
        dim_phi_actor_input = 2 * self.dim_goal + self.dim_body + dim_input_objects
        dim_phi_actor_output = 3 * dim_phi_actor_input
        dim_rho_actor_input = dim_phi_actor_output
        dim_rho_actor_output = self.dim_act

        dim_phi_critic_input = dim_phi_actor_input + self.dim_act
        dim_phi_critic_output = 3 * dim_phi_critic_input
        dim_rho_critic_input = dim_phi_critic_output
        dim_rho_critic_output = 1

        self.critic = SemanticCritic(self.nb_objects, dim_phi_critic_input, dim_phi_critic_output, dim_rho_critic_input, dim_rho_critic_output,
                                     self.n_permutations, self.dim_body, self.dim_object, self.symmetry_trick)
        self.critic_target = SemanticCritic(self.nb_objects, dim_phi_critic_input, dim_phi_critic_output, dim_rho_critic_input,
                                            dim_rho_critic_output, self.n_permutations, self.dim_body, self.dim_object, self.symmetry_trick,
                                            )
        self.actor = SemanticActor(self.nb_objects, dim_phi_actor_input, dim_phi_actor_output, dim_rho_actor_input, dim_rho_actor_output,
                                   self.dim_body, self.dim_object, self.symmetry_trick)

    def policy_forward_pass(self, obs, ag, g, no_noise=False):
        if not no_noise:
            self.pi_tensor, self.log_prob, _ = self.actor.sample(obs, ag, g)
        else:
            _, self.log_prob, self.pi_tensor = self.actor.sample(obs, ag, g)

    def forward_pass(self, obs, ag, g, actions=None):

        self.pi_tensor, self.log_prob, _ = self.actor.sample(obs, ag, g)

        if actions is not None:
            self.q1_pi_tensor, self.q2_pi_tensor = self.critic.forward(obs, ag, g, self.pi_tensor)
            return self.critic.forward(obs, ag, g, actions)
        else:
            with torch.no_grad():
                self.target_q1_pi_tensor, self.target_q2_pi_tensor = self.critic_target.forward(obs, ag, g, self.pi_tensor)
            self.q1_pi_tensor, self.q2_pi_tensor = None, None
