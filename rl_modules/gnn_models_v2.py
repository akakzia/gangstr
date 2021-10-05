import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from itertools import permutations
import numpy as np
from rl_modules.networks import GnnMessagePassing, PhiCriticDeepSet, PhiActorDeepSet, RhoActorDeepSet, RhoCriticDeepSet

epsilon = 1e-6


class GnnCritic(nn.Module):
    def __init__(self, nb_objects, dim_body, dim_object, dim_mp_interaction, dim_edge_interaction, dim_phi_interaction, dim_node_interaction,
                 dim_phi_critic_input, dim_rho_critic_output):
        super(GnnCritic, self).__init__()

        # self.one_hot_encodings = [torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.])]
        self.nb_objects = nb_objects
        self.dim_body = dim_body
        self.dim_object = dim_object

        dim_mp_input = dim_mp_interaction
        dim_mp_output = dim_edge_interaction

        dim_phi_interaction_input = dim_phi_interaction
        dim_phi_interaction_output = dim_node_interaction

        self.mp_interaction = GnnMessagePassing(dim_mp_input, dim_mp_output)
        self.phi_interaction = PhiActorDeepSet(dim_phi_interaction_input, 256, dim_phi_interaction_output)

        # self.mp_star = GnnMessagePassing(10+9+6, 6)

        self.phi_critic = PhiCriticDeepSet(dim_phi_critic_input, 256, 3 * dim_phi_critic_input)
        self.rho_critic = RhoCriticDeepSet(3 * dim_phi_critic_input, dim_rho_critic_output)

        self.edge_ids = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]

    def forward_interaction(self, obs, edge_features):
        batch_size = obs.shape[0]
        assert batch_size == len(obs)

        # obs_body = obs[:, :self.dim_body]
        obs_objects = [obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]
                       for i in range(self.nb_objects)]

        inp = torch.stack([torch.cat([obj[:, :3], torch.max(edge_features[self.edge_ids[i], :, :], dim=0).values], dim=1)
                           for i, obj in enumerate(obs_objects)])

        output_phi_interaction = self.phi_interaction(inp)

        # inp_mp_star = torch.stack([torch.cat([obs_objects[i][:, 9:], obs_body, output_phi_interaction[i, :, :]], dim=-1) for i in range(3)])

        # output_mp_star = self.mp_star(inp_mp_star)

        return output_phi_interaction

    def forward(self, obs, act, nodes):
        batch_size = obs.shape[0]
        assert batch_size == len(obs)

        obs_body = obs[:, :self.dim_body]
        # obs_objects = [torch.cat((torch.cat(batch_size * [self.one_hot_encodings[i]]).reshape(obs_body.shape[0], self.nb_objects),
        #                           obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]), dim=1)
        #                for i in range(self.nb_objects)]
        obs_objects = [obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]
                       for i in range(self.nb_objects)]

        # inp = torch.stack([torch.cat([act, obs_body, obj, torch.max(edge_features[self.edge_ids[i], :, :], dim=0).values], dim=1)
        #                    for i, obj in enumerate(obs_objects)])

        inp_phi = torch.stack([torch.cat([act, obs_body, obj, nodes[i, :, :]], dim=-1) for i, obj in enumerate(obs_objects)])

        # inp = torch.cat([act, obs_body, torch.max(aa, dim=0).values], dim=1)

        output_phi_critic_1, output_phi_critic_2 = self.phi_critic(inp_phi)
        output_phi_critic_1 = output_phi_critic_1.sum(dim=0)
        output_phi_critic_2 = output_phi_critic_2.sum(dim=0)
        q1_pi_tensor, q2_pi_tensor = self.rho_critic(output_phi_critic_1, output_phi_critic_2)
        return q1_pi_tensor, q2_pi_tensor

    def message_passing(self, obs, ag, g):
        batch_size = obs.shape[0]
        assert batch_size == len(ag)

        obs_body = obs[:, :self.dim_body]
        # obs_objects = [torch.cat((torch.cat(batch_size * [self.one_hot_encodings[i]]).reshape(obs_body.shape[0], self.nb_objects),
        #                                       obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]), dim=1)
        #                            for i in range(self.nb_objects)]
        obs_objects = [obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]
                       for i in range(self.nb_objects)]

        obj_ids = [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]
        goal_ids = [[0, 3], [1, 4], [0, 5], [2, 6], [1, 7], [2, 8]]

        delta_g = g - ag

        inp_mp = torch.stack([torch.cat([delta_g[:, goal_ids[i]], obs_objects[obj_ids[i][0]][:, :3] - obs_objects[obj_ids[i][1]][:, :3],
                                         obs_objects[obj_ids[i][0]][:, :3], obs_objects[obj_ids[i][1]][:, :3]],
                                        dim=-1) for i in range(6)])

        # inp_mp = torch.stack([torch.cat([ag[:, goal_ids[i]], g[:, goal_ids[i]], obs_objects[obj_ids[i][0]][:, :3],
        #                                  obs_objects[obj_ids[i][1]][:, :3]], dim=-1) for i in range(6)])

        # inp_mp = torch.stack([torch.cat([g, ag, obj[0], obj[1]], dim=-1) for obj in permutations(obs_objects, 2)])

        output_mp = self.mp_interaction(inp_mp)

        return output_mp


class GnnActor(nn.Module):
    def __init__(self, nb_objects, dim_body, dim_object, dim_phi_actor_input, dim_rho_actor_output):
        super(GnnActor, self).__init__()

        self.nb_objects = nb_objects
        self.dim_body = dim_body
        self.dim_object = dim_object

        self.phi_actor = PhiActorDeepSet(dim_phi_actor_input, 256, 3 * dim_phi_actor_input)
        self.rho_actor = RhoActorDeepSet(3 * dim_phi_actor_input, dim_rho_actor_output)

        self.edge_ids = [np.array([0, 2]), np.array([1, 4]), np.array([3, 5])]

        # self.one_hot_encodings = [torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.])]

    def forward(self, obs, nodes):
        batch_size = obs.shape[0]
        assert batch_size == len(obs)

        obs_body = obs[:, :self.dim_body]
        # obs_objects = [torch.cat((torch.cat(batch_size * [self.one_hot_encodings[i]]).reshape(obs_body.shape[0], self.nb_objects),
        #                           obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]), dim=1)
        #                for i in range(self.nb_objects)]
        obs_objects = [obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]
                       for i in range(self.nb_objects)]

        inp_phi = torch.stack([torch.cat([obs_body, obj, nodes[i, :, :]], dim=-1) for i, obj in enumerate(obs_objects)])

        # inp = torch.cat([obs_body, torch.max(aa, dim=0).values], dim=1)

        output_phi_actor = self.phi_actor(inp_phi)
        output_phi_actor = output_phi_actor.sum(dim=0)
        mean, logstd = self.rho_actor(output_phi_actor)
        return mean, logstd

    def sample(self, obs, nodes):
        mean, log_std = self.forward(obs, nodes)
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


class GnnSemantic:
    def __init__(self, env_params, args):
        self.dim_body = 10
        self.dim_object = 15
        self.dim_goal = env_params['goal']
        self.dim_act = env_params['action']
        self.nb_objects = args.n_blocks

        self.q1_pi_tensor = None
        self.q2_pi_tensor = None
        self.target_q1_pi_tensor = None
        self.target_q2_pi_tensor = None
        self.pi_tensor = None
        self.log_prob = None

        # dim_input_objects = 2 * (self.nb_objects + self.dim_object)
        # dim_mp_input = 6 + 2
        # dim_mp_output = 3 * dim_mp_input
        #
        # dim_phi_actor_input = self.dim_body + self.dim_object + dim_mp_output
        # dim_phi_actor_output = 3 * dim_phi_actor_input
        # dim_rho_actor_input = dim_phi_actor_output
        # dim_rho_actor_output = self.dim_act
        #
        # dim_phi_critic_input = self.dim_body + self.dim_object + dim_mp_output + self.dim_act
        # dim_phi_critic_output = 3 * dim_phi_critic_input
        # dim_rho_critic_input = dim_phi_critic_output
        # dim_rho_critic_output = 1

        # Interaction Network dimensions
        interaction_dim_edge = 3 + 2
        interaction_dim_node = 3
        interaction_dim_mp_i = interaction_dim_edge + 2 * interaction_dim_node
        interaction_dim_phi_i = interaction_dim_node + interaction_dim_edge

        # Critic Network dimensions
        dim_phi_critic_input = self.dim_body + self.dim_act + interaction_dim_node + self.dim_object
        dim_rho_critic_output = 1

        # Actor Network dimensions
        dim_phi_actor_input = self.dim_body + interaction_dim_node + self.dim_object
        dim_rho_actor_output = self.dim_act

        self.critic = GnnCritic(self.nb_objects, self.dim_body, self.dim_object, interaction_dim_mp_i, interaction_dim_edge, interaction_dim_phi_i,
                                interaction_dim_node, dim_phi_critic_input, dim_rho_critic_output)
        self.critic_target = GnnCritic(self.nb_objects, self.dim_body, self.dim_object, interaction_dim_mp_i, interaction_dim_edge, interaction_dim_phi_i,
                                       interaction_dim_node, dim_phi_critic_input, dim_rho_critic_output)
        self.actor = GnnActor(self.nb_objects, self.dim_body, self.dim_object, dim_phi_actor_input, dim_rho_actor_output)

    def policy_forward_pass(self, obs, ag, g, no_noise=False):
        edge_features = self.critic.message_passing(obs, ag, g)
        updated_nodes = self.critic.forward_interaction(obs, edge_features)
        if not no_noise:
            self.pi_tensor, self.log_prob, _ = self.actor.sample(obs, updated_nodes)
        else:
            _, self.log_prob, self.pi_tensor = self.actor.sample(obs, updated_nodes)

    def forward_pass(self, obs, ag, g, actions=None):
        edge_features = self.critic.message_passing(obs, ag, g)
        updated_nodes = self.critic.forward_interaction(obs, edge_features)

        self.pi_tensor, self.log_prob, _ = self.actor.sample(obs, updated_nodes)

        if actions is not None:
            self.q1_pi_tensor, self.q2_pi_tensor = self.critic.forward(obs, self.pi_tensor, updated_nodes)
            return self.critic.forward(obs, actions, updated_nodes)
        else:
            with torch.no_grad():
                self.target_q1_pi_tensor, self.target_q2_pi_tensor = self.critic_target.forward(obs, self.pi_tensor, updated_nodes)
            self.q1_pi_tensor, self.q2_pi_tensor = None, None
