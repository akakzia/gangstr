import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from itertools import permutations
import numpy as np
from rl_modules.networks import GnnMessagePassing, PhiCriticDeepSet, PhiActorDeepSet, RhoActorDeepSet, RhoCriticDeepSet
from utils import get_graph_structure

epsilon = 1e-6


class GnnCritic(nn.Module):
    def __init__(self, nb_objects, edges, incoming_edges, predicate_ids, aggregation, readout, dim_body, dim_object, dim_mp_input,
                 dim_mp_output, dim_phi_critic_input, dim_phi_critic_output, dim_rho_critic_input, dim_rho_critic_output):
        super(GnnCritic, self).__init__()

        self.nb_objects = nb_objects
        self.dim_body = dim_body
        self.dim_object = dim_object

        self.n_permutations = self.nb_objects * (self.nb_objects - 1)

        self.aggregation = aggregation
        self.readout = readout

        self.mp_critic = GnnMessagePassing(dim_mp_input, dim_mp_output)
        self.phi_critic = PhiCriticDeepSet(dim_phi_critic_input, 256, dim_phi_critic_output)
        self.rho_critic = RhoCriticDeepSet(dim_rho_critic_input, dim_rho_critic_output)

        self.edges = edges
        self.incoming_edges = incoming_edges
        self.predicate_ids = predicate_ids

    def forward(self, obs, act, edge_features):
        batch_size = obs.shape[0]
        assert batch_size == len(obs)

        obs_body = obs[:, :self.dim_body]
        obs_objects = [obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]
                       for i in range(self.nb_objects)]

        if self.aggregation == 'max':
            inp = torch.stack([torch.cat([act, obs_body, obj, torch.max(edge_features[self.incoming_edges[i], :, :], dim=0).values], dim=1)
                               for i, obj in enumerate(obs_objects)])
        elif self.aggregation == 'sum':
            inp = torch.stack([torch.cat([act, obs_body, obj, torch.sum(edge_features[self.incoming_edges[i], :, :], dim=0)], dim=1)
                               for i, obj in enumerate(obs_objects)])
        elif self.aggregation == 'mean':
            inp = torch.stack([torch.cat([act, obs_body, obj, torch.mean(edge_features[self.incoming_edges[i], :, :], dim=0)], dim=1)
                               for i, obj in enumerate(obs_objects)])
        else:
            raise NotImplementedError

        output_phi_critic_1, output_phi_critic_2 = self.phi_critic(inp)
        if self.readout == 'sum':
            output_phi_critic_1 = output_phi_critic_1.sum(dim=0)
            output_phi_critic_2 = output_phi_critic_2.sum(dim=0)
        elif self.readout == 'mean':
            output_phi_critic_1 = output_phi_critic_1.mean(dim=0)
            output_phi_critic_2 = output_phi_critic_2.mean(dim=0)
        elif self.readout == 'max':
            output_phi_critic_1 = output_phi_critic_1.max(dim=0).values
            output_phi_critic_2 = output_phi_critic_2.max(dim=0).values
        q1_pi_tensor, q2_pi_tensor = self.rho_critic(output_phi_critic_1, output_phi_critic_2)
        return q1_pi_tensor, q2_pi_tensor

    def message_passing(self, obs, ag, g):
        batch_size = obs.shape[0]
        assert batch_size == len(ag)

        obs_objects = [obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]
                       for i in range(self.nb_objects)]

        delta_g = g - ag

        inp_mp = torch.stack([torch.cat([delta_g[:, self.predicate_ids[i]], obs_objects[self.edges[i][0]][:, :3],
                                         obs_objects[self.edges[i][1]][:, :3]], dim=-1) for i in range(self.n_permutations)])

        output_mp = self.mp_critic(inp_mp)

        return output_mp


class GnnActor(nn.Module):
    def __init__(self, nb_objects, incoming_edges, aggregation, readout, dim_body, dim_object, dim_phi_actor_input, dim_phi_actor_output, dim_rho_actor_input,
                 dim_rho_actor_output):
        super(GnnActor, self).__init__()

        self.nb_objects = nb_objects
        self.dim_body = dim_body
        self.dim_object = dim_object

        self.aggregation = aggregation
        self.readout = readout

        self.phi_actor = PhiActorDeepSet(dim_phi_actor_input, 256, dim_phi_actor_output)
        self.rho_actor = RhoActorDeepSet(dim_rho_actor_input, dim_rho_actor_output)

        self.incoming_edges = incoming_edges

    def forward(self, obs, edge_features):
        batch_size = obs.shape[0]
        assert batch_size == len(obs)

        obs_body = obs[:, :self.dim_body]
        obs_objects = [obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]
                       for i in range(self.nb_objects)]

        if self.aggregation == 'max':
            inp = torch.stack([torch.cat([obs_body, obj, torch.max(edge_features[self.incoming_edges[i], :, :], dim=0).values], dim=1)
                                       for i, obj in enumerate(obs_objects)])
        elif self.aggregation == 'sum':
            inp = torch.stack([torch.cat([obs_body, obj, torch.sum(edge_features[self.incoming_edges[i], :, :], dim=0)], dim=1)
                               for i, obj in enumerate(obs_objects)])
        elif self.aggregation == 'mean':
            inp = torch.stack([torch.cat([obs_body, obj, torch.mean(edge_features[self.incoming_edges[i], :, :], dim=0)], dim=1)
                               for i, obj in enumerate(obs_objects)])
        else:
            raise NotImplementedError

        output_phi_actor = self.phi_actor(inp)
        if self.readout == 'sum':
            output_phi_actor = output_phi_actor.sum(dim=0)
        elif self.readout == 'mean':
            output_phi_actor = output_phi_actor.mean(dim=0)
        elif self.readout == 'max':
            output_phi_actor = output_phi_actor.max(dim=0).values
        else:
            raise NotImplementedError

        mean, logstd = self.rho_actor(output_phi_actor)
        return mean, logstd

    def sample(self, obs, edge_features):
        mean, log_std = self.forward(obs, edge_features)
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

        self.aggregation = args.aggregation_fct
        self.readout = args.readout_fct

        self.q1_pi_tensor = None
        self.q2_pi_tensor = None
        self.target_q1_pi_tensor = None
        self.target_q2_pi_tensor = None
        self.pi_tensor = None
        self.log_prob = None

        # Process indexes for graph construction
        self.edges, self.incoming_edges, self.predicate_ids = get_graph_structure(self.nb_objects)

        dim_mp_input = 6 + 2  # 2 * nb_position_dimensions + nb_predicates
        dim_mp_output = 3 * dim_mp_input

        dim_phi_actor_input = self.dim_body + self.dim_object + dim_mp_output
        dim_phi_actor_output = 3 * dim_phi_actor_input
        dim_rho_actor_input = dim_phi_actor_output
        dim_rho_actor_output = self.dim_act

        dim_phi_critic_input = self.dim_body + self.dim_object + dim_mp_output + self.dim_act
        dim_phi_critic_output = 3 * dim_phi_critic_input
        dim_rho_critic_input = dim_phi_critic_output
        dim_rho_critic_output = 1

        self.critic = GnnCritic(self.nb_objects, self.edges, self.incoming_edges, self.predicate_ids, self.aggregation,
                                self.readout, self.dim_body, self.dim_object, dim_mp_input, dim_mp_output,
                                dim_phi_critic_input, dim_phi_critic_output, dim_rho_critic_input, dim_rho_critic_output)
        self.critic_target = GnnCritic(self.nb_objects, self.edges, self.incoming_edges, self.predicate_ids, self.aggregation,
                                       self.readout, self.dim_body, self.dim_object, dim_mp_input, dim_mp_output,
                                       dim_phi_critic_input, dim_phi_critic_output, dim_rho_critic_input, dim_rho_critic_output)
        self.actor = GnnActor(self.nb_objects, self.incoming_edges, self.aggregation, self.readout, self.dim_body, self.dim_object,
                              dim_phi_actor_input, dim_phi_actor_output, dim_rho_actor_input, dim_rho_actor_output)

    def policy_forward_pass(self, obs, ag, g, no_noise=False):
        edge_features = self.critic.message_passing(obs, ag, g)
        if not no_noise:
            self.pi_tensor, self.log_prob, _ = self.actor.sample(obs, edge_features)
        else:
            _, self.log_prob, self.pi_tensor = self.actor.sample(obs, edge_features)

    def forward_pass(self, obs, ag, g, actions=None):
        edge_features = self.critic.message_passing(obs, ag, g)

        self.pi_tensor, self.log_prob, _ = self.actor.sample(obs, edge_features)

        if actions is not None:
            self.q1_pi_tensor, self.q2_pi_tensor = self.critic.forward(obs, self.pi_tensor, edge_features)
            return self.critic.forward(obs, actions, edge_features)
        else:
            with torch.no_grad():
                self.target_q1_pi_tensor, self.target_q2_pi_tensor = self.critic_target.forward(obs, self.pi_tensor, edge_features)
            self.q1_pi_tensor, self.q2_pi_tensor = None, None
