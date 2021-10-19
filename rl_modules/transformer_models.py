import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import math
from rl_modules.networks import PhiCriticDeepSet, PhiActorDeepSet, RhoActorDeepSet, RhoCriticDeepSet
from utils import get_graph_structure, get_idxs_per_object

epsilon = 1e-6


class AttentionModule(nn.Module):
    def __init__(self, nb_attention_heads, nb_objects, dim_object, dim_body, dim_mixed_output):
        super(AttentionModule, self).__init__()

        self.num_attention_heads = nb_attention_heads

        self.nb_objects = nb_objects
        self.dim_object = dim_object
        self.dim_body = dim_body

        self.attention_head_size = dim_mixed_output

        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.key_layer = nn.Linear(self.dim_object, self.all_head_size)
        self.query_layer = nn.Linear(self.dim_object, self.all_head_size)
        self.value_layer = nn.Linear(self.dim_object, self.all_head_size)

        self.dense = nn.Linear(self.all_head_size, self.attention_head_size)

        self.semantic_ids = get_idxs_per_object(n=self.nb_objects)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, obs, g):
        batch_size = obs.shape[0]
        assert batch_size == len(g)

        obs_objects = [obs[:, self.dim_body + 15 * i: self.dim_body + 15 * (i + 1)]
                       for i in range(self.nb_objects)]

        inp = torch.stack([torch.cat([obs_objects[i], g[:, self.semantic_ids[i]]], dim=-1) for i in range(self.nb_objects)], dim=1)

        k_inp = self.key_layer(inp)
        q_inp = self.query_layer(inp)
        v_inp = self.value_layer(inp)

        query_layer = self.transpose_for_scores(q_inp)
        key_layer = self.transpose_for_scores(k_inp)
        value_layer = self.transpose_for_scores(v_inp)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [Batch_size x Seq_length x Num_of_heads x Head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # [Batch_size x Seq_length x Hidden_size]
        context_layer = context_layer.view(*new_context_layer_shape)

        output = self.dense(context_layer)

        return output

class ActorReadoutModule(nn.Module):
    def __init__(self, nb_objects, dim_body, dim_phi_actor_input, dim_phi_actor_output, dim_rho_actor_input,
                 dim_rho_actor_output):
        super(ActorReadoutModule, self).__init__()

        self.nb_objects = nb_objects
        self.dim_body = dim_body

        self.phi_actor = PhiActorDeepSet(dim_phi_actor_input, 256, dim_phi_actor_output)
        self.rho_actor = RhoActorDeepSet(dim_rho_actor_input, dim_rho_actor_output)

    def forward(self, obs, context_layer):
        batch_size = obs.shape[0]
        assert batch_size == len(obs)

        obs_body = obs[:, :self.dim_body]

        inp = torch.stack([torch.cat([obs_body, context_layer[:, i, :]], dim=1) for i in range(self.nb_objects)])

        output_phi_actor = self.phi_actor(inp)

        output_phi_actor = output_phi_actor.sum(dim=0)

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


class CriticReadoutModule(nn.Module):
    def __init__(self, nb_objects, dim_body, dim_phi_critic_input, dim_phi_critic_output, dim_rho_critic_input,
                 dim_rho_critic_output):
        super(CriticReadoutModule, self).__init__()

        self.nb_objects = nb_objects
        self.dim_body = dim_body

        self.phi_critic = PhiCriticDeepSet(dim_phi_critic_input, 256, dim_phi_critic_output)
        self.rho_critic = RhoCriticDeepSet(dim_rho_critic_input, dim_rho_critic_output)

    def forward(self, obs, act, context_layer):
        batch_size = obs.shape[0]
        assert batch_size == len(obs)

        obs_body = obs[:, :self.dim_body]

        inp = torch.stack([torch.cat([obs_body, act, context_layer[:, i, :]], dim=1) for i in range(self.nb_objects)])

        output_phi_critic_1, output_phi_critic_2 = self.phi_critic(inp)

        output_phi_critic_1 = output_phi_critic_1.sum(dim=0)
        output_phi_critic_2 = output_phi_critic_2.sum(dim=0)

        q1_pi_tensor, q2_pi_tensor = self.rho_critic(output_phi_critic_1, output_phi_critic_2)
        return q1_pi_tensor, q2_pi_tensor

class TransformerSemantic:
    def __init__(self, env_params, args):
        self.dim_body = 10
        self.dim_object = 15
        self.dim_goal = env_params['goal']
        self.dim_act = env_params['action']
        self.nb_objects = args.n_blocks

        self.dim_object_features = self.dim_object + 3*(self.nb_objects - 1)

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

        self.dim_mixed_output = 9

        self.nb_attention_heads = args.nb_attention_heads

        self.dim_phi_actor_input = self.dim_body + self.dim_mixed_output
        self.dim_phi_actor_output = 3 * self.dim_phi_actor_input

        self.dim_rho_actor_input = self.dim_phi_actor_output
        self.dim_rho_actor_output = self.dim_act

        self.dim_phi_critic_input = self.dim_body + self.dim_mixed_output + self.dim_act
        self.dim_phi_critic_output = 3 * self.dim_phi_actor_input

        self.dim_rho_critic_input = self.dim_phi_actor_output
        self.dim_rho_critic_output = 1

        self.attention_module = AttentionModule(self.nb_attention_heads, self.nb_objects, self.dim_object_features, self.dim_body, self.dim_mixed_output)
        self.critic = CriticReadoutModule(self.nb_objects, self.dim_body, self.dim_phi_critic_input, self.dim_phi_critic_output,
                                        self.dim_rho_critic_input, self.dim_rho_critic_output)
        self.critic_target = CriticReadoutModule(self.nb_objects, self.dim_body, self.dim_phi_critic_input, self.dim_phi_critic_output,
                                        self.dim_rho_critic_input, self.dim_rho_critic_output)
        self.actor = ActorReadoutModule(self.nb_objects, self.dim_body, self.dim_phi_actor_input, self.dim_phi_actor_output,
                                        self.dim_rho_actor_input, self.dim_rho_actor_output)

    def policy_forward_pass(self, obs, ag, g, no_noise=False):
        context_layer = self.attention_module(obs, g)
        if not no_noise:
            self.pi_tensor, self.log_prob, _ = self.actor.sample(obs, context_layer)
        else:
            _, self.log_prob, self.pi_tensor = self.actor.sample(obs, context_layer)

    def forward_pass(self, obs, ag, g, actions=None):
        context_layer = self.attention_module(obs, g)

        self.pi_tensor, self.log_prob, _ = self.actor.sample(obs, context_layer)

        if actions is not None:
            self.q1_pi_tensor, self.q2_pi_tensor = self.critic.forward(obs, self.pi_tensor, context_layer)
            return self.critic.forward(obs, actions, context_layer)
        else:
            with torch.no_grad():
                self.target_q1_pi_tensor, self.target_q2_pi_tensor = self.critic_target.forward(obs, self.pi_tensor, context_layer)
            self.q1_pi_tensor, self.q2_pi_tensor = None, None
