from copy import deepcopy
from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam


class Agent:
    """Agent that can interact with environment from pettingzoo"""

    def __init__(self, obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr, critic_layer_norm=False,
                 redq_n=1, redq_m=1):
        assert redq_n > 0
        assert redq_m <= redq_n
        self.m = redq_m
        self.n = redq_n
        self.actor = MLPNetwork(obs_dim, act_dim)

        # critic input all the observations and actions
        # if there are 3 agents for example, the input for critic is (obs1, obs2, obs3, act1, act2, act3)
        self.critics = [MLPNetwork(global_obs_dim, 1, layer_norm=critic_layer_norm) for _ in range(self.n)]
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizers = [Adam(self.critics[i].parameters(), lr=critic_lr) for i in range(self.n)]
        self.target_actor = deepcopy(self.actor)
        self.target_critics = deepcopy(self.critics)

    @staticmethod
    def gumbel_softmax(logits, tau=1.0, eps=1e-20):
        # NOTE that there is a function like this implemented in PyTorch(torch.nn.functional.gumbel_softmax),
        # but as mention in the doc, it may be removed in the future, so i implement it myself
        epsilon = torch.rand_like(logits)
        logits += -torch.log(-torch.log(epsilon + eps) + eps)
        return F.softmax(logits / tau, dim=-1)

    def action(self, obs, model_out=False):
        # this method is called in the following two cases:
        # a) interact with the environment
        # b) calculate action when update actor, where input(obs) is sampled from replay buffer with size:
        # torch.Size([batch_size, state_dim])

        logits = self.actor(obs)  # torch.Size([batch_size, action_size])
        # action = self.gumbel_softmax(logits)
        action = F.gumbel_softmax(logits, hard=True)
        if model_out:
            return action, logits
        return action

    def target_action(self, obs):
        # when calculate target critic value in MADDPG,
        # we use target actor to get next action given next states,
        # which is sampled from replay buffer with size torch.Size([batch_size, state_dim])

        logits = self.target_actor(obs)  # torch.Size([batch_size, action_size])
        # action = self.gumbel_softmax(logits)
        action = F.gumbel_softmax(logits, hard=True)
        return action.squeeze(0).detach()

    def critic_values(self, state_list: List[Tensor], act_list: List[Tensor], indices=None):
        x = torch.cat(state_list + act_list, 1)
        return [critic(x).squeeze(1) for critic in (self.critics if indices is None else (self.critics[i] for i in indices))]

    def target_critic_values(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return [target_critic(x).squeeze(1) for target_critic in self.target_critics]  # tensor with a given length

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critics(self, losses):
        for c_opt in self.critic_optimizers:
            c_opt.zero_grad()
        for loss in losses:
            loss.backward()
        for critic in self.critics:
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
        for c_opt in self.critic_optimizers:
            c_opt.step()


class MLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, non_linear=nn.ReLU(), layer_norm=False):
        super(MLPNetwork, self).__init__()

        layers = [nn.Linear(in_dim, hidden_dim),
                  non_linear,
                  nn.Linear(hidden_dim, hidden_dim),
                  non_linear,
                  nn.Linear(hidden_dim, out_dim)]
        if layer_norm:
            layers.append(nn.LayerNorm(out_dim))
            layers.append(non_linear)

        self.net = nn.Sequential(*layers).apply(self.init)

    @staticmethod
    def init(m):
        """init parameter of the module"""
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.net(x)
