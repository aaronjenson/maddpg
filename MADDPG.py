import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from Agent import Agent
from Buffer import Buffer


def setup_logger(filename):
    """ set up logger with filename. """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(filename, mode='w')
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s--%(levelname)s--%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


class MADDPG:
    """A MADDPG(Multi Agent Deep Deterministic Policy Gradient) agent"""

    def __init__(self, dim_info, capacity, batch_size, actor_lr, critic_lr, res_dir, device, offline_data=None, layer_norm=False, num_qs=1, num_min_qs=1):
        assert num_qs > 0
        assert num_qs >= num_min_qs

        self.device = device
        self.offline_data = offline_data
        self.num_qs = num_qs
        self.num_min_qs = num_min_qs
        # sum all the dims of each agent to get input dim for critic
        global_obs_act_dim = sum(sum(val) for val in dim_info.values())
        # create Agent(actor-critic) and replay buffer for each agent
        self.agents = {}
        self.buffers = {}
        self.offline_buffers = {}
        for agent_id, (obs_dim, act_dim) in dim_info.items():
            self.agents[agent_id] = Agent(obs_dim, act_dim, global_obs_act_dim, actor_lr, critic_lr, self.device,
                                          critic_layer_norm=layer_norm, redq_n=num_qs, redq_m=num_min_qs)
            self.buffers[agent_id] = Buffer(capacity, obs_dim, act_dim, self.device)
            if offline_data is not None:
                self.offline_buffers[agent_id] = Buffer(capacity, obs_dim, act_dim, self.device)
        self.dim_info = dim_info

        self.batch_size = batch_size
        self.res_dir = res_dir  # directory to save the training result
        self.logger = setup_logger(os.path.join(res_dir, 'maddpg.log'))

        if offline_data is not None:
            for data in offline_data:
                obs, action, next_obs, reward, terminations, truncations, info = data
                self.add(obs, action, reward, next_obs, terminations, truncations, offline=True)

    def add(self, obs, action, reward, next_obs, terminated, truncated, offline=False):
        buffers = self.offline_buffers if offline else self.buffers
        # NOTE that the experience is a dict with agent name as its key
        for agent_id in obs.keys():
            o = obs[agent_id]
            a = action[agent_id]
            if isinstance(a, int):
                # the action from env.action_space.sample() is int, we have to convert it to onehot
                a = np.eye(self.dim_info[agent_id][1])[a]

            r = reward[agent_id]
            next_o = next_obs[agent_id]
            term = terminated[agent_id]
            trunc = truncated[agent_id]
            buffers[agent_id].add(o, a, r, next_o, term, trunc)

    def sample(self, batch_size):
        """sample experience from all the agents' buffers, and collect data for network input"""
        # get the total num of transitions, these buffers should have same number of transitions
        offline_indices = None
        if self.offline_data is not None:
            batch_size //= 2
            total_offline = len(self.offline_buffers['agent_0'])
            if batch_size > total_offline:
                offline_indices = np.random.choice(total_offline, size=total_offline, replace=False)
                batch_size = (2 * batch_size) - total_offline
            else:
                offline_indices = np.random.choice(total_offline, size=batch_size, replace=False)
        total_num = len(self.buffers['agent_0'])
        indices = np.random.choice(total_num, size=batch_size, replace=False)

        # NOTE that in MADDPG, we need the obs and actions of all agents
        # but only the reward and done of the current agent is needed in the calculation
        obs, act, reward, next_obs, terminated, truncated, next_act = {}, {}, {}, {}, {}, {}, {}
        for agent_id, buffer in self.buffers.items():
            o, a, r, n_o, term, trunc = buffer.sample(indices)
            obs[agent_id] = o
            act[agent_id] = a
            reward[agent_id] = r
            next_obs[agent_id] = n_o
            terminated[agent_id] = term
            truncated[agent_id] = trunc
            # calculate next_action using target_network and next_state
            next_act[agent_id] = self.agents[agent_id].target_action(n_o)

        if self.offline_data is not None:
            for agent_id, buffer in self.offline_buffers.items():
                o, a, r, n_o, term, trunc = buffer.sample(offline_indices)
                obs[agent_id] = torch.cat((obs[agent_id], o))
                act[agent_id] = torch.cat((act[agent_id], a))
                reward[agent_id] = torch.cat((reward[agent_id], r))
                next_obs[agent_id] = torch.cat((next_obs[agent_id], n_o))
                terminated[agent_id] = torch.cat((terminated[agent_id], term))
                truncated[agent_id] = torch.cat((truncated[agent_id], trunc))
                # calculate next_action using target_network and next_state
                next_act[agent_id] = torch.cat((next_act[agent_id], self.agents[agent_id].target_action(n_o)))

        return obs, act, reward, next_obs, terminated, truncated, next_act

    def select_action(self, obs, noise=False):
        actions = {}
        for agent, o in obs.items():
            o = torch.from_numpy(o).unsqueeze(0).float()
            a = self.agents[agent].action(o, noise=noise)  # torch.Size([1, action_size])
            # NOTE that the output is a tensor, convert it to int before input to the environment
            actions[agent] = a.squeeze(0).argmax().item()
            self.logger.info(f'{agent} action: {actions[agent]}')
        return actions

    def learn(self, batch_size, gamma):
        for agent_id, agent in self.agents.items():
            obs, act, reward, next_obs, terminated, truncated, next_act = self.sample(batch_size)
            # update critic
            critic_values = agent.critic_values(list(obs.values()), list(act.values()))

            # calculate target critic value
            next_target_critic_values = agent.target_critic_values(list(next_obs.values()),
                                                                 list(next_act.values()))
            target_values = [reward[agent_id] + gamma * next_target_critic_value * \
                (1 - terminated[agent_id]) * (1 - truncated[agent_id]) for next_target_critic_value in next_target_critic_values]

            critic_losses = [F.mse_loss(critic_values[i], target_values[i].detach(), reduction='mean') for i in range(len(critic_values))]
            agent.update_critics(critic_losses)

            # sample num_min_qs
            indices = torch.randperm(self.num_qs)[:self.num_min_qs] if self.num_min_qs < self.num_qs else None

            # update actor
            # action of the current agent is calculated using its actor
            action, logits = agent.action(obs[agent_id], model_out=True)
            act[agent_id] = action
            actor_losses = agent.critic_values(list(obs.values()), list(act.values()), indices)
            actor_loss = max([-loss.mean() for loss in actor_losses])
            actor_loss_pse = torch.pow(logits, 2).mean()
            agent.update_actor(actor_loss + 1e-3 * actor_loss_pse)
            # self.logger.info(f'agent{i}: critic loss: {critic_loss.item()}, actor loss: {actor_loss.item()}')

    def update_target(self, tau):
        def soft_update(from_network, to_network):
            """ copy the parameters of `from_network` to `to_network` with a proportion of tau"""
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        for agent in self.agents.values():
            soft_update(agent.actor, agent.target_actor)
            for i in range(self.num_qs):
                soft_update(agent.critics[i], agent.target_critics[i])

    def save(self, reward):
        """save actor parameters of all agents and training reward to `res_dir`"""
        torch.save(
            {name: agent.actor.state_dict() for name, agent in self.agents.items()},  # actor parameter
            os.path.join(self.res_dir, 'model.pt')
        )
        with open(os.path.join(self.res_dir, 'rewards.pkl'), 'wb') as f:  # save training data
            pickle.dump({'rewards': reward}, f)

    @classmethod
    def load(cls, dim_info, file, device='cpu'):
        """init maddpg using the model saved in `file`"""
        instance = cls(dim_info, 0, 0, 0, 0, os.path.dirname(file), device)
        data = torch.load(file)
        for agent_id, agent in instance.agents.items():
            agent.actor.load_state_dict(data[agent_id])
        return instance
