#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from ..utils import *
import time
from .BaseAgent import *

class QuantileRegressionDQNAgent_mod(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.criterion = nn.MSELoss()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.policy = config.policy_fn()
        self.total_steps = 0
        self.quantile_weight = 1.0 / self.config.num_quantiles
        self.cumulative_density = tensor(
            (2 * np.arange(self.config.num_quantiles) + 1) / (2.0 * self.config.num_quantiles))
        self.mem_update_rate = 0.1;


    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1 - cond)

    def evaluation_action(self, state, x_mem):
        value = self.network.predict(np.stack([self.config.state_normalizer(state)]), np.stack([self.config.state_normalizer(x_mem)])).squeeze(0).detach()
        value = (value * self.quantile_weight).sum(-1).cpu().detach().numpy().flatten()
        return np.argmax(value)

    def episode(self, deterministic = False):
        episode_start_time = time.time()
        state = self.task.reset()
        memory = np.asarray(state)
        total_reward = 0.0
        steps = 0
        action_list = []
        while True:
#            memory = np.asarray((1-self.mem_update_rate)*memory+self.mem_update_rate*np.asarray(state))
            memory = (np.multiply(1-self.mem_update_rate,memory)+np.multiply(self.mem_update_rate,np.asarray(state))).astype(np.uint8)
            value = self.network.predict(np.stack([self.config.state_normalizer(state)]),np.stack([self.config.state_normalizer(memory)])).squeeze(0).detach()
            value = (value * self.quantile_weight).sum(-1).cpu().detach().numpy().flatten()
            if deterministic:
                action = np.argmax(value)
            elif self.total_steps < self.config.exploration_steps:
                action = np.random.randint(0, len(value))
            else:
                action = self.policy.sample(value)
            next_state, reward, done, _ = self.task.step(action)
            total_reward +=  reward
            reward = self.config.reward_normalizer(reward)
            if not deterministic:
                self.replay.feed([state, memory, action, reward, next_state, int(done)])
                self.total_steps +=  1
            steps +=  1
            state = next_state
            if not deterministic and self.total_steps > self.config.exploration_steps \
                    and self.total_steps % self.config.sgd_update_frequency ==  0:
                experiences = self.replay.sample()
                states, memories, actions, rewards, next_states, terminals = experiences
                states = self.config.state_normalizer(states)
                next_states = self.config.state_normalizer(next_states)
                memories = self.config.state_normalizer(memories)
                quantiles_next = self.target_network.predict(next_states,memories).detach()
                q_next = (quantiles_next * self.quantile_weight).sum(-1)
                _, a_next = torch.max(q_next, dim = 1)
                a_next = a_next.view(-1, 1, 1).expand(-1, -1, quantiles_next.size(2))
                quantiles_next = quantiles_next.gather(1, a_next).squeeze(1)

                rewards = tensor(rewards)
                terminals = tensor(terminals)
                quantiles_next = rewards.view(-1, 1) + self.config.discount * (1 - terminals.view(-1, 1)) * quantiles_next

                quantiles = self.network.predict(states,memories)
                actions = tensor(actions).long()
                actions = actions.view(-1, 1, 1).expand(-1, -1, quantiles.size(2))
                quantiles = quantiles.gather(1, actions).squeeze(1)

                quantiles_next = quantiles_next.t().unsqueeze(-1)
                diff = quantiles_next - quantiles
                loss = self.huber(diff) * (self.cumulative_density.view(1, -1) - (diff.detach() < 0).float()).abs()

                self.optimizer.zero_grad()
                loss.mean(0).mean(1).sum().backward()
                self.optimizer.step()

            self.evaluate()
            if not deterministic and self.total_steps % self.config.target_network_update_freq ==  0:
                self.target_network.load_state_dict(self.network.state_dict())
            if not deterministic and self.total_steps > self.config.exploration_steps:
                self.policy.update_epsilon()

            if done:
                break

        episode_time = time.time() - episode_start_time
        self.config.logger.debug('episode steps %d, episode time %f, time per step %f' %
                          (steps, episode_time, episode_time / float(steps)))
        return total_reward, steps

class QuantileRegressionDQNAgent_mod_surp(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.criterion = nn.MSELoss()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.policy = config.policy_fn()
        self.total_steps = 0
        self.quantile_weight = 1.0 / self.config.num_quantiles
        self.cumulative_density = tensor(
            (2 * np.arange(self.config.num_quantiles) + 1) / (2.0 * self.config.num_quantiles))
        self.mem_update_rate = 0.1;


    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1 - cond)

    def evaluation_action(self, state, x_mem):
        value = self.network.predict(np.stack([self.config.state_normalizer(state)]), np.stack([self.config.state_normalizer(x_mem)])).squeeze(0).detach()
        value = (value * self.quantile_weight).sum(-1).cpu().detach().numpy().flatten()
        return np.argmax(value)

    def episode(self, deterministic = False):
        episode_start_time = time.time()
        state = self.task.reset()
        memory = np.asarray(state)
        total_reward = 0.0
        steps = 0
        while True:
            memory = (np.multiply(1-self.mem_update_rate,memory)+np.multiply(self.mem_update_rate,np.asarray(state))).astype(np.uint8)
            novel_x = state - memory
            value = self.network.predict(np.stack([self.config.state_normalizer(state)]),np.stack([self.config.state_normalizer(novel_x)])).squeeze(0).detach()
            value = (value * self.quantile_weight).sum(-1).cpu().detach().numpy().flatten()
            if deterministic:
                action = np.argmax(value)
            elif self.total_steps < self.config.exploration_steps:
                action = np.random.randint(0, len(value))
            else:
                action = self.policy.sample(value)
            next_state, reward, done, _ = self.task.step(action)
            total_reward +=  reward
            reward = self.config.reward_normalizer(reward)
            if not deterministic:
                self.replay.feed([state, novel_x, action, reward, next_state, int(done)])
                self.total_steps +=  1
            steps +=  1
            state = next_state
            if not deterministic and self.total_steps > self.config.exploration_steps \
                    and self.total_steps % self.config.sgd_update_frequency ==  0:
                experiences = self.replay.sample()
                states, novel_xs, actions, rewards, next_states, terminals = experiences
                states = self.config.state_normalizer(states)
                next_states = self.config.state_normalizer(next_states)
                novel_xs = self.config.state_normalizer(novel_xs)
                quantiles_next = self.target_network.predict(next_states,novel_xs).detach()
                q_next = (quantiles_next * self.quantile_weight).sum(-1)
                _, a_next = torch.max(q_next, dim = 1)
                a_next = a_next.view(-1, 1, 1).expand(-1, -1, quantiles_next.size(2))
                quantiles_next = quantiles_next.gather(1, a_next).squeeze(1)

                rewards = tensor(rewards)
                terminals = tensor(terminals)
                quantiles_next = rewards.view(-1, 1) + self.config.discount * (1 - terminals.view(-1, 1)) * quantiles_next

                quantiles = self.network.predict(states,novel_xs)
                actions = tensor(actions).long()
                actions = actions.view(-1, 1, 1).expand(-1, -1, quantiles.size(2))
                quantiles = quantiles.gather(1, actions).squeeze(1)

                quantiles_next = quantiles_next.t().unsqueeze(-1)
                diff = quantiles_next - quantiles
                loss = self.huber(diff) * (self.cumulative_density.view(1, -1) - (diff.detach() < 0).float()).abs()

                self.optimizer.zero_grad()
                loss.mean(0).mean(1).sum().backward()
                self.optimizer.step()

            self.evaluate()
            if not deterministic and self.total_steps % self.config.target_network_update_freq ==  0:
                self.target_network.load_state_dict(self.network.state_dict())
            if not deterministic and self.total_steps > self.config.exploration_steps:
                self.policy.update_epsilon()

            if done:
                break

        episode_time = time.time() - episode_start_time
        self.config.logger.debug('episode steps %d, episode time %f, time per step %f' %
                          (steps, episode_time, episode_time / float(steps)))
        return total_reward, steps

class QuantileRegressionDQNAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.criterion = nn.MSELoss()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.policy = config.policy_fn()
        self.total_steps = 0
        self.quantile_weight = 1.0 / self.config.num_quantiles
        self.cumulative_density = tensor(
            (2 * np.arange(self.config.num_quantiles) + 1) / (2.0 * self.config.num_quantiles))

    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1 - cond)

    def evaluation_action(self, state):
        value = self.network.predict(np.stack([self.config.state_normalizer(state)])).squeeze(0).detach()
        value = (value * self.quantile_weight).sum(-1).cpu().detach().numpy().flatten()
        return np.argmax(value)

    def episode(self, deterministic = False,episode = 0):
        self.target_network.zero_grad()
        self.network.zero_grad()
        episode_start_time = time.time()
        state = self.task.reset()[0]
        state = np.expand_dims(state,axis = 0)
        total_reward = 0.0
        steps = 0
        action_list = []
        while True:
            prediction_tensor = np.stack([self.config.state_normalizer(state)])
            value = self.network.predict(prediction_tensor).squeeze(0).detach()
            value = (value * self.quantile_weight).sum(-1).cpu().detach().numpy().flatten()
            if deterministic:
                action = np.argmax(value)
            elif self.total_steps < self.config.exploration_steps:
                action = np.random.randint(0, len(value))
            else:
                action = self.policy.sample(value)
            next_state, reward, done, info = self.task.step(action)
            action_list.append(action)
            if len(next_state.shape)<3:
                next_state = np.expand_dims(next_state,axis = 0)
            total_reward +=  reward
            reward = self.config.reward_normalizer(reward)
            #if episode>240000 and (action == 1 or action == 0):
            #    print(value)
            #    print(action_list)
            #    print(reward)
            #    time.sleep(2)
            if not deterministic:
                self.replay.feed([state, action, reward, next_state, int(done), self.network.body.repackage_hidden(self.network.body.hidden)])
                self.total_steps +=  1
            steps +=  1
            state = next_state

            if not deterministic and self.total_steps > self.config.exploration_steps \
                    and self.total_steps % self.config.sgd_update_frequency ==  0:
                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals, hidden = experiences
                states = self.config.state_normalizer(states)
                next_states = self.config.state_normalizer(next_states)
                hidden = (hidden[0][0],hidden[0][1])
                self.target_network.body.hidden = hidden
                quantiles_next = self.target_network.predict(next_states).detach()
                q_next = (quantiles_next * self.quantile_weight).sum(-1)
                _, a_next = torch.max(q_next, dim = 1)
                a_next = a_next.view(-1, 1, 1).expand(-1, -1, quantiles_next.size(2))
                quantiles_next = quantiles_next.gather(1, a_next).squeeze(1)

                rewards = tensor(rewards)
                terminals = tensor(terminals)
                quantiles_next = rewards.view(-1, 1) + self.config.discount * (1 - terminals.view(-1, 1)) * quantiles_next
                hidden_temp = self.network.body.hidden
                #self.network.body.hidden = self.network.body.repackage_hidden(hidden_temp)
                self.network.body.hidden = hidden
                quantiles = self.network.predict(states)
                self.network.body.hidden = self.network.body.repackage_hidden(hidden)
                self.network.body.hidden = hidden_temp
                actions = tensor(actions).long()
                actions = actions.view(-1, 1, 1).expand(-1, -1, quantiles.size(2))
                quantiles = quantiles.gather(1, actions).squeeze(1)

                quantiles_next = quantiles_next.t().unsqueeze(-1)
                diff = quantiles_next - quantiles
                loss = self.huber(diff) * (self.cumulative_density.view(1, -1) - (diff.detach() < 0).float()).abs()

                self.optimizer.zero_grad()
                loss.mean(0).mean(1).sum().backward()
                self.optimizer.step()
                self.target_network.body.reset_flag = True

            self.evaluate()
            if not deterministic and self.total_steps % self.config.target_network_update_freq ==  0:
                self.target_network.load_state_dict(self.network.state_dict())
            if not deterministic and self.total_steps > self.config.exploration_steps:
                self.policy.update_epsilon()

            if done:
                break

        episode_time = time.time() - episode_start_time
        self.config.logger.debug('episode steps %d, episode time %f, time per step %f' %
                          (steps, episode_time, episode_time / float(steps)))
        return total_reward, steps
