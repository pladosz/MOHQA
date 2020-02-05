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


class ApnnDQNAgent(BaseAgent):
    """Reponsible for running the graph and calling appropriate network functions"""
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

    def episode(self, deterministic = False,episode = 0):
        episode_start_time = time.time()
        state = self.task.reset()[0]
        state=np.transpose(state,(2,0,1))
        total_reward = 0.0
        steps = 0
        self.network.reset()
        y_apnn_out_stack = np.zeros((self.network.output_dim))
        y_apnn_in_stack = np.zeros((self.network.body.feature_dim))
        y_dqn_out_stack = np.zeros((self.network.output_dim))
        y_conv_layers_stack = np.zeros((256))
        state_type_list = []
        Eligbility_traces_stack = np.zeros((self.network.output_dim,self.network.body.feature_dim))
        while True:
            value = self.network.predict(np.stack([self.config.state_normalizer(state)]),steps, True).flatten()
            value = value+self.network.y_apnn_out.cpu().detach().numpy().flatten()
            if deterministic:
                action = np.argmax(value)
            elif self.total_steps < self.config.exploration_steps:
                action = np.random.randint(0, len(value))
            else:
                action = self.policy.sample(value)
            self.network.compute_eligbility_traces_action(value,action)
            next_state, reward, done, info = self.task.step(action)
            next_state=np.transpose(next_state,(2,0,1))
            if steps == 2:
                action_storage = action
                self.network.action_storage = action_storage
            y_apnn_out_stack = np.vstack([y_apnn_out_stack, self.network.y_apnn_out.cpu()])
            y_apnn_in_stack = np.vstack([y_apnn_in_stack, self.network.y_apnn_in.cpu()])
            y_dqn_out_stack = np.vstack([y_dqn_out_stack, self.network.y_dqn_out_detached.cpu()])
            Eligbility_traces_stack = np.dstack([Eligbility_traces_stack, self.network.Eligbility_traces.cpu()])
            total_reward +=  reward
            reward = self.config.reward_normalizer(reward)
            if not deterministic:
                self.replay.feed([state, action, reward, next_state, int(done),\
                 self.network.Eligbility_traces, self.network.modulation,\
                  self.network.step, self.network.step_previous,episode, \
                  self.network.Theta, self.network.running_average])
                self.total_steps +=  1
            steps +=  1
            self.network.update(reward,steps,episode)
            state = next_state
            if not deterministic and self.total_steps > self.config.exploration_steps \
                    and self.total_steps % self.config.sgd_update_frequency ==  0:
                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals, \
                Eligbility_traces, modulation, network_step, \
                network_step_previous,episode_replay, thetas, \
                running_average = experiences
                states = self.config.state_normalizer(states)
                next_states = self.config.state_normalizer(next_states)
                self.target_network.running_average = running_average
                q_next = self.target_network.predict(next_states, network_step, False).detach()
                q_next = q_next+self.target_network.y_apnn_out.detach()
                if self.config.double_q:
                    _, best_actions = self.network.predict(next_states).detach().max(1)
                    q_next = q_next.gather(1, best_actions.unsqueeze(1)).squeeze(1)
                else:
                    q_next, max_indecies = q_next.max(1)
                terminals = tensor(terminals)
                rewards = tensor(rewards)
                q_next = self.config.discount * q_next * (1 - terminals)
                q_next.add_(rewards)
                actions = tensor(actions).unsqueeze(1).long()
                Eligbility_traces_temp = self.network.Eligbility_traces
                modulation_temp = self.network.modulation
                network_step_previous_temp = self.network.step_previous
                running_average_temp = self.network.running_average
                theta_temp = self.network.Theta
                self.network.Eligbility_traces = Eligbility_traces
                self.network.modulation = modulation
                self.network.step_previous = network_step_previous
                self.network.Theta = thetas
                self.network.running_average = running_average
                q = self.network.predict(states,network_step, False)
                q = q.gather(1, actions).squeeze(1)
                loss = self.criterion(q, q_next)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
                self.optimizer.step()
                self.network.Eligbility_traces = Eligbility_traces_temp
                self.network.modulation = modulation_temp
                self.network.step_previous = network_step_previous_temp
                self.network.Theta = theta_temp
                self.network.running_average = running_average_temp
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
        if(episode % 100 ==  0):
            file_path = self.config.log_dir+"/debug_matrix/"
            np.save(file_path+"y_apnn_out_stack_{0}".format(episode),y_apnn_out_stack)
            np.save(file_path+"y_dqn_out_stack_{0}".format(episode),y_dqn_out_stack)
            np.save(file_path+"y_apnn_in_stack_{0}".format(episode),y_apnn_in_stack)
            np.save(file_path+"Eligbility_traces_stack_{0}".format(episode),Eligbility_traces_stack)
            np.save(file_path+"apnn_weights_{0}".format(episode),
                    self.network.apnn_head_output.weight.data.cpu().detach())
            np.save(file_path+"reward_{0}".format(episode),total_reward)
        return total_reward, steps
