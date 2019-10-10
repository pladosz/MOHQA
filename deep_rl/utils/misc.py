#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import pickle
import os
import datetime
import torch
from .torch_utils import *
#from io import BytesIO
#import scipy.misc
#import torchvision

try:
    # python >=  3.5
    from pathlib import Path
except:
    # python ==  2.7
    from pathlib2 import Path

def run_episodes(agent):
    config = agent.config
    random_seed(config.seed)
    window_size = 100
    ep = 0
    rewards = []
    steps = []
    avg_test_rewards = []
    agent_type = agent.__class__.__name__
    while True:
        ep +=  1
        reward, step = agent.episode(episode = ep)
        agent.network.body.reset_flag = True
        torch.cuda.empty_cache()
        #agent.network.body.hidden = repackage_hidden(agent.network.body.hidden)
        rewards.append(reward)
        steps.append(step)
        avg_reward = np.mean(rewards[-window_size:])
        config.logger.info('episode %d, reward %f, avg reward %f, total steps %d, episode step %d' % (
            ep, reward, avg_reward, agent.total_steps, step))
        #L2M changes:
        config.logger.scalar_summary('reward', reward)
        config.logger.scalar_summary('max reward', np.max(rewards[-window_size:]))
        config.logger.scalar_summary('avg reward', avg_reward)
        if config.save_interval and ep % config.save_interval ==  0:
            with open(config.log_dir + '/%s-%s-online-stats-%s.bin' % (
                    agent_type, config.tag, agent.task.name), 'wb') as f:
                pickle.dump([steps, rewards], f)
            agent.save(config.log_dir + '/%s-%s-model-%s.bin' % (agent_type, config.tag, agent.task.name))
            np.save(config.log_dir + '/rewards',rewards)
        #    for tag, value in agent.network.named_parameters():
        #        tag = tag.replace('.', '/')
        #        config.logger.histo_summary(tag, np.nan_to_num(value.data.cpu().numpy()))
            if config.log_modulation:
                mod_avg = torch.mean(agent.network.body.y_mod0) + torch.mean(agent.network.body.y_mod1) + torch.mean(agent.network.body.y_mod2)
                mod_std = torch.std(agent.network.body.y_mod0) + torch.std(agent.network.body.y_mod1) + torch.std(agent.network.body.y_mod2)
                mod_max_l0 = torch.max(agent.network.body.y_mod0)
                mod_max_l1 = torch.max(agent.network.body.y_mod1)
                mod_max_l2 = torch.max(agent.network.body.y_mod2)
                mod_min_l1 = torch.min(agent.network.body.y_mod1)
                config.logger.scalar_summary('z_mod avg', mod_avg/3)
                config.logger.scalar_summary('z_mod std', mod_std/3)
                config.logger.scalar_summary('z_mod min l1', mod_min_l1)
                config.logger.scalar_summary('z_mod max l0', mod_max_l0)
                config.logger.scalar_summary('z_mod max l1', mod_max_l1)
                config.logger.scalar_summary('z_mod max l2', mod_max_l2)

                conv1MW = agent.network.body.conv1_mem_features.weight
                conv2MW = agent.network.body.conv2_mem_features.weight
                conv3MW = agent.network.body.conv2_mem_features.weight
                conv1 = agent.network.body.conv1.weight
                conv2 = agent.network.body.conv2.weight
                conv3 = agent.network.body.conv3.weight
                mod0 = agent.network.body.y_mod0
                mod1 = agent.network.body.y_mod1
                mod2 = agent.network.body.y_mod2

                #agent.network.body.conv1_mem_features.weight[0:32,0].detach().numpy()}
            #    batch_tensor = conv1MW[0:10,0]
            #    grid_img = torchvision.utils.make_grid(batch_tensor, nrow = 5)
                for i in range(2):
                    config.logger.image_summary('conv1_mem_f' + str(i), conv1MW[i,0])
                    config.logger.image_summary('conv1' + str(i), conv1[i,0])
#                    print('mod0.shape', mod0.shape)
                    config.logger.image_summary('mod0' + str(i), mod0[0,i])
                for i in range(2):
                    config.logger.image_summary('conv2_mem_f' + str(i), conv2MW[i,0])
                    config.logger.image_summary('conv2' + str(i), conv2[i,0])
                    config.logger.image_summary('mod1' + str(i), mod1[0,i])
                for i in range(2):
                    config.logger.image_summary('conv3_mem_f' + str(i), conv3MW[i,0])
                    config.logger.image_summary('conv3' + str(i), conv3[i,0])
                    config.logger.image_summary('mod2' + str(i), mod2[0,i])

    #            import matplotlib.pyplot as plt
    #            fig, axs = plt.subplots(nrows = 5, ncols = 5, figsize = (12.3, 9),
    #                            subplot_kw = {'xticks': [], 'yticks': []})

    #            fig.subplots_adjust(left = 0.03, right = 0.97, hspace = 0.3, wspace = 0.05)
    #            numbers = np.arange(25)
    #            for ax, image_nr in zip(axs.flat, numbers):
                # this line shows the standard set
    #            ax.imshow(imageDataset.getNoisyImage(image_nr))
    #                ax.imshow(conv1MW[image_nr,0].detach().numpy())
    #            plt.tight_layout()
    #            plt.show()
    #            input()

        if config.episode_limit and ep > config.episode_limit:
            with open(config.log_dir + '/%s-%s-online-stats-%s.bin' % (
                    agent_type, config.tag, agent.task.name), 'wb') as f:
                pickle.dump([steps, rewards], f)
            agent.save(config.log_dir + '/%s-%s-model-%s.bin' % (agent_type, config.tag, agent.task.name))
            break

        if config.max_steps and agent.total_steps > config.max_steps:
            with open(config.log_dir + '/%s-%s-online-stats-%s.bin' % (
                    agent_type, config.tag, agent.task.name), 'wb') as f:
                pickle.dump([steps, rewards], f)
            agent.save(config.log_dir + '/%s-%s-model-%s.bin' % (agent_type, config.tag, agent.task.name))
            break

    agent.close()
    return steps, rewards, avg_test_rewards

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def run_iterations(agent):
    config = agent.config
    random_seed(config.seed)
    agent_name = agent.__class__.__name__
    iteration = 0
    steps = []
    rewards = []
    while True:
        agent.iteration()
        steps.append(agent.total_steps)
        rewards.append(np.mean(agent.last_episode_rewards))
        if iteration % config.iteration_log_interval ==  0:
            config.logger.info('total steps %d, mean/max/min reward %f/%f/%f' % (
                agent.total_steps, np.mean(agent.last_episode_rewards),
                np.max(agent.last_episode_rewards),
                np.min(agent.last_episode_rewards)
        ))
            config.logger.scalar_summary('avg reward', np.mean(agent.last_episode_rewards))
            config.logger.scalar_summary('max reward', np.max(agent.last_episode_rewards))
            config.logger.scalar_summary('min reward', np.min(agent.last_episode_rewards))

        if iteration % (config.iteration_log_interval * 20) ==  0:

            with open(config.log_dir + '/%s-%s-online-stats-%s.bin' % (agent_name, config.tag, agent.task.name), 'wb') as f:
                pickle.dump({'rewards': rewards,
                             'steps': steps}, f)
            agent.save(config.log_dir + '/%s-%s-model-%s.bin' % (agent_name, config.tag, agent.task.name))
            for tag, value in agent.network.named_parameters():
                tag = tag.replace('.', '/')
                config.logger.histo_summary(tag, value.data.cpu().numpy())
            if config.log_modulation:
#                print(dir(agent.network.network))
#                   input()
                mod_avg = torch.mean(agent.network.network.phi_body.y_mod0) + torch.mean(agent.network.network.phi_body.y_mod1) + torch.mean(agent.network.network.phi_body.y_mod2)
                mod_std = torch.std(agent.network.network.phi_body.y_mod0) + torch.std(agent.network.network.phi_body.y_mod1) + torch.std(agent.network.network.phi_body.y_mod2)
                mod_max_l1 = torch.max(agent.network.network.phi_body.y_mod1)
                mod_min_l1 = torch.min(agent.network.network.phi_body.y_mod1)
                config.logger.scalar_summary('z_mod avg', mod_avg/3)
                config.logger.scalar_summary('z_mod std', mod_std/3)
                config.logger.scalar_summary('z_mod min l1', mod_min_l1)
                config.logger.scalar_summary('z_mod max l1', mod_max_l1)

                conv1MW = agent.network.network.phi_body.conv1_mem_features.weight
                for i in range(10):
                    config.logger.image_summary('conv1_sample' + str(i), conv1MW[i,0])


        iteration +=  1
        if config.max_steps and agent.total_steps >=  config.max_steps:
            with open(config.log_dir + '/%s-%s-online-stats-%s.bin' % (
                agent_name, config.tag, agent.task.name), 'wb') as f:
                pickle.dump([steps, rewards], f)
                agent.save(config.log_dir + '/%s-%s-model-%s.bin' % (agent_name, config.tag, agent.task.name))
            agent.close()
            break

    return steps, rewards

def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")

def get_default_log_dir(name):
    return './log/%s/%s' % (name, get_time_str())

def sync_grad(target_network, src_network):
    for param, src_param in zip(target_network.parameters(), src_network.parameters()):
        param._grad = src_param.grad.clone()

def mkdir(path):
    Path(path).mkdir(parents = True, exist_ok = True)

class Batcher:
    def __init__(self, batch_size, data):
        self.batch_size = batch_size
        self.data = data
        self.num_entries = len(data[0])
        self.reset()

    def reset(self):
        self.batch_start = 0
        self.batch_end = self.batch_start + self.batch_size

    def end(self):
        return self.batch_start >=  self.num_entries

    def next_batch(self):
        batch = []
        for d in self.data:
            batch.append(d[self.batch_start: self.batch_end])
        self.batch_start = self.batch_end
        self.batch_end = min(self.batch_start + self.batch_size, self.num_entries)
        return batch

    def shuffle(self):
        indices = np.arange(self.num_entries)
        np.random.shuffle(indices)
        self.data = [d[indices] for d in self.data]
