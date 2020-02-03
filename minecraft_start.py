#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

'''
 Copyright (C) 2019 Andrea Soltoggio, Pawel Ladosz, Eseoghene Ben-Iwhiwhu Jeffrey Dick

MOHQA implemented at Loughborough University.'''

import matplotlib
matplotlib.use("Pdf")
from deep_rl import *
from deep_rl.agent.ApnnDQN_agent import ApnnDQNAgent
import os
#import run_maze_with_trained_model
from shutil import copy
from gym_CTgraph import CTgraph_env
from gym_CTgraph.CTgraph_plot import CTgraph_plot
from gym_CTgraph.CTgraph_conf import CTgraph_conf
from gym_CTgraph.CTgraph_images import CTgraph_images
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def dqn_minecraft(name):
    config = Config()
    #config.seed = 123451
    #config.seed = 213124
    #config.seed = 457875
    #config.seed = 578578
    config.seed = 1
#    config.max_steps = 2 * 1000000
    config.expType = \
    "dqn_pa_gs_mine_{0}_v2".format(config.seed)
    apnn_config_file_directory="./apnn_parameters.json"
    apnn_configuration = apnn_conf.apnn_conf(apnn_config_file_directory)
    config.apnn_conf_data = apnn_configuration.getParameters()
    config.expID = "baseline"
    config.log_dir = get_default_log_dir(config.expType) + config.expID
    config.episode_limit = 1400000
    config.history_length = 1
    config.task_fn = lambda: Minecraft_imaze(name, history_length=config.history_length, log_dir=config.log_dir, conf_data=None)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.001, alpha=0.95, eps=0.01)
    # config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody())
    config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody())
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0.0,3.0e6))
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.gradient_clip = 5
    config.discount = 0.99
    config.target_network_update_freq = 5000
    config.exploration_steps= 50000
    # config.double_q = True
    config.double_q = False
    config.logger = get_logger(log_dir=config.log_dir)
    copy("./minecraft_start.py",config.log_dir)
    #copy maze json file for future references
    #copy(maze_conf_file_directory,config.log_dir)
    run_episodes(DQNAgent(config))

def apnn_minecraft(name):
    config = Config()
    config.seed = 1
    #config.seed = 213124
    #config.seed = 457875
    #config.seed = 578578
    #config.seed = 54542
    #tracker = SummaryTracker()
#    config.max_steps = 2 * 1000000
    config.expType = \
    "apnn_pa_gs_mine_{0}_v2".format(config.seed)
    apnn_config_file_directory="./apnn_parameters.json"
    apnn_configuration = apnn_conf.apnn_conf(apnn_config_file_directory)
    config.apnn_conf_data = apnn_configuration.getParameters()
    config.expID = "apnn"
    config.log_dir = get_default_log_dir(config.expType) + config.expID
    config.episode_limit = 1400000
    config.history_length = 1
    config.task_fn = lambda: Minecraft_imaze(name, history_length=config.history_length, log_dir=config.log_dir, conf_data=None)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.001, alpha=0.95, eps=0.01)
    # config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody())
        # config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody())
    config.network_fn = lambda state_dim, action_dim: \
            ApnnNet(action_dim, ApnnConvBody(),config.apnn_conf_data)
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1, 0, 3.0e6))
    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.gradient_clip = 0.05
    config.discount = 0.6
    config.target_network_update_freq = 5000
    config.exploration_steps= 50000
    # config.double_q = True
    config.double_q = False
    config.logger = get_logger(log_dir=config.log_dir)
    #copy maze json and apnn config file for future references
    copy(apnn_config_file_directory,config.log_dir)
    copy("./minecraft_start.py",config.log_dir)
    #create debugging folder
    os.makedirs(config.log_dir+"/debug_matrix/")
    os.makedirs(config.log_dir+"/debug_matrix/loss")
    ## use run_episodes for training or following three lines for testing trained network with network parameter file in log directory.
    run_episodes(ApnnDQNAgent(config))
    #log="/home/lunet/ttpl6/STELLAR/ct_maze_work/DeepRL-0.3/log/dqn_pa_gs_1_d_2_bf_2_DelayProb_0.5_seed_123451CTgraph-v0/190618-155407apnn/ApnnDQNAgent-vanilla-model-CTgraph-v0.bin"
    #agent=ApnnDQNAgent(config)
    #run_maze_with_trained_model.play_the_maze(agent,log,name)
      

if __name__ ==  '__main__':
    mkdir('data/video')
    mkdir('dataset')
    mkdir('log')
    set_one_thread()
    select_device(1)
    # select different implementations of algorithms (use examples.py for further references)
    apnn_maze('CTgraph-v0')
