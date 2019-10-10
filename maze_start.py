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


def dqn_maze(name):
    config = Config()
    config.seed = 123451
#    config.max_steps = 2 * 1000000
    maze_conf_file_directory = "./DynamicMazeEnv/graph.json"
    configuration = CTgraph_conf(maze_conf_file_directory)
    config.conf_data = configuration.getParameters()
    print(config.conf_data)
    config.expType = \
    "dqn_pa_gs_{0}_d_{1}_bf_{2}_DelayProb_{3}_seed_{4}".format(config.conf_data['general_seed'],\
                                                               config.conf_data['graph_shape']['depth'],\
                                                               config.conf_data['graph_shape']['branching_factor'],\
                                                               config.conf_data['graph_shape']['delay_prob'],\
                                                               config.seed) \
                                                               + name
    config.expID = "baseline"
    config.log_dir = get_default_log_dir(config.expType) + config.expID
    config.episode_limit = 1000000
    config.history_length = 1
    config.task_fn = lambda: Maze(name, history_length = config.history_length,
                                 log_dir = config.log_dir, conf_data = config.conf_data)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr = 0.001, alpha = 0.95, eps = 0.01)
    # config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody())
    config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody())
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0.0, 0.5e6))
    config.replay_fn = lambda: Replay(memory_size = int(1e6), batch_size = 32)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.gradient_clip = 0.1
    config.discount = 0.6
    config.target_network_update_freq = 10000
    config.exploration_steps =  50000
    # config.double_q = True
    config.double_q = False
    config.logger = get_logger(log_dir = config.log_dir)
    #copy maze json file for future references
    copy(maze_conf_file_directory,config.log_dir)
    run_episodes(DQNAgent(config))

def ddqn_maze(name):
    config = Config()
    #config.seed = 123451
    #config.seed = 213124
    #config.seed = 457875
    #config.seed = 578578
    config.seed = 123451
#    config.max_steps = 2 * 1000000
    maze_conf_file_directory = "./DynamicMazeEnv/graph.json"
    configuration = CTgraph_conf(maze_conf_file_directory)
    config.conf_data = configuration.getParameters()
    print(config.conf_data)
    config.expType = \
    "dqn_pa_gs_{0}_d_{1}_bf_{2}_DelayProb_{3}_seed_{4}".format(config.conf_data['general_seed'],\
                                                               config.conf_data['graph_shape']['depth'],\
                                                               config.conf_data['graph_shape']['branching_factor'],\
                                                               config.conf_data['graph_shape']['delay_prob'],\
                                                               config.seed) \
                                                               + name
    config.expID = "ddqn"
    config.log_dir = get_default_log_dir(config.expType) + config.expID
    config.episode_limit = 1000000
    config.history_length = 1
    config.task_fn = lambda: Maze(name, history_length = config.history_length,
                                  log_dir = config.log_dir, conf_data = config.conf_data)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr = 0.001, alpha = 0.95, eps = 0.01)
    # config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody())
    config.network_fn = lambda state_dim, action_dim: DuelingNet(action_dim, NatureConvBody())
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0.0, 0.5e6))
    config.replay_fn = lambda: Replay(memory_size = int(1e6), batch_size = 32)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.gradient_clip = 0.5
    config.discount = 0.8
    config.target_network_update_freq = 10000
    config.exploration_steps =  50000
    # config.double_q = True
    config.double_q = False
    config.logger = get_logger(log_dir = config.log_dir)
    #copy maze json file for future references
    copy(maze_conf_file_directory,config.log_dir)
    run_episodes(DQNAgent(config))

def qrdqn_maze(name):
    config = Config()
    #tracker = SummaryTracker()
#    config.max_steps = 2 * 1000000
    maze_conf_file_directory = "./DynamicMazeEnv/maze.json"
    configuration = CTMaze_conf(maze_conf_file_directory)
    config.conf_data = configuration.getParameters()
    print(config.conf_data)
    config.expType = \
    "dqn_pa_gs_{0}_d_{1}_bf_{2}_DelayProb_{3}_seed_{4}".format(config.conf_data['general_seed'],\
                                                               config.conf_data['graph_shape']['depth'],\
                                                               config.conf_data['graph_shape']['branching_factor'],\
                                                               config.conf_data['graph_shape']['delay_prob'],\
                                                               config.seed) \
                                                               + name
    config.expID = "qrdqn"
    config.log_dir = get_default_log_dir(config.expType) + config.expID
    config.episode_limit = 1000000
    config.history_length = 1
    config.task_fn = lambda: Maze(name, history_length = config.history_length,
                                  log_dir = config.log_dir, conf_data = config.conf_data)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr = 0.001, alpha = 0.95, eps = 0.01)
    #config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr = 0.001, alpha = 0.95, eps = 0.01)
    # config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody())
    config.network_fn = lambda state_dim, action_dim: \
            QuantileNet(action_dim, config.num_quantiles, NatureConvBody())
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0, 0.5e6))
    config.replay_fn = lambda: Replay(memory_size = int(1e6), batch_size = 32)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.gradient_clip = 0.1
    config.discount = 0.6
    config.target_network_update_freq = 1000
    config.exploration_steps =  15000
    # config.double_q = True
    config.double_q = False
    config.logger = get_logger(log_dir = config.log_dir)
    #copy maze json file for future references
    copy(maze_conf_file_directory,config.log_dir)
    run_episodes(QuantileRegressionDQNAgent(config))

def qrdqn_maze_lstm(name):
    config = Config()
    maze_conf_file_directory = "./DynamicMazeEnv/graph.json"
    configuration = CTMaze_conf(maze_conf_file_directory)
    config.conf_data = configuration.getParameters()
    print(config.conf_data)
    config.expType = \
    "dqn_pa_gs_{0}_d_{1}_bf_{2}_DelayProb_{3}_seed_{4}".format(config.conf_data['general_seed'],\
                                                               config.conf_data['graph_shape']['depth'],\
                                                               config.conf_data['graph_shape']['branching_factor'],\
                                                               config.conf_data['graph_shape']['delay_prob'],\
                                                               config.seed) \
                                                               + name
    config.expID = "qrdqn_lstm"
    config.log_dir = get_default_log_dir(config.expType) + config.expID
    config.episode_limit = 1000000
    config.history_length = 1
    config.task_fn = lambda: Maze(name, history_length = config.history_length,
                                 log_dir = config.log_dir, conf_data = config.conf_data)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr = 0.01, alpha = 0.95, eps = 0.01)
    config.network_fn = lambda state_dim, action_dim: \
            QuantileNet(action_dim, config.num_quantiles, NatureConvBody_lstm())
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1.0, 0, 0.5e6))
    config.replay_fn = lambda: Replay(memory_size = int(1e6), batch_size = 1)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps =  15000
    # config.double_q = True
    config.double_q = False
    config.logger = get_logger(log_dir = config.log_dir)
    copy(maze_conf_file_directory,config.log_dir)
    run_episodes(QuantileRegressionDQNAgent(config))


def apnn_maze(name):
    """The main function to run the MOHQA (note sometimes APNN is used as a name for MOHQA).

    inputs:  The name argument is the name of the environment.
    Note: this implementation is only tested in ct-graph environment
    """
    config = Config()
    config.seed = 123451
    maze_conf_file_directory = "./DynamicMazeEnv/graph.json"
    apnn_config_file_directory = "./apnn_parameters.json"
    configuration = CTgraph_conf(maze_conf_file_directory)
    apnn_configuration = CTgraph_conf(apnn_config_file_directory)
    config.conf_data = configuration.getParameters()
    config.apnn_conf_data = apnn_configuration.getParameters()
    print(config.conf_data)
    config.expType = \
    "dqn_pa_gs_{0}_d_{1}_bf_{2}_DelayProb_{3}_seed_{4}".format(config.conf_data['general_seed'],\
                                                               config.conf_data['graph_shape']['depth'],\
                                                               config.conf_data['graph_shape']['branching_factor'],\
                                                               config.conf_data['graph_shape']['delay_prob'],\
                                                               config.seed) \
                                                               + name
    config.expID = "apnn"
    config.log_dir = get_default_log_dir(config.expType) + config.expID
    config.episode_limit = 1000000
    config.history_length = 1
    config.task_fn = lambda: Maze(name, history_length = config.history_length,
                                  log_dir = config.log_dir, conf_data = config.conf_data)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr = 0.001, alpha = 0.95, eps = 0.01)
    config.network_fn = lambda state_dim, action_dim: \
            ApnnNet(action_dim, ApnnConvBody(),config.apnn_conf_data)
    config.policy_fn = lambda: GreedyPolicy(LinearSchedule(1, 0, 1e6))
    config.replay_fn = lambda: Replay(memory_size = int(1e6), batch_size = 32)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.gradient_clip = 0.1
    config.discount = 0.6
    config.target_network_update_freq = 1000
    config.exploration_steps =  15000
    config.double_q = False
    config.logger = get_logger(log_dir = config.log_dir)
    #copy maze json and apnn config file for future references
    copy(maze_conf_file_directory,config.log_dir)
    copy(apnn_config_file_directory,config.log_dir)
    #create debugging folder
    os.makedirs(config.log_dir+"/debug_matrix/")
    os.makedirs(config.log_dir+"/debug_matrix/loss")
    run_episodes(ApnnDQNAgent(config))

if __name__ ==  '__main__':
    mkdir('data/video')
    mkdir('dataset')
    mkdir('log')
    set_one_thread()
    select_device(1)
    # select different implementations of algorithms (use examples.py for further references)
    apnn_maze('CTgraph-v0')
