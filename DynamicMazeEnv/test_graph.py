"""
CT-graph.
 Copyright (C) 2019 Andrea Soltoggio, Pawel Ladosz, Eseoghene Ben-Iwhiwhu

Launch script to test one single navigation episode in automatic mode and manual mode"""
import numpy as np
import gym
from gym_CTgraph import CTgraph_env
from gym_CTgraph.CTgraph_plot import CTgraph_plot
from gym_CTgraph.CTgraph_conf import CTgraph_conf
from gym_CTgraph.CTgraph_images import CTgraph_images
import argparse
import json
import random
import matplotlib.pyplot as plt
import timeit

def printout(p_obs, p_reward, p_act, p_done, p_info, p_counter):
    """Print out the navigation variable at each step"""
    print("Feeding action: ", p_act)
    print("Step:", p_counter)
#    print("Observation: ", p_obs)
    print("Reward: ", p_reward)
    print("Done: ", p_done)
    print("--\nInfo: ", p_info)

parser = argparse.ArgumentParser(description = 'Process some integers.')
parser.add_argument('-c', '--case', default = 0, dest = "CASE",
                    help = 'exectution mode')

args = parser.parse_args()

# fetch the parameters from the json file
configuration = CTgraph_conf("graph.json")
conf_data = configuration.getParameters()
# print configration data
print(json.dumps(conf_data, indent = 3))

# instantiate the maze
start = timeit.timeit()
env = gym.make('CTgraph-v0')
end = timeit.timeit()
#print(end - start)

imageDataset = CTgraph_images(conf_data)

# initialise and get initial observation and info
observation, reward, done, info = env.init(conf_data, imageDataset)

#plotting: uncomment the following line to plot the observations
CTgraph_plot.plotImages(imageDataset, False)

# get a random path from the maze
high_reward_path = env.get_random_path()
# use this random path to set the path to the high reward. Note that the maze would have already a high_reward_path from the initialisation
env.set_high_reward_path(high_reward_path)

print("*--- Testing script ----*")

action = 0
counter = 0
print_results = True

CASE = int(args.CASE)
#interactive case: step-by-step with operator inputs
if CASE ==  0:
    print("The test script sets the high reward path to: ", env.get_high_reward_path())
    printout(observation, reward, action, done, info, counter)

    start = timeit.timeit()
    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (3, 5))
    #plt.figure(figsize = (2,2))
    axs.imshow(observation)
    plt.show(block = False)
    while not done:
        action = int(input("Action: "))
        observation, reward, done, info = env.step(action)
        counter = counter  + 1
        if print_results:
            printout(observation, reward, action, done, info, counter)
            axs.imshow(observation)
            plt.draw()
            plt.show(block = False)
    print("close images to end")
    plt.show(block = True)

#automated: for testing many episodes
if CASE ==  1:
    #tesing high rewards
    total_reward = 0
    nr_episodes = 1000
    probDelayCrash = 0.0
    probDecisionPointCrash = 0.0
    probWrongDecision = 0.5

    for test in range(0,nr_episodes):
        done = False
        observation, reward, done, info = env.complete_reset()
        high_reward_path = env.get_random_path()
        env.set_high_reward_path(high_reward_path)
        index_decision_point_actions = 0
        print("E:%d" % test, end = '')
        print(" testing path:", high_reward_path, end = '\n')
        while not done:
            # check if I'm in a delay or root stateType
            if "1" in info or "0" in info:
                action = 0
                if random.random() < probDelayCrash:
                    action = np.randint(1,env.BRANCH+2)
                print('x%d' % env.step_counter, end = '')
                observation, reward, done, info = env.step(action)
                total_reward = total_reward + reward

            if "2" in info:
                # correct action
                action = high_reward_path[index_decision_point_actions] + 1
                if random.random() < probDecisionPointCrash: #do something wrong with a small prob
                    action = 0

                if random.random() < probWrongDecision: #do something wrong with a small prob, cycling through actions
                    action = action % (env.BRANCH + 1) + 1
                print('(a:%d)' % action, end = '')
                observation, reward, done, info = env.step(action)
                index_decision_point_actions = index_decision_point_actions + 1
                total_reward = total_reward + reward

            if "3" in info:
                print("-E, R:%0.1f" % reward ," in %d" % env.step_counter, "steps")
                observation, reward, done, info = env.step(0)
                total_reward = total_reward + reward

            if "4" in info:
                print("Crash at step", env.step_counter, end = '\n')
    print("total reward: ", total_reward)
