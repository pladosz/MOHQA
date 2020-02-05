#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
from .atari_wrapper import *
import multiprocessing as mp
import sys
from .bench import Monitor
from ..utils import *
import uuid
#from gym_CTgraph import CTgraph_env
#from gym_CTgraph.CTgraph_plot import CTgraph_plot
#from gym_CTgraph.CTgraph_conf import CTgraph_conf
#from gym_CTgraph.CTgraph_images import CTgraph_images
import gym
from mcgridenv import MazeDepth2v4Grid


class BaseTask:
    def __init__(self):
        pass

    def set_monitor(self, env, log_dir):
        if log_dir is None:
            return env
        mkdir(log_dir)
        return Monitor(env, '%s/%s' % (log_dir, uuid.uuid4()))

    def reset(self):
        #print 'base task reset called'
        return self.env.reset()

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        #print 'base task step called'
        #print self.env
        #print done
        if done:
            next_state = self.env.reset()[0]
        return next_state, reward, done, info

    def seed(self, random_seed):
        return self.env.seed(random_seed)

class ClassicalControl(BaseTask):
    def __init__(self, name = 'CartPole-v0', max_steps = 200, log_dir = None):
        BaseTask.__init__(self)
        self.name = name
        self.env = gym.make(self.name)
        self.env._max_episode_steps = max_steps
        self.action_dim = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]
        self.env = self.set_monitor(self.env, log_dir)

class Maze(BaseTask):
    def __init__(self, name, seed = 0, log_dir = None,
                 frame_skip = 4, history_length = 4, dataset = False, conf_data = {}):
        #attempt to load default dictionary if no config is given
        if not conf_data:
            configuration = CTgraph_conf("./DynamicMazeEnv/graph.json")
            conf_data = configuration.getParameters()
        BaseTask.__init__(self)
        env = gym.make(name)
        imageDataset = CTgraph_images(conf_data)
        observation, reward, done, info = env.init(conf_data, imageDataset)
        if dataset:
            env = DatasetEnv(env)
            self.dataset_env = env
        env = self.set_monitor(env, log_dir)
        #env = wrap_deepmind(env, history_length = history_length)
        self.env = env
        self.action_dim  = 3#self.env.action_space.n
        self.state_dim = 1#self.env.observation_space.shape
        self.name = name
        self.conf_data = conf_data


class Minecraft_imaze(BaseTask):
    def __init__(self, name, seed=0, log_dir=None,
                     frame_skip=4, history_length=4, dataset=False, conf_data={}):
        #define parameters
        BaseTask.__init__(self)
        env = MazeDepth2v4Grid(max_num_turns=5, max_nav_actions=70)
        print(env.get_goals())
        env.set_goal(np.array([10,11]))
        obs, reward, done, info  = env.reset()
        self.env = env
        self.action_dim =3#self.env.action_space.n
        self.state_dim = 3#self.env.observation_space.shape
        self.name = name
        self.conf_data=conf_data

    def step(self, action):
        if action == 0:
            action = 0
        elif action == 1:
            action = 2
        elif action == 2:
            action = 3
        print(action)
        next_state, reward, done, info = self.env.step(action)
        #print 'base task step called'
        #print self.env
        if done:
            next_state = self.env.reset()[0]
        return next_state, reward, done, info
        
class PixelAtari(BaseTask):
    def __init__(self, name, seed = 0, log_dir = None,
                 frame_skip = 4, history_length = 4, dataset = False):
        BaseTask.__init__(self)
        env = make_atari(name, frame_skip)
        env.seed(seed)
        if dataset:
            env = DatasetEnv(env)
            self.dataset_env = env
        env = self.set_monitor(env, log_dir)
        env = wrap_deepmind(env, history_length = history_length)
        self.env = env
        self.action_dim = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape
        self.name = name

class RamAtari(BaseTask):
    def __init__(self, name, no_op, frame_skip, log_dir = None):
        BaseTask.__init__(self)
        self.name = name
        env = gym.make(name)
        assert 'NoFrameskip' in env.spec.id
        env = self.set_monitor(env, log_dir)
        env = EpisodicLifeEnv(env)
        env = NoopResetEnv(env, noop_max = no_op)
        env = SkipEnv(env, skip = frame_skip)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        self.env = env
        self.action_dim = self.env.action_space.n
        self.state_dim = 128

class Pendulum(BaseTask):
    def __init__(self, log_dir = None):
        BaseTask.__init__(self)
        self.name = 'Pendulum-v0'
        self.env = gym.make(self.name)
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]
        self.env = self.set_monitor(self.env, log_dir)

    def step(self, action):
        return BaseTask.step(self, np.clip(2 * action, -2, 2))

class Box2DContinuous(BaseTask):
    def __init__(self, name, log_dir = None):
        BaseTask.__init__(self)
        self.name = name
        self.env = gym.make(self.name)
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]
        self.env = self.set_monitor(self.env, log_dir)

    def step(self, action):
        return BaseTask.step(self, np.clip(action, -1, 1))

class Roboschool(BaseTask):
    def __init__(self, name, log_dir = None):
        import roboschool
        BaseTask.__init__(self)
        self.name = name
        self.env = gym.make(self.name)
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]
        self.env = self.set_monitor(self.env, log_dir)

    def step(self, action):
        return BaseTask.step(self, np.clip(action, -1, 1))

class Bullet(BaseTask):
    def __init__(self, name, log_dir = None):
        import pybullet_envs
        BaseTask.__init__(self)
        self.name = name
        self.env = gym.make(name)
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]
        self.env = self.set_monitor(self.env, log_dir)

    def step(self, action):
        return BaseTask.step(self, np.clip(action, -1, 1))

class PixelBullet(BaseTask):
    def __init__(self, name, seed = 0, log_dir = None, frame_skip = 4, history_length = 4):
        import pybullet_envs
        self.name = name
        env = gym.make(name)
        env.seed(seed)
        env = RenderEnv(env)
        env = self.set_monitor(env, log_dir)
        env = SkipEnv(env, skip = frame_skip)
        env = WarpFrame(env)
        env = WrapPyTorch(env)
        if history_length:
            env = StackFrame(env, history_length)
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape
        self.env = env

class ProcessTask:
    def __init__(self, task_fn, log_dir = None):
        self.pipe, worker_pipe = mp.Pipe()
        self.worker = ProcessWrapper(worker_pipe, task_fn, log_dir)
        self.worker.start()
        self.pipe.send([ProcessWrapper.SPECS, None])
        self.state_dim, self.action_dim, self.name = self.pipe.recv()

    def step(self, action):
        self.pipe.send([ProcessWrapper.STEP, action])
        return self.pipe.recv()

    def reset(self):
        self.pipe.send([ProcessWrapper.RESET, None])
        return self.pipe.recv()

    def close(self):
        self.pipe.send([ProcessWrapper.EXIT, None])

class ProcessWrapper(mp.Process):
    STEP = 0
    RESET = 1
    EXIT = 2
    SPECS = 3
    def __init__(self, pipe, task_fn, log_dir):
        mp.Process.__init__(self)
        self.pipe = pipe
        self.task_fn = task_fn
        self.log_dir = log_dir

    def run(self):
        np.random.seed()
        seed = np.random.randint(0, sys.maxsize)
        task = self.task_fn(log_dir = self.log_dir)
        task.seed(seed)
        while True:
            op, data = self.pipe.recv()
            if op ==  self.STEP:
                self.pipe.send(task.step(data))
            elif op ==  self.RESET:
                self.pipe.send(task.reset())
            elif op ==  self.EXIT:
                self.pipe.close()
                return
            elif op ==  self.SPECS:
                self.pipe.send([task.state_dim, task.action_dim, task.name])
            else:
                raise Exception('Unknown command')

class ParallelizedTask:
    def __init__(self, task_fn, num_workers, log_dir = None, single_process = False):
        if single_process:
            self.tasks = [task_fn(log_dir = log_dir) for _ in range(num_workers)]
        else:
            self.tasks = [ProcessTask(task_fn, log_dir) for _ in range(num_workers)]
        self.state_dim = self.tasks[0].state_dim
        self.action_dim = self.tasks[0].action_dim
        self.name = self.tasks[0].name
        self.single_process = single_process

    def step(self, actions):
        results = [task.step(action) for task, action in zip(self.tasks, actions)]
        results = map(lambda x: np.stack(x), zip(*results))
        return results

    def reset(self):
        results = [task.reset() for task in self.tasks]
        return np.stack(results)

    def close(self):
        if self.single_process:
            return
        for task in self.tasks: task.close()
