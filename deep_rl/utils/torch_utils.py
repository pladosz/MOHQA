#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .config import *
import torch
import os

def select_device(gpu_id):
    if torch.cuda.is_available() and gpu_id >=  0:
        #onfig.DEVICE = torch.device('cuda:%d' % (gpu_id))
        Config.DEVICE = torch.device('cuda')
    else:
        Config.DEVICE = torch.device('cpu')

def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = torch.tensor(x, device = Config.DEVICE, dtype = torch.float32)
    return x

def range_tensor(end):
    return torch.arange(end).to(Config.DEVICE)

def to_np(t):
    return t.cpu().detach().numpy()

def random_seed(seed):
#    np.random.seed()
#    torch.manual_seed(np.random.randint(int(1e6)))
    np.random.seed(seed)
    torch.manual_seed(seed)

def set_one_thread():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
