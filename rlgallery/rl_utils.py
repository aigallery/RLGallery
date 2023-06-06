import tqdm
import numpy as np
import torch
import collections
import random
import wandb
from .registry import LOGGRS, UTILS

@LOGGRS.register_module()
class Logger:
    """
    logger for all info. using wandb
    """

    def __init__(self, wandb_project="xxx") -> None:
        self.wandb_project = wandb_project

    def create_true_instance(self):
        wandb.init(project=self.wandb_project)
        return self

    def watch_model(self, model):
        wandb.watch(model, log_freq=100)

    @property
    def save_dir(self):
        return wandb.run.dir

    def save(self, paths):
        for path in paths:
            wandb.save(path)

    def log_info(self, opt_info):

        wandb.log(opt_info)

@UTILS.register_module()
class ReplayBuffer:
    """
    经验回放池,主要包括加入数据、采样数据两大函数
    """

    def __init__(self, buffer_size=100):
        """队列，先进先出
        Args:
            buffer_size (int): 队列大小
        """
        self.buffer_size = buffer_size

    def create_true_instance(self):
        self.buffer = collections.deque(maxlen=self.buffer_size)
        return self
    
    def add(self, state, action, reward, next_state, done):
        """将数据加入buffer
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """从buffer中采样数据, 数据量为batch_size
        """
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        """目前buffer中数据的数量
        """
        return len(self.buffer)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] -
              cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
# 以上为 rl_utils.py
