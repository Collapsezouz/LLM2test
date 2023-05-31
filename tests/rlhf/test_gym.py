### Init
# > pip install 'gymnasium[classic-control]'
### Test
# python -m tests.rlhf.test_gym
### Debug
# export DEBUG_PORT=5679
# REMOTE_DEBUG=1 python -m tests.rlhf.test_gym
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from tests import logger


# 设置随机种子
np.random.seed(1)
torch.manual_seed(1)

def _show_img(rgb_array):
    plt.figure(3)
    plt.clf()
    plt.imshow(rgb_array)
    plt.title("env img")
    plt.axis('off')
    plt.show()

def main(name='CartPole-v1', render_mode='rgb_array'):
    env = gym.make(name, render_mode=render_mode)
    state, _ = env.reset()
    action = 0
    logger.debug('state=%s, action=%s, %s', state, action, _)
    next_state, reward, terminated, truncated, _ = env.step(action)
    logger.debug('next_state=%s, reward=%s, terminated=%s, truncated=%s, %s', next_state, reward, terminated, truncated, _)
    x = env.render()
    # logger.debug('render: %s', x)
    _show_img(x)

if __name__ == "__main__":
    import fire
    fire.Fire(main)