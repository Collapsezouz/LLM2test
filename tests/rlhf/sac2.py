# -*- coding: utf-8 -*-
"""
Proximal Policy Optimization (PPO)
代码来源: https://wandb.ai/takuyamagata/RLRG_AC/reports/Proximal-Policy-Optimization-PPO-on-OpenAI-Gym--VmlldzoxMTY2ODU

# Init
> pip install 'gymnasium[classic-control]'
> pip install 'gymnasium[box2d]'

# Run
> python -m tests.rlhf.sac2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import argparse
import gymnasium as gym

def add_noise_to_weights(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            m.weight.add_(torch.randn(m.weight.size()) * 0.1)
            
def get_update_rate(prev_param_dict, new_param_dict):
    out = {}
    if len(prev_param_dict)!=0:
        for k in prev_param_dict.keys():
            param = prev_param_dict[k].reshape(-1)
            p_mpow = np.mean(param.detach().numpy()**2)
            diffs = new_param_dict[k].reshape(-1) - prev_param_dict[k].reshape(-1)
            d_mpow = np.mean(diffs.detach().numpy()**2)
            if d_mpow==0:
                update_rate = -1000
            elif p_mpow==0:
                update_rate = +1000
            else:
                update_rate = 10*np.log10( d_mpow / p_mpow )
            out[k] = update_rate
    # copy new_ to prev_
    prev_param_dict = new_param_dict
    for k in prev_param_dict:
        prev_param_dict[k] = prev_param_dict[k].clone()
    return out, prev_param_dict
# ==============================================================================
# Network definition
# =========================================cda=====================================
# Value function approx. netowrk (Critic's network)
class ValueNet(nn.Module):
    def __init__(self, dStates, nHidden):
        super(ValueNet, self).__init__()
        self.dStates = dStates
        self.nHidden = nHidden
        layers = [
            nn.Linear(dStates, nHidden),
            nn.Linear(nHidden, nHidden),
            nn.Linear(nHidden, nHidden),
            nn.Linear(nHidden, 1)
        ]
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = F.tanh(self.layers[0](x))
        x = F.tanh(self.layers[1](x))
        x = F.tanh(self.layers[2](x))
        out = self.layers[3](x)  # linear activation function at the final layer
        return out
    
# Policy function network - output mean and log(var.)
class PolicyNet(nn.Module):
    def __init__(self, dStates, dActions, nHidden):
        super(PolicyNet, self).__init__()
        self.dStates = dStates
        self.dActions = dActions
        self.nHidden = nHidden
        layers = [
            nn.Linear(dStates, nHidden),
            nn.Linear(nHidden, nHidden),
            nn.Linear(nHidden, nHidden),
            nn.Linear(nHidden, dActions*2)
        ]
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = F.relu(self.layers[0](x))
        x = F.relu(self.layers[1](x))
        x = F.relu(self.layers[2](x))
        out = self.layers[3](x)
        return out
    
# ==============================================================================
# # Memory - handling replay buffer
# ==============================================================================
class Memory:
    samples = []  # replay buffer
    episode_samples = []  # episode buffer

    def __init__(self, capacity, gamma):
        self.capacity = capacity
        self.gamma = gamma

    # add sample to the buffer
    def add(self, sample, done):
        self.episode_samples.append(sample)
        if done:
            R = 0
            for n in range(len(self.episode_samples)):
                o = self.episode_samples[-n - 1]
                R = o[2] + self.gamma * R
                self.samples.append((o[0], o[1], o[2], o[3], o[4], R))  # (s,a,r,s',log_p, R)
                if len(self.samples) > self.capacity:
                    self.samples.pop(0)
            self.episode_samples = []
            
    # clear buffer
    def clear(self):
        self.samples = []

    # get randomly selected n samples from the buffer
    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)
    
# ==============================================================================
# # Agent - agent top level
# ==============================================================================
# agent parameters
MAX_MEM = 30000
GAMMA = 0.99
class ACAgent():
    def __init__(self, dStates, dActions,
                 a_range=1,
                 nHidden=256,
                 nEpoc=1,
                 batch_size=64,
                 lr_value=1e-3,
                 lr_policy=1e-3
                 ):
        self.dStates = dStates
        self.dActions = dActions
        self.a_range = a_range
        self.nHidden = nHidden
        self.lr_value = lr_value
        self.lr_policy = lr_policy
        self.nEpoc = nEpoc
        self.batch_size = batch_size
        self.memory = Memory(MAX_MEM, GAMMA)
        self.policynet  = PolicyNet(dStates, dActions, nHidden)
        self.valuenet = ValueNet(dStates, nHidden)
        self.optim_policy = torch.optim.Adam(self.policynet.parameters(), lr=lr_policy)
        self.optim_value  = torch.optim.Adam(self.valuenet.parameters(), lr=lr_value)
        self.huber_loss = nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean')
        self.softpuls = nn.Softplus()

    def act(self, s):
        out = self.policynet(torch.tensor(s).view(1, self.dStates))
        out = out.detach().squeeze(0)
        mean = out[0:self.dActions]
        std = self.softpuls(out[self.dActions:]) + 1e-5
        noise = torch.randn(self.dActions)
        a = (std * noise + mean)
        #log_p = -(a - mean) ** 2 / (2 * std ** 2) - torch.log(np.sqrt(2 * np.pi) * std)
        log_p = -noise ** 2 / 2 - torch.log(np.sqrt(2 * np.pi) * std)
        return a, log_p
    
    def observe(self, sample, done):
        self.memory.add(sample, done)

    def replay(self):
        batch = self.memory.sample(self.batch_size)
        batchLen = len(batch)
        if batchLen == 0:
            return
        s = torch.tensor([o[0] for o in batch])
        a = torch.tensor([o[1] for o in batch])
        r = torch.tensor([o[2] for o in batch])
        s_= torch.tensor([o[3] for o in batch])
        log_p_old = torch.tensor([o[4] for o in batch])
        R = torch.tensor([o[5] for o in batch])
        # update value function
        self.valuenet.train()
        self.optim_value.zero_grad()
        v4s  = self.valuenet(s)
        v4s_ = self.valuenet(s_).detach()
        loss_value = self.huber_loss(r + GAMMA * v4s_.squeeze(1), v4s.squeeze(1))
        loss_value.backward()
        self.optim_value.step()
        # update policy
        out = self.policynet(s)
        mean = out[:, 0:self.dActions]
        std = self.softpuls(out[:, self.dActions:]) + 1e-5
        log_p = -(a-mean)**2 / (2*std**2) - torch.log(np.sqrt(2*np.pi)*std)
        p_ratio = torch.exp(torch.clamp(log_p - log_p_old, -1, +1))
        A = r + GAMMA * v4s_.squeeze(1) - v4s.squeeze(1)
        A = A.detach()
        loss_policy_1 = - torch.mul(p_ratio.squeeze(0), A.unsqueeze(1) )
        loss_policy_2 = - torch.mul(torch.clamp(p_ratio.squeeze(0), 1.0-0.2, 1.0+0.2), A.unsqueeze(1) )
        loss_policy = torch.max(loss_policy_1, loss_policy_2)
        loss_policy.sum().backward()
        if torch.isnan( loss_policy.sum() ):
            print("NaN in loss_policy")
        if torch.isinf( loss_policy.sum() ):
            print("INF in loss_policy")
        torch.nn.utils.clip_grad_norm_(self.policynet.parameters(), 2.0)  # clip gradient
        self.optim_policy.step()
        if torch.any( torch.isnan(self.policynet.layers[1].weight) ):
            print("NaN in policy net weight")

    # run one episode
    def play(self, env:gym.Env, max_steps=10000, render=False):
        s, _ = env.reset()
        s = s.astype('float32')
        r = 0  # reward
        logs = {"video_frames": [],
                "actions": [],
                "observations": [],
                "rewards": []}
        totRW = 0  # total reward in this episode
        done = False  # episode completion flag
        for j in range(max_steps):
            # call agent
            action, log_p = self.act(s)
            a = action.data.numpy()
            log_p = log_p.data.numpy()
            if render:
                img = env.render()
                logs["video_frames"].append(img)
                logs["actions"].append(a)
                logs["observations"].append(s)
                logs["rewards"].append(r)
            # if done==True, then finish this episode.
            if done:
                break
            # call environment
            o, r, terminated, truncated, info = env.step( np.tanh(a)*self.a_range )
            done = terminated or truncated
            # store experience into replay buffer
            s_ = o.astype('float32')
            self.observe((s, a, r, s_, log_p), done)
            s = s_
            # accumrate total reward
            totRW += r
        # learaning
        for n in range(self.nEpoc):
            self.replay()
        return totRW, logs
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO')
    parser.add_argument('--env', type=str, default='LunarLanderContinuous-v2', help='Environment name [str]') # "MountainCarContinuous-v0", "Pendulum-v0", "sinWaveTracking-v0", "CartPole-v0",
    parser.add_argument('--nHidden', type=int, default=64, help='Number of hidden layer units [int]')
    parser.add_argument('--episodes-to-run', type=int, default=5000, help='Number of episodes to run [int]')
    parser.add_argument('--lr-policy', type=float, default=2e-5, help='Learning rate for policy net')
    parser.add_argument('--lr-value', type=float, default=2e-4, help='Learning rate for value net')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--nEpoc', type=int, default=32, help='Number of epoch for training')
    parser.add_argument('--test-interval', type=int, default=100, help='Interval between test episodes')
    parser.add_argument('--clear-buffer', default=False, action='store_true', help='Enable clearing replay buffer at every episode')
    config = parser.parse_args()
    config_dict = vars(config)
    # print out parameters
    for k in config_dict.keys():
        print(f"{k} = {config_dict[k]}")
    env = gym.make(config.env, render_mode='rgb_array')
    dActions = env.action_space.shape[0]
    dStates = env.observation_space.shape[0]
    action_range = (env.action_space.high - env.action_space.low)/2
    agent = ACAgent(dStates, dActions, a_range=action_range,
                         nHidden=config.nHidden,
                         nEpoc=config.nEpoc,
                         batch_size=config.batch_size,
                         lr_policy=config.lr_policy,
                         lr_value=config.lr_value
                         )
    prev_policynet_dict = {}
    prev_valuenet_dict = {}
    render_enable = False
    R = []
    EPISODES_TO_RUN = config.episodes_to_run
    for n in range(EPISODES_TO_RUN):
        # generate render condition
        render_enable = (n > 100) and (n%config.test_interval==0)
        r, logs = agent.play(env, render=render_enable)
        if config.clear_buffer:
            agent.memory.clear()
        R.append(r)
        if (n % 10 == 0):
            print('{}: ave r: {} '.format(n, np.mean(R[-100:])))