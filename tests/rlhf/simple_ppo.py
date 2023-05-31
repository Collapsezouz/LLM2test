### Init
# > pip install 'gymnasium[classic-control]'
### Test
# python -m tests.rlhf.simple_ppo --output_path=./logs/simple_actor_critic.bin
### Debug
# export DEBUG_PORT=5679
# REMOTE_DEBUG=1 python -m tests.rlhf.simple_ppo
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque

from tests import logger


# 设置随机种子
np.random.seed(1)
torch.manual_seed(1)

class ActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.fc4 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy = F.softmax(self.fc3(x), dim=1)
        value = self.fc4(x)
        return policy, value
    
def train(env_name=None, output_path=None, hidden_size=32, lr=1e-3, num_epochs=50, batch_size=64,
        gamma=0.99, clip_ratio=0.2, max_steps=200, early_stop=False, render=False):
    env_name = env_name or 'CartPole-v1'
    # 创建 CartPole-v0 环境
    env = gym.make(env_name)
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    # 创建 Actor-Critic 网络
    net = ActorCritic(input_size, hidden_size, output_size)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # 初始化参数
    total_rewards = []
    for epoch in range(num_epochs):
        state, _ = env.reset()
        done = False
        rewards = []
        values = []
        log_probs = []
        
        # 收集一批经验数据
        for step in range(max_steps):
            if render:
                env.render()
                
            # 预测动作概率和状态值
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            policy, value = net(state_tensor)
            
            # 选择行动并计算log概率
            action_tensor = policy.multinomial(1)
            log_prob = F.log_softmax(policy, dim=1)[0, action_tensor]
            
            # 执行行动并记录结果
            action = action_tensor.item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            # 如果到达终止状态，停止游戏并记录总奖励
            if done:
                total_reward = sum(rewards)
                total_rewards.append(total_reward)
                break
                
            state = next_state
        
        # 计算优势函数
        returns = []
        advantages = []
        prev_return = 0
        prev_value = 0
        for i in reversed(range(len(rewards))):
            next_return = rewards[i] + gamma * prev_return
            returns.insert(0, next_return)
            delta = rewards[i] + gamma * prev_value - values[i]
            advantages.insert(0, delta)
            prev_return = next_return
            prev_value = values[i]
        
        # 转换为张量
        returns_tensor = torch.tensor(returns, dtype=torch.float)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float)
        log_probs_tensor = torch.stack(log_probs)
        
        # 计算策略损失和价值损失
        for i in range(batch_size):
            # 随机选择一个样本
            index = np.random.randint(0, len(rewards))
            
            # 计算策略概率和log概率
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            policy, value = net(state_tensor)
            action_tensor = policy.multinomial(1)
            log_prob = F.log_softmax(policy, dim=1)[0, action_tensor]
            # 计算重要性采样比率和Clipped Surrogate Objective
            ratio = torch.exp(log_prob - log_probs_tensor[index])
            clipped_ratio = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio)
            policy_loss = -torch.min(ratio*advantages_tensor[index], clipped_ratio*advantages_tensor[index])
            
            # 计算价值损失
            returns_tensor[index] = returns_tensor[index].detach()
            value_loss = F.smooth_l1_loss(value, returns_tensor[index])
            
            # 计算总损失和进行反向传播更新
            loss = policy_loss + value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 判断是否早期终止
        if early_stop and np.mean(total_rewards[-10:]) >= 195:
            print(f'Early stop at epoch {epoch}, average reward: {np.mean(total_rewards[-10:])}')
            break
        
        # 打印训练进度
        print(f'Epoch {epoch}, average reward: {np.mean(total_rewards[-10:])}')
    
    env.close()

    if output_path:
        net.state_dict(destination=output_path)
    return net


def main(output_path=None, env_name=None):
    net = train(
        env_name=env_name,
        output_path=output_path
    )

if __name__ == "__main__":
    import fire
    fire.Fire(main)