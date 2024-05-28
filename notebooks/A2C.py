#!/usr/bin/env python
# coding: utf-8



import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from multiprocessing import Process, Pipe
import argparse
import gym


# ## 建立Actor和Critic网络



class ActorCritic(nn.Module):
    ''' A2C网络模型，包含一个Actor和Critic
    '''
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1),
        )
    def save_critic(self, name):
        torch.save(self.critic.state_dict,name)
    
    def save_actor(self, name):
        torch.save(self.actor.state_dict, name)

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist  = Categorical(probs)
        return dist, value

class A2C:
    ''''''
    def __init__(self,n_states,n_actions,cfg) -> None:
        self.gamma = cfg.gamma
        self.device = cfg.device
        self.model = ActorCritic(n_states, n_actions, cfg.hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())

    def compute_returns(self,next_value, rewards, masks):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * masks[step]
            returns.insert(0, R)
        return returns


def make_envs(env_name):
    def _thunk():
        env = gym.make(env_name)
        env.seed(2)
        return env
    return _thunk
def test_env(env,model,vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(cfg.device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        # sum up every step reward
        total_reward += reward
    return total_reward

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def train(cfg,envs):
    print('Start training!')
    print(f'Env:{cfg.env_name}, Algorithm:{cfg.algo_name}, Device:{cfg.device}')
    # use for test_env
    env = gym.make(cfg.env_name) # a single env
    env.seed(10)
    n_states  = envs.observation_space.shape[0]
    n_actions = envs.action_space.n
    model = ActorCritic(n_states, n_actions, cfg.hidden_dim).to(cfg.device)
    optimizer = optim.Adam(model.parameters())
    step_idx    = 0
    test_rewards = []
    test_ma_rewards = []
    state = envs.reset()
    while step_idx < cfg.max_steps:
        log_probs = []
        values    = []
        rewards   = []
        masks     = []
        entropy = 0
        # rollout trajectory update every n_steps
        for _ in range(cfg.n_steps):
            state = torch.FloatTensor(state).to(cfg.device)
            dist, value = model(state)
            action = dist.sample()
            next_state, reward, done, _ = envs.step(action.cpu().numpy())
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(cfg.device))
            # if done==1 , mask--> 0
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(cfg.device))
            state = next_state
            step_idx += 1
            # each 200 step_idx, will test 
            if step_idx % 200 == 0:
                # average total reward for test_steps for ten envs
                test_reward = np.mean([test_env(env,model) for _ in range(10)])
                print(f"step_idx:{step_idx}, test_reward:{test_reward}")
                # compare in the different test period 
                test_rewards.append(test_reward)
                if test_ma_rewards:
                    # the latest one wight 0.9, the current weight 0.1
                    test_ma_rewards.append(0.9*test_ma_rewards[-1]+0.1*test_reward)
                else:
                    test_ma_rewards.append(test_reward) 
                # plot(step_idx, test_rewards)   
        # each trajectory(every n_steps) will update model once
        next_state = torch.FloatTensor(next_state).to(cfg.device)
        # the current step
        _, next_value = model(next_state)
        returns = compute_returns(next_value, rewards, masks)
        log_probs = torch.cat(log_probs)
        returns   = torch.cat(returns).detach()
        values    = torch.cat(values)
        advantage = returns - values
        # 最大化 log 概率选择优势高的动作,等同于最小化 -log_probs*advantage,来更新优化策略。
        actor_loss  = -(log_probs * advantage.detach()).mean()
        # same as ppo value loss
        critic_loss = advantage.pow(2).mean()

        # 这部分loss计算Actor网络预测的动作概率分布π(a|s)与真实采样的动作之间的交叉熵。目的是最小化π(a|s)与样本数据的差距,从而优化Actor网络输出更高真实动作概率的π(a|s)。
        # 这部分loss通过Critic网络预测的状态价值V(s)与经验回报的差别来计算。采用TD(λ)算法估计的目标值与V(s)之间的平方差,来优化Critic网络对状态价值的预测能力。
        # 加入动作熵项可以避免策略过早收敛,保持探索能力。熵损失项鼓励π(a|s)的熵值越大越好,分布更加均匀

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Finish training！')
    return test_rewards, test_ma_rewards



import matplotlib.pyplot as plt
import seaborn as sns 
def plot_rewards(rewards, ma_rewards, cfg, tag='train'):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve on {} of {} for {}".format(
        cfg.device, cfg.algo_name, cfg.env_name))
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(ma_rewards, label='ma rewards')
    plt.legend()
    plt.show()


import easydict
from common.multiprocessing_env import SubprocVecEnv
cfg = easydict.EasyDict({
        "algo_name": 'A2C',
        "env_name": 'CartPole-v0',
        "n_envs": 8,
        "max_steps": 20000,
        "n_steps":5,
        "gamma":0.99,
        "lr": 1e-3,
        "hidden_dim": 256,
        "device":torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
})
envs = [make_envs(cfg.env_name) for i in range(cfg.n_envs)]
envs = SubprocVecEnv(envs) 
rewards,ma_rewards = train(cfg,envs)
plot_rewards(rewards, ma_rewards, cfg, tag="train") # 画出结果

