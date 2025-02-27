import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from mario_rl_tutorial import JoypadSpace,ResizeObservation, SkipFrame,GrayScaleObservation,Mario
# 加载训练好的模型权重

model_path = Path("checkpoints/2025-01-17T13-57-19/mario_net_0.chkpt")  # 替换为你的模型文件路径
state_dict = torch.load(model_path)



# 创建一个新的环境并使用Monitor包装器
eval_env = gym_super_mario_bros.make("SuperMarioBros-2-2-v3", render_mode='human', apply_api_compatibility=True)
eval_env = JoypadSpace(eval_env, [["right"], ["right", "A"]])
eval_env = SkipFrame(eval_env, skip=4)
eval_env = GrayScaleObservation(eval_env)
eval_env = ResizeObservation(eval_env, shape=84)
eval_env = FrameStack(eval_env, num_stack=4)

# 使用Monitor包装器记录游戏过程
# eval_env = Monitor(eval_env, directory="mario_videos", force=True)
# from gym.wrappers.record_video import RecordVideo
# eval_env = RecordVideo(eval_env, video_folder="mario_videos_replay")

 
mario = Mario(state_dim=(4, 84, 84), action_dim=eval_env.action_space.n, save_dir=None)
mario.net.load_state_dict(state_dict['model'])
mario.net.eval()  # 设置模型为评估模式

# 设置评估的episode数量
num_eval_episodes = 10

for episode in range(num_eval_episodes):
    state = eval_env.reset()
    total_reward = 0

    while True:
        eval_env.render()
        # 使用模型选择动作
        action = mario.act(state)

        # 执行动作
        next_state, reward, done, trunc, info = eval_env.step(action)

        # 累计奖励
        total_reward += reward

        # 更新状态
        state = next_state

        # 检查是否结束
        if done or info["flag_get"]:
            break

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# 关闭环境
eval_env.close()