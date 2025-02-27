import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Assuming continuous actions
        return x

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class PPO:
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, clip_ratio=0.2, ppo_epochs=10, batch_size=64):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

    def select_action(self, state):
        state = torch.FloatTensor(state)
        action = self.policy(state)
        return action.detach().numpy()

    def compute_returns(self, rewards, dones, next_values):
        returns = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_values[step] * (1 - dones[step]) - self.value(torch.FloatTensor(states[step])).item()
            gae = delta + self.gamma * 0.95 * gae * (1 - dones[step])
            returns.insert(0, gae + self.value(torch.FloatTensor(states[step])).item())
        return returns

    def train(self, states, actions, rewards, dones, next_states):
        returns = self.compute_returns(rewards, dones, self.value(torch.FloatTensor(next_states)).detach().numpy())
        returns = torch.FloatTensor(returns)
        advantages = returns - self.value(torch.FloatTensor(states)).detach()

        for _ in range(self.ppo_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(len(states))), self.batch_size, drop_last=False):
                states_batch = torch.FloatTensor(states[index])
                actions_batch = torch.FloatTensor(actions[index])
                returns_batch = returns[index]
                advantages_batch = advantages[index]

                old_log_probs = self.policy(states_batch).gather(1, actions_batch.long()).detach()

                log_probs = self.policy(states_batch).gather(1, actions_batch.long())
                ratio = torch.exp(log_probs - old_log_probs)

                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = (returns_batch - self.value(states_batch)).pow(2).mean()

                self.optimizer_policy.zero_grad()
                policy_loss.backward()
                self.optimizer_policy.step()

                self.optimizer_value.zero_grad()
                value_loss.backward()
                self.optimizer_value.step()


import torch
import numpy as np
from typing import Dict, NamedTuple

class ActionPredictionTuple(NamedTuple):
    continuous: torch.Tensor
    discrete: torch.Tensor

class CuriosityNetwork(torch.nn.Module):
    EPSILON = 1e-10

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self._state_encoder = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )

        self._action_flattener = torch.nn.Linear(action_dim, hidden_dim)

        self.inverse_model_action_encoding = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, 256),
            torch.nn.ReLU()
        )

        self.continuous_action_prediction = torch.nn.Linear(256, action_dim)
        self.forward_model_next_state_prediction = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + hidden_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, hidden_dim)
        )

    def get_current_state(self, states):
        return self._state_encoder(states)

    def get_next_state(self, next_states):
        return self._state_encoder(next_states)

    def predict_action(self, states, next_states):
        inverse_model_input = torch.cat((states, next_states), dim=1)
        hidden = self.inverse_model_action_encoding(inverse_model_input)
        continuous_pred = self.continuous_action_prediction(hidden)
        return ActionPredictionTuple(continuous_pred, None)

    def predict_next_state(self, states, actions):
        flattened_action = self._action_flattener(actions)
        forward_model_input = torch.cat((states, flattened_action), dim=1)
        return self.forward_model_next_state_prediction(forward_model_input)

    def compute_inverse_loss(self, states, next_states, actions):
        predicted_action = self.predict_action(states, next_states)
        sq_difference = (actions - predicted_action.continuous) ** 2
        sq_difference = torch.sum(sq_difference, dim=1)
        return torch.mean(sq_difference)

    def compute_reward(self, states, next_states,actions):
        predicted_next_state = self.predict_next_state(states, actions)
        target = self.get_next_state(next_states)
        sq_difference = 0.5 * (target - predicted_next_state) ** 2
        sq_difference = torch.sum(sq_difference, dim=1)
        return sq_difference

    def compute_forward_loss(self, states, next_states, actions):
        return torch.mean(self.compute_reward(states, next_states,actions))

class CuriosityRewardProvider:
    beta = 0.2  # Forward vs Inverse loss weight
    loss_multiplier = 10.0  # Loss multiplier

    def __init__(self, state_dim, action_dim, learning_rate=0.0003):
        self._network = CuriosityNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self._network.parameters(), lr=learning_rate)
        self._has_updated_once = False

    def evaluate(self, states, next_states, actions):
        with torch.no_grad():
            rewards = self._network.compute_reward(states, next_states)
        rewards = np.minimum(rewards.numpy(), 1.0 / 1.0)  # Assuming strength=1.0
        return rewards * self._has_updated_once

    def update(self, states, next_states, actions):
        self._has_updated_once = True
        forward_loss = self._network.compute_forward_loss(states, next_states, actions)
        inverse_loss = self._network.compute_inverse_loss(states, next_states, actions)# St,St_+1

        loss = self.loss_multiplier * (
            self.beta * forward_loss + (1.0 - self.beta) * inverse_loss
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "Losses/Curiosity Forward Loss": forward_loss.item(),
            "Losses/Curiosity Inverse Loss": inverse_loss.item(),
        }
    

class PPOWithCuriosity:
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, clip_ratio=0.2, ppo_epochs=10, batch_size=64):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        self.curiosity_reward_provider = CuriosityRewardProvider(state_dim, action_dim)

    def select_action(self, state):
        state = torch.FloatTensor(state)
        action = self.policy(state)
        return action.detach().numpy()

    def compute_returns(self, rewards, dones, next_values):
        returns = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_values[step] * (1 - dones[step]) - self.value(torch.FloatTensor(states[step])).item()
            gae = delta + self.gamma * 0.95 * gae * (1 - dones[step])
            returns.insert(0, gae + self.value(torch.FloatTensor(states[step])).item())
        return returns

    def train(self, states, actions, rewards, dones, next_states):
        curiosity_rewards = self.curiosity_reward_provider.evaluate(states, next_states, actions)
        rewards = np.array(rewards) + curiosity_rewards

        returns = self.compute_returns(rewards, dones, self.value(torch.FloatTensor(next_states)).detach().numpy())
        returns = torch.FloatTensor(returns)
        advantages = returns - self.value(torch.FloatTensor(states)).detach()

        for _ in range(self.ppo_epochs):
            # minibatch update the policy and value net
            for index in BatchSampler(SubsetRandomSampler(range(len(states))), self.batch_size, drop_last=False):
                states_batch = torch.FloatTensor(states[index])
                actions_batch = torch.FloatTensor(actions[index])
                returns_batch = returns[index]
                advantages_batch = advantages[index]

                old_log_probs = self.policy(states_batch).gather(1, actions_batch.long()).detach()

                log_probs = self.policy(states_batch).gather(1, actions_batch.long())
                ratio = torch.exp(log_probs - old_log_probs)

                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = (returns_batch - self.value(states_batch)).pow(2).mean()

                self.optimizer_policy.zero_grad()
                policy_loss.backward()
                self.optimizer_policy.step()

                self.optimizer_value.zero_grad()
                value_loss.backward()
                self.optimizer_value.step()
        # every epoch, update the curiosity module
        self.curiosity_reward_provider.update(states, next_states, actions)



import gym
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

# Initialize environment
env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Initialize PPO with Curiosity
ppo = PPOWithCuriosity(state_dim, action_dim)

# Training loop
num_episodes = 1000
max_steps = 200
states, actions, rewards, dones, next_states = [], [], [], [], []

for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0

    for step in range(max_steps):
        action = ppo.select_action(state)
        next_state, reward, done, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        next_states.append(next_state)

        state = next_state
        episode_reward += reward

        if done:
            break

    print(f"Episode {episode}: Reward {episode_reward}")

    ppo.train(states, actions, rewards, dones, next_states)
    states, actions, rewards, dones, next_states = [], [], [], [], []

