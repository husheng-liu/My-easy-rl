import numpy as np
import random
import math
import sys
import gym
from gym import spaces
import colorama

class GameEnvironment(gym.Env):
    def __init__(self, m=8, n=8, target={1: 9}, max_steps=20, element_low=1, \
                 element_high=4, seed=None):
        super(GameEnvironment, self).__init__()
        self.m = m
        self.n = n
        self.element_low = element_low
        self.element_high = element_high
        self.target = target
        self.max_steps = max_steps
        self.board = self.initialize_board(seed=seed)
        self.score = 0
        self.steps = 0
        self.eliminated_count = 0
        self.current_target = {key: target[key] for key in target}
        self.current_score = 0
        self.score_rule = self.get_score_rule(target, element_low, element_high)

        # Define action and observation space
        self.action_space = spaces.Discrete((m-1)*n + m*(n-1))
        self.observation_space = spaces.Box(low=0, high=self.element_high, shape=(self.m, self.n), dtype=np.int32)

    def get_score_rule(self, target, element_low, element_high):
        socre_rule = {}
        for element in range(element_low, element_high+1):
            if element in target:
                socre_rule[element] = 0.3
            else:
                socre_rule[element] = 0.1
        return socre_rule
            
    def initialize_board(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        while True:
            board = np.random.randint(self.element_low, self.element_high+1, (self.m, self.n))
            if not self.find_matches(board):
                return board

    def get_legal_moves(self):
        moves = []
        for i in range(self.m):
            for j in range(self.n):
                # Check horizontal swap
                if j < self.n - 1:
                    self.board[i][j], self.board[i][j + 1] = self.board[i][j + 1], self.board[i][j]
                    if self.find_matches():
                        moves.append((i, j, 'H'))
                    self.board[i][j], self.board[i][j + 1] = self.board[i][j + 1], self.board[i][j]  # Revert the swap

                # Check vertical swap
                if i < self.m - 1:
                    self.board[i][j], self.board[i + 1][j] = self.board[i + 1][j], self.board[i][j]
                    if self.find_matches():
                        moves.append((i, j, 'V'))
                    self.board[i][j], self.board[i + 1][j] = self.board[i + 1][j], self.board[i][j]  # Revert the swap

        return moves

    def get_action_mask(self):
        action_mask = np.zeros(self.action_space.n)
        legal_moves = self.get_legal_moves()
        print(f"legal_moves: {legal_moves}")
        for move in legal_moves:
            print(f"move: {move}")
            if move[2] == 'H':
                action_mask[move[0] * (self.n - 1) + move[1]] = True
            elif move[2] == 'V':
                action_mask[move[0]* self.n+ move[1]+self.m*(self.n-1)]
                # action_mask[self.m * self.n + move[0] * (self.n - 1) + move[1]] = True
        return action_mask

    def make_move(self, move):
        i, j, direction = move
        if direction == 'H':
            self.board[i][j], self.board[i][j + 1] = self.board[i][j + 1], self.board[i][j]
        elif direction == 'V':
            self.board[i][j], self.board[i + 1][j] = self.board[i + 1][j], self.board[i][j]
        self.update_board()
        self.steps += 1

    def update_board(self):
        eliminated = True
        while eliminated:
            eliminated = False
            to_eliminate = self.find_matches()
            if to_eliminate:
                self.update_target(to_eliminate)
                self.update_score(to_eliminate)
                eliminated = True
                self.eliminated_count += len(to_eliminate)
                for (i, j) in to_eliminate:
                    self.board[i][j] = 0

                self.drop_elements()

    def drop_elements(self):
        for col in range(self.n):
            non_empty_elements = [self.board[i][col] for i in range(self.m) if self.board[i][col] != 0]
            for idx, element in enumerate(reversed(non_empty_elements)):
                self.board[self.m - 1 - idx][col] = element
            num_new_elements = self.m - len(non_empty_elements)
            if num_new_elements > 0:
                new_elements = np.random.randint(1, self.element_high, num_new_elements)
                for idx, new_element in enumerate(new_elements):
                    self.board[idx][col] = new_element

    def find_matches(self, board=None):
        if board is None:
            board = self.board
        matches = set()
        for i in range(self.m):
            for j in range(self.n):
                if board[i][j] != 0:
                    if j < self.n - 2 and board[i][j] == board[i][j + 1] == board[i][j + 2]:
                        matches.update([(i, j), (i, j + 1), (i, j + 2)])
                    if i < self.m - 2 and board[i][j] == board[i + 1][j] == board[i + 2][j]:
                        matches.update([(i, j), (i + 1, j), (i + 2, j)])
        return matches

    def update_target(self, to_eliminate):
        for (i, j) in to_eliminate:
            cell_type = self.board[i][j]
            if cell_type in self.current_target:
                self.current_target[cell_type] = max(0, self.current_target[cell_type] - 1)
                if self.is_game_over():
                    self.calculate_settlement_reward()

    
    def update_score(self, to_eliminate):
        for (i, j) in to_eliminate:
            ele_type = self.board[i][j]
            self.step_score += self.score_rule[ele_type]
            self.current_score+=self.score_rule[ele_type]

    def is_game_over(self):
        if self.steps >= self.max_steps or all(count == 0 for count in self.current_target.values()):
            # print("游戏结束...")
            return True
        else:
            # print("游戏没有结束")
            return False

    def is_level_completed(self):
        return all(count == 0 for count in self.current_target.values()) and self.steps < self.max_steps

    def get_score(self):
        
        return self.current_score

    def calculate_settlement_reward(self):
        if self.is_level_completed():
            remaining_steps = self.max_steps - self.steps
            return self.current_score + remaining_steps * 30
        return 0

    def is_valid_move(self):
        pass
    
    def move_to_action(self, direction, row, col):
        max_rows, max_cols = self.m,self.n
        
        if direction == "H":
            # Ensure column is within valid range for right moves.
            if col >= max_cols - 1:
                raise ValueError("Column index out of range for right move.")
            return row * (max_cols - 1) + col
        elif direction == "V":
            # No need to check column range for up moves as all columns are valid.
            right_moves_count = (max_cols - 1) * max_rows
            offset = col + row * max_cols
            return right_moves_count + offset
        else:
            raise ValueError("Invalid direction.")


    def action_to_move(self, action):
        max_rows, max_cols = self.m,self.n 
        
        right_moves_count = (max_cols - 1) * max_rows
        
        if action < right_moves_count:
            # Move is to the right
            direction = "H"
            row = action // (max_cols - 1)
            col = action % (max_cols - 1)
        else:
            # Move is upward
            direction = "V" 
            offset = action - right_moves_count
            row = offset // max_cols
            col = offset % max_cols
        
        return row, col, direction
    
    def reset(self):
        self.board = self.initialize_board(seed=32)
        self.score = 0
        self.steps = 0
        self.eliminated_count = 0
        self.current_target = {key: self.target[key] for key in self.target}
        action_mask = self.get_action_mask()
        print("action_mask: ", action_mask)
        return self.board, action_mask

    def step(self, action):
        # legal_moves = self.get_legal_moves()
        # if action >= len(legal_moves):
            # raise ValueError("Invalid action")
        # move = legal_moves[action]
        reward = 0
        self.step_score = 0
        move = self.action_to_move(action)
        self.make_move(move)
        # 通关给1分，不通关给0分
        # 1.过于稀疏 2.中间过程给与积分
        
        if self.is_level_completed():
            reward += self.step_score
        else:
            reward = self.step_score
        done = self.is_game_over()
        action_mask = self.get_action_mask()
        return self.board, reward, done, {'action_mask': action_mask}

    def render(self, mode='human'):
        colorama.init(autoreset=True)
        color_map = {
            0: '\033[90m',  # 灰色
            1: '\033[91m',  # 红色
            2: '\033[92m',  # 绿色
            3: '\033[93m',  # 黄色
            4: '\033[94m',  # 蓝色
            5: '\033[95m',  # 紫色
            6: '\033[96m',  # 青色
        }
        reset_color = '\033[0m'  # 重置颜色

        for row in self.board:
            for cell in row:
                print(f"{color_map.get(cell, reset_color)}{cell}{reset_color}", end=' ')
            print()

    def close(self):
        pass


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * input_shape[1] * input_shape[2], 256)
        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x, action_mask=None):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        print(f"action_mask: {action_mask}")
        if action_mask is not None:
            x = x.masked_fill(action_mask == 0, float('-inf'))
            print(f"x: {x}")
        return x

class DQNAgent:
    def __init__(self, state_shape, num_actions, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, batch_size=32, memory_size=10000):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        
        self.model = DQN(state_shape, num_actions)
        self.target_model = DQN(state_shape, num_actions)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done, action_mask):
        self.memory.append((state, action, reward, next_state, done, action_mask))

    def act(self, state, action_mask):
        if np.random.rand() <= self.epsilon:
            legal_actions = np.where(action_mask == 1)[0]
            return np.random.choice(legal_actions)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_mask = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0)
        q_values = self.model(state, action_mask=action_mask)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done, action_mask in minibatch:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            action_mask = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0)
            target = self.model(state, action_mask=action_mask)
            target_next = self.target_model(next_state).detach()
            if done:
                target[0][action] = reward
            else:
                target[0][action] = reward + self.gamma * torch.max(target_next[0])
            self.optimizer.zero_grad()
            loss = self.criterion(target, self.model(state, action_mask=action_mask))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.target_model.load_state_dict(torch.load(filename))

    
def train_dqn(env: GameEnvironment, agent:DQNAgent, episodes=1000):
    for episode in range(episodes):
        state, action_mask = env.reset()
        state = np.expand_dims(state, axis=0)  # Add batch dimension
        
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state, action_mask)
            next_state, reward, done, info = env.step(action)
            next_state = np.expand_dims(next_state, axis=0)  # Add batch dimension
            next_action_mask = info['action_mask']
            agent.remember(state, action, reward, next_state, done, action_mask)
            state = next_state
            action_mask = next_action_mask
            total_reward += reward
            agent.replay()
        agent.update_target_model()
        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

if __name__ == "__main__":
    # 游戏环境主要有2个随机行，第一个时随机初始化棋盘，第二个时随机生成元素。
    
    game = GameEnvironment(m=6, n=6, target={1: 8}, \
                            max_steps=20, element_low=1, \
                            element_high=4, seed=42)
    game.render()
    game.reset()
    print(game.render())
    game.reset()
    print(game.render())
    # for i in range(24):
    #     print(game.action_to_move(i))
    # state_shape = (1, game.m, game.n)  # Add channel dimension
    # num_actions = (game.m-1)*game.n + game.m*(game.n-1)
    # agent = DQNAgent(state_shape, num_actions, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, batch_size=32, memory_size=10000)
    
    # train_dqn(game, agent, episodes=1000)