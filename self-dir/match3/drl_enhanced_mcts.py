import numpy as np

class Match3Environment:
    def __init__(self, grid_size=6):
        self.grid_size = grid_size
        self.grid = self._initialize_grid()
        self.done = False

    def _initialize_grid(self):
        """随机初始化棋盘"""
        return np.random.randint(1, 5, size=(self.grid_size, self.grid_size))

    def step(self, action):
        """
        执行动作并返回下一步状态、奖励和是否结束。
        :param action: 动作 (x1, y1, x2, y2)，表示交换两个相邻方块的位置。
        :return: 新状态、奖励、是否结束
        """
        x1, y1, x2, y2 = action
        if not self._is_valid_move(x1, y1, x2, y2):
            return self.grid.copy(), -10, self.done  # 非法移动惩罚

        # 交换方块
        self.grid[x1, y1], self.grid[x2, y2] = self.grid[x2, y2], self.grid[x1, y1]

        # 检查并消除匹配项
        reward = self._check_and_remove_matches()

        # 如果没有消除，则还原交换
        if reward == 0:
            self.grid[x1, y1], self.grid[x2, y2] = self.grid[x2, y2], self.grid[x1, y1]
            return self.grid.copy(), -1, self.done  # 无效移动惩罚

        # 填充空位
        self._fill_empty_spaces()

        # 检查游戏是否结束
        if np.all(self.grid > 0):  # 如果棋盘上还有方块
            self.done = False
        else:
            self.done = True

        return self.grid.copy(), reward, self.done

    def _is_valid_move(self, x1, y1, x2, y2):
        """检查动作是否合法"""
        return (abs(x1 - x2) + abs(y1 - y2) == 1) and (0 <= x1 < self.grid_size) and (0 <= y1 < self.grid_size) and \
               (0 <= x2 < self.grid_size) and (0 <= y2 < self.grid_size)

    def _check_and_remove_matches(self):
        """检查并移除匹配项，返回奖励"""
        matches = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # 水平匹配
                if j <= self.grid_size - 3 and len(set(self.grid[i, j:j+3])) == 1:
                    matches.extend([(i, j), (i, j+1), (i, j+2)])
                # 垂直匹配
                if i <= self.grid_size - 3 and len(set(self.grid[i:i+3, j])) == 1:
                    matches.extend([(i, j), (i+1, j), (i+2, j)])

        if matches:
            for match in matches:
                self.grid[match] = 0  # 将匹配项置为 0
            return len(matches) * 10  # 每个匹配项奖励 10 分
        return 0

    def _fill_empty_spaces(self):
        """填充空位"""
        for col in range(self.grid_size):
            empty = np.where(self.grid[:, col] == 0)[0]
            if len(empty) > 0:
                self.grid[empty[0]:, col] = np.roll(self.grid[empty[0]:, col], len(empty))
                self.grid[:len(empty), col] = np.random.randint(1, 5, len(empty))

    def reset(self):
        """重置环境"""
        self.grid = self._initialize_grid()
        self.done = False
        return self.grid.copy()

    def render(self):
        """打印当前棋盘"""
        print(self.grid)

import torch
import torch.nn as nn
import torch.optim as optim

class DRLModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(DRLModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.fc(x)

# 初始化模型
grid_size = 6
input_size = grid_size * grid_size  # 棋盘展平后的大小
output_size = 4  # 动作空间大小（简化为上下左右）
model = DRLModel(input_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()


import math
from collections import defaultdict

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state.flatten()  # 展平棋盘状态
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0
        self.untried_actions = self.get_legal_actions()

    def get_legal_actions(self):
        """获取所有合法动作"""
        legal_actions = []
        grid_size = int(math.sqrt(len(self.state)))
        for i in range(grid_size):
            for j in range(grid_size):
                if j < grid_size - 1:  # 水平交换
                    legal_actions.append((i, j, i, j + 1))
                if i < grid_size - 1:  # 垂直交换
                    legal_actions.append((i, j, i + 1, j))
        return legal_actions

    def expand(self):
        """扩展节点"""
        action = self.untried_actions.pop()
        next_state, _, _ = env.step(action)
        child_node = Node(next_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def is_fully_expanded(self):
        """判断是否完全扩展"""
        return len(self.untried_actions) == 0

    def best_child(self, c=1.4):
        """选择最优子节点"""
        weights = [(child.value / (child.visits + 1e-6)) + c * math.sqrt(math.log(self.visits) / (child.visits + 1e-6))
                   for child in self.children]
        return self.children[np.argmax(weights)]

    def rollout(self):
        """模拟直到游戏结束"""
        current_state = self.state.reshape((grid_size, grid_size))
        env = Match3Environment(grid_size=grid_size)
        env.grid = current_state.copy()
        total_reward = 0
        done = False

        while not done:
            legal_actions = self.get_legal_actions()
            action = legal_actions[np.random.randint(len(legal_actions))]
            _, reward, done = env.step(action)
            total_reward += reward

        return total_reward

    def backpropagate(self, reward):
        """反向传播奖励"""
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(reward)

def mcts_search(root, simulations=1000):
    """MCTS 搜索"""
    for _ in range(simulations):
        node = root
        # 选择
        while node.is_fully_expanded():
            node = node.best_child()
        # 扩展
        if not node.is_terminal():
            node = node.expand()
        # 模拟
        reward = node.rollout()
        # 反向传播
        node.backpropagate(reward)

    # 返回访问次数最多的动作
    visits = [child.visits for child in root.children]
    best_action = root.children[np.argmax(visits)].action
    return best_action


def train_drl_mcts(env, model, episodes=100, simulations=100):
    for episode in range(episodes):
        state = env.reset().flatten()
        done = False
        total_reward = 0

        while not done:
            root = Node(state.reshape((grid_size, grid_size)))
            action = mcts_search(root, simulations=simulations)
            next_state, reward, done = env.step(action)
            total_reward += reward

            # 更新神经网络
            # TD update
            target = reward + 0.99 * model(torch.tensor(next_state, dtype=torch.float32)).max().item()
            predicted = model(torch.tensor(state, dtype=torch.float32))
            loss = criterion(predicted, torch.tensor([target], dtype=torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state.flatten()

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

def test_drl_mcts(env, model, episodes=10):
    total_rewards = []
    for episode in range(episodes):
        state = env.reset().flatten()
        done = False
        episode_reward = 0

        while not done:
            with torch.no_grad():
                q_values = model(torch.tensor(state, dtype=torch.float32))
                action_idx = torch.argmax(q_values).item()
                # 将动作索引映射到实际动作
                actions = root.get_legal_actions()
                action = actions[action_idx] if actions else None

            next_state, reward, done = env.step(action)
            episode_reward += reward
            state = next_state.flatten()

        total_rewards.append(episode_reward)
        print(f"Test Episode {episode + 1}, Total Reward: {episode_reward}")

    print(f"Average Test Reward: {np.mean(total_rewards)}")

# 还有另一块工作,F3P

if __name__ == "__main__":
    # 
    env = Match3Environment(grid_size=6)
    train_drl_mcts(env, model, episodes=50, simulations=50)
    test_drl_mcts(env, model, episodes=10)