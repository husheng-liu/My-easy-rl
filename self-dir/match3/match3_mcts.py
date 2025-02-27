import numpy as np
import random
import math
import sys


class MCTSNode:
    def __init__(self, parent=None, move=None, game_env=None):
        self.parent = parent
        self.move = move
        self.game_env = game_env
        self.visits = 0
        self.wins = 0
        self.children = []
        self.untried_moves = self.game_env.get_legal_moves()

    def select_child(self):
        """选择最优子节点 UCB1 平衡利用和探索"""
        # return max(self.children, key=lambda c: c.wins / c.visits + math.sqrt(2 * math.log(self.visits) / c.visits))
        return max(self.children, key=lambda c: (c.wins / c.visits if c.visits > 0 else 0) + math.sqrt(2 * math.log(self.visits) / (c.visits if c.visits > 0 else 1)))
    def expand(self):
        """扩展一个未访问的子节点"""
        
        if self.untried_moves:
            move = self.untried_moves.pop()
            new_game_env = self.game_env.copy()
            # 执行动作，变成下一个状态，也就是子节点
            new_game_env.make_move(move)
            child_node = MCTSNode(parent=self, move=move, game_env=new_game_env)
            self.children.append(child_node)
            return child_node
        return None

    def simulate(self):
        """模拟游戏结果，模拟的时候随机选择合法动作，直到游戏结束（成功或者失败）"""
        current_env = self.game_env.copy()
        while not current_env.is_game_over():
            legal_moves = current_env.get_legal_moves()
            if not legal_moves:
                break
            move = random.choice(legal_moves)
            current_env.make_move(move)
        return 1 if current_env.is_level_completed() else 0

    def backpropagate(self, result):
        """回溯更新节点信息"""
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)

class MCTS:
    def __init__(self, game_env, iterations=1000):
        self.game_env = game_env
        self.iterations = iterations

    def search(self):
        # 环境reset之后，就会出现多个legal action, 搜索的过程是
        # 1 把当前状态作为root node
        # 2 扩展时会复制环境，并把复制出来的作为当前状态的子节点
        # 3 然后模拟和反向传播
        # 之后继续循环，等到全部legal action 被弹出时， 
        # 利用ucb1 选择一个子节点（从之前子节点集合种选择）
        # 之后，以这个子节点进行扩展，模拟，反向传播，
        # 再之后，依然需要利用ucb1 再选择子节点
        # 一开始时reset 状态作为root node ,然后依次在其 legal action 下扩展四个（假设）
        # 子节点（其实是四个state）之后模拟，传播，更新ucb需要的参数，
        # 然后开始四选一（依据ucb1),然后再以选中的节点的legal action 
        # 进行扩展，模拟和传播，继续更新ucb参数, 一直循环选择这些子节点（上述四个），然后
        # 更新相关ucb 参数，直到循环结束，然后按照胜率选择最好的那个子节点。
        # 上面的循环都是在演进不同节点选择后模拟是否会成功。
        root_node = MCTSNode(game_env=self.game_env)
        for i in range(self.iterations):
            node = root_node
            # Selection,满足条件是中间节点
            # 如果当前状态没有合法移动那么就是叶子节点
            while node.untried_moves == [] and node.children != []:
                node = node.select_child()
            # Expansion: 如果存在未尝试的动作，那就执行一个这种动作，然后得到一个新的
            # 局面，再把这个局面作为新的root
            if node.untried_moves:
                # 每次expand 都会pop 一个legal_action
                node = node.expand()
            # Simulation
            print(f"NO.{i} simulate")
            result = node.simulate()
            # Backpropagation
            node.backpropagate(result)
        # Select the best move
        # best_child = max(root_node.children, key=lambda c: c.visits)
        best_child = max(root_node.children, key=lambda c: (c.wins / c.visits if c.visits > 0 else 0))
        return best_child.move

class GameEnvironment:
    def __init__(self, m=8, n=8, target={1: 9}, max_steps=20, element_low=0, element_high=5):
        self.m = m
        self.n = n
        self.element_low = element_low
        self.element_high = element_high
        self.target = target
        self.max_steps = max_steps
        self.board = self.initialize_board()
        self.score = 0
        self.steps = 0
        self.eliminated_count = 0
        self.current_target = {key: target[key] for key in target}

    def initialize_board(self):
        while True:
            board = np.random.randint(self.element_low, self.element_high, (self.m, self.n))
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
                new_elements = np.random.randint(1, 4, num_new_elements)
                for idx, new_element in enumerate(new_elements):
                    self.board[idx][col] = new_element
        print(f"after drop: {self.board}")

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
                self.is_game_over()
        print(self.current_target)

    def is_game_over(self):
        if self.steps >= self.max_steps or all(count == 0 for count in self.current_target.values()):
            print("游戏结束...")
            return True
        else:
            print("游戏没有结束")
            return False
            # import sys
            # sys.exit()
        
        # if all(count == 0 for count in self.current_target.values()):
            # print("成功!!!")
            # return True
            # import sys
            # sys.exit()

    def is_level_completed(self):
        return all(count == 0 for count in self.current_target.values()) and self.steps < self.max_steps

    def get_score(self):
        return self.score

    def copy(self):
        new_game = GameEnvironment(self.m, self.n, self.target, self.max_steps)
        new_game.board = self.board.copy()
        new_game.score = self.score
        new_game.steps = self.steps
        new_game.eliminated_count = self.eliminated_count
        new_game.current_target = {key: self.current_target[key] for key in self.current_target}
        return new_game

    def print_board(self):
        import colorama
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

    def make_manual_move(self):
        while True:
            try:
                print(f"target: {self.current_target}")
                print(f"valid move: {self.get_legal_moves()}")
                user_input = input("Enter your move (row1 col1 row2 col2): ")
                row1, col1, row2, col2 = map(int, user_input.split())
                if row1 < 0 or row1 >= self.m or col1 < 0 or col1 >= self.n or row2 < 0 or row2 >= self.m or col2 < 0 or col2 >= self.n:
                    print("Invalid move. Please enter valid row and column indices.")
                    continue

                if abs(row1 - row2) + abs(col1 - col2) != 1:
                    print("Invalid move. You can only swap adjacent cells.")
                    continue

                move = (min(row1, row2), min(col1, col2), 'H' if col1 != col2 else 'V')
                if move in self.get_legal_moves():
                    self.make_move(move)
                    self.print_board()
                else:
                    print("Invalid move. No match found after swap.")
            except ValueError:
                print("Invalid input format. Please enter row1 col1 row2 col2.")

if __name__ == "__main__":
    game = GameEnvironment(m=4, n=4, target={1: 15, 2: 18}, max_steps=20, element_low=1, element_high=4)
    game.print_board()
    print(game.target)
    mcts = MCTS(game_env=game, iterations=10)

    while not game.is_game_over():
        print("start searching")
        best_move = mcts.search()
        print(f"Best move: {best_move}")
        game.make_move(best_move)
        game.print_board()
        import time
        time.sleep(10000)
