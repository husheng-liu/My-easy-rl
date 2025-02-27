import numpy as np
import random

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
        # 交换，发现匹配，消除匹配，元素下落，生成元素，
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
            print("消除")
            eliminated = False
            to_eliminate = self.find_matches()
            print(f"to_eliminate: {to_eliminate}")
            if to_eliminate:
                self.update_target(to_eliminate)
                eliminated = True
                self.eliminated_count += len(to_eliminate)
                for (i, j) in to_eliminate:
                    self.board[i][j] = 0

                self.drop_elements()

    def drop_elements(self):
        for col in range(self.n):
            # 收集当前列中非零元素
            non_empty_elements = [self.board[i][col] for i in range(self.m) if self.board[i][col] != 0]
            
            # 将非零元素下落到列的底部
            for idx, element in enumerate(reversed(non_empty_elements)):
                self.board[self.m - 1 - idx][col] = element
            
            # 计算需要生成的新元素数量
            num_new_elements = self.m - len(non_empty_elements)
            
            # 在列的顶部生成新的随机元素
            if num_new_elements > 0:
                new_elements = np.random.randint(1, 7, num_new_elements)
                for idx, new_element in enumerate(new_elements):
                    self.board[idx][col] = new_element
        print(f"after drop: {self.print_board()}")
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
        # print(f"to_eliminate: {to_eliminate}")
        for (i, j) in to_eliminate:
            cell_type = self.board[i][j]
            if cell_type in self.current_target:
                self.current_target[cell_type] = max(0, self.current_target[cell_type] - 1)
                self.is_game_over()
                # print(f"self.current_target: {self.current_target}")
    def is_game_over(self):
        if self.steps >= self.max_steps:
            print("失败...")
            import sys
            sys.exit()
        if all(count == 0 for count in self.current_target.values()):
            print("成功!!!")
            import sys
            sys.exit()

    def is_level_completed(self):
        return all(count == 0 for count in self.current_target.values()) and self.steps < self.max_steps

    def get_score(self):
        return self.score

    # def copy(self):
    #     new_game = GameEnvironment(self.m, self.n, self.target, self.max_steps)
    #     new_game.board = self.board.copy()
    #     new_game.score = self.score
    #     new_game.steps = self.steps
    #     new_game.eliminated_count = self.eliminated_count
    #     new_game.current_target = {key: self.current_target[key] for key in self.current_target}
    #     return new_game

    def print_board(self):
            import colorama
            colorama.init(autoreset=True)
            # 定义颜色映射
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
    game = GameEnvironment(m=4, n=4, target={1: 3, 2: 3}, max_steps=20, element_low=1, element_high=4)
#     print(game.find_matches(board=[[2, 3, 2, 2],
# [1, 2, 2, 1],
# [2, 1, 3, 1],
# [2, 3, 2, 1]]))
    game.print_board()
    # game.make_manual_move()
    while not game.is_game_over():
        legal_moves = game.get_legal_moves()
        print(legal_moves)
        move = random.choice(legal_moves)
        print(f"move: {move}")
        game.make_move(move)
    #     game.make_manual_move()
    #     print(f"Steps remaining: {game.max_steps - game.steps}")
    #     print(f"Current target: {game.current_target}")

    # if game.is_level_completed():
    #     print("Level completed!")
    # else:
    #     print("Game over!")