import random

class Agent:
    def __init__(self, name, elo=1200):
        self.name = name
        self.elo = elo

    def __repr__(self):
        return f"{self.name}: {self.elo}"

class EloRankingSystem:
    def __init__(self, agents):
        self.agents = [Agent(name) for name in agents]  # 初始化agent及其Elo分

    def simulate_match(self, agent1, agent2):
        """
        模拟两方对战，返回胜者和败者。
        """
        prob1 = 1 / (1 + 10 ** ((agent2.elo - agent1.elo) / 400))
        prob2 = 1 - prob1

        # 随机决定胜负
        if random.random() < prob1:
            winner, loser = agent1, agent2
        else:
            winner, loser = agent2, agent1

        return winner, loser

    def adjust_elo(self, winner, loser, k_factor=32):
        """
        根据对战结果调整Elo分。
        """
        expected_win = 1 / (1 + 10 ** ((loser.elo - winner.elo) / 400))
        expected_loss = 1 - expected_win

        winner.elo += k_factor * (1 - expected_win)
        loser.elo += k_factor * (0 - expected_loss)

    def select_agents_by_rank(self):
        """
        根据排名选择两个相邻的Agent进行对战。
        """
        rankings = sorted(self.agents, key=lambda x: x.elo, reverse=True)
        num_agents = len(rankings)

        # 如果有两个或更多Agent，则随机选择一对相邻的Agent
        if num_agents >= 2:
            idx = random.randint(0, num_agents - 2)  # 确保不会超出索引范围
            return rankings[idx], rankings[idx + 1]
        else:
            raise ValueError("至少需要两个Agent才能进行比赛。")

    def run_tournament(self, num_matches):
        """
        运行指定数量的比赛,按照排名相近进行对战
        """
        for _ in range(num_matches):
            try:
                # 根据排名选择两个相邻的Agent
                agent1, agent2 = self.select_agents_by_rank()
                winner, loser = self.simulate_match(agent1, agent2)
                self.adjust_elo(winner, loser)
            except ValueError as e:
                print(e)
                break

    def get_rankings(self):
        """
        返回按Elo分排序的排名表。
        """
        return sorted(self.agents, key=lambda x: x.elo, reverse=True)

# 主程序
if __name__ == "__main__":
    # 初始化agents
    agents = ["Alice", "Bob", "Charlie", "David", "Eve"]

    # 创建Elo排名系统
    ranking_system = EloRankingSystem(agents)

    # 运行比赛（例如：50场比赛）
    num_matches = 50
    ranking_system.run_tournament(num_matches)

    # 输出最终排名表
    rankings = ranking_system.get_rankings()
    print("最终排名表:")
    for rank, agent in enumerate(rankings, start=1):
        print(f"{rank}. {agent}")