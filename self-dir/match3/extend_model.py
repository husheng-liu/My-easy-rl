import numpy as np

# Step 1: 定义初始化函数
def initialize_population(num_players=2000, skill_mean=50, skill_std=10, persistence_mean=5, persistence_std=1):
    """
    初始化玩家群体。
    :param num_players: 玩家数量
    :param skill_mean: 技能均值
    :param skill_std: 技能标准差
    :param persistence_mean: 坚持度均值
    :param persistence_std: 坚持度标准差
    :return: 玩家群体（技能、坚持度、无聊倾向）
    这些数据应该从真实玩家数据中采集，但是skill, persistence, boredom 怎样量化的？
    """
    skills = np.random.normal(skill_mean, skill_std, num_players)  # 技能正态分布
    persistences = np.random.normal(persistence_mean, persistence_std, num_players)  # 坚持度正态分布
    boredom_tendencies = np.random.uniform(0.1, 0.5, num_players)  # 无聊倾向均匀分布
    return list(zip(skills, persistences, boredom_tendencies))


# Step 2: 定义通过关卡逻辑
def pass_level(player, level_difficulty, learning_rate=0.1, max_attempts=10):
    """
    模拟玩家尝试通过关卡的行为。
    技能水平高于游戏难度就通关，否则就多次尝试提高技能但是也会增加无聊程度
    :param player: 玩家 (skill, persistence, boredom)
    :param level_difficulty: 关卡难度
    :param learning_rate: 学习速率
    :param max_attempts: 最大尝试次数
    :return: 是否通过关卡, 更新后的技能值, 尝试次数
    """
    skill, persistence, boredom = player
    attempts = 0
    while attempts < max_attempts:
        if skill >= level_difficulty:
            return True, skill, attempts + 1 # 技能水平高于游戏难度就通关
        skill += learning_rate * (level_difficulty - skill)  # 更新技能
        attempts += 1
    return False, skill, attempts


# Step 3: 定义流失逻辑
def churn_player(player, attempts, max_attempts, boredom_threshold=0.7):
    """
    模拟玩家是否流失。
    尝试次数高于最大尝试次数，具有一定概率流失
    另外无聊程度高于无聊阈值，那么会有无聊程度相当的可能性流失
    :param player: 玩家 (skill, persistence, boredom)
    :param attempts: 尝试次数
    :param max_attempts: 最大尝试次数
    :param boredom_threshold: 无聊阈值
    :return: 是否流失
    流失概率应该怎样赋值？
    """
    
    _, persistence, boredom = player
    if attempts >= max_attempts:
        return np.random.rand() < 0.5  # 如果尝试次数过多，以 50% 概率流失
    if boredom > boredom_threshold:
        return np.random.rand() < boredom  # 根据无聊倾向决定流失概率
    return False


# Step 4: 模拟整个关卡流程
def simulate_levels(players, level_difficulties, learning_rate=0.1, max_attempts=10, boredom_threshold=0.7):
    """
    模拟所有关卡的行为。
    :param players: 初始玩家群体
    :param level_difficulties: 每个关卡的难度列表
    :param learning_rate: 学习速率
    :param max_attempts: 最大尝试次数
    :param boredom_threshold: 无聊阈值
    :return: 每个关卡的通过率和流失率
    """
    pass_rates = []
    churn_rates = []
    
    for level_difficulty in level_difficulties:
        remaining_players = []
        passed_count = 0
        churned_count = 0
        
        for player in players:
            passed, updated_skill, attempts = pass_level(player, level_difficulty, learning_rate, max_attempts)
            if passed:
                passed_count += 1
            if churn_player(player, attempts, max_attempts, boredom_threshold):
                churned_count += 1
            else: # 没有流失的就是剩余的
                remaining_players.append((updated_skill, player[1], player[2]))  # 更新技能并保留玩家
        
        pass_rate = passed_count / len(players) if len(players) > 0 else 0
        churn_rate = churned_count / len(players) if len(players) > 0 else 0
        pass_rates.append(pass_rate)
        churn_rates.append(churn_rate)
        
        players = remaining_players  # 更新剩余玩家群体
    
    return pass_rates, churn_rates


# Step 5: 主程序
if __name__ == "__main__":
    # 初始化参数
    num_players = 2000
    num_levels = 168
    # 这里的关卡难度是由 baseline 模型的输出得到的
    level_difficulties = np.linspace(30, 70, num_levels)  # 假设关卡难度从 30 到 70 线性递增

    # 初始化玩家群体
    players = initialize_population(num_players)

    # 模拟所有关卡
    pass_rates, churn_rates = simulate_levels(players, level_difficulties)

    # 输出结果
    print("Pass Rates:", len(pass_rates))
    print("Churn Rates:", len(churn_rates))

    # 可视化
    import matplotlib.pyplot as plt # type: ignore

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_levels + 1), pass_rates, label="Pass Rate")
    plt.plot(range(1, num_levels + 1), churn_rates, label="Churn Rate")
    plt.xlabel("Level")
    plt.ylabel("Rate")
    plt.title("Pass and Churn Rates Over Levels")
    plt.legend()
    plt.show()


    # 绘制散点图
    plt.figure(figsize=(12, 8))

    # 定义颜色和形状
    colors = ['red', 'green', 'blue', 'orange']
    markers = ['o', 's', '^', 'D']

    # 每40关为一组
    for i in range(0, num_levels, 40):
        if i ==120:
            end = num_levels
        else:
            end = min(i + 40, num_levels)

        plt.scatter(pass_rates[i:end], churn_rates[i:end], color=colors[i//40], marker=markers[i//40], label=f'Levels {i+1}-{end}')
        if i ==120:
            break
    plt.xlabel("Pass Rate")
    plt.ylabel("Churn Rate")
    plt.title("Pass and Churn Rates Over Levels (Grouped by 40 Levels)")
    plt.legend()
    plt.grid(True)
    plt.show()