import numpy as np
import pandas as pd
from scipy.stats import norm, uniform
# 从真实玩家数据中拟合技能分布,坚持度分布和无聊程度分布,然后每个玩家的skill, persistence, boredom
# 都是从上述分布中采样出来的
 
# 假设我们有一个包含真实玩家数据的 DataFrame
# 数据包括：尝试次数（attempts）、是否通过（pass_level）、是否流失（churn）
data = pd.DataFrame({
    'attempts': [3, 5, 2, 4, 6, 1, 2, 3, 5, 7],  # 每个玩家尝试某个关卡的次数
    'pass_level': [1, 0, 1, 1, 0, 1, 1, 0, 0, 1],  # 是否通过关卡 (1: 通过, 0: 未通过)
    'churn': [0, 1, 0, 0, 1, 0, 0, 1, 1, 0]  # 是否流失 (1: 流失, 0: 未流失)
})

# Step 1: 拟合技能分布
# 技能分布可以通过完成关卡的成功率来估计
skill_mean = data['pass_level'].mean() * 100  # 假设技能值范围为 0-100
skill_std = data['pass_level'].std() * 100
print(f"Skill Distribution: Mean={skill_mean}, Std={skill_std}")

# Step 2: 拟合坚持度分布
# 坚持度分布可以通过尝试次数来估计
persistence_mean = data['attempts'].mean()
persistence_std = data['attempts'].std()
print(f"Persistence Distribution: Mean={persistence_mean}, Std={persistence_std}")

# Step 3: 拟合无聊倾向分布
# 无聊倾向分布可以通过流失率来估计，假设均匀分布在 [0, 1]
boredom_tendency = data['churn'].mean()
boredom_distribution = uniform(loc=0, scale=boredom_tendency)
print(f"Boredom Tendency Distribution: Uniform(0, {boredom_tendency})")

# Step 4: 拟合学习速率
# 学习速率可以通过观察玩家多次尝试后技能提升的速度来估计
# 假设学习速率为每次尝试提升 5%
learning_rate = 0.05
print(f"Learning Rate: {learning_rate}")

# Step 5: 拟合流失速率
# 流失速率可以通过统计流失比例来估计
churn_rate = data['churn'].mean()
print(f"Churn Rate: {churn_rate}")


import numpy as np
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: 数据准备
# 假设我们有一个包含 AI 游戏代理生成的游戏数据的 DataFrame
# 数据包括：清除目标百分比（clear_percentage）、剩余步数比例（remaining_steps_ratio）、是否通过（pass_level）
data = pd.DataFrame({
    'level': [1, 2, 3, 4, 5],  # 关卡编号
    'clear_percentage': [0.8, 0.7, 0.6, 0.5, 0.4],  # 清除目标百分比(9种指标)
    'remaining_steps_ratio': [0.9, 0.8, 0.7, 0.6, 0.5],  # 剩余步数比例(5种指标)
    'agent_pass_rate':[0.97, 0.90, 0.76,0.74,0.58],# (2种统计指标) 
    'pass_rate': [0.85, 0.75, 0.65, 0.55, 0.45]  # 真实通过率（用于训练模型）
})

# 特征和目标变量
X = data[['clear_percentage', 'remaining_steps_ratio']]  # 输入特征
y = data['pass_rate']  # 目标变量（真实通过率）

# Step 2: 模型训练
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# Step 3: 模型评估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse}")

# Step 4: 对每个关卡进行预测
# 假设我们要预测新的关卡数据
new_data = pd.DataFrame({
    'level': [6, 7],
    'clear_percentage': [0.3, 0.2],
    'remaining_steps_ratio': [0.4, 0.3]
})

# 预测通过率
predicted_pass_rates = model.predict(new_data[['clear_percentage', 'remaining_steps_ratio']])
new_data['predicted_pass_rate'] = predicted_pass_rates

print("Predicted Pass Rates for New Levels:")
print(new_data)