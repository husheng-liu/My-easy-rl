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
    'clear_percentage': [0.8, 0.7, 0.6, 0.5, 0.4],  # 清除目标百分比
    'remaining_steps_ratio': [0.9, 0.8, 0.7, 0.6, 0.5],  # 剩余步数比例
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