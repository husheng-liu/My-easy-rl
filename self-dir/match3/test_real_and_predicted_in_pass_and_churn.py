from scipy.stats import ks_2samp
from sklearn.metrics import mean_squared_error
import numpy as np


# 假设我们有以下数据：
# - 预测的通过率和流失率分布
# - 真实的通过率和流失率分布

# 示例数据
predicted_pass_rates = np.array([0.8, 0.7, 0.6, 0.5, 0.4])  # 预测的通过率分布
real_pass_rates = np.array([0.82, 0.68, 0.65, 0.52, 0.39])  # 真实的通过率分布

predicted_churn_rates = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # 预测的流失率分布
real_churn_rates = np.array([0.12, 0.21, 0.28, 0.39, 0.48])  # 真实的流失率分布

# Step 1: 使用 Kolmogorov-Smirnov 检验比较分布
ks_stat_pass, p_value_pass = ks_2samp(predicted_pass_rates, real_pass_rates)
ks_stat_churn, p_value_churn = ks_2samp(predicted_churn_rates, real_churn_rates)

print(f"KS Test for Pass Rates: Statistic={ks_stat_pass}, P-value={p_value_pass}")
print(f"KS Test for Churn Rates: Statistic={ks_stat_churn}, P-value={p_value_churn}")

# Step 2: 使用均方误差（MSE）比较预测值与真实值
mse_pass = mean_squared_error(real_pass_rates, predicted_pass_rates)
mse_churn = mean_squared_error(real_churn_rates, predicted_churn_rates)

print(f"MSE for Pass Rates: {mse_pass}")
print(f"MSE for Churn Rates: {mse_churn}")

# Step 3: 可视化比较分布
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

# 绘制通过率分布
plt.subplot(1, 2, 1)
plt.plot(predicted_pass_rates, label="Predicted Pass Rates")
plt.plot(real_pass_rates, label="Real Pass Rates")
plt.title("Pass Rates Comparison")
plt.legend()

# 绘制流失率分布
plt.subplot(1, 2, 2)
plt.plot(predicted_churn_rates, label="Predicted Churn Rates")
plt.plot(real_churn_rates, label="Real Churn Rates")
plt.title("Churn Rates Comparison")
plt.legend()

plt.tight_layout()
plt.show()