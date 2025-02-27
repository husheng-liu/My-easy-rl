import numpy as np
import scipy.stats as stats

# 测试者通关次数
# successes = [8, 7, 6, 9, 5, 4, 8, 7, 6, 5]
np.random.seed(1)
successes = np.random.randint(8,100, [100,])
print(successes)
attempts = 100  # 每人尝试10次

# 计算每个测试者的通关率
rates = [s / attempts for s in successes]

# 平均通关率
mean_rate = np.mean(rates)
print(f"平均通关率: {mean_rate}")

# 计算95%置信区间
confidence_interval = stats.t.interval(0.95, len(rates)-1, loc=np.mean(rates), scale=stats.sem(rates))
print(f"95%置信区间: {confidence_interval}")