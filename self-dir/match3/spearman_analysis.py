import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# 示例数据：假设我们有一个包含多个特征的数据集
data = {
    'level': [1, 2, 3, 4, 5],  # 关卡编号
    'clear_percentage': [0.8, 0.7, 0.6, 0.5, 0.4],  # 清除目标百分比
    'remaining_steps_ratio': [0.9, 0.8, 0.7, 0.6, 0.5],  # 剩余步数比例
    'level_difficulty': [30, 40, 50, 60, 70],  # 关卡难度
    'pass_rate': [0.9, 0.8, 0.7, 0.6, 0.5],  # 通过率
    'churn_rate': [0.1, 0.2, 0.3, 0.4, 0.5]   # 流失率
}

# 转换为 DataFrame
df = pd.DataFrame(data)

# 定义函数：计算 Spearman 相关性
def calculate_spearman_correlation(df, target_columns, feature_columns):
    """
    计算特征与目标变量之间的 Spearman 相关性。
    :param df: 数据框
    :param target_columns: 目标变量列名列表
    :param feature_columns: 特征列名列表
    :return: 包含 Spearman 相关系数和 p 值的结果 DataFrame
    """
    results = []
    for target in target_columns:
        for feature in feature_columns:
            correlation, p_value = spearmanr(df[feature], df[target])
            results.append({
                'Target': target,
                'Feature': feature,
                'Spearman Correlation': correlation,
                'P-Value': p_value
            })
    return pd.DataFrame(results)

# 定义目标变量和特征变量
target_columns = ['pass_rate', 'churn_rate']  # 目标变量
feature_columns = ['clear_percentage', 'remaining_steps_ratio', 'level_difficulty']  # 特征变量

# 计算 Spearman 相关性
spearman_results = calculate_spearman_correlation(df, target_columns, feature_columns)

# 打印结果
print("Spearman Correlation Results:")
print(spearman_results)


