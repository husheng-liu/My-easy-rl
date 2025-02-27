import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# 示例数据：假设我们有一个包含多个特征的数据集
data = {
    'level': [1, 2, 3, 4, 5],  # 关卡编号
    'clear_percentage': [0.8, 0.7, 0.6, 0.5, 0.4],  # 清除目标百分比
    'remaining_steps_ratio': [0.9, 0.8, 0.7, 0.6, 0.5],  # 剩余步数比例
    'level_difficulty': [30, 40, 50, 60, 70],  # 关卡难度
    'goal_score': [80, 70, 60, 50, 40],  # 目标分数
    'bonus_collected': [0.85, 0.75, 0.65, 0.55, 0.45],  # 收集的奖励比例
    'pass_rate': [0.9, 0.8, 0.7, 0.6, 0.5],  # 通过率
    'churn_rate': [0.1, 0.2, 0.3, 0.4, 0.5]   # 流失率
}

# 转换为 DataFrame
df = pd.DataFrame(data)

# 定义函数：计算 Spearman 相关性
def calculate_spearman_correlation(feature_values, target_values):
    """
    计算特征值与目标值之间的 Spearman 相关性。
    :param feature_values: 特征值列表
    :param target_values: 目标值列表
    :return: Spearman 相关系数和 p 值
    """
    correlation, p_value = spearmanr(feature_values, target_values)
    return correlation, p_value

# 定义函数：生成特征的子集并进行相关性分析
def analyze_feature_subsets(df, target_columns, all_features, subset_sizes=[5, 10, 25, 50, 100]):
    """
    根据特征值的排序生成子集，并计算 Spearman 相关性。
    :param df: 数据框
    :param target_columns: 目标变量列名列表
    :param all_features: 所有特征列名列表
    :param subset_sizes: 子集大小百分比列表
    :return: 包含子集分析结果的 DataFrame
    """
    all_results = []
    
    for feature in all_features:
        # 对特征值按降序排序
        sorted_df = df.sort_values(by=feature, ascending=False)
        
        for size in subset_sizes:
            # 计算子集大小
            subset_count = max(1, int(len(sorted_df) * size / 100))  # 确保至少选择一个样本
            subset_df = sorted_df.iloc[:subset_count]
            
            for target in target_columns:
                # 提取子集的特征值和目标值
                feature_values = subset_df[feature].values
                target_values = subset_df[target].values
                
                # 计算 Spearman 相关性
                correlation, p_value = calculate_spearman_correlation(feature_values, target_values)
                
                # 记录结果
                all_results.append({
                    'Feature': feature,
                    'Target': target,
                    'Subset Size (%)': size,
                    'Spearman Correlation': correlation,
                    'P-Value': p_value
                })
    
    return pd.DataFrame(all_results)

# 定义目标变量和所有特征变量
target_columns = ['pass_rate', 'churn_rate']  # 目标变量
all_features = ['clear_percentage', 'remaining_steps_ratio', 'level_difficulty', 'goal_score', 'bonus_collected']  # 所有特征

# 分析不同子集的特征
subset_analysis_results = analyze_feature_subsets(df, target_columns, all_features, subset_sizes=[5, 10, 25, 50, 100])

# 打印结果
print("Subset Analysis Results:")
print(subset_analysis_results)