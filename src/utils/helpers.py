import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed=42):
    """設置隨機種子以確保結果可重現"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def generate_random_lottery_numbers(min_num=1, max_num=49, count=5):
    """生成隨機彩票號碼"""
    return sorted(random.sample(range(min_num, max_num + 1), count))

def calculate_hit_rate(predicted, actual):
    """計算命中率"""
    hits = sum(1 for num in predicted if num in actual)
    return hits / len(actual)

def plot_number_frequency(data, column_prefix='num', num_columns=5, figsize=(12, 8), save_path=None):
    """繪製號碼頻率分佈圖"""
    plt.figure(figsize=figsize)
    
    # 合併所有號碼列
    all_numbers = []
    for i in range(1, num_columns + 1):
        all_numbers.extend(data[f'{column_prefix}{i}'].tolist())
    
    # 計算頻率
    number_counts = pd.Series(all_numbers).value_counts().sort_index()
    
    # 繪製頻率圖
    sns.barplot(x=number_counts.index, y=number_counts.values)
    plt.title('Number Frequency Distribution')
    plt.xlabel('Number')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_correlation_matrix(data, columns=None, figsize=(10, 8), save_path=None):
    """繪製相關性矩陣"""
    if columns is None:
        # 排除非數值列
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # 計算相關性
    corr = data[columns].corr()
    
    # 繪製相關性矩陣
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', square=True, linewidths=.5)
    plt.title('Correlation Matrix')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def create_lag_features(data, columns, lag_periods=[1, 2, 3, 5, 10]):
    """創建滯後特徵"""
    df = data.copy()
    
    for col in columns:
        for lag in lag_periods:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    return df

def create_rolling_features(data, columns, windows=[5, 10, 20, 50]):
    """創建滾動特徵"""
    df = data.copy()
    
    for col in columns:
        for window in windows:
            df[f'{col}_mean_{window}'] = df[col].rolling(window=window).mean()
            df[f'{col}_std_{window}'] = df[col].rolling(window=window).std()
            df[f'{col}_min_{window}'] = df[col].rolling(window=window).min()
            df[f'{col}_max_{window}'] = df[col].rolling(window=window).max()
    
    return df