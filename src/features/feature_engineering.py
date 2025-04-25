import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
from functools import lru_cache
import time
from datetime import datetime
from statsmodels.tsa.stattools import acf, pacf

logger = logging.getLogger(__name__)

class LotteryFeatureEngineering:
    def __init__(self, data_path=None):
        """
        初始化特徵工程類
        
        參數:
        data_path: 數據文件路徑，如果提供則自動載入數據
        """
        self.scaler = StandardScaler()
        self.data_path = data_path
        self.data = None
        self.features = None
        
        # 如果提供了數據路徑，則自動載入數據
        if data_path:
            self.load_data()
    
    def load_data(self):
        """載入彩球歷史數據"""
        logger.info(f"正在從 {self.data_path} 載入數據...")
        try:
            self.data = pd.read_excel(self.data_path)
            # 確保列名正確
            if len(self.data.columns) >= 6:
                self.data.columns = ['date'] + [f'num{i}' for i in range(1, len(self.data.columns))]
            # 轉換日期格式
            self.data['date'] = pd.to_datetime(self.data['date'])
            logger.info(f"成功載入數據，共 {len(self.data)} 條記錄")
            return self.data
        except Exception as e:
            logger.error(f"載入數據失敗: {str(e)}")
            return None
    
    @lru_cache(maxsize=32)
    def _calculate_frequency_features(self, data_tuple):
        """計算頻率特徵（使用緩存優化）"""
        # 將元組轉換回DataFrame
        data = pd.DataFrame(data_tuple, columns=[f'num{i+1}' for i in range(len(data_tuple[0]))])
        
        # 計算每個號碼的出現頻率
        freq_dict = {}
        for col in data.columns:
            for num in data[col]:
                freq_dict[num] = freq_dict.get(num, 0) + 1
        
        # 將頻率轉換為比率
        total_draws = len(data)
        freq_ratio = {num: count / total_draws for num, count in freq_dict.items()}
        
        return freq_ratio
    
    def _calculate_recency_features(self, data):
        """計算最近性特徵（優化版本）"""
        # 使用numpy數組加速計算
        num_range = range(1, 50)
        last_seen = {num: -1 for num in num_range}
        recency = {num: [] for num in num_range}
        
        # 使用向量化操作
        for idx, draw in enumerate(data):
            for num in num_range:
                if num in draw:
                    recency_val = idx - last_seen[num]
                    recency[num].append(recency_val)
                    last_seen[num] = idx
        
        # 計算平均最近性
        avg_recency = {}
        for num in num_range:
            if recency[num]:
                avg_recency[num] = sum(recency[num]) / len(recency[num])
            else:
                avg_recency[num] = len(data)  # 如果從未出現，設為最大值
        
        return avg_recency
    
    def _calculate_pattern_features(self, data, window_size=5):
        """計算模式特徵（優化版本）"""
        # 預先分配記憶體
        patterns = {}
        
        # 只考慮最近的數據以提高效率
        recent_data = data[-window_size*10:] if len(data) > window_size*10 else data
        
        for i in range(len(recent_data) - window_size + 1):
            window = recent_data[i:i+window_size]
            # 使用frozenset作為鍵以提高查找效率
            window_key = tuple(sorted(frozenset(num for draw in window for num in draw)))
            patterns[window_key] = patterns.get(window_key, 0) + 1
        
        # 找出最常見的模式
        common_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return common_patterns
    
    def _calculate_hot_cold_features(self, data, hot_threshold=0.3, cold_threshold=0.1):
        """計算熱門和冷門號碼特徵"""
        # 使用已優化的頻率計算
        freq_ratio = self._calculate_frequency_features(tuple(map(tuple, data)))
        
        # 識別熱門和冷門號碼
        hot_numbers = [num for num, ratio in freq_ratio.items() if ratio >= hot_threshold]
        cold_numbers = [num for num, ratio in freq_ratio.items() if ratio <= cold_threshold]
        
        return hot_numbers, cold_numbers
    
    def _calculate_gap_features(self, data):
        """計算號碼間隔特徵（優化版本）"""
        # 使用numpy加速計算
        gaps = {}
        for num in range(1, 50):
            # 找出該號碼出現的所有位置
            positions = []
            for i, draw in enumerate(data):
                if num in draw:
                    positions.append(i)
            
            # 計算間隔
            if len(positions) > 1:
                draw_gaps = np.diff(positions)
                gaps[num] = np.mean(draw_gaps)
            else:
                gaps[num] = len(data)  # 如果只出現一次或從未出現，設為最大值
        
        return gaps
    
    def _calculate_pair_features(self, data):
        """計算配對特徵（優化版本）"""
        # 使用字典加速查找
        pairs = {}
        
        # 只考慮最近的數據以提高效率
        recent_data = data[-100:] if len(data) > 100 else data
        
        for draw in recent_data:
            # 使用組合而不是雙重循環
            for i in range(len(draw)):
                for j in range(i+1, len(draw)):
                    pair = (min(draw[i], draw[j]), max(draw[i], draw[j]))
                    pairs[pair] = pairs.get(pair, 0) + 1
        
        # 找出最常見的配對
        common_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:20]
        
        return common_pairs
    
    def _is_prime(self, n):
        """判斷一個數是否為質數"""
        if n <= 1:
            return 0
        if n <= 3:
            return 1
        if n % 2 == 0 or n % 3 == 0:
            return 0
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return 0
            i += 6
        return 1
    
    def _count_consecutive(self, row):
        """計算連續數字的數量"""
        numbers = sorted([row[f'num{i}'] for i in range(1, len(row)) if f'num{i}' in row])
        consecutive_count = 0
        for i in range(len(numbers) - 1):
            if numbers[i + 1] - numbers[i] == 1:
                consecutive_count += 1
        return consecutive_count
    
    def _calculate_entropy(self, numbers):
        """計算數字組合的熵"""
        try:
            _, counts = np.unique(numbers, return_counts=True)
            probabilities = counts / len(numbers)
            return -np.sum(probabilities * np.log2(probabilities))
        except:
            return 0  # 出錯時返回0
    
    def _get_sequence_pattern(self, numbers):
        """獲取數字序列的模式"""
        try:
            sorted_numbers = sorted(numbers)
            diffs = [sorted_numbers[i+1] - sorted_numbers[i] for i in range(len(sorted_numbers)-1)]
            pattern = ''.join([str(d) for d in diffs])
            return hash(pattern) % 100  # 返回模式的哈希值，限制在0-99範圍內
        except:
            return 0  # 出錯時返回0
    
    def _calculate_distribution_score(self, numbers):
        """計算數字分佈的均勻性分數"""
        try:
            sorted_numbers = sorted(numbers)
            ideal_gap = 49 / (len(numbers) + 1)  # 假設彩球範圍是1-49
            actual_gaps = [sorted_numbers[0]] + [sorted_numbers[i+1] - sorted_numbers[i] for i in range(len(sorted_numbers)-1)] + [49 - sorted_numbers[-1]]
            return np.std(actual_gaps) / ideal_gap  # 標準差與理想間隔的比值
        except:
            return 0  # 出錯時返回0
    
    def _calculate_complexity(self, numbers):
        """計算數字組合的複雜性分數"""
        try:
            # 結合多個複雜性指標
            entropy = self._calculate_entropy(numbers)
            distribution = self._calculate_distribution_score(numbers)
            
            # 創建臨時字典以便調用 _count_consecutive
            temp_dict = {f'num{i+1}': num for i, num in enumerate(numbers)}
            consecutive = self._count_consecutive(temp_dict)
            
            # 綜合分數
            return entropy * (1 + distribution) / (1 + consecutive)
        except:
            return 0  # 出錯時返回0
    
    def create_basic_features(self):
        """創建基本特徵"""
        if self.data is None:
            logger.error("未載入數據，無法創建特徵")
            return None
            
        logger.info("開始創建基本特徵...")
        df = self.data.copy()
        
        # 創建所有基本特徵的字典
        features_dict = {}
        
        # 時間相關特徵
        features_dict['year'] = df['date'].dt.year
        features_dict['month'] = df['date'].dt.month
        features_dict['day'] = df['date'].dt.day
        features_dict['day_of_week'] = df['date'].dt.dayofweek
        features_dict['quarter'] = df['date'].dt.quarter
        features_dict['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        features_dict['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        
        # 數字統計特徵
        num_cols = [col for col in df.columns if col.startswith('num')]
        for col in num_cols:
            features_dict[f'{col}_is_prime'] = df[col].apply(self._is_prime)
            features_dict[f'{col}_is_odd'] = df[col].apply(lambda x: x % 2 != 0).astype(int)
        
        # 數字組合特徵
        features_dict['sum_all'] = df[num_cols].sum(axis=1)
        features_dict['mean_all'] = df[num_cols].mean(axis=1)
        features_dict['std_all'] = df[num_cols].std(axis=1)
        features_dict['max_minus_min'] = df[num_cols].max(axis=1) - df[num_cols].min(axis=1)
        
        # 連續數字特徵
        features_dict['consecutive_count'] = df.apply(self._count_consecutive, axis=1)
        
        # 一次性將所有特徵添加到DataFrame
        new_features = pd.DataFrame(features_dict, index=df.index)
        df = pd.concat([df, new_features], axis=1)
        
        self.features = df
        logger.info(f"基本特徵創建完成，共 {len(new_features.columns)} 個特徵")
        return df
    
    def create_complex_features(self):
        """創建複雜特徵（包括基本特徵和進階特徵）
        
        返回:
            features: 包含所有特徵的 DataFrame
        """
        # 創建基本特徵
        self.create_basic_features()
        
        # 創建進階特徵
        self.create_advanced_features()
        
        # 返回特徵
        return self.features


    def extract_features(self, data=None):
        """提取特徵（優化版本）"""
        if data is None:
            if self.data is None:
                logger.error("未提供數據，無法提取特徵")
                return None
            data = self.data.copy()
        
        start_time = time.time()
        logger.info("開始特徵提取...")
        
        # 提取數字列
        num_cols = [col for col in data.columns if col.startswith('num')]
        
        # 將數據轉換為列表的列表格式
        data_list = data[num_cols].values.tolist()
        
        # 將數據轉換為元組以便緩存
        data_tuple = tuple(map(tuple, data_list))
        
        # 並行計算各種特徵
        freq_ratio = self._calculate_frequency_features(data_tuple)
        avg_recency = self._calculate_recency_features(data_list)
        hot_numbers, cold_numbers = self._calculate_hot_cold_features(data_list)
        gaps = self._calculate_gap_features(data_list)
        common_pairs = self._calculate_pair_features(data_list)
        
        # 為每個可能的號碼創建特徵向量
        features = {}
        for num in range(1, 50):
            features[num] = [
                freq_ratio.get(num, 0),  # 頻率
                avg_recency.get(num, len(data_list)),  # 最近性
                1 if num in hot_numbers else 0,  # 是否為熱門號碼
                1 if num in cold_numbers else 0,  # 是否為冷門號碼
                gaps.get(num, len(data_list)),  # 間隔
                sum(1 for pair, _ in common_pairs if num in pair)  # 配對頻率
            ]
        
        # 將特徵轉換為DataFrame
        feature_df = pd.DataFrame.from_dict(features, orient='index')
        feature_df.columns = ['frequency', 'recency', 'is_hot', 'is_cold', 'gap', 'pair_frequency']
        
        # 標準化特徵
        feature_df = pd.DataFrame(self.scaler.fit_transform(feature_df), 
                                 index=feature_df.index, 
                                 columns=feature_df.columns)
        
        logger.info(f"特徵提取完成，耗時 {time.time() - start_time:.2f} 秒")
        return feature_df
    
    def create_advanced_features(self):
        """創建進階特徵"""
        if self.data is None:
            logger.error("未載入數據，無法創建進階特徵")
            return None
            
        logger.info("開始創建進階特徵...")
        df = self.data.copy()
        
        # 如果已經創建了基本特徵，則使用它
        if self.features is not None:
            df = self.features.copy()
        
        # 創建所有進階特徵的字典
        features_dict = {}
        
        # 數字列
        num_cols = [col for col in df.columns if col.startswith('num')]
        
        # 創建滯後特徵 (前N期的號碼)
        for lag in range(1, 4):  # 創建前3期的滯後特徵
            for col in num_cols:
                features_dict[f'{col}_lag{lag}'] = df[col].shift(lag)
        
        # 創建滾動統計特徵
        windows = [3, 5, 10]
        for window in windows:
            features_dict[f'sum_all_roll{window}'] = df[[c for c in df.columns if c.startswith('sum_all')]].rolling(window).mean().iloc[:, 0]
            features_dict[f'mean_all_roll{window}'] = df[[c for c in df.columns if c.startswith('mean_all')]].rolling(window).mean().iloc[:, 0]
            features_dict[f'std_all_roll{window}'] = df[[c for c in df.columns if c.startswith('std_all')]].rolling(window).mean().iloc[:, 0]
        
        # 創建複雜性特徵
        features_dict['entropy'] = df.apply(lambda row: self._calculate_entropy([row[col] for col in num_cols]), axis=1)
        features_dict['sequence_pattern'] = df.apply(lambda row: self._get_sequence_pattern([row[col] for col in num_cols]), axis=1)
        features_dict['distribution_score'] = df.apply(lambda row: self._calculate_distribution_score([row[col] for col in num_cols]), axis=1)
        features_dict['complexity_score'] = df.apply(lambda row: self._calculate_complexity([row[col] for col in num_cols]), axis=1)
        
        # 一次性將所有特徵添加到DataFrame
        new_features = pd.DataFrame(features_dict, index=df.index)
        df = pd.concat([df, new_features], axis=1)
        
        # 填充NaN值
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        self.features = df
        logger.info(f"進階特徵創建完成，共 {len(new_features.columns)} 個特徵")
        return df
    
    def get_latest_features(self, n_previous=5):
        """獲取最新的特徵用於預測"""
        if self.features is None:
            logger.error("未創建特徵，無法獲取最新特徵")
            return None
            
        # 獲取最新的n_previous期數據
        latest_features = self.features.iloc[-n_previous:].copy()
        
        # 移除目標變數（下一期的號碼）
        target_cols = [col for col in latest_features.columns if col.startswith('next_')]
        if target_cols:
            latest_features = latest_features.drop(columns=target_cols)
        
        return latest_features
    
    def prepare_train_test_data(self, test_size=0.2, target_cols=None):
        """準備訓練和測試數據"""
        if self.features is None:
            logger.error("未創建特徵，無法準備訓練數據")
            return None, None, None, None
            
        # 如果未指定目標列，則使用所有num開頭的列
        if target_cols is None:
            target_cols = [col for col in self.features.columns if col.startswith('num')]
        
        # 創建目標變數（下一期的號碼）
        y = self.features[target_cols].shift(-1)
        
        # 移除目標變數和不需要的列
        X = self.features.drop(columns=target_cols + ['date'])
        
        # 移除含有NaN的行
        valid_idx = ~y.isnull().any(axis=1)
        X = X[valid_idx]
        y = y[valid_idx]
        
        # 分割訓練集和測試集
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"準備訓練和測試數據完成，訓練集大小: {X_train.shape}, 測試集大小: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def get_training_data(self):
        """獲取訓練數據
        
        返回:
            X: 特徵矩陣
            y: 目標變數
        """
        if self.features is None:
            # 如果特徵尚未創建，先創建基本和進階特徵
            self.create_basic_features()
            self.create_advanced_features()
        
        # 獲取號碼列作為目標變數
        target_cols = [col for col in self.data.columns if col.startswith('num')]
        
        # 創建目標變數（當前期的號碼）
        y = self.data[target_cols]
        
        # 移除目標變數和不需要的列
        X = self.features.drop(columns=target_cols + ['date'])
        
        # 移除含有NaN的行
        valid_idx = ~X.isnull().any(axis=1) & ~y.isnull().any(axis=1)
        X = X[valid_idx]
        y = y[valid_idx]
        
        # 記錄日誌
        logger.info(f"獲取訓練數據完成，特徵矩陣大小: {X.shape}, 目標變數大小: {y.shape}")
        
        return X, y


    def analyze_correlations(self):
        """分析號碼之間的相關性"""
        # 載入數據
        df = self.load_data()
        
        # 只選擇號碼列
        number_cols = [col for col in df.columns if col.startswith('num')]
        
        # 創建一個新的DataFrame來存儲每個號碼的出現情況
        all_numbers = range(1, 50)  # 假設彩球範圍是1-49
        occurrence_df = pd.DataFrame(index=df.index, columns=[f'num_{i}' for i in all_numbers])
        
        # 填充出現情況 (1表示出現，0表示未出現)
        for idx, row in df.iterrows():
            drawn_numbers = [row[col] for col in number_cols if pd.notna(row[col])]
            for num in all_numbers:
                occurrence_df.loc[idx, f'num_{num}'] = 1 if num in drawn_numbers else 0
        
        # 計算相關性矩陣
        corr_matrix = occurrence_df.corr()
        
        # 找出最強的正相關和負相關
        strongest_positive = []
        strongest_negative = []
        
        # 遍歷相關性矩陣的上三角部分
        for i in range(len(all_numbers)):
            for j in range(i+1, len(all_numbers)):
                num1 = i + 1
                num2 = j + 1
                col1 = f'num_{num1}'
                col2 = f'num_{num2}'
                
                if col1 in corr_matrix.columns and col2 in corr_matrix.columns:
                    corr = corr_matrix.loc[col1, col2]
                    
                    pair = f"{num1}-{num2}"
                    if corr > 0:
                        strongest_positive.append({"pair": pair, "correlation": corr})
                    else:
                        strongest_negative.append({"pair": pair, "correlation": corr})
        
        # 排序
        strongest_positive = sorted(strongest_positive, key=lambda x: x["correlation"], reverse=True)[:5]
        strongest_negative = sorted(strongest_negative, key=lambda x: x["correlation"])[:5]
        
        return {
            "strongest_positive": strongest_positive,
            "strongest_negative": strongest_negative
        }

    def analyze_periodicity(self):
        """分析號碼的週期性"""
        # 載入數據
        df = self.load_data()
        
        # 只選擇號碼列
        number_cols = [col for col in df.columns if col.startswith('num')]
        
        # 計算每個可能號碼的週期性
        all_numbers = range(1, 50)  # 假設彩球範圍是1-49
        periodicity = []
        
        for num in all_numbers:
            # 找出該號碼出現的所有期數
            appearances = []
            for idx, row in df.iterrows():
                drawn_numbers = [row[col] for col in number_cols if pd.notna(row[col])]
                if num in drawn_numbers:
                    appearances.append(idx)
            
            # 計算間隔
            if len(appearances) > 1:
                intervals = [appearances[i+1] - appearances[i] for i in range(len(appearances)-1)]
                # 計算變異係數 (標準差/平均值)
                mean_interval = np.mean(intervals)
                if mean_interval > 0:
                    cv = np.std(intervals) / mean_interval
                    periodicity.append({"number": num, "cv": cv})
        
        # 排序，CV 越高表示週期性越強
        periodicity = sorted(periodicity, key=lambda x: x["cv"], reverse=True)[:10]
        
        return periodicity

    def analyze_trends(self):
        """分析號碼的趨勢"""
        # 載入數據
        df = self.load_data()
        
        # 只選擇號碼列
        number_cols = [col for col in df.columns if col.startswith('num')]
        
        # 計算每個可能號碼的趨勢
        all_numbers = range(1, 50)  # 假設彩球範圍是1-49
        rising = []
        falling = []
        
        # 定義早期和晚期的窗口大小
        window_size = min(50, len(df) // 2)
        
        for num in all_numbers:
            # 計算早期窗口中號碼的出現頻率
            early_count = 0
            for idx in range(window_size):
                if idx < len(df):
                    drawn_numbers = [df.iloc[idx][col] for col in number_cols if pd.notna(df.iloc[idx][col])]
                    if num in drawn_numbers:
                        early_count += 1
            freq_early = early_count / window_size if window_size > 0 else 0
            
            # 計算晚期窗口中號碼的出現頻率
            late_count = 0
            for idx in range(len(df) - window_size, len(df)):
                if idx >= 0:
                    drawn_numbers = [df.iloc[idx][col] for col in number_cols if pd.notna(df.iloc[idx][col])]
                    if num in drawn_numbers:
                        late_count += 1
            freq_late = late_count / window_size if window_size > 0 else 0
            
            # 計算趨勢 (頻率變化)
            trend = freq_late - freq_early
            
            if trend > 0:
                rising.append({"number": num, "trend": trend})
            else:
                falling.append({"number": num, "trend": -trend})
        
        # 排序
        rising = sorted(rising, key=lambda x: x["trend"], reverse=True)[:5]
        falling = sorted(falling, key=lambda x: x["trend"], reverse=True)[:5]
        
        return {
            "rising": rising,
            "falling": falling
        }