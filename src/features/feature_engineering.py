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
        feature_df_scaled = pd.DataFrame(
            self.scaler.fit_transform(feature_df),
            index=feature_df.index,
            columns=feature_df.columns
        )
        
        # 添加高級特徵
        if self.features is not None:
            # 添加時間特徵
            if 'date' in self.features.columns:
                date_features = self.features[['date']].copy()
                date_features['year'] = date_features['date'].dt.year
                date_features['month'] = date_features['date'].dt.month
                date_features['day_of_week'] = date_features['date'].dt.dayofweek
                
                # 將時間特徵與號碼特徵結合
                for num in range(1, 50):
                    if num in feature_df_scaled.index:
                        for date_col in ['year', 'month', 'day_of_week']:
                            feature_df_scaled.loc[num, f'{date_col}_effect'] = date_features[date_col].corr(
                                self.features.apply(lambda row: 1 if num in row.values else 0, axis=1)
                            )
        
        end_time = time.time()
        logger.info(f"特徵提取完成，耗時: {end_time - start_time:.2f} 秒")
        
        return feature_df_scaled
    
    def create_advanced_features(self):
        """創建進階特徵"""
        if self.data is None:
            logger.error("未載入數據，無法創建特徵")
            return None
            
        df = self.features.copy() if self.features is not None else self.create_basic_features()
        logger.info("開始創建進階特徵...")
        
        # 創建所有進階特徵的字典
        features_dict = {}
        
        # 獲取數字列
        num_cols = [col for col in df.columns if col.startswith('num') and len(col) <= 4]  # 排除衍生特徵
        
        # 歷史統計特徵
        for col in num_cols:
            # 計算每個數字的歷史出現頻率
            for window in [5, 10, 20]:
                if len(df) > window:
                    features_dict[f'{col}_freq_{window}'] = df[col].rolling(window=window).apply(
                        lambda x: (x == df[col].iloc[-1]).mean() if len(x) > 0 and not pd.isna(df[col].iloc[-1]) else 0,
                        raw=True
                    ).fillna(0)
        
        # 計算每個數字與前N期的差異
        for col in num_cols:
            for lag in [1, 2, 3]:
                if len(df) > lag:
                    features_dict[f'{col}_lag_{lag}'] = df[col].shift(lag).fillna(-1).astype(int)
                    features_dict[f'{col}_diff_{lag}'] = df[col] - features_dict[f'{col}_lag_{lag}']
        
        # 數字區間特徵
        for col in num_cols:
            features_dict[f'{col}_range'] = pd.cut(
                df[col], 
                bins=[0, 10, 20, 30, 40, 50], 
                labels=[1, 2, 3, 4, 5]
            ).astype(int)
        
        # 自相關特徵
        for col in num_cols:
            # 計算自相關係數
            try:
                if len(df) > 10:  # 確保有足夠的數據
                    acf_values = acf(df[col].dropna(), nlags=min(5, len(df)-1))
                    for lag in range(1, min(6, len(acf_values))):
                        features_dict[f'{col}_acf_{lag}'] = np.full(len(df), acf_values[lag])
            except Exception as e:
                logger.warning(f"計算自相關係數時出錯: {str(e)}")
                # 如果計算失敗，填充0
                for lag in range(1, 6):
                    features_dict[f'{col}_acf_{lag}'] = np.full(len(df), 0)
        
        # 數字組合模式
        odd_count_cols = [f'{col}_is_odd' for col in num_cols]
        prime_count_cols = [f'{col}_is_prime' for col in num_cols]
        
        if all(col in df.columns for col in odd_count_cols):
            features_dict['odd_count'] = df[odd_count_cols].sum(axis=1)
        
        if all(col in df.columns for col in prime_count_cols):
            features_dict['prime_count'] = df[prime_count_cols].sum(axis=1)
        
        # 數字間隔特徵
        for i, col1 in enumerate(num_cols):
            for col2 in num_cols[i+1:]:
                features_dict[f'diff_{col1}_{col2}'] = df[col2] - df[col1]
        
        # 熱門數字特徵
        for col in num_cols:
            window_size = min(100, len(df))
            if window_size > 0:
                features_dict[f'{col}_popularity'] = df[col].rolling(window=window_size).apply(
                    lambda x: pd.Series(x).value_counts().get(df[col].iloc[-1], 0) if not pd.isna(df[col].iloc[-1]) else 0,
                    raw=True
                ).fillna(0)
        
        # 一次性將所有特徵添加到DataFrame
        new_features = pd.DataFrame(features_dict, index=df.index)
        df = pd.concat([df, new_features], axis=1)
        
        self.features = df
        logger.info(f"進階特徵創建完成，共 {len(new_features.columns)} 個特徵")
        return df
    
    def create_complex_features(self):
        """創建複雜特徵"""
        if self.data is None:
            logger.error("未載入數據，無法創建特徵")
            return None
            
        df = self.features.copy() if self.features is not None else self.create_advanced_features()
        logger.info("開始創建複雜特徵...")
        
        # 創建所有複雜特徵的字典
        features_dict = {}
        
        # 獲取數字列
        num_cols = [col for col in df.columns if col.startswith('num') and len(col) <= 4]  # 排除衍生特徵
        
        # 數字組合熵
        features_dict['number_entropy'] = df.apply(
            lambda x: self._calculate_entropy([x[col] for col in num_cols if col in x]), 
            axis=1
        )
        
        # 數字序列模式
        features_dict['sequence_pattern'] = df.apply(
            lambda x: self._get_sequence_pattern([x[col] for col in num_cols if col in x]), 
            axis=1
        )
        
        # 數字分佈特徵
        features_dict['distribution_score'] = df.apply(
            lambda x: self._calculate_distribution_score([x[col] for col in num_cols if col in x]), 
            axis=1
        )
        
        # 數字重複模式
        for window in [10, 20]:
            window = min(window, len(df) - 1)  # 確保窗口大小不超過數據長度
            if window > 0:
                for col in num_cols:
                    # 計算過去window期內，當前數字重複出現的次數
                    features_dict[f'{col}_repeat_in_{window}'] = df[col].rolling(window=window).apply(
                        lambda x: (x == x[-1]).sum() - 1 if len(x) > 0 and not pd.isna(x[-1]) else 0,
                        raw=True
                    ).fillna(0)
        
        # 歷史趨勢特徵
        for col in num_cols:
            # 計算過去10期的趨勢斜率，確保窗口大小合適
            window_size = min(10, len(df) - 1)
            if window_size > 1:  # 至少需要2個點來計算趨勢
                try:
                    features_dict[f'{col}_trend_{window_size}'] = df[col].rolling(window=window_size).apply(
                        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
                        raw=True
                    ).fillna(0)
                except Exception as e:
                    logger.warning(f"計算趨勢特徵時出錯: {str(e)}")
                    # 如果計算失敗，填充0
                    features_dict[f'{col}_trend_{window_size}'] = np.zeros(len(df))
        
        # 數字組合的複雜性
        features_dict['complexity_score'] = df.apply(
            lambda x: self._calculate_complexity([x[col] for col in num_cols if col in x]), 
            axis=1
        )
        
        # 一次性將所有特徵添加到DataFrame
        new_features = pd.DataFrame(features_dict, index=df.index)
        df = pd.concat([df, new_features], axis=1)
        
        self.features = df
        logger.info(f"複雜特徵創建完成，共 {len(new_features.columns)} 個特徵")
        return df
    
    def get_training_data(self, target_cols=None, lag_periods=1):
        """準備訓練數據，使用歷史數據預測未來"""
        if self.data is None:
            logger.error("未載入數據，無法準備訓練數據")
            return None, None
            
        if self.features is None:
            self.create_complex_features()
        
        logger.info("開始準備訓練數據...")
        df = self.features.copy()
        
        if target_cols is None:
            # 獲取所有數字列作為目標
            target_cols = [col for col in df.columns if col.startswith('num') and len(col) <= 4]
        
        # 創建目標變數（下一期的號碼）
        target_dict = {}
        for col in target_cols:
            target_dict[f'next_{col}'] = df[col].shift(-lag_periods)
        
        # 一次性添加所有目標變數
        df = pd.concat([df, pd.DataFrame(target_dict, index=df.index)], axis=1)
        
        # 移除包含NaN的行
        df = df.dropna()
        
        # 分離特徵和目標
        X = df.drop(columns=['date'] + [f'next_{col}' for col in target_cols] + target_cols)
        y = df[[f'next_{col}' for col in target_cols]]
        
        logger.info(f"訓練數據準備完成，特徵數量: {X.shape[1]}, 樣本數量: {X.shape[0]}")
        return X, y
    
    def combine_feature_sets(self):
        """結合所有特徵集，創建完整的特徵集"""
        if self.data is None:
            logger.error("未載入數據，無法創建特徵")
            return None
            
        logger.info("開始結合所有特徵集...")
        
        # 創建基本特徵
        self.create_basic_features()
        
        # 創建進階特徵
        self.create_advanced_features()
        
        # 創建複雜特徵
        self.create_complex_features()
        
        # 提取優化特徵
        optimized_features = self.extract_features()
        
        # 將優化特徵添加到主特徵集
        if isinstance(optimized_features, pd.DataFrame):
            # 為每個號碼添加優化特徵
            num_cols = [col for col in self.features.columns if col.startswith('num') and len(col) <= 4]
            for col in num_cols:
                for feat_col in optimized_features.columns:
                    # 為每個數字列添加對應的優化特徵
                    self.features[f'{col}_{feat_col}'] = self.features[col].map(
                        lambda x: optimized_features.loc[x, feat_col] if x in optimized_features.index else 0
                    )
        
        logger.info(f"特徵結合完成，總特徵數量: {self.features.shape[1]}")
        return self.features