import numpy as np
import random
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)

class DiversityEnhancer:
    def __init__(self, diversity_degree=0.3):
        """初始化多樣性增強器
        
        引數:
            diversity_degree: 多樣性程度 (0-1)
        """
        self.diversity_degree = diversity_degree
    
    def enhance_diversity(self, predictions, num_sets=None, method='hybrid'):
        """增強預測結果的多樣性
        
        引數:
            predictions: 原始預測結果
            num_sets: 需要返回的預測集數量
            method: 多樣性增強方法 ('mutation', 'clustering', 'hybrid')
        
        返回:
            增強多樣性後的預測結果
        """
        logger.info(f"使用 {method} 方法增強預測多樣性，程度: {self.diversity_degree}")
        
        # 檢查預測結果格式並標準化
        formatted_predictions = self._format_predictions(predictions)
        
        # 檢查是否所有預測都相同
        all_same = True
        if formatted_predictions and len(formatted_predictions) > 1:
            first_pred = str(formatted_predictions[0])
            for pred in formatted_predictions[1:]:
                if str(pred) != first_pred:
                    all_same = False
                    break
        
        # 如果所有預測都相同，強制使用更高的多樣性
        temp_diversity_degree = self.diversity_degree
        if all_same:
            temp_diversity_degree = max(0.8, self.diversity_degree * 2)  # 更激進的多樣性增強
            logger.warning(f"檢測到所有預測相同，增加多樣性程度至 {temp_diversity_degree}")
        
        if method == 'mutation':
            enhanced = self._enhance_by_mutation(formatted_predictions, diversity_level=temp_diversity_degree, num_sets=num_sets)
        elif method == 'clustering':
            enhanced = self._enhance_by_clustering(formatted_predictions, diversity_level=temp_diversity_degree)
        elif method == 'hybrid':
            # 先使用聚類，然後對結果進行變異
            clustered = self._enhance_by_clustering(formatted_predictions, diversity_level=temp_diversity_degree)
            enhanced = self._enhance_by_mutation(clustered, diversity_level=temp_diversity_degree, num_sets=num_sets)
        else:
            logger.warning(f"未知的多樣性增強方法: {method}，使用原始預測")
            enhanced = formatted_predictions
        
        # 如果指定了返回集數，則擷取相應數量
        if num_sets is not None and num_sets > 0:
            # 確保有足夠的預測集
            while len(enhanced) < num_sets:
                # 生成隨機預測集
                random_set = []
                for _ in range(len(formatted_predictions[0][0]) if formatted_predictions and formatted_predictions[0] else 6):
                    # 生成1-49之間的不重複隨機數
                    available_numbers = [n for n in range(1, 50) if n not in random_set]
                    if available_numbers:
                        random_set.append(random.choice(available_numbers))
                random_set.sort()
                enhanced.append([random_set])
            
            # 擷取指定數量
            if len(enhanced) > num_sets:
                enhanced = enhanced[:num_sets]
        
        # 最後檢查是否仍然有重複的預測
        if enhanced:
            unique_preds = set()
            for i, pred_set in enumerate(enhanced):
                pred_str = str(pred_set)
                if pred_str in unique_preds:
                    # 如果有重複，生成一個新的變異版本
                    new_pred_set = []
                    for numbers in pred_set:
                        new_numbers = list(numbers)
                        # 變異2-3個號碼
                        mutation_count = random.randint(2, 3)
                        positions = random.sample(range(len(new_numbers)), min(mutation_count, len(new_numbers)))
                        
                        for pos in positions:
                            available_numbers = [n for n in range(1, 50) if n not in new_numbers]
                            if available_numbers:
                                new_numbers[pos] = random.choice(available_numbers)
                        
                        new_pred_set.append(sorted(new_numbers))
                    
                    enhanced[i] = new_pred_set
                else:
                    unique_preds.add(pred_str)
        
        # 最後檢查增強後的預測是否仍然全部相同
        all_same_after = True
        if enhanced and len(enhanced) > 1:
            first_enhanced = str(enhanced[0])
            for e in enhanced[1:]:
                if str(e) != first_enhanced:
                    all_same_after = False
                    break
        
        # 如果仍然全部相同，強制進行更激進的變異
        if all_same_after:
            logger.warning("增強後的預測仍然全部相同，進行強制多樣化")
            forced_diverse = []
            base_pred = enhanced[0]
            
            for i in range(len(enhanced)):
                if i == 0:
                    forced_diverse.append(base_pred)  # 保留一個原始預測
                    continue
                    
                # 為其他預測建立顯著不同的變體
                new_pred = []
                for numbers in base_pred:
                    new_numbers = list(numbers)
                    # 變異至少一半的號碼
                    mutation_count = max(len(new_numbers) // 2, 2)
                    positions = random.sample(range(len(new_numbers)), mutation_count)
                    
                    for pos in positions:
                        available_numbers = [n for n in range(1, 50) if n not in new_numbers]
                        if available_numbers:
                            new_numbers[pos] = random.choice(available_numbers)
                    
                    new_pred.append(sorted(new_numbers))
                
                forced_diverse.append(new_pred)
            
            enhanced = forced_diverse
        
        # 列印增強後的預測結果，用於除錯
        logger.debug(f"增強後的預測結果: {enhanced}")
        
        return enhanced


    def _format_predictions(self, predictions):
        """標準化預測結果格式
        
        將各種格式的預測結果轉換為標準格式：
        [
            [[num1, num2, ...], [num1, num2, ...], ...],  # 第一組預測
            [[num1, num2, ...], [num1, num2, ...], ...],  # 第二組預測
            ...
        ]
        """
        if not predictions:
            return []
        
        # 如果是單個整數，轉換為標準格式
        if isinstance(predictions, int):
            return [[[predictions]]]
        
        # 如果是單個列表且包含整數，轉換為標準格式
        if isinstance(predictions, list) and all(isinstance(x, int) for x in predictions):
            return [[predictions]]
        
        # 如果是二維列表，檢查是否需要進一步轉換
        if isinstance(predictions, list) and all(isinstance(x, list) for x in predictions):
            # 檢查第二層是否包含整數
            if all(all(isinstance(y, int) for y in x) for x in predictions):
                return [predictions]
            
            # 檢查第二層是否包含列表
            if all(all(isinstance(y, list) for y in x) for x in predictions if x):
                return predictions
            
            # 混合情況，進行標準化
            formatted = []
            for pred_set in predictions:
                if not pred_set:
                    continue
                
                if all(isinstance(x, int) for x in pred_set):
                    formatted.append([pred_set])
                elif all(isinstance(x, list) for x in pred_set):
                    formatted.append(pred_set)
                else:
                    # 嘗試轉換混合型別
                    new_set = []
                    for item in pred_set:
                        if isinstance(item, int):
                            new_set.append([item])
                        elif isinstance(item, list):
                            new_set.append(item)
                        else:
                            try:
                                new_set.append(list(item))
                            except:
                                new_set.append([item])
                    formatted.append(new_set)
            
            return formatted
        
        # 其他情況，嘗試最佳轉換
        try:
            # 嘗試轉換為列表
            pred_list = list(predictions)
            return self._format_predictions(pred_list)
        except:
            logger.warning(f"無法處理的預測格式: {type(predictions)}")
            return []
    
    def _enhance_by_mutation(self, predictions, diversity_level=0.2, num_sets=None):
        """透過變異增強多樣性"""
        if not predictions:
            return []
        
        enhanced_predictions = []
        
        # 如果指定了 num_sets，則生成指定數量的預測組
        target_sets = num_sets if num_sets is not None else len(predictions)
        
        # 確保我們有足夠的預測組作為基礎
        base_predictions = predictions
        while len(base_predictions) < target_sets:
            base_predictions.extend(predictions[:target_sets-len(base_predictions)])
        
        for pred_set in base_predictions[:target_sets]:
            enhanced_set = []
            
            for numbers in pred_set:
                # 確保 numbers 是列表
                if not isinstance(numbers, list):
                    try:
                        numbers = list(numbers)
                    except:
                        numbers = [numbers]
                
                # 複製原始號碼
                new_numbers = list(numbers)
                
                # 根據多樣性程度決定變異的號碼數量
                # 增加變異數量，確保至少有1-2個號碼變異
                mutation_count = max(2, int(len(numbers) * diversity_level))  # 確保至少變異2個號碼
                
                if mutation_count > 0:
                    # 隨機選擇要變異的位置
                    positions = random.sample(range(len(numbers)), min(mutation_count, len(numbers)))
                    
                    # 變異選定的號碼
                    for pos in positions:
                        # 生成一個不在當前組閤中的新號碼
                        available_numbers = [n for n in range(1, 50) if n not in new_numbers]
                        if available_numbers:
                            new_numbers[pos] = random.choice(available_numbers)
                
                # 確保號碼是排序的
                enhanced_set.append(sorted(new_numbers))
            
            enhanced_predictions.append(enhanced_set)
        
        return enhanced_predictions
    
    def _enhance_by_clustering(self, predictions, diversity_level=0.2, num_sets=None):
        """透過聚類增強多樣性"""
        if not predictions:
            return []
        
        # 檢查是否所有預測都相同
        all_same = True
        if len(predictions) > 1:
            first_pred = str(predictions[0])
            for pred in predictions[1:]:
                if str(pred) != first_pred:
                    all_same = False
                    break
        
        # 如果所有預測都相同或只有一組預測，直接使用變異法
        if all_same or len(predictions) == 1:
            return self._enhance_by_mutation(predictions, diversity_level=max(0.5, diversity_level*2), num_sets=num_sets)
        
        # 將所有預測展平為特徵向量
        all_numbers = []
        for pred_set in predictions:
            for numbers in pred_set:
                # 確保 numbers 是列表
                if not isinstance(numbers, list):
                    try:
                        numbers = list(numbers)
                    except:
                        numbers = [numbers]
                
                # 建立一個49維的向量，表示每個號碼是否被選中
                vector = np.zeros(49)
                for num in numbers:
                    try:
                        num_int = int(num)
                        if 1 <= num_int <= 49:
                            vector[num_int-1] = 1
                    except (ValueError, TypeError):
                        # 忽略無法轉換為整數的值
                        continue
                
                all_numbers.append((numbers, vector))
        
        if len(all_numbers) <= 1:
            # 如果只有一組預測，直接生成多個變異版本
            base_numbers = all_numbers[0][0]
            enhanced_predictions = []
            
            for _ in range(len(predictions)):
                pred_set = []
                for _ in range(len(predictions[0])):
                    # 建立變異版本
                    new_numbers = list(base_numbers)
                    # 隨機變異2-3個號碼
                    mutation_count = random.randint(2, 3)
                    positions = random.sample(range(len(new_numbers)), mutation_count)
                    
                    for pos in positions:
                        available_numbers = [n for n in range(1, 50) if n not in new_numbers]
                        if available_numbers:
                            new_numbers[pos] = random.choice(available_numbers)
                    
                    pred_set.append(sorted(new_numbers))
                
                enhanced_predictions.append(pred_set)
            
            return enhanced_predictions
        
        # 提取特徵向量
        vectors = np.array([v for _, v in all_numbers])
        
        # 使用K-means聚類，但減少聚類數量以避免警告
        n_clusters = min(3, len(vectors))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(vectors)
        
        # 從每個聚類中選擇代表
        cluster_representatives = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_representatives:
                cluster_representatives[cluster_id] = all_numbers[i][0]
        
        # 構建新的預測結果
        enhanced_predictions = []
        representatives = list(cluster_representatives.values())
        
        # 確保我們有足夠的代表
        while len(representatives) < len(predictions) * len(predictions[0]):
            # 新增一些隨機變異的代表
            for rep in list(representatives):  # 使用副本避免無限迴圈
                if len(representatives) >= len(predictions) * len(predictions[0]):
                    break
                
                # 建立變異版本
                new_rep = list(rep)
                # 變異多個位置以增加多樣性
                mutation_count = random.randint(2, 3)
                positions = random.sample(range(len(new_rep)), min(mutation_count, len(new_rep)))
                
                for pos in positions:
                    available_numbers = [n for n in range(1, 50) if n not in new_rep]
                    if available_numbers:
                        new_rep[pos] = random.choice(available_numbers)
                
                representatives.append(sorted(new_rep))
        
        # 重新組織為原始預測的格式
        for i in range(len(predictions)):
            pred_set = []
            for j in range(len(predictions[i])):
                idx = i * len(predictions[i]) + j
                if idx < len(representatives):
                    pred_set.append(representatives[idx])
                else:
                    # 如果沒有足夠的代表，使用原始預測
                    pred_set.append(predictions[i][j])
            enhanced_predictions.append(pred_set)
        
        return enhanced_predictions
    
    def _enhance_by_frequency(self, predictions):
        """基於頻率的多樣性增強"""
        if not predictions:
            return []
        
        # 計算所有號碼的出現頻率
        frequency = np.zeros(50)
        for pred_set in predictions:
            for numbers in pred_set:
                # 確保 numbers 是列表
                if not isinstance(numbers, list):
                    try:
                        numbers = list(numbers)
                    except:
                        numbers = [numbers]
                
                for num in numbers:
                    try:
                        num_int = int(num)
                        if 1 <= num_int <= 49:
                            frequency[num_int] += 1
                    except (ValueError, TypeError):
                        # 忽略無法轉換為整數的值
                        continue
        
        # 標準化頻率
        if np.sum(frequency) > 0:
            frequency = frequency / np.sum(frequency)
        
        enhanced_predictions = []
        
        for pred_set in predictions:
            enhanced_set = []
            
            for numbers in pred_set:
                # 確保 numbers 是列表
                if not isinstance(numbers, list):
                    try:
                        numbers = list(numbers)
                    except:
                        numbers = [numbers]
                
                # 複製原始號碼
                new_numbers = list(numbers)
                
                # 根據多樣性程度決定變異的號碼數量
                mutation_count = int(len(numbers) * self.diversity_degree)
                
                if mutation_count > 0:
                    # 選擇出現頻率最高的號碼進行變異
                    numbers_freq = []
                    for num in new_numbers:
                        try:
                            num_int = int(num)
                            if 1 <= num_int <= 49:
                                numbers_freq.append((num_int, frequency[num_int]))
                        except (ValueError, TypeError):
                            # 忽略無法轉換為整數的值
                            continue
                    
                    numbers_freq.sort(key=lambda x: x[1], reverse=True)
                    
                    # 變異選定的號碼
                    for i in range(min(mutation_count, len(numbers_freq))):
                        num_to_replace = numbers_freq[i][0]
                        pos = new_numbers.index(num_to_replace)
                        
                        # 選擇出現頻率最低的號碼作為替換
                        available_numbers = []
                        for n in range(1, 50):
                            if n not in new_numbers:
                                available_numbers.append((n, frequency[n]))
                        
                        if available_numbers:
                            available_numbers.sort(key=lambda x: x[1])
                            new_numbers[pos] = available_numbers[0][0]
                
                # 確保號碼是排序的
                enhanced_set.append(sorted(new_numbers))
            
            enhanced_predictions.append(enhanced_set)
        
        return enhanced_predictions
    
    def _enhance_by_distance(self, predictions):
        """基於距離的多樣性增強"""
        if not predictions:
            return []
        
        # 將所有預測展平為特徵向量
        all_numbers = []
        for pred_set in predictions:
            for numbers in pred_set:
                # 確保 numbers 是列表
                if not isinstance(numbers, list):
                    try:
                        numbers = list(numbers)
                    except:
                        numbers = [numbers]
                
                # 建立一個49維的向量，表示每個號碼是否被選中
                vector = np.zeros(49)
                for num in numbers:
                    try:
                        num_int = int(num)
                        if 1 <= num_int <= 49:
                            vector[num_int-1] = 1
                    except (ValueError, TypeError):
                        # 忽略無法轉換為整數的值
                        continue
                
                all_numbers.append((numbers, vector))
        
        if len(all_numbers) <= 1:
            return predictions
        
        # 提取特徵向量
        vectors = np.array([v for _, v in all_numbers])
        
        # 選擇距離最大的向量組合
        max_distance_indices = []
        
        # 貪心演算法：每次選擇與已選向量距離和最大的向量
        if len(vectors) > 0:
            max_distance_indices.append(0)  # 從第一個向量開始
            
            while len(max_distance_indices) < len(vectors):
                max_dist = -1
                max_idx = -1
                
                for i in range(len(vectors)):
                    if i in max_distance_indices:
                        continue
                    
                    # 計算與已選向量的距離和
                    dist_sum = 0
                    for j in max_distance_indices:
                        dist_sum += np.sum((vectors[i] - vectors[j]) ** 2)
                    
                    if dist_sum > max_dist:
                        max_dist = dist_sum
                        max_idx = i
                
                if max_idx != -1:
                    max_distance_indices.append(max_idx)
        
        # 構建新的預測結果
        enhanced_predictions = []
        selected_numbers = [all_numbers[i][0] for i in max_distance_indices]
        
        # 重新組織為原始預測的格式
        for i in range(len(predictions)):
            pred_set = []
            for j in range(len(predictions[i])):
                idx = i * len(predictions[i]) + j
                if idx < len(selected_numbers):
                    pred_set.append(selected_numbers[idx])
                else:
                    # 如果沒有足夠的選擇，使用原始預測
                    pred_set.append(predictions[i][j])
            enhanced_predictions.append(pred_set)
        
        return enhanced_predictions
    
    def calculate_diversity_score(self, predictions):
        """計算預測結果的多樣性分數"""
        if not predictions or len(predictions) <= 1:
            return 0.0
        
        # 將預測結果轉換為標準格式
        formatted_predictions = self._format_predictions(predictions)
        
        # 計算組內多樣性
        intra_set_diversity = self.calculate_intra_set_diversity(formatted_predictions)
        
        # 計算組間多樣性
        inter_set_diversity = self.calculate_inter_set_diversity(formatted_predictions)
        
        # 綜合分數 (加權平均)
        return 0.4 * intra_set_diversity + 0.6 * inter_set_diversity
    
    def calculate_intra_set_diversity(self, predictions):
        """計算組內多樣性"""
        if not predictions:
            return 0.0
        
        diversity_scores = []
        
        for pred_set in predictions:
            if len(pred_set) <= 1:
                continue
            
            # 計算組內每對號碼之間的差異
            total_diff = 0
            count = 0
            
            for i in range(len(pred_set)):
                for j in range(i+1, len(pred_set)):
                    # 計算兩組號碼的差異
                    set1 = set(pred_set[i])
                    set2 = set(pred_set[j])
                    
                    # Jaccard距離 = 1 - (交集大小 / 並集大小)
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    
                    if union > 0:
                        jaccard_dist = 1 - (intersection / union)
                        total_diff += jaccard_dist
                        count += 1
            
            if count > 0:
                diversity_scores.append(total_diff / count)
        
        # 返回平均組內多樣性
        if diversity_scores:
            return sum(diversity_scores) / len(diversity_scores)
        else:
            return 0.0
    
    def calculate_inter_set_diversity(self, predictions):
        """計算組間多樣性"""
        if len(predictions) <= 1:
            return 0.0
        
        # 計算每組預測的平均號碼
        avg_numbers = []
        
        for pred_set in predictions:
            # 將每組預測中的所有號碼合併
            all_nums = []
            for numbers in pred_set:
                all_nums.extend(numbers)
            
            # 計算平均值
            if all_nums:
                avg_numbers.append(sum(all_nums) / len(all_nums))
            else:
                avg_numbers.append(0)
        
        # 計算平均號碼之間的標準差
        if avg_numbers:
            return np.std(avg_numbers)
        else:
            return 0.0