import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from scipy.stats import entropy, ks_2samp
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
import os
import logging

logger = logging.getLogger(__name__)

class LotteryEvaluator:
    def __init__(self, output_dir='results'):
        """
        初始化評估器
        
        參數:
        output_dir: 輸出結果的目錄
        """
        self.output_dir = output_dir
        
        # 確保輸出目錄存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 設置中文字體
        try:
            # 嘗試使用系統中的中文字體
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
            logger.info("成功設置中文字體")
        except Exception as e:
            logger.warning(f"無法設置中文字體，圖表中的中文可能無法正確顯示: {str(e)}")
    
    def calculate_hit_rate(self, predictions, actual_numbers):
        """計算命中率"""
        hit_counts = []
        hit_rates = []
        
        for pred_set in predictions:
            set_hits = []
            set_rates = []
            for numbers in pred_set:
                # 計算命中的號碼數量
                hits = len(set(numbers) & set(actual_numbers))
                set_hits.append(hits)
                
                # 計算命中率
                hit_rate = hits / len(actual_numbers)
                set_rates.append(hit_rate)
            
            # 取每組預測中的最大命中數
            hit_counts.append(max(set_hits))
            hit_rates.append(max(set_rates))
        
        # 計算平均命中率和命中分佈
        avg_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0
        hit_distribution = {i: hit_counts.count(i) for i in range(len(actual_numbers) + 1)}
        hit_count_3_plus = sum(1 for count in hit_counts if count >= 3)
        
        return {
            "avg_hit_rate": avg_hit_rate,
            "hit_count_3_plus": hit_count_3_plus,
            "hit_distribution": hit_distribution,
            "hit_counts": hit_counts,
            "hit_rates": hit_rates
        }
    
    def calculate_distribution_similarity(self, predictions, historical_data):
        """計算預測號碼與歷史數據的分布相似度"""
        # 將所有預測號碼合併為一個列表
        all_predicted_numbers = []
        for pred_set in predictions:
            for numbers in pred_set:
                all_predicted_numbers.extend(numbers)
        
        # 計算預測號碼的分布
        pred_dist = np.zeros(50)  # 假設號碼範圍是1-49
        for num in all_predicted_numbers:
            try:
                num = int(num)  # 確保 num 是整數
                if 1 <= num <= 49:
                    pred_dist[num] += 1
            except (ValueError, TypeError):
                # 處理無法轉換為整數的情況
                continue
        pred_dist = pred_dist[1:] / (sum(pred_dist[1:]) or 1)  # 避免除零
        
        # 計算歷史數據的分布
        hist_dist = np.zeros(50)
        # 從歷史數據中提取號碼
        for _, row in historical_data.iterrows():
            for col in historical_data.columns:
                if col.startswith('num'):
                    try:
                        num = int(row[col]) if pd.notna(row[col]) else None
                        if num is not None and 1 <= num <= 49:
                            hist_dist[num] += 1
                    except (ValueError, TypeError):
                        # 處理無法轉換為整數的情況
                        continue
        hist_dist = hist_dist[1:] / (sum(hist_dist[1:]) or 1)  # 避免除零
        
        # 計算KL散度（越小越相似）
        kl_div = entropy(hist_dist, pred_dist + 1e-10)  # 添加小值避免除零
        
        # 計算Kolmogorov-Smirnov統計量（越小越相似）
        ks_stat, _ = ks_2samp(pred_dist, hist_dist)
        
        # 計算餘弦相似度（越大越相似）
        cosine_sim = np.dot(pred_dist, hist_dist) / (np.linalg.norm(pred_dist) * np.linalg.norm(hist_dist) + 1e-10)
        
        return {
            "kl_divergence": kl_div,
            "ks_statistic": ks_stat,
            "cosine_similarity": cosine_sim
        }
    
    def calculate_diversity_score(self, predictions):
        """計算預測結果的多樣性分數"""
        if not predictions or not predictions[0]:
            return {"intra_set_diversity": 0, "inter_set_diversity": 0}
            
        diversity_scores = []
        
        for pred_set in predictions:
            # 計算預測組內的多樣性
            unique_numbers = set()
            for numbers in pred_set:
                unique_numbers.update(numbers)
            
            # 多樣性分數 = 唯一號碼數 / 總號碼數
            total_numbers = sum(len(numbers) for numbers in pred_set)
            diversity_score = len(unique_numbers) / total_numbers if total_numbers > 0 else 0
            diversity_scores.append(diversity_score)
        
        # 計算預測組間的多樣性
        all_sets = []
        for pred_set in predictions:
            flat_set = set()
            for numbers in pred_set:
                flat_set.update(numbers)
            all_sets.append(flat_set)
        
        inter_set_diversity = 0
        if len(all_sets) > 1:
            # 計算所有組合的Jaccard距離的平均值
            jaccard_distances = []
            for i in range(len(all_sets)):
                for j in range(i+1, len(all_sets)):
                    intersection = len(all_sets[i] & all_sets[j])
                    union = len(all_sets[i] | all_sets[j])
                    jaccard_distance = 1 - (intersection / union) if union > 0 else 0
                    jaccard_distances.append(jaccard_distance)
            
            inter_set_diversity = sum(jaccard_distances) / len(jaccard_distances) if jaccard_distances else 0
        
        return {
            "intra_set_diversity": sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0,
            "inter_set_diversity": inter_set_diversity
        }
    
    def evaluate_predictions(self, predictions, actual_numbers, historical_data):
        """綜合評估預測結果"""
        logger.info("開始評估預測結果...")
        
        hit_results = self.calculate_hit_rate(predictions, actual_numbers)
        logger.info(f"命中率計算完成: 平均命中率 {hit_results['avg_hit_rate']:.4f}")
        
        distribution_similarity = self.calculate_distribution_similarity(predictions, historical_data)
        logger.info(f"分布相似度計算完成: KL散度 {distribution_similarity['kl_divergence']:.4f}")
        
        diversity_score = self.calculate_diversity_score(predictions)
        logger.info(f"多樣性分數計算完成: 組內多樣性 {diversity_score['intra_set_diversity']:.4f}, 組間多樣性 {diversity_score['inter_set_diversity']:.4f}")
        
        # 計算綜合評分
        # 可以根據需要調整各指標的權重
        hit_weight = 0.5
        dist_weight = 0.3
        div_weight = 0.2
        
        # 標準化分數
        hit_score = hit_results["avg_hit_rate"]
        dist_score = 1 / (1 + distribution_similarity["kl_divergence"])  # 轉換為越大越好
        div_score = (diversity_score["intra_set_diversity"] + diversity_score["inter_set_diversity"]) / 2
        
        composite_score = (hit_weight * hit_score +
                           dist_weight * dist_score +
                           div_weight * div_score)
        
        logger.info(f"綜合評分計算完成: {composite_score:.4f}")
        
        return {
            "hit_results": hit_results,
            "distribution_similarity": distribution_similarity,
            "diversity_score": diversity_score,
            "composite_score": composite_score
        }
    
    def plot_hit_distribution(self, predictions, actual_numbers, save=True):
        """繪製命中分佈圖"""
        hit_results = self.calculate_hit_rate(predictions, actual_numbers)
        hit_distribution = hit_results['hit_distribution']
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(hit_distribution.keys()), y=list(hit_distribution.values()))
        plt.title('命中數分佈')
        plt.xlabel('命中數量')
        plt.ylabel('頻率')
        plt.xticks(range(len(actual_numbers) + 1))
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'hit_distribution.png'))
            plt.close()
        else:
            plt.show()
    
    def plot_prediction_vs_actual(self, predictions, actual_numbers, save=True):
        """繪製預測值與實際值的對比圖"""
        # 將預測和實際值轉換為適合繪圖的格式
        all_numbers = set(range(1, 50))  # 假設號碼範圍是1-49
        
        # 計算每個號碼在預測和實際中的出現頻率
        pred_freq = {num: 0 for num in all_numbers}
        for pred_set in predictions:
            for numbers in pred_set:
                for num in numbers:
                    if num in all_numbers:
                        pred_freq[num] = pred_freq.get(num, 0) + 1
        
        actual_freq = {num: 0 for num in all_numbers}
        for num in actual_numbers:
            if num in all_numbers:
                actual_freq[num] = actual_freq.get(num, 0) + 1
        
        # 繪製對比圖
        plt.figure(figsize=(12, 6))
        
        x = list(sorted(all_numbers))
        y_pred = [pred_freq.get(num, 0) for num in x]
        y_actual = [actual_freq.get(num, 0) for num in x]
        
        plt.bar(x, y_pred, alpha=0.5, label='預測頻率')
        plt.bar(x, y_actual, alpha=0.5, label='實際頻率')
        
        plt.title('預測號碼與實際號碼頻率對比')
        plt.xlabel('號碼')
        plt.ylabel('頻率')
        plt.legend()
        plt.grid(True)
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'prediction_vs_actual.png'))
            plt.close()
        else:
            plt.show()
    
    def generate_evaluation_report(self, predictions, actual_numbers, historical_data):
        """生成評估報告"""
        evaluation_results = self.evaluate_predictions(predictions, actual_numbers, historical_data)
        hit_results = evaluation_results['hit_results']
        
        report = {
            'avg_hit_rate': hit_results['avg_hit_rate'],
            'hit_distribution': hit_results['hit_distribution'],
            'hit_count_3_plus': hit_results['hit_count_3_plus'],
            'hit_count_3_plus_percentage': hit_results['hit_count_3_plus'] / len(hit_results['hit_counts']) if hit_results['hit_counts'] else 0,
            'distribution_similarity': evaluation_results['distribution_similarity'],
            'diversity_score': evaluation_results['diversity_score'],
            'composite_score': evaluation_results['composite_score']
        }
        
        # 保存報告
        report_df = pd.DataFrame([{
            'avg_hit_rate': report['avg_hit_rate'],
            'hit_count_3_plus_percentage': report['hit_count_3_plus_percentage'],
            'kl_divergence': report['distribution_similarity']['kl_divergence'],
            'ks_statistic': report['distribution_similarity']['ks_statistic'],
            'cosine_similarity': report['distribution_similarity']['cosine_similarity'],
            'intra_set_diversity': report['diversity_score']['intra_set_diversity'],
            'inter_set_diversity': report['diversity_score']['inter_set_diversity'],
            'composite_score': report['composite_score']
        }])
        
        report_df.to_csv(os.path.join(self.output_dir, 'evaluation_report.csv'), index=False)
        
        # 繪製圖表
        self.plot_hit_distribution(predictions, actual_numbers)
        self.plot_prediction_vs_actual(predictions, actual_numbers)
        
        logger.info(f"評估報告已生成並保存至 {self.output_dir}")
        
        return report