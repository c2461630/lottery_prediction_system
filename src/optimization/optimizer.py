import json
import random
import numpy as np
import logging
from sklearn.model_selection import ParameterGrid
from pathlib import Path

logger = logging.getLogger(__name__)

class LotteryOptimizer:
    def __init__(self, model_trainer, evaluator):
        self.model_trainer = model_trainer
        self.evaluator = evaluator
        self.optimal_params_file = Path("models/optimal_lottery_params.json")
        self.optimal_params = self._load_optimal_params()
        
    def _load_optimal_params(self):
        """載入最佳參數"""
        if self.optimal_params_file.exists():
            with open(self.optimal_params_file, 'r') as f:
                return json.load(f)
        return {"model_name": "ensemble", "hit_rate": 0, "params": {}}
    
    def _save_optimal_params(self, params):
        """保存最佳參數"""
        self.optimal_params_file.parent.mkdir(exist_ok=True)
        with open(self.optimal_params_file, 'w') as f:
            json.dump(params, f)
        print(f"參數已保存到: {self.optimal_params_file}")
    
    def get_param_space(self):
        """獲取參數搜索空間"""
        # 為每個模型定義特定的超參數空間
        param_space = {
            "random_forest": {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "bootstrap": [True, False]
            },
            "xgboost": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7, 9],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "gamma": [0, 0.1, 0.2]
            },
            "lightgbm": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7, 9],
                "num_leaves": [31, 63, 127],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0]
            },
            "catboost": {
                "iterations": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "depth": [4, 6, 8, 10],
                "l2_leaf_reg": [1, 3, 5, 7],
                "random_strength": [0.1, 1, 10]
            },
            "ensemble": {
                "weights": [
                    [0.25, 0.25, 0.25, 0.25],  # 平均權重
                    [0.4, 0.3, 0.2, 0.1],      # 偏重第一個模型
                    [0.1, 0.2, 0.3, 0.4],      # 偏重最後一個模型
                    [0.4, 0.4, 0.1, 0.1],      # 偏重前兩個模型
                    [0.1, 0.1, 0.4, 0.4]       # 偏重後兩個模型
                ],
                "voting_method": ["soft", "hard"]
            }
        }
        return param_space
    
    def optimize_parameters(self, X, y, actual_numbers, trials=100):
        """優化參數"""
        logger.info(f"使用 {trials} 次試驗尋找最佳參數")
        
        param_space = self.get_param_space()
        models = list(param_space.keys())
        
        best_hit_rate = 0
        best_params = None
        best_model_name = None
        best_predictions = None
        
        # 追蹤進度
        progress_interval = max(1, trials // 10)
        
        for trial in range(1, trials + 1):
            # 隨機選擇一個模型
            model_name = random.choice(models)
            
            # 為選定的模型隨機選擇參數
            model_param_space = param_space[model_name]
            
            # 使用ParameterGrid生成所有可能的參數組合
            param_grid = list(ParameterGrid(model_param_space))
            
            # 隨機選擇一組參數
            params = random.choice(param_grid)
            
            # 訓練模型
            self.model_trainer.train_model(model_name, X, y, params)
            
            # 生成預測
            latest_data = self.model_trainer.prepare_latest_data()
            predictions = self.model_trainer.generate_predictions(model_name, latest_data, num_sets=5)
            
            # 評估預測
            hit_results = self.evaluator.calculate_hit_rate(predictions, actual_numbers)
            hit_rate = hit_results["avg_hit_rate"]
            
            # 更新最佳參數
            if hit_rate > best_hit_rate:
                best_hit_rate = hit_rate
                best_params = params
                best_model_name = model_name
                best_predictions = predictions
                
                # 保存最佳參數
                self.optimal_params = {
                    "model_name": best_model_name,
                    "hit_rate": best_hit_rate,
                    "params": best_params,
                    "predictions": [[list(row) for row in pred_set] for pred_set in best_predictions]
                }
                self._save_optimal_params(self.optimal_params)
                
                print(f"更新最佳彩票預測參數 - 模型: {best_model_name}, 命中率: {best_hit_rate*100:.2f}%, 試驗: {trial}/{trials}")
            
            # 顯示進度
            if trial % progress_interval == 0:
                print(f"已完成 {trial}/{trials} 次試驗，當前最佳命中率: {best_hit_rate*100:.2f}%")
        
        print(f"優化完成，最佳模型: {best_model_name}, 命中率: {best_hit_rate*100:.2f}%")
        print(f"最終最佳參數已保存到: {self.optimal_params_file}")
        
        return self.optimal_params