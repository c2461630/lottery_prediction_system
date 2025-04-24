import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import optuna
import joblib
import os
import random
import matplotlib.pyplot as plt
import pickle
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import matplotlib
from datetime import datetime
import json

matplotlib.use('Agg')  # 使用非交互式後端，避免 Tkinter 問題

class LotteryModelTrainer:
    def __init__(self, X, y, model_dir='models'):
        self.X = X
        self.y = y
        self.model_dir = model_dir
        self.models = {}
        self.ensemble_model = None
        self.feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        self.best_params = {}
        
        # 確保模型目錄存在
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def train_test_split(self, test_size=0.2, random_state=42):
        """分割訓練集和測試集"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_random_forest(self, optimize=True):
        """訓練隨機森林模型"""
        if optimize:
            # 使用Optuna優化超參數
            study = optuna.create_study(direction='minimize')
            study.optimize(self._objective_rf, n_trials=50)
            
            # 使用最佳參數訓練模型
            best_params = study.best_params
            self.best_params['random_forest'] = best_params
            model = RandomForestRegressor(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                min_samples_split=best_params['min_samples_split'],
                min_samples_leaf=best_params['min_samples_leaf'],
                random_state=42
            )
        else:
            model = RandomForestRegressor(random_state=42)
        
        model.fit(self.X_train, self.y_train)
        self.models['random_forest'] = model
        
        # 保存模型
        joblib.dump(model, os.path.join(self.model_dir, 'random_forest_model.pkl'))
        
        return model
    
    def train_xgboost(self, optimize=True):
        """訓練XGBoost模型"""
        if optimize:
            # 使用Optuna優化超參數
            study = optuna.create_study(direction='minimize')
            study.optimize(self._objective_xgb, n_trials=50)
            
            # 使用最佳參數訓練模型
            best_params = study.best_params
            self.best_params['xgboost'] = best_params
            model = xgb.XGBRegressor(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                learning_rate=best_params['learning_rate'],
                subsample=best_params['subsample'],
                colsample_bytree=best_params['colsample_bytree'],
                random_state=42
            )
        else:
            model = xgb.XGBRegressor(random_state=42)
        
        # 對每個目標列單獨訓練模型
        models = {}
        for i, col in enumerate(self.y_train.columns):
            model_i = model.fit(self.X_train, self.y_train.iloc[:, i])
            models[col] = model_i
        
        self.models['xgboost'] = models
        
        # 保存模型
        joblib.dump(models, os.path.join(self.model_dir, 'xgboost_model.pkl'))
        
        return models
    
    def train_lightgbm(self, optimize=True):
        """訓練LightGBM模型"""
        if optimize:
            # 使用Optuna優化超參數
            study = optuna.create_study(direction='minimize')
            study.optimize(self._objective_lgb, n_trials=50)
            
            # 使用最佳參數訓練模型
            best_params = study.best_params
            self.best_params['lightgbm'] = best_params
            model = lgb.LGBMRegressor(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                learning_rate=best_params['learning_rate'],
                subsample=best_params['subsample'],
                colsample_bytree=best_params['colsample_bytree'],
                random_state=42
            )
        else:
            model = lgb.LGBMRegressor(random_state=42)
        
        # 對每個目標列單獨訓練模型
        models = {}
        for i, col in enumerate(self.y_train.columns):
            model_i = model.fit(self.X_train, self.y_train.iloc[:, i])
            models[col] = model_i
        
        self.models['lightgbm'] = models
        
        # 保存模型
        joblib.dump(models, os.path.join(self.model_dir, 'lightgbm_model.pkl'))
        
        return models
    
    def train_catboost(self, optimize=True):
        """訓練CatBoost模型"""
        if optimize:
            # 使用Optuna優化超參數
            study = optuna.create_study(direction='minimize')
            study.optimize(self._objective_catboost, n_trials=30)
            
            # 使用最佳參數訓練模型
            best_params = study.best_params
            self.best_params['catboost'] = best_params
            model = CatBoostRegressor(
                iterations=best_params['iterations'],
                depth=best_params['depth'],
                learning_rate=best_params['learning_rate'],
                random_strength=best_params['random_strength'],
                bagging_temperature=best_params['bagging_temperature'],
                random_seed=42,
                verbose=0
            )
        else:
            model = CatBoostRegressor(random_seed=42, verbose=0)
        
        # 對每個目標列單獨訓練模型
        models = {}
        for i, col in enumerate(self.y_train.columns):
            model_i = model.fit(self.X_train, self.y_train.iloc[:, i])
            models[col] = model_i
        
        self.models['catboost'] = models
        
        # 保存模型
        joblib.dump(models, os.path.join(self.model_dir, 'catboost_model.pkl'))
        
        return models
    
    def train_neural_network(self, optimize=True):
        """訓練深度學習模型"""
        if optimize:
            # 使用Optuna優化超參數
            study = optuna.create_study(direction='minimize')
            study.optimize(self._objective_nn, n_trials=30)
            
            # 使用最佳參數訓練模型
            best_params = study.best_params
            self.best_params['neural_network'] = best_params
            models = self._build_nn_models(
                hidden_layers=best_params['hidden_layers'],
                neurons=best_params['neurons'],
                dropout=best_params['dropout'],
                learning_rate=best_params['learning_rate']
            )
        else:
            models = self._build_nn_models()
        
        self.models['neural_network'] = models
        
        # 保存模型
        for i, col in enumerate(self.y_train.columns):
            models[col].save(os.path.join(self.model_dir, f'nn_model_{col}.h5'))
        
        return models
    
    def _build_nn_models(self, hidden_layers=2, neurons=64, dropout=0.2, learning_rate=0.001):
        """構建深度學習模型"""
        models = {}
        
        for i, col in enumerate(self.y_train.columns):
            model = Sequential()
            
            # 輸入層
            model.add(Dense(neurons, activation='relu', input_shape=(self.X_train.shape[1],)))
            model.add(Dropout(dropout))
            
            # 隱藏層
            for _ in range(hidden_layers):
                model.add(Dense(neurons, activation='relu'))
                model.add(Dropout(dropout))
            
            # 輸出層
            model.add(Dense(1))
            
            # 編譯模型
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(loss='mse', optimizer=optimizer)
            
            # 訓練模型
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model.fit(
                self.X_train, self.y_train.iloc[:, i],
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
            
            models[col] = model
        
        return models
    
    def train_ensemble(self):
        """訓練集成模型"""
        # 確保已經訓練了基礎模型
        if not self.models:
            self.train_random_forest()
            self.train_xgboost()
            self.train_lightgbm()
        
        # 對每個目標列創建集成模型
        ensemble_models = {}
        
        for i, col in enumerate(self.y_train.columns):
            # 準備基礎模型
            base_models = []
            
            # 添加XGBoost模型
            if 'xgboost' in self.models:
                base_models.append(('xgboost', self.models['xgboost'][col]))
            
            # 添加LightGBM模型
            if 'lightgbm' in self.models:
                base_models.append(('lightgbm', self.models['lightgbm'][col]))
            
            # 添加CatBoost模型
            if 'catboost' in self.models:
                base_models.append(('catboost', self.models['catboost'][col]))
            
            # 創建集成模型
            ensemble = VotingRegressor(base_models)
            ensemble.fit(self.X_train, self.y_train.iloc[:, i])
            
            ensemble_models[col] = ensemble
        
        self.ensemble_model = ensemble_models
        
        # 保存模型
        joblib.dump(ensemble_models, os.path.join(self.model_dir, 'ensemble_model.pkl'))
        
        return ensemble_models
    
    def evaluate_model(self, model_name, X=None, y=None):
        """評估模型性能"""
        if X is None:
            X = self.X_test
        if y is None:
            y = self.y_test
        
        if model_name == 'ensemble' and self.ensemble_model:
            models = self.ensemble_model
        elif model_name in self.models:
            models = self.models[model_name]
        else:
            raise ValueError(f"Model {model_name} not found. Please train it first.")
        
        results = {}
        
        # 對每個目標列評估模型
        for i, col in enumerate(y.columns):
            if model_name in ['xgboost', 'lightgbm', 'catboost', 'ensemble']:
                y_pred = models[col].predict(X)
            elif model_name == 'neural_network':
                y_pred = models[col].predict(X).flatten()
            else:  # random_forest
                y_pred = models.predict(X)[:, i]
            
            # 計算評估指標
            mse = mean_squared_error(y.iloc[:, i], y_pred)
            mae = mean_absolute_error(y.iloc[:, i], y_pred)
            r2 = r2_score(y.iloc[:, i], y_pred)
            
            results[col] = {
                'mse': mse,
                'mae': mae,
                'r2': r2
            }
        
        return results
    
    def cross_validate(self, model_name, cv=5):
        """使用交叉驗證評估模型"""
        if model_name not in ['random_forest', 'xgboost', 'lightgbm', 'catboost']:
            raise ValueError("Cross-validation is only supported for tree-based models.")
        
        results = {}
        
        # 對每個目標列進行交叉驗證
        for i, col in enumerate(self.y.columns):
            if model_name == 'random_forest':
                model = RandomForestRegressor(random_state=42)
            elif model_name == 'xgboost':
                model = xgb.XGBRegressor(random_state=42)
            elif model_name == 'lightgbm':
                model = lgb.LGBMRegressor(random_state=42)
            elif model_name == 'catboost':
                model = CatBoostRegressor(random_seed=42, verbose=0)
            
            # 執行交叉驗證
            mse_scores = -cross_val_score(model, self.X, self.y.iloc[:, i],
                                         cv=cv, scoring='neg_mean_squared_error')
            mae_scores = -cross_val_score(model, self.X, self.y.iloc[:, i],
                                         cv=cv, scoring='neg_mean_absolute_error')
            r2_scores = cross_val_score(model, self.X, self.y.iloc[:, i],
                                       cv=cv, scoring='r2')
            
            results[col] = {
                'mse_mean': mse_scores.mean(),
                'mse_std': mse_scores.std(),
                'mae_mean': mae_scores.mean(),
                'mae_std': mae_scores.std(),
                'r2_mean': r2_scores.mean(),
                'r2_std': r2_scores.std()
            }
        
        return results
    
    def predict(self, X, model_name='ensemble'):
        """使用模型進行預測"""
        if model_name == 'ensemble' and self.ensemble_model:
            models = self.ensemble_model
        elif model_name in self.models:
            models = self.models[model_name]
        else:
            raise ValueError(f"Model {model_name} not found. Please train it first.")
        
        predictions = {}
        
        # 對每個目標列進行預測
        for i, col in enumerate(self.y.columns.tolist()):
            if model_name in ['xgboost', 'lightgbm', 'catboost', 'ensemble']:
                pred = models[col].predict(X)
            elif model_name == 'neural_network':
                pred = models[col].predict(X).flatten()
            else:  # random_forest
                pred = models.predict(X)[:, i]
            
            predictions[col] = pred
        
        return pd.DataFrame(predictions)
    
    def generate_lottery_numbers(self, X, model_name='ensemble', num_sets=1):
        """生成彩票號碼預測"""
        predictions = self.predict(X, model_name)
        
        # 生成多組預測號碼
        lottery_sets = []
        for _ in range(num_sets):
            # 對每個預測值添加一些隨機性
            noisy_predictions = {}
            for col in predictions.columns:
                # 添加少量噪聲以增加多樣性
                noise = np.random.normal(0, 2, size=len(predictions))
                noisy_predictions[col] = predictions[col] + noise
            
            # 將預測轉換為有效的彩票號碼（假設範圍是1-49）
            lottery_numbers = []
            
            # 重設索引以確保可以使用位置索引
            noisy_preds_df = pd.DataFrame(noisy_predictions).reset_index(drop=True)
            
            for i in range(len(noisy_preds_df)):
                # 獲取一行預測值
                row_preds = [noisy_preds_df[col].iloc[i] for col in noisy_preds_df.columns]
                
                # 將預測值轉換為1-49範圍內的整數
                numbers = [max(1, min(49, int(round(pred)))) for pred in row_preds]
                
                # 確保沒有重複的號碼
                while len(set(numbers)) < len(numbers):
                    # 找出重複的號碼
                    seen = set()
                    duplicates = [x for x in numbers if x in seen or seen.add(x)]
                    
                    # 替換重複的號碼
                    for idx, num in enumerate(numbers):
                        if num in duplicates:
                            # 生成一個不在當前集合中的隨機數
                            new_num = random.randint(1, 49)
                            while new_num in numbers:
                                new_num = random.randint(1, 49)
                            numbers[idx] = new_num
                            break
                
                # 排序號碼
                numbers.sort()
                lottery_numbers.append(numbers)
            
            lottery_sets.append(lottery_numbers)
        
        return lottery_sets
    
    def evaluate_hit_rate(self, predictions, actual_numbers):
        """評估命中率"""
        hit_rates = []
        
        for pred_set in predictions:
            for pred_numbers in pred_set:
                # 計算命中的號碼數量
                hits = sum(1 for num in pred_numbers if num in actual_numbers)
                
                # 計算命中率
                hit_rate = hits / len(actual_numbers)
                hit_rates.append(hit_rate)
        
        # 計算平均命中率
        avg_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0
        
        return {
            'hit_rates': hit_rates,
            'avg_hit_rate': avg_hit_rate,
            'hit_count_3_plus': sum(1 for rate in hit_rates if rate >= 0.6)  # 命中3個或以上 (3/5 = 0.6)
        }
    
    def find_optimal_parameters(self, X_sample, actual_numbers, trials=100):
        """尋找能達到高命中率的最佳參數"""
        best_hit_rate = 0
        best_params = None
        best_predictions = None
        
        for _ in range(trials):
            # 隨機選擇模型
            model_name = random.choice(['ensemble', 'xgboost', 'lightgbm', 'catboost', 'random_forest'])
            
            # 生成預測
            predictions = self.generate_lottery_numbers(X_sample, model_name, num_sets=5)
            
            # 評估命中率
            hit_rate_results = self.evaluate_hit_rate(predictions, actual_numbers)
            
            # 更新最佳參數
            if hit_rate_results['avg_hit_rate'] > best_hit_rate:
                best_hit_rate = hit_rate_results['avg_hit_rate']
                best_params = {
                    'model_name': model_name
                }
                best_predictions = predictions
            
            # 如果命中率達到40%以上，提前結束
            if hit_rate_results['hit_count_3_plus'] > 0:
                print(f"Found parameters with hit rate >= 60% (3+ numbers): {model_name}")
                return {
                    'model_name': model_name,
                    'hit_rate': hit_rate_results['avg_hit_rate'],
                    'predictions': predictions
                }
        
        return {
            'model_name': best_params['model_name'] if best_params else None,
            'hit_rate': best_hit_rate,
            'predictions': best_predictions
        }
    
    def save_best_model_config(self, model_name, hit_rate):
        """保存最佳模型配置"""
        config = {
            'model_name': model_name,
            'hit_rate': hit_rate,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(os.path.join(self.model_dir, 'best_model_config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"最佳模型配置已保存: {model_name}, 命中率: {hit_rate:.2f}%")
    
    # Optuna目標函數
    def _objective_rf(self, trial):
        """Random Forest的超參數優化目標函數"""
        # 定義超參數空間
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
        }
        
        # 創建模型
        model = RandomForestRegressor(**params, random_state=42)
        
        # 計算交叉驗證分數
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in kf.split(self.X_train):
            X_train_fold, X_val_fold = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_train_fold, y_val_fold = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            
            # 計算MSE
            mse = mean_squared_error(y_val_fold, y_pred)
            scores.append(mse)
        
        # 返回平均MSE
        return np.mean(scores)
    
    def _objective_xgb(self, trial):
        """XGBoost的超參數優化目標函數"""
        # 定義超參數空間
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
        }
        
        # 計算交叉驗證分數
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in kf.split(self.X_train):
            X_train_fold, X_val_fold = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_train_fold, y_val_fold = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            
            # 對每個目標列單獨訓練模型
            fold_mse = []
            for i in range(y_train_fold.shape[1]):
                model = xgb.XGBRegressor(**params, random_state=42)
                model.fit(X_train_fold, y_train_fold.iloc[:, i])
                y_pred = model.predict(X_val_fold)
                mse = mean_squared_error(y_val_fold.iloc[:, i], y_pred)
                fold_mse.append(mse)
            
            # 計算平均MSE
            scores.append(np.mean(fold_mse))
        
        # 返回平均MSE
        return np.mean(scores)
    
    def _objective_lgb(self, trial):
        """LightGBM的超參數優化目標函數"""
        # 定義超參數空間
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
        }
        
        # 計算交叉驗證分數
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in kf.split(self.X_train):
            X_train_fold, X_val_fold = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_train_fold, y_val_fold = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            
            # 對每個目標列單獨訓練模型
            fold_mse = []
            for i in range(y_train_fold.shape[1]):
                model = lgb.LGBMRegressor(**params, random_state=42)
                model.fit(X_train_fold, y_train_fold.iloc[:, i])
                y_pred = model.predict(X_val_fold)
                mse = mean_squared_error(y_val_fold.iloc[:, i], y_pred)
                fold_mse.append(mse)
            
            # 計算平均MSE
            scores.append(np.mean(fold_mse))
        
        # 返回平均MSE
        return np.mean(scores)
    
    def _objective_catboost(self, trial):
        """CatBoost的超參數優化目標函數"""
        # 定義超參數空間
        params = {
            'iterations': trial.suggest_int('iterations', 50, 300),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'random_strength': trial.suggest_float('random_strength', 1e-9, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0)
        }
        
        # 計算交叉驗證分數
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in kf.split(self.X_train):
            X_train_fold, X_val_fold = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_train_fold, y_val_fold = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            
            # 對每個目標列單獨訓練模型
            fold_mse = []
            for i in range(y_train_fold.shape[1]):
                model = CatBoostRegressor(**params, random_seed=42, verbose=0)
                model.fit(X_train_fold, y_train_fold.iloc[:, i])
                y_pred = model.predict(X_val_fold)
                mse = mean_squared_error(y_val_fold.iloc[:, i], y_pred)
                fold_mse.append(mse)
            
            # 計算平均MSE
            scores.append(np.mean(fold_mse))
        
        # 返回平均MSE
        return np.mean(scores)
    
    def _objective_nn(self, trial):
        """神經網絡的超參數優化目標函數"""
        # 定義超參數空間
        params = {
            'hidden_layers': trial.suggest_int('hidden_layers', 1, 3),
            'neurons': trial.suggest_int('neurons', 32, 128),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'epochs': trial.suggest_int('epochs', 50, 200)
        }
        
        # 計算交叉驗證分數
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in kf.split(self.X_train):
            X_train_fold, X_val_fold = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_train_fold, y_val_fold = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            
            # 對每個目標列單獨訓練模型
            fold_mse = []
            for i in range(y_train_fold.shape[1]):
                # 創建神經網絡模型
                model = self._create_nn_model(
                    input_dim=X_train_fold.shape[1],
                    hidden_layers=params['hidden_layers'],
                    neurons=params['neurons'],
                    dropout=params['dropout'],
                    learning_rate=params['learning_rate']
                )
                
                # 訓練模型
                model.fit(
                    X_train_fold, y_train_fold.iloc[:, i],
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    verbose=0
                )
                
                # 評估模型
                y_pred = model.predict(X_val_fold)
                mse = mean_squared_error(y_val_fold.iloc[:, i], y_pred)
                fold_mse.append(mse)
            
            # 計算平均MSE
            scores.append(np.mean(fold_mse))
        
        # 返回平均MSE
        return np.mean(scores)
    
    def _create_nn_model(self, input_dim, hidden_layers=2, neurons=64, dropout=0.2, learning_rate=0.001):
        """創建神經網絡模型"""
        model = Sequential()
        
        # 輸入層
        model.add(Dense(neurons, input_dim=input_dim, activation='relu'))
        model.add(Dropout(dropout))
        
        # 隱藏層
        for _ in range(hidden_layers):
            model.add(Dense(neurons, activation='relu'))
            model.add(Dropout(dropout))
        
        # 輸出層
        model.add(Dense(1))
        
        # 編譯模型
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(loss='mse', optimizer=optimizer)
        
        return model
    
    def optimize_hyperparameters(self, model_name, n_trials=100):
        """使用Optuna優化超參數"""
        print(f"開始優化 {model_name} 的超參數...")
        
        # 選擇目標函數
        if model_name == 'random_forest':
            objective = self._objective_rf
        elif model_name == 'xgboost':
            objective = self._objective_xgb
        elif model_name == 'lightgbm':
            objective = self._objective_lgb
        elif model_name == 'catboost':
            objective = self._objective_catboost
        elif model_name == 'neural_network':
            objective = self._objective_nn
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        # 創建Optuna研究
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # 獲取最佳參數
        best_params = study.best_params
        print(f"最佳參數: {best_params}")
        print(f"最佳MSE: {study.best_value}")
        
        # 保存最佳參數
        params_path = os.path.join(self.model_dir, f'{model_name}_best_params.json')
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        
        print(f"最佳參數已保存到: {params_path}")
        
        return best_params
    
    def load_best_params(self, model_name):
        """加載最佳超參數"""
        params_path = os.path.join(self.model_dir, f'{model_name}_best_params.json')
        
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                best_params = json.load(f)
            print(f"已加載 {model_name} 的最佳參數")
            return best_params
        else:
            print(f"找不到 {model_name} 的最佳參數，將使用默認參數")
            return None
    
    def feature_importance(self, model_name='random_forest'):
        """計算特徵重要性"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Please train it first.")
        
        if model_name == 'random_forest':
            # 對於RandomForest，直接獲取特徵重要性
            importances = self.models[model_name].feature_importances_
            feature_names = self.X.columns
            
            # 創建特徵重要性DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })
            
            # 按重要性排序
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            return importance_df
        
        elif model_name in ['xgboost', 'lightgbm', 'catboost']:
            # 對於樹模型，計算每個目標的特徵重要性並取平均
            feature_names = self.X.columns
            importances = np.zeros(len(feature_names))
            
            for col in self.y.columns:
                if model_name == 'xgboost':
                    model_importances = self.models[model_name][col].feature_importances_
                elif model_name == 'lightgbm':
                    model_importances = self.models[model_name][col].feature_importances_
                elif model_name == 'catboost':
                    model_importances = self.models[model_name][col].feature_importances_
                
                importances += model_importances
            
            # 計算平均重要性
            importances /= len(self.y.columns)
            
            # 創建特徵重要性DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })
            
            # 按重要性排序
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            return importance_df
        
        elif model_name == 'neural_network':
            # 對於神經網絡，使用排列重要性
            feature_names = self.X.columns
            importances = np.zeros(len(feature_names))
            
            for i, col in enumerate(self.y.columns):
                # 使用驗證集計算排列重要性
                result = permutation_importance(
                    self.models[model_name][col], self.X_val, self.y_val.iloc[:, i],
                    n_repeats=10, random_state=42
                )
                importances += result.importances_mean
            
            # 計算平均重要性
            importances /= len(self.y.columns)
            
            # 創建特徵重要性DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })
            
            # 按重要性排序
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            return importance_df
        
        else:
            raise ValueError(f"不支持的模型類型: {model_name}")
    
    def plot_feature_importance(self, model_name='random_forest', top_n=10):
        """繪製特徵重要性圖"""
        importance_df = self.feature_importance(model_name)
        
        # 只顯示前N個特徵
        importance_df = importance_df.head(top_n)
        
        # 繪製條形圖
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('重要性')
        plt.ylabel('特徵')
        plt.title(f'{model_name} 模型的特徵重要性 (Top {top_n})')
        plt.tight_layout()
        
        # 保存圖片
        plt.savefig(os.path.join(self.model_dir, f'{model_name}_feature_importance.png'))
        plt.close()
        
        return importance_df
    
    def plot_learning_curves(self, model_name='random_forest'):
        """繪製學習曲線"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Please train it first.")
        
        # 設置交叉驗證
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # 設置訓練集大小
        train_sizes = np.linspace(0.1, 1.0, 5)
        
        plt.figure(figsize=(15, 10))
        
        for i, col in enumerate(self.y.columns):
            plt.subplot(2, 3, i+1)
            
            if model_name == 'random_forest':
                model = self.models[model_name]
                y_col = self.y.iloc[:, i]
            elif model_name in ['xgboost', 'lightgbm', 'catboost', 'neural_network']:
                model = self.models[model_name][col]
                y_col = self.y[col]
            
            # 計算學習曲線
            train_sizes, train_scores, test_scores = learning_curve(
                model, self.X, y_col, cv=cv, train_sizes=train_sizes,
                scoring='neg_mean_squared_error', n_jobs=-1
            )
            
            # 計算平均值和標準差
            train_scores_mean = -np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = -np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            
            # 繪製學習曲線
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1, color="r")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1, color="g")
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="訓練集MSE")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="驗證集MSE")
            
            plt.title(f'目標 {col} 的學習曲線')
            plt.xlabel('訓練樣本數')
            plt.ylabel('MSE')
            plt.legend(loc="best")
            plt.grid()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, f'{model_name}_learning_curves.png'))
        plt.close()

    def plot_prediction_vs_actual(self, X_test, y_test, model_name='ensemble'):
        """繪製預測值與實際值的對比圖"""
        # 獲取預測值
        y_pred = self.predict(X_test, model_name)
        
        plt.figure(figsize=(15, 10))
        
        for i, col in enumerate(y_test.columns):
            plt.subplot(2, 3, i+1)
            
            # 繪製散點圖
            plt.scatter(y_test[col], y_pred[col], alpha=0.5)
            
            # 添加對角線（理想情況）
            min_val = min(y_test[col].min(), y_pred[col].min())
            max_val = max(y_test[col].max(), y_pred[col].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            # 計算評估指標
            mse = mean_squared_error(y_test[col], y_pred[col])
            r2 = r2_score(y_test[col], y_pred[col])
            
            plt.title(f'目標 {col} (MSE: {mse:.2f}, R²: {r2:.2f})')
            plt.xlabel('實際值')
            plt.ylabel('預測值')
            plt.grid()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, f'{model_name}_prediction_vs_actual.png'))
        plt.close()
    
    def plot_residuals(self, X_test, y_test, model_name='ensemble'):
        """繪製殘差圖"""
        # 獲取預測值
        y_pred = self.predict(X_test, model_name)
        
        plt.figure(figsize=(15, 10))
        
        for i, col in enumerate(y_test.columns):
            plt.subplot(2, 3, i+1)
            
            # 計算殘差
            residuals = y_test[col] - y_pred[col]
            
            # 繪製殘差散點圖
            plt.scatter(y_pred[col], residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            
            plt.title(f'目標 {col} 的殘差圖')
            plt.xlabel('預測值')
            plt.ylabel('殘差')
            plt.grid()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, f'{model_name}_residuals.png'))
        plt.close()
    
    def plot_error_distribution(self, X_test, y_test, model_name='ensemble'):
        """繪製誤差分佈圖"""
        # 獲取預測值
        y_pred = self.predict(X_test, model_name)
        
        plt.figure(figsize=(15, 10))
        
        for i, col in enumerate(y_test.columns):
            plt.subplot(2, 3, i+1)
            
            # 計算誤差
            errors = y_test[col] - y_pred[col]
            
            # 繪製誤差直方圖
            plt.hist(errors, bins=20, alpha=0.7)
            
            # 添加正態分佈擬合
            mu, sigma = norm.fit(errors)
            x = np.linspace(min(errors), max(errors), 100)
            p = norm.pdf(x, mu, sigma)
            plt.plot(x, p * len(errors) * (max(errors) - min(errors)) / 20, 'r--', linewidth=2)
            
            plt.title(f'目標 {col} 的誤差分佈 (μ={mu:.2f}, σ={sigma:.2f})')
            plt.xlabel('誤差')
            plt.ylabel('頻率')
            plt.grid()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, f'{model_name}_error_distribution.png'))
        plt.close()
    
    def plot_model_comparison(self, X_test, y_test):
        """比較不同模型的性能"""
        # 獲取所有已訓練的模型
        model_names = [name for name in self.models.keys()]
        if self.ensemble_model:
            model_names.append('ensemble')
        
        # 計算每個模型的MSE和R²
        results = {}
        
        for model_name in model_names:
            y_pred = self.predict(X_test, model_name)
            
            model_mse = []
            model_r2 = []
            
            for col in y_test.columns:
                mse = mean_squared_error(y_test[col], y_pred[col])
                r2 = r2_score(y_test[col], y_pred[col])
                
                model_mse.append(mse)
                model_r2.append(r2)
            
            results[model_name] = {
                'mse_mean': np.mean(model_mse),
                'r2_mean': np.mean(model_r2)
            }
        
        # 創建比較圖
        plt.figure(figsize=(12, 10))
        
        # MSE比較
        plt.subplot(2, 1, 1)
        model_names_sorted = sorted(results.keys(), key=lambda x: results[x]['mse_mean'])
        mse_values = [results[name]['mse_mean'] for name in model_names_sorted]
        
        plt.bar(model_names_sorted, mse_values)
        plt.title('不同模型的平均MSE比較')
        plt.xlabel('模型')
        plt.ylabel('平均MSE')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        
        # R²比較
        plt.subplot(2, 1, 2)
        model_names_sorted = sorted(results.keys(), key=lambda x: -results[x]['r2_mean'])
        r2_values = [results[name]['r2_mean'] for name in model_names_sorted]
        
        plt.bar(model_names_sorted, r2_values)
        plt.title('不同模型的平均R²比較')
        plt.xlabel('模型')
        plt.ylabel('平均R²')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'model_comparison.png'))
        plt.close()
        
        return results
    
    def generate_model_report(self, X_test, y_test):
        """生成模型報告"""
        # 比較不同模型的性能
        model_comparison = self.plot_model_comparison(X_test, y_test)
        
        # 找出最佳模型
        best_model = min(model_comparison.items(), key=lambda x: x[1]['mse_mean'])[0]
        
        # 為最佳模型生成詳細報告
        print(f"生成 {best_model} 模型的詳細報告...")
        
        # 繪製特徵重要性
        if best_model != 'ensemble':
            self.plot_feature_importance(best_model)
        
        # 繪製學習曲線
        if best_model != 'ensemble':
            self.plot_learning_curves(best_model)
        
        # 繪製預測值與實際值的對比圖
        self.plot_prediction_vs_actual(X_test, y_test, best_model)
        
        # 繪製殘差圖
        self.plot_residuals(X_test, y_test, best_model)
        
        # 繪製誤差分佈圖
        self.plot_error_distribution(X_test, y_test, best_model)
        
        # 生成報告文本
        report = f"""
        # 彩票預測模型報告
        
        ## 模型比較
        
        | 模型 | 平均MSE | 平均R² |
        |------|---------|-------|
        """
        
        for model_name, metrics in sorted(model_comparison.items(), key=lambda x: x[1]['mse_mean']):
            report += f"| {model_name} | {metrics['mse_mean']:.4f} | {metrics['r2_mean']:.4f} |\n"
        
        report += f"""
        ## 最佳模型: {best_model}
        
        最佳模型 {best_model} 的平均MSE為 {model_comparison[best_model]['mse_mean']:.4f}，平均R²為 {model_comparison[best_model]['r2_mean']:.4f}。
        
        ## 詳細評估
        
        詳細的評估圖表已保存在模型目錄中，包括：
        - 特徵重要性圖
        - 學習曲線
        - 預測值與實際值對比圖
        - 殘差圖
        - 誤差分佈圖
        
        ## 結論與建議
        
        根據模型評估結果，{best_model} 模型在預測彩票號碼方面表現最佳。然而，彩票抽獎本質上具有隨機性，模型的預測能力有限。
        
        建議：
        1. 將模型預測作為參考，而非確定性指導
        2. 結合歷史統計數據和模型預測進行決策
        3. 定期更新模型，納入最新的抽獎數據
        """
        
        # 保存報告
        with open(os.path.join(self.model_dir, 'model_report.md'), 'w') as f:
            f.write(report)
        
        print(f"模型報告已保存到: {os.path.join(self.model_dir, 'model_report.md')}")
        
        return best_model    