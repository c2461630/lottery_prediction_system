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
from tensorflow.keras.optimizers import Adam
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
    
    def generate_lottery_numbers(self, X_new, model_name='ensemble', num_sets=5):
        """生成彩票號碼預測"""
        if model_name == 'ensemble' and self.ensemble_model:
            models = self.ensemble_model
        elif model_name in self.models:
            models = self.models[model_name]
        else:
            raise ValueError(f"Model {model_name} not found. Please train it first.")
        
        predictions = []
        
        for _ in range(num_sets):
            # 對每個目標列進行預測
            numbers = []
            
            for i, col in enumerate(self.y.columns):
                if model_name in ['xgboost', 'lightgbm', 'catboost', 'ensemble']:
                    pred = models[col].predict(X_new)[0]
                elif model_name == 'neural_network':
                    pred = models[col].predict(X_new)[0][0]
                else:  # random_forest
                    pred = models.predict(X_new)[0][i]
                
                # 將預測值轉換為1-49之間的整數
                num = max(1, min(49, int(round(pred))))
                
                # 確保不重複
                while num in numbers:
                    num = max(1, min(49, num + 1))
                
                numbers.append(num)
            
            # 排序號碼
            numbers.sort()
            predictions.append(numbers)
        
        return predictions
    
    def _objective_rf(self, trial):
        """隨機森林模型的目標函數"""
        # 定義超參數空間
        n_estimators = trial.suggest_int('n_estimators', 100, 500)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        
        # 創建模型
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        
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
        """XGBoost模型的目標函數"""
        # 定義超參數空間
        n_estimators = trial.suggest_int('n_estimators', 100, 500)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        subsample = trial.suggest_float('subsample', 0.6, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
        
        # 創建模型
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42
        )
        
        # 計算交叉驗證分數
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in kf.split(self.X_train):
            X_train_fold, X_val_fold = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_train_fold, y_val_fold = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            
            # 對每個目標列單獨訓練模型
            mse_fold = 0
            for i in range(y_train_fold.shape[1]):
                model.fit(X_train_fold, y_train_fold.iloc[:, i])
                y_pred = model.predict(X_val_fold)
                mse_fold += mean_squared_error(y_val_fold.iloc[:, i], y_pred)
            
            # 計算平均MSE
            mse_fold /= y_train_fold.shape[1]
            scores.append(mse_fold)
        
        # 返回平均MSE
        return np.mean(scores)
    
    def _objective_lgb(self, trial):
        """LightGBM模型的目標函數"""
        # 定義超參數空間
        n_estimators = trial.suggest_int('n_estimators', 100, 500)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        subsample = trial.suggest_float('subsample', 0.6, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
        
        # 創建模型
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42
        )
        
        # 計算交叉驗證分數
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in kf.split(self.X_train):
            X_train_fold, X_val_fold = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_train_fold, y_val_fold = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            
            # 對每個目標列單獨訓練模型
            mse_fold = 0
            for i in range(y_train_fold.shape[1]):
                model.fit(X_train_fold, y_train_fold.iloc[:, i])
                y_pred = model.predict(X_val_fold)
                mse_fold += mean_squared_error(y_val_fold.iloc[:, i], y_pred)
            
            # 計算平均MSE
            mse_fold /= y_train_fold.shape[1]
            scores.append(mse_fold)
        
        # 返回平均MSE
        return np.mean(scores)
    
    def _objective_catboost(self, trial):
        """CatBoost模型的目標函數"""
        # 定義超參數空間
        iterations = trial.suggest_int('iterations', 100, 500)
        depth = trial.suggest_int('depth', 4, 10)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        random_strength = trial.suggest_float('random_strength', 1e-9, 10, log=True)
        bagging_temperature = trial.suggest_float('bagging_temperature', 0, 10)
        
        # 創建模型
        model = CatBoostRegressor(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            random_strength=random_strength,
            bagging_temperature=bagging_temperature,
            random_seed=42,
            verbose=0
        )
        
        # 計算交叉驗證分數
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in kf.split(self.X_train):
            X_train_fold, X_val_fold = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_train_fold, y_val_fold = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            
            # 對每個目標列單獨訓練模型
            mse_fold = 0
            for i in range(y_train_fold.shape[1]):
                model.fit(X_train_fold, y_train_fold.iloc[:, i])
                y_pred = model.predict(X_val_fold)
                mse_fold += mean_squared_error(y_val_fold.iloc[:, i], y_pred)
            
            # 計算平均MSE
            mse_fold /= y_train_fold.shape[1]
            scores.append(mse_fold)
        
        # 返回平均MSE
        return np.mean(scores)
    
    def _objective_nn(self, trial):
        """神經網絡模型的目標函數"""
        # 定義超參數空間
        hidden_layers = trial.suggest_int('hidden_layers', 1, 3)
        neurons = trial.suggest_categorical('neurons', [32, 64, 128])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        
        # 計算交叉驗證分數
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in kf.split(self.X_train):
            X_train_fold, X_val_fold = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_train_fold, y_val_fold = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            
            # 對每個目標列單獨訓練模型
            mse_fold = 0
            for i in range(y_train_fold.shape[1]):
                # 創建模型
                model = Sequential()
                model.add(Dense(neurons, activation='relu', input_shape=(X_train_fold.shape[1],)))
                model.add(Dropout(dropout))
                
                for _ in range(hidden_layers):
                    model.add(Dense(neurons, activation='relu'))
                    model.add(Dropout(dropout))
                
                model.add(Dense(1))
                
                # 編譯模型
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                model.compile(loss='mse', optimizer=optimizer)
                
                # 訓練模型
                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                model.fit(
                    X_train_fold, y_train_fold.iloc[:, i],
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                # 評估模型
                y_pred = model.predict(X_val_fold).flatten()
                mse_fold += mean_squared_error(y_val_fold.iloc[:, i], y_pred)
            
            # 計算平均MSE
            mse_fold /= y_train_fold.shape[1]
            scores.append(mse_fold)
        
        # 返回平均MSE
        return np.mean(scores)
    
    def optimize_hyperparameters(self, model_name='random_forest', n_trials=50):
        """優化超參數"""
        print(f"開始優化 {model_name} 模型的超參數...")
        
        if model_name == 'random_forest':
            study = optuna.create_study(direction='minimize')
            study.optimize(self._objective_rf, n_trials=n_trials)
            best_params = study.best_params
            best_score = study.best_value
            
            print(f"最佳參數: {best_params}")
            print(f"最佳MSE: {best_score}")
            
            # 保存最佳參數
            params_path = os.path.join(self.model_dir, 'random_forest_best_params.json')
            with open(params_path, 'w') as f:
                json.dump(best_params, f)
            print(f"最佳參數已保存到: {params_path}")
            
            return best_params, best_score, 'random_forest'
            
        elif model_name == 'xgboost':
            study = optuna.create_study(direction='minimize')
            study.optimize(self._objective_xgb, n_trials=n_trials)
            best_params = study.best_params
            best_score = study.best_value
            
            print(f"最佳參數: {best_params}")
            print(f"最佳MSE: {best_score}")
            
            # 保存最佳參數
            params_path = os.path.join(self.model_dir, 'xgboost_best_params.json')
            with open(params_path, 'w') as f:
                json.dump(best_params, f)
            print(f"最佳參數已保存到: {params_path}")
            
            return best_params, best_score, 'xgboost'
            
        elif model_name == 'lightgbm':
            study = optuna.create_study(direction='minimize')
            study.optimize(self._objective_lgb, n_trials=n_trials)
            best_params = study.best_params
            best_score = study.best_value
            print(f"最佳參數: {best_params}")
            print(f"最佳MSE: {best_score}")
            
            # 保存最佳參數
            params_path = os.path.join(self.model_dir, 'lightgbm_best_params.json')
            with open(params_path, 'w') as f:
                json.dump(best_params, f)
            print(f"最佳參數已保存到: {params_path}")
            
            return best_params, best_score, 'lightgbm'
            
        elif model_name == 'catboost':
            study = optuna.create_study(direction='minimize')
            study.optimize(self._objective_catboost, n_trials=n_trials)
            best_params = study.best_params
            best_score = study.best_value
            
            print(f"最佳參數: {best_params}")
            print(f"最佳MSE: {best_score}")
            
            # 保存最佳參數
            params_path = os.path.join(self.model_dir, 'catboost_best_params.json')
            with open(params_path, 'w') as f:
                json.dump(best_params, f)
            print(f"最佳參數已保存到: {params_path}")
            
            return best_params, best_score, 'catboost'
            
        elif model_name == 'neural_network':
            study = optuna.create_study(direction='minimize')
            study.optimize(self._objective_nn, n_trials=n_trials)
            best_params = study.best_params
            best_score = study.best_value
            
            print(f"最佳參數: {best_params}")
            print(f"最佳MSE: {best_score}")
            
            # 保存最佳參數
            params_path = os.path.join(self.model_dir, 'neural_network_best_params.json')
            with open(params_path, 'w') as f:
                json.dump(best_params, f)
            print(f"最佳參數已保存到: {params_path}")
            
            return best_params, best_score, 'neural_network'
            
        else:
            raise ValueError(f"不支持的模型: {model_name}")
    
    def save_optimal_parameters(self, model_name, params, hit_rate):
        """保存最佳參數"""
        optimal_params = {
            'model_name': model_name,
            'parameters': params,
            'hit_rate': hit_rate
        }
        
        params_path = os.path.join(self.model_dir, 'optimal_parameters.json')
        with open(params_path, 'w') as f:
            json.dump(optimal_params, f)
        
        return params_path
    
    def load_optimal_parameters(self):
        """載入最佳參數"""
        params_path = os.path.join(self.model_dir, 'optimal_parameters.json')
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                optimal_params = json.load(f)
            print(f"已載入最佳引數 - 模型: {optimal_params['model_name']}, 命中率: {optimal_params['hit_rate']:.2f}%")
            return optimal_params
        else:
            return None
    
    def train_model_with_params(self, model_name, params):
        """使用指定參數訓練模型
        
        參數:
            model_name: 模型名稱 ('random_forest', 'xgboost', 'lightgbm', 'catboost', 'neural_network')
            params: 模型參數字典
        
        返回:
            訓練好的模型
        """
        if model_name == 'random_forest':
            model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            model.fit(self.X_train, self.y_train)
            self.models['random_forest'] = model
            
            # 保存模型
            model_path = os.path.join(self.model_dir, 'random_forest_model.pkl')
            joblib.dump(model, model_path)
            
            return model
            
        elif model_name == 'xgboost':
            # 對每個目標列單獨訓練模型
            models = {}
            for i, col in enumerate(self.y_train.columns):
                model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
                model.fit(self.X_train, self.y_train.iloc[:, i])
                models[col] = model
            
            self.models['xgboost'] = models
            
            # 保存模型
            model_path = os.path.join(self.model_dir, 'xgboost_model.pkl')
            joblib.dump(models, model_path)
            
            return models
            
        elif model_name == 'lightgbm':
            # 對每個目標列單獨訓練模型
            models = {}
            for i, col in enumerate(self.y_train.columns):
                model = lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1)
                model.fit(self.X_train, self.y_train.iloc[:, i])
                models[col] = model
            
            self.models['lightgbm'] = models
            
            # 保存模型
            model_path = os.path.join(self.model_dir, 'lightgbm_model.pkl')
            joblib.dump(models, model_path)
            
            return models
            
        elif model_name == 'catboost':
            # 對每個目標列單獨訓練模型
            models = {}
            for i, col in enumerate(self.y_train.columns):
                model = CatBoostRegressor(**params, random_seed=42, thread_count=-1, verbose=0)
                model.fit(self.X_train, self.y_train.iloc[:, i])
                models[col] = model
            
            self.models['catboost'] = models
            
            # 保存模型
            model_path = os.path.join(self.model_dir, 'catboost_model.pkl')
            joblib.dump(models, model_path)
            
            return models
            
        elif model_name == 'neural_network':
            # 神經網絡模型需要特殊處理
            input_dim = self.X_train.shape[1]
            
            # 從參數中獲取神經網絡配置
            hidden_layers = params.get('hidden_layers', 2)
            neurons = params.get('neurons', 64)
            dropout = params.get('dropout', 0.2)
            learning_rate = params.get('learning_rate', 0.001)
            
            # 對每個目標列單獨訓練模型
            models = {}
            for i, col in enumerate(self.y_train.columns):
                # 創建模型
                model = Sequential()
                model.add(Dense(neurons, activation='relu', input_shape=(input_dim,)))
                model.add(Dropout(dropout))
                
                for _ in range(hidden_layers):
                    model.add(Dense(neurons, activation='relu'))
                    model.add(Dropout(dropout))
                
                model.add(Dense(1))
                
                # 編譯模型
                optimizer = Adam(learning_rate=learning_rate)
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
            
            self.models['neural_network'] = models
            
            # 保存模型
            for col, model in models.items():
                model_path = os.path.join(self.model_dir, f'nn_model_{col}.h5')
                model.save(model_path)
            
            return models
            
        else:
            raise ValueError(f"不支持的模型: {model_name}")