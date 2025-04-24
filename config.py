import os

# 基本路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# 數據文件
DATA_FILE = os.path.join(DATA_DIR, 'lottery_history.xlsx')

# 模型參數
DEFAULT_RANDOM_STATE = 42
CV_FOLDS = 5
TEST_SIZE = 0.2

# 特徵工程參數
LAG_PERIODS = [1, 2, 3, 5, 10]
ROLLING_WINDOWS = [5, 10, 20, 50]

# 模型訓練參數
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': DEFAULT_RANDOM_STATE
}

XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': DEFAULT_RANDOM_STATE
}

LGBM_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': DEFAULT_RANDOM_STATE
}

CATBOOST_PARAMS = {
    'iterations': 100,
    'depth': 6,
    'learning_rate': 0.1,
    'random_seed': DEFAULT_RANDOM_STATE
}

NN_PARAMS = {
    'hidden_layers': 2,
    'neurons': 64,
    'dropout': 0.2,
    'learning_rate': 0.001,
    'epochs': 50,
    'batch_size': 32,
    'validation_split': 0.2
}

# 預測參數
DEFAULT_NUM_SETS = 5
DEFAULT_TRIALS = 100