import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify,send_from_directory
import joblib
import traceback
import logging
from datetime import datetime
import json
import random
import time
import optuna
from src.features.feature_engineering import LotteryFeatureEngineering
from src.models.model_trainer import LotteryModelTrainer
from src.evaluation.evaluator import LotteryEvaluator
from src.diversity_enhancer import DiversityEnhancer

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ])
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 全局變數
DATA_PATH = 'data/lottery_history.xlsx'
MODEL_DIR = 'models'
RESULTS_DIR = 'results'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 確保目錄存在
for directory in [MODEL_DIR, RESULTS_DIR, 'data']:
    if not os.path.exists(directory):
        os.makedirs(directory)

# 初始化多樣性增強器
diversity_enhancer = DiversityEnhancer()

# 全局變數用於追蹤訓練和優化進度
training_progress = {
    'progress': 0,
    'status': '準備訓練...',
    'current_model': None,
    'completed': False,
    'cancelled': False
}

optimization_progress = {
    'progress': 0,
    'status': '準備優化...',
    'completed': False,
    'cancelled': False
}

# 全局變數用於存儲當前的訓練器和優化器
current_trainer = None
current_study = None

# 模擬數據
def generate_sample_data(n_samples=200):
    data = []
    for i in range(n_samples):
        # 生成隨機號碼 (1-49)
        numbers = sorted(random.sample(range(1, 50), 5))
        # 添加日期 (過去n天)
        date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        date = date.replace(day=date.day - i)
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'num1': numbers[0],
            'num2': numbers[1],
            'num3': numbers[2],
            'num4': numbers[3],
            'num5': numbers[4]
        })
    return data

# 如果沒有數據文件，則生成模擬數據
if not os.path.exists(DATA_PATH):
    sample_data = generate_sample_data()
    df = pd.DataFrame(sample_data)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_excel(DATA_PATH, index=False)
    logger.info(f"已生成模擬數據並保存至 {DATA_PATH}")

@app.route('/')
def index():
    """首頁"""
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_models():
    """訓練模型"""
    try:
        global training_progress, current_trainer
        
        # 重置訓練進度
        training_progress = {
            'progress': 0,
            'status': '準備訓練...',
            'current_model': None,
            'completed': False,
            'cancelled': False
        }
        
        logger.info("開始訓練模型...")
        
        # 檢查數據文件是否存在
        if not os.path.exists(DATA_PATH):
            logger.error(f"數據文件不存在: {DATA_PATH}")
            return jsonify({
                'status': 'error',
                'message': f"數據文件不存在: {DATA_PATH}"
            }), 400
        
        # 載入並處理數據
        logger.info("載入並處理數據...")
        try:
            fe = LotteryFeatureEngineering(DATA_PATH)
            data = fe.load_data()
            logger.info(f"數據載入成功，共 {len(data)} 條記錄")
            training_progress['progress'] = 5
            training_progress['status'] = '數據載入成功'
        except Exception as e:
            logger.error(f"載入數據時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"載入數據時出錯: {str(e)}"
            }), 500
        
        # 創建特徵
        logger.info("創建特徵...")
        try:
            features = fe.create_complex_features()
            logger.info(f"特徵創建成功，共 {features.shape[1]} 個特徵")
            training_progress['progress'] = 10
            training_progress['status'] = '特徵創建成功'
        except Exception as e:
            logger.error(f"創建特徵時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"創建特徵時出錯: {str(e)}"
            }), 500
        
        # 準備訓練數據
        logger.info("準備訓練數據...")
        try:
            X, y = fe.get_training_data()
            logger.info(f"訓練數據準備完成，特徵數量: {X.shape[1]}，樣本數量: {X.shape[0]}")
            training_progress['progress'] = 15
            training_progress['status'] = '訓練數據準備完成'
        except Exception as e:
            logger.error(f"準備訓練數據時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"準備訓練數據時出錯: {str(e)}"
            }), 500
        
        # 初始化模型訓練器
        logger.info("初始化模型訓練器...")
        try:
            trainer = LotteryModelTrainer(X, y, model_dir=MODEL_DIR)
            current_trainer = trainer  # 保存訓練器的引用
            logger.info("模型訓練器初始化成功")
            training_progress['progress'] = 20
            training_progress['status'] = '模型訓練器初始化成功'
        except Exception as e:
            logger.error(f"初始化模型訓練器時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"初始化模型訓練器時出錯: {str(e)}"
            }), 500
        
        # 分割訓練集和測試集
        logger.info("分割訓練集和測試集...")
        try:
            X_train, X_test, y_train, y_test = trainer.train_test_split()
            logger.info(f"數據分割完成，訓練集: {X_train.shape[0]} 樣本，測試集: {X_test.shape[0]} 樣本")
            training_progress['progress'] = 25
            training_progress['status'] = '數據分割完成'
        except Exception as e:
            logger.error(f"分割數據時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"分割數據時出錯: {str(e)}"
            }), 500
        
        # 訓練模型
        logger.info("訓練模型...")
        models = {}
        results = {}
        cv_results = {}
        
        # 檢查是否應該停止訓練
        if trainer.should_stop:
            logger.info("訓練已被用戶取消")
            training_progress['cancelled'] = True
            training_progress['status'] = '訓練已取消'
            return jsonify({
                'status': 'cancelled',
                'message': '訓練已被用戶取消'
            }), 499
        
        # 訓練隨機森林模型
        try:
            logger.info("訓練隨機森林模型...")
            training_progress['current_model'] = 'random_forest'
            training_progress['status'] = '訓練隨機森林模型'
            training_progress['progress'] = 30
            
            models['random_forest'] = trainer.train_random_forest(optimize=True)
            logger.info("隨機森林模型訓練完成")
            training_progress['progress'] = 40
        except Exception as e:
            logger.error(f"訓練隨機森林模型時出錯: {str(e)}")
            logger.error(traceback.format_exc())
        
        # 檢查是否應該停止訓練
        if trainer.should_stop:
            logger.info("訓練已被用戶取消")
            training_progress['cancelled'] = True
            training_progress['status'] = '訓練已取消'
            return jsonify({
                'status': 'cancelled',
                'message': '訓練已被用戶取消'
            }), 499
        
        # 訓練XGBoost模型
        try:
            logger.info("訓練XGBoost模型...")
            training_progress['current_model'] = 'xgboost'
            training_progress['status'] = '訓練XGBoost模型'
            training_progress['progress'] = 45
            
            models['xgboost'] = trainer.train_xgboost(optimize=True)
            logger.info("XGBoost模型訓練完成")
            training_progress['progress'] = 55
        except Exception as e:
            logger.error(f"訓練XGBoost模型時出錯: {str(e)}")
            logger.error(traceback.format_exc())
        
        # 檢查是否應該停止訓練
        if trainer.should_stop:
            logger.info("訓練已被用戶取消")
            training_progress['cancelled'] = True
            training_progress['status'] = '訓練已取消'
            return jsonify({
                'status': 'cancelled',
                'message': '訓練已被用戶取消'
            }), 499
        
        # 訓練LightGBM模型
        try:
            logger.info("訓練LightGBM模型...")
            training_progress['current_model'] = 'lightgbm'
            training_progress['status'] = '訓練LightGBM模型'
            training_progress['progress'] = 60
            
            models['lightgbm'] = trainer.train_lightgbm(optimize=True)
            logger.info("LightGBM模型訓練完成")
            training_progress['progress'] = 70
        except Exception as e:
            logger.error(f"訓練LightGBM模型時出錯: {str(e)}")
            logger.error(traceback.format_exc())
        
        # 檢查是否應該停止訓練
        if trainer.should_stop:
            logger.info("訓練已被用戶取消")
            training_progress['cancelled'] = True
            training_progress['status'] = '訓練已取消'
            return jsonify({
                'status': 'cancelled',
                'message': '訓練已被用戶取消'
            }), 499
        
        # 訓練CatBoost模型
        try:
            logger.info("訓練CatBoost模型...")
            training_progress['current_model'] = 'catboost'
            training_progress['status'] = '訓練CatBoost模型'
            training_progress['progress'] = 75
            
            models['catboost'] = trainer.train_catboost(optimize=True)
            logger.info("CatBoost模型訓練完成")
            training_progress['progress'] = 85
        except Exception as e:
            logger.error(f"訓練CatBoost模型時出錯: {str(e)}")
            logger.error(traceback.format_exc())
        
        # 檢查是否應該停止訓練
        if trainer.should_stop:
            logger.info("訓練已被用戶取消")
            training_progress['cancelled'] = True
            training_progress['status'] = '訓練已取消'
            return jsonify({
                'status': 'cancelled',
                'message': '訓練已被用戶取消'
            }), 499
        
        # 訓練神經網絡模型
        try:
            logger.info("訓練神經網絡模型...")
            training_progress['current_model'] = 'neural_network'
            training_progress['status'] = '訓練神經網絡模型'
            training_progress['progress'] = 90
            
            models['neural_network'] = trainer.train_neural_network(optimize=True)
            logger.info("神經網絡模型訓練完成")
            training_progress['progress'] = 95
        except Exception as e:
            logger.error(f"訓練神經網絡模型時出錯: {str(e)}")
            logger.error(traceback.format_exc())
        
        # 檢查是否應該停止訓練
        if trainer.should_stop:
            logger.info("訓練已被用戶取消")
            training_progress['cancelled'] = True
            training_progress['status'] = '訓練已取消'
            return jsonify({
                'status': 'cancelled',
                'message': '訓練已被用戶取消'
            }), 499
        
        # 訓練集成模型
        if len(models) >= 2:  # 至少需要2個模型才能集成
            try:
                logger.info("訓練集成模型...")
                training_progress['current_model'] = 'ensemble'
                training_progress['status'] = '訓練集成模型'
                training_progress['progress'] = 97
                
                models['ensemble'] = trainer.train_ensemble()
                logger.info("集成模型訓練完成")
                training_progress['progress'] = 99
            except Exception as e:
                logger.error(f"訓練集成模型時出錯: {str(e)}")
                logger.error(traceback.format_exc())
        else:
            logger.warning("可用模型不足，無法訓練集成模型")
        
        # 評估模型
        logger.info("評估模型...")
        training_progress['status'] = '評估模型'
        
        for model_name in models.keys():
            try:
                results[model_name] = trainer.evaluate_model(model_name)
                logger.info(f"{model_name} 模型評估完成")
            except Exception as e:
                logger.error(f"評估 {model_name} 模型時出錯: {str(e)}")
                logger.error(traceback.format_exc())
        
        # 交叉驗證
        logger.info("執行交叉驗證...")
        training_progress['status'] = '執行交叉驗證'
        
        for model_name in ['random_forest', 'xgboost', 'lightgbm', 'catboost']:
            if model_name in models:
                try:
                    cv_results[model_name] = trainer.cross_validate(model_name)
                    logger.info(f"{model_name} 模型交叉驗證完成")
                except Exception as e:
                    logger.error(f"交叉驗證 {model_name} 模型時出錯: {str(e)}")
                    logger.error(traceback.format_exc())
        
        # 保存評估結果
        logger.info("保存評估結果...")
        training_progress['status'] = '保存評估結果'
        
        try:
            results_path = os.path.join(RESULTS_DIR, 'evaluation_results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
            
            cv_results_path = os.path.join(RESULTS_DIR, 'cv_results.json')
            with open(cv_results_path, 'w') as f:
                json.dump(cv_results, f, indent=4)
            
            logger.info(f"評估結果已保存至 {results_path} 和 {cv_results_path}")
        except Exception as e:
            logger.error(f"保存評估結果時出錯: {str(e)}")
            logger.error(traceback.format_exc())
        
        # 訓練完成
        training_progress['progress'] = 100
        training_progress['status'] = '訓練完成'
        training_progress['completed'] = True
        logger.info("模型訓練完成")
        
        return jsonify({
            'status': 'success',
            'message': '模型訓練成功',
            'models': list(models.keys()),
            'results': results
        })
    
    except Exception as e:
        logger.error(f"訓練模型時出錯: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"訓練模型時出錯: {str(e)}"
        }), 500

@app.route('/stop_training', methods=['POST'])
def stop_training():
    """終止訓練過程"""
    try:
        global current_trainer, training_progress
        
        if current_trainer:
            current_trainer.should_stop = True
            logger.info("已發送終止訓練請求")
            training_progress['status'] = '正在終止訓練...'
            
            return jsonify({
                'status': 'success',
                'message': '已發送終止訓練請求'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': '沒有正在進行的訓練'
            }), 400
    
    except Exception as e:
        logger.error(f"終止訓練時出錯: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"終止訓練時出錯: {str(e)}"
        }), 500

@app.route('/training_progress', methods=['GET'])
def get_training_progress():
    """獲取訓練進度"""
    global training_progress
    return jsonify(training_progress)

@app.route('/train_advanced', methods=['POST'])
def train_advanced():
    """高級訓練模型，可以指定更多參數"""
    try:
        global training_progress, current_trainer
        
        # 獲取請求參數
        data = request.json
        epochs = data.get('epochs', 100)
        batch_size = data.get('batch_size', 32)
        
        # 重置訓練進度
        training_progress = {
            'progress': 0,
            'status': '準備高級訓練...',
            'current_model': None,
            'completed': False,
            'cancelled': False
        }
        
        logger.info(f"開始高級訓練模型，epochs={epochs}, batch_size={batch_size}...")
        
        # 檢查數據文件是否存在
        if not os.path.exists(DATA_PATH):
            logger.error(f"數據文件不存在: {DATA_PATH}")
            return jsonify({
                'status': 'error',
                'message': f"數據文件不存在: {DATA_PATH}"
            }), 400
        
        # 載入並處理數據
        logger.info("載入並處理數據...")
        try:
            fe = LotteryFeatureEngineering(DATA_PATH)
            data = fe.load_data()
            logger.info(f"數據載入成功，共 {len(data)} 條記錄")
            training_progress['progress'] = 5
            training_progress['status'] = '數據載入成功'
        except Exception as e:
            logger.error(f"載入數據時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"載入數據時出錯: {str(e)}"
            }), 500
        
        # 創建特徵
        logger.info("創建特徵...")
        try:
            features = fe.create_complex_features()
            logger.info(f"特徵創建成功，共 {features.shape[1]} 個特徵")
            training_progress['progress'] = 10
            training_progress['status'] = '特徵創建成功'
        except Exception as e:
            logger.error(f"創建特徵時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"創建特徵時出錯: {str(e)}"
            }), 500
        
        # 準備訓練數據
        logger.info("準備訓練數據...")
        try:
            X, y = fe.get_training_data()
            logger.info(f"訓練數據準備完成，特徵數量: {X.shape[1]}，樣本數量: {X.shape[0]}")
            training_progress['progress'] = 15
            training_progress['status'] = '訓練數據準備完成'
        except Exception as e:
            logger.error(f"準備訓練數據時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"準備訓練數據時出錯: {str(e)}"
            }), 500
        
        # 初始化模型訓練器
        logger.info("初始化模型訓練器...")
        try:
            trainer = LotteryModelTrainer(X, y, model_dir=MODEL_DIR)
            current_trainer = trainer  # 保存訓練器的引用
            logger.info("模型訓練器初始化成功")
            training_progress['progress'] = 20
            training_progress['status'] = '模型訓練器初始化成功'
        except Exception as e:
            logger.error(f"初始化模型訓練器時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"初始化模型訓練器時出錯: {str(e)}"
            }), 500
        
        # 分割訓練集和測試集
        logger.info("分割訓練集和測試集...")
        try:
            X_train, X_test, y_train, y_test = trainer.train_test_split()
            logger.info(f"數據分割完成，訓練集: {X_train.shape[0]} 樣本，測試集: {X_test.shape[0]} 樣本")
            training_progress['progress'] = 25
            training_progress['status'] = '數據分割完成'
        except Exception as e:
            logger.error(f"分割數據時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"分割數據時出錯: {str(e)}"
            }), 500
        
        # 訓練神經網絡模型
        try:
            logger.info("訓練神經網絡模型...")
            training_progress['current_model'] = 'neural_network'
            training_progress['status'] = '訓練神經網絡模型'
            training_progress['progress'] = 30
            
            # 這裡可以添加更多的高級訓練參數
            models = trainer.train_neural_network(optimize=True)
            logger.info("神經網絡模型訓練完成")
            training_progress['progress'] = 90
        except Exception as e:
            logger.error(f"訓練神經網絡模型時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"訓練神經網絡模型時出錯: {str(e)}"
            }), 500
        
        # 評估模型
        logger.info("評估神經網絡模型...")
        training_progress['status'] = '評估神經網絡模型'
        training_progress['progress'] = 95
        
        try:
            results = trainer.evaluate_model('neural_network')
            logger.info("神經網絡模型評估完成")
        except Exception as e:
            logger.error(f"評估神經網絡模型時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"評估神經網絡模型時出錯: {str(e)}"
            }), 500
        
        # 訓練完成
        training_progress['progress'] = 100
        training_progress['status'] = '高級訓練完成'
        training_progress['completed'] = True
        logger.info("高級模型訓練完成")
        
        return jsonify({
            'status': 'success',
            'message': '高級模型訓練成功',
            'model': 'neural_network',
            'results': results
        })
    
    except Exception as e:
        logger.error(f"高級訓練模型時出錯: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"高級訓練模型時出錯: {str(e)}"
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """生成預測"""
    try:
        data = request.json
        model_name = data.get('model_name', 'ensemble')
        num_sets = data.get('num_sets', 5)
        
        logger.info(f"使用 {model_name} 模型生成 {num_sets} 組預測...")
        
        # 載入特徵工程器
        fe = LotteryFeatureEngineering(DATA_PATH)
        
        # 獲取最新數據作為預測輸入
        X_new = fe.get_latest_features()
        
        # 載入模型
        if model_name == 'random_forest':
            model_path = os.path.join(MODEL_DIR, 'random_forest_model.pkl')
            model = joblib.load(model_path)
        elif model_name == 'xgboost':
            model_path = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
            model = joblib.load(model_path)
        elif model_name == 'lightgbm':
            model_path = os.path.join(MODEL_DIR, 'lightgbm_model.pkl')
            model = joblib.load(model_path)
        elif model_name == 'catboost':
            model_path = os.path.join(MODEL_DIR, 'catboost_model.pkl')
            model = joblib.load(model_path)
        elif model_name == 'neural_network':
            # 神經網絡模型需要特殊處理
            model = {}
            for i in range(5):  # 假設有5個目標列
                model_path = os.path.join(MODEL_DIR, f'nn_model_num{i+1}.h5')
                if os.path.exists(model_path):
                    model[f'num{i+1}'] = joblib.load(model_path)
        else:  # 默認使用集成模型
            model_path = os.path.join(MODEL_DIR, 'ensemble_model.pkl')
            model = joblib.load(model_path)
        
        # 初始化模型訓練器
        X, y = fe.get_training_data()
        trainer = LotteryModelTrainer(X, y, model_dir=MODEL_DIR)
        
        # 生成預測
        predictions = []
        for _ in range(num_sets):
            pred_set = trainer.generate_lottery_numbers(X_new, model_name=model_name)
            predictions.append(pred_set)
        
        # 應用多樣性增強
        diversity_method = request.cookies.get('diversityMethod', 'hybrid')
        diversity_level = float(request.cookies.get('diversityLevel', 0.2))
        
        if diversity_method != 'none' and diversity_level > 0:
            predictions = diversity_enhancer.enhance_diversity(
                predictions, 
                method=diversity_method, 
                level=diversity_level
            )
        
        logger.info(f"成功生成 {len(predictions)} 組預測")
        
        return jsonify({
            'status': 'success',
            'model': model_name,
            'predictions': predictions
        })
    
    except Exception as e:
        logger.error(f"生成預測時出錯: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"生成預測時出錯: {str(e)}"
        }), 500

@app.route('/predict_with_best_params', methods=['POST'])
def predict_with_best_params():
    try:
        data = request.get_json()
        num_sets = data.get('num_sets', 5)
        
        app.logger.info(f"使用最佳參數生成 {num_sets} 組預測...")
        
        # 載入特徵工程器
        fe = LotteryFeatureEngineering(DATA_PATH)
        
        # 創建特徵
        fe.create_basic_features()
        fe.create_advanced_features()
        X_train, y_train = fe.get_training_data()
        
        # 確保 X_new 是正確的格式
        X_new = X_train.iloc[-1:].copy()
        app.logger.info(f"X_new 形狀: {X_new.shape}, 類型: {type(X_new)}")
        
        # 檢查 X_new 是否包含 NaN 值
        if X_new.isna().any().any():
            app.logger.warning("X_new 包含 NaN 值，將進行填充")
            X_new = X_new.fillna(X_train.mean())
        
        # 初始化模型訓練器
        trainer = LotteryModelTrainer(X_train, y_train, model_dir=MODEL_DIR)
        
        # 載入最佳參數
        optimal_params = trainer.load_optimal_parameters()
        if optimal_params:
            model_name = optimal_params.get('model_name', 'random_forest')
            hit_rate = optimal_params.get('hit_rate', 0)
        else:
            # 如果沒有最佳參數文件，使用默認值
            params_path = os.path.join(MODEL_DIR, 'optimal_parameters.json')
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    optimal_params = json.load(f)
                model_name = optimal_params.get('model_name', 'random_forest')
                hit_rate = optimal_params.get('hit_rate', 0)
            else:
                model_name = 'random_forest'
                hit_rate = 0
        
        app.logger.info(f"使用模型 {model_name}，歷史命中率: {hit_rate}%")
        
        # 嘗試載入模型
        try:
            model_path = os.path.join(MODEL_DIR, f'{model_name}_model.pkl')
            if os.path.exists(model_path):
                if model_name == 'random_forest':
                    trainer.models[model_name] = joblib.load(model_path)
                else:
                    trainer.models[model_name] = joblib.load(model_path)
                app.logger.info(f"已載入 {model_name} 模型")
            else:
                app.logger.error(f"找不到 {model_name} 模型文件")
                return jsonify({'error': f"找不到 {model_name} 模型文件，請先訓練模型"}), 500
        except Exception as e:
            app.logger.error(f"載入 {model_name} 模型時出錯: {str(e)}")
            return jsonify({'error': f"載入模型時出錯: {str(e)}"}), 500
        
        # 在這裡添加修正，確保 X_new 有正確的特徵名稱
        # 從訓練數據中獲取特徵名稱
        feature_names = X_train.columns.tolist()
        
        # 確保 X_new 使用相同的特徵名稱
        X_new = pd.DataFrame(X_new.values, columns=feature_names)
        
        # 生成預測
        try:
            predictions = []
            for i in range(num_sets):
                # 確保 X_new 是正確的格式
                app.logger.info(f"生成預測 {i+1}/{num_sets}")
                pred_set = trainer.generate_lottery_numbers(X_new, model_name=model_name)
                app.logger.info(f"預測結果: {pred_set}")
                predictions.append({
                    'set_number': i + 1,
                    'numbers': pred_set
                })
        except Exception as e:
            app.logger.error(f"生成預測時出錯: {str(e)}")
            app.logger.error(traceback.format_exc())
            return jsonify({'error': f"生成預測時出錯: {str(e)}"}), 500
        
        # 增強多樣性
        diversity_method = request.cookies.get('diversityMethod', 'hybrid')
        diversity_level = float(request.cookies.get('diversityLevel', '0.2'))
        
        diversity_enhancer = DiversityEnhancer()
        
        if diversity_method != 'none' and diversity_level > 0:
            enhanced_predictions = []
            for i, pred in enumerate(diversity_enhancer.enhance_diversity(
            [p['numbers'] for p in predictions],
            method=diversity_method,
            num_sets=len(predictions)  # 正確的參數名
        )):
                enhanced_predictions.append({
                    'set_number': i + 1,
                    'numbers': pred
                })
            predictions = enhanced_predictions
        
        app.logger.info(f"成功生成 {len(predictions)} 組預測")
        
        return jsonify({
            'model': model_name,
            'hit_rate': hit_rate,
            'predictions': predictions
        })
    except Exception as e:
        app.logger.error(f"使用最佳參數生成預測時出錯: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate():
    """評估預測結果"""
    try:
        data = request.json
        predictions = data.get('predictions', [])
        actual_numbers = data.get('actual_numbers', [])
        
        logger.info(f"評估預測結果，實際號碼: {actual_numbers}")
        
        if not predictions:
            return jsonify({
                'status': 'error',
                'message': "沒有提供預測結果"
            }), 400
        
        if not actual_numbers:
            return jsonify({
                'status': 'error',
                'message': "沒有提供實際號碼"
            }), 400
        
        # 初始化評估器
        evaluator = LotteryEvaluator()
        
        # 評估命中率
        hit_results = evaluator.evaluate_hit_rate(predictions, actual_numbers)
        
        # 生成評估報告
        evaluation_report = evaluator.generate_evaluation_report(predictions, actual_numbers)
        
        logger.info(f"評估完成，平均命中率: {hit_results['avg_hit_rate']:.2f}")
        
        return jsonify({
            'status': 'success',
            'hit_results': hit_results,
            'evaluation_report': evaluation_report
        })
    
    except Exception as e:
        logger.error(f"評估預測結果時出錯: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"評估預測結果時出錯: {str(e)}"
        }), 500

@app.route('/optimize', methods=['POST'])
def optimize():
    """優化參數"""
    try:
        global optimization_progress, current_study
        
        data = request.json
        model_name = data.get('model_name')
        n_trials = data.get('n_trials', 50)
        actual_numbers = data.get('actual_numbers', [])
        
        # 重置優化進度
        optimization_progress = {
            'progress': 0,
            'status': '準備優化...',
            'completed': False,
            'cancelled': False
        }
        
        logger.info(f"開始優化參數，模型: {model_name}, 試驗次數: {n_trials}")
        
        # 載入特徵工程器
        fe = LotteryFeatureEngineering(DATA_PATH)
        
        # 準備訓練數據
        X, y = fe.get_training_data()
        
        # 初始化模型訓練器
        trainer = LotteryModelTrainer(X, y, model_dir=MODEL_DIR)
        
        # 分割訓練集和測試集
        X_train, X_test, y_train, y_test = trainer.train_test_split()
        
        # 如果指定了模型名稱，則優化該模型
        if model_name:
            # 創建優化研究
            study = optuna.create_study(direction='minimize')
            current_study = study
            
            # 定義回調函數來更新進度
            def callback(study, trial):
                global optimization_progress
                progress = int((trial.number + 1) / n_trials * 100)
                optimization_progress['progress'] = min(progress, 99)  # 保留最後1%給最終處理
                
                # 檢查是否應該取消優化
                if optimization_progress['cancelled']:
                    study.stop()
            
            # 優化參數
            best_params, best_score, best_model = trainer.optimize_hyperparameters(
                model_name=model_name, 
                n_trials=n_trials
            )
            
            # 更新優化進度
            optimization_progress['progress'] = 100
            optimization_progress['completed'] = True
            
            logger.info(f"參數優化完成，最佳模型: {best_model}, 最佳分數: {best_score:.4f}")
            
            return jsonify({
                'status': 'success',
                'best_model': best_model,
                'best_params': best_params,
                'best_score': best_score
            })
        
        # 如果提供了實際號碼，則優化預測結果
        elif actual_numbers:
            # 獲取最新數據作為預測輸入
            X_new = fe.get_latest_features()
            
            # 初始化評估器
            evaluator = LotteryEvaluator()
            
            # 測試不同模型
            models_to_test = ['random_forest', 'xgboost', 'lightgbm', 'catboost', 'neural_network', 'ensemble']
            best_hit_rate = 0
            best_model = None
            best_predictions = None
            
            for i, model in enumerate(models_to_test):
                try:
                    # 更新進度
                    progress = int((i + 1) / len(models_to_test) * 100)
                    optimization_progress['progress'] = min(progress, 99)
                    optimization_progress['status'] = f'測試模型: {model}'
                    
                    # 檢查是否應該取消優化
                    if optimization_progress['cancelled']:
                        break
                    
                    # 生成預測
                    predictions = []
                    for _ in range(10):  # 生成10組預測
                        pred_set = trainer.generate_lottery_numbers(X_new, model_name=model)
                        predictions.append(pred_set)
                    
                    # 評估命中率
                    hit_results = evaluator.evaluate_hit_rate(predictions, actual_numbers)
                    hit_rate = hit_results['avg_hit_rate']
                    
                    logger.info(f"模型 {model} 的命中率: {hit_rate:.4f}")
                    
                    # 更新最佳模型
                    if hit_rate > best_hit_rate:
                        best_hit_rate = hit_rate
                        best_model = model
                        best_predictions = predictions
                
                except Exception as e:
                    logger.error(f"測試模型 {model} 時出錯: {str(e)}")
                    logger.error(traceback.format_exc())
            
            # 如果找到了最佳模型
            if best_model:
                # 保存最佳參數
                optimal_params = {
                    'model_name': best_model,
                    'hit_rate': best_hit_rate,
                    'predictions': best_predictions
                }
                
                params_path = trainer.save_optimal_parameters(best_model, {}, best_hit_rate)
                
                # 更新優化進度
                optimization_progress['progress'] = 100
                optimization_progress['completed'] = True
                
                logger.info(f"優化完成，最佳模型: {best_model}, 命中率: {best_hit_rate:.4f}")
                
                return jsonify({
                    'status': 'success',
                    'optimal_params': optimal_params
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': "未找到有效的模型"
                }), 400
        
        else:
            return jsonify({
                'status': 'error',
                'message': "請提供模型名稱或實際號碼"
            }), 400
    
    except Exception as e:
        logger.error(f"優化參數時出錯: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"優化參數時出錯: {str(e)}"
        }), 500

@app.route('/stop_optimization', methods=['POST'])
def stop_optimization():
    """終止優化過程"""
    try:
        global current_study, optimization_progress
        
        if current_study:
            current_study.stop()
            optimization_progress['cancelled'] = True
            logger.info("已發送終止優化請求")
            
            return jsonify({
                'status': 'success',
                'message': '已發送終止優化請求'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': '沒有正在進行的優化'
            }), 400
    
    except Exception as e:
        logger.error(f"終止優化時出錯: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"終止優化時出錯: {str(e)}"
        }), 500

@app.route('/optimization_progress', methods=['GET'])
def get_optimization_progress():
    """獲取優化進度"""
    global optimization_progress
    return jsonify(optimization_progress)

@app.route('/data', methods=['GET'])
def get_data_summary():
    """獲取數據摘要"""
    try:
        logger.info("獲取數據摘要...")
        
        # 載入特徵工程器
        fe = LotteryFeatureEngineering(DATA_PATH)
        
        # 載入數據
        data = fe.load_data()
        
        # 基本統計信息
        total_records = len(data)
        
        # 獲取熱門號碼和冷門號碼
        number_counts = {}
        for _, row in data.iterrows():
            # 從 num1 到 num5 列獲取號碼
            numbers = []
            for col in ['num1', 'num2', 'num3', 'num4', 'num5']:
                if col in row.index and not pd.isna(row[col]):
                    numbers.append(int(row[col]))
            
            for num in numbers:
                if num in number_counts:
                    number_counts[num] += 1
                else:
                    number_counts[num] = 1
        
        # 排序號碼頻率
        sorted_numbers = sorted(number_counts.items(), key=lambda x: x[1], reverse=True)
        
        # 獲取前10個熱門號碼和後10個冷門號碼
        hot_numbers = [num for num, _ in sorted_numbers[:10]]
        cold_numbers = [num for num, _ in sorted_numbers[-10:]]
        
        # 獲取最近10次開獎結果
        recent_draws = []
        for _, row in data.iloc[-10:].iterrows():
            numbers = []
            for col in ['num1', 'num2', 'num3', 'num4', 'num5']:
                if col in row.index and not pd.isna(row[col]):
                    numbers.append(int(row[col]))
            recent_draws.append(numbers)
        
        # 計算號碼頻率
        total_draws = len(data)
        number_frequencies = []
        for num in range(1, 50):  # 假設號碼範圍是1-49
            count = number_counts.get(num, 0)
            percentage = (count / total_draws) * 100
            number_frequencies.append({
                'number': num,
                'count': count,
                'percentage': percentage
            })
        
        logger.info("數據摘要獲取成功")
        
        return jsonify({
            'status': 'success',
            'total_records': total_records,
            'hot_numbers': hot_numbers,
            'cold_numbers': cold_numbers,
            'recent_draws': recent_draws,
            'number_frequencies': number_frequencies
        })
    
    except Exception as e:
        logger.error(f"獲取數據摘要時出錯: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"獲取數據摘要時出錯: {str(e)}"
        }), 500

@app.route('/advanced_analysis', methods=['GET'])
def advanced_analysis():
    """執行高級數據分析"""
    try:
        logger.info("執行高級數據分析...")
        
        # 載入特徵工程器
        fe = LotteryFeatureEngineering(DATA_PATH)
        
        # 載入數據
        data = fe.load_data()
        
        # 相關性分析
        correlations = fe.analyze_correlations()
        
        # 週期性分析
        periodicity = fe.analyze_periodicity()
        
        # 趨勢分析
        trends = fe.analyze_trends()
        
        logger.info("高級數據分析完成")
        
        return jsonify({
            'status': 'success',
            'correlations': correlations,
            'periodicity': periodicity,
            'trends': trends
        })
    
    except Exception as e:
        logger.error(f"執行高級數據分析時出錯: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"執行高級數據分析時出錯: {str(e)}"
        }), 500

@app.route('/diversity_settings', methods=['POST'])
def set_diversity_settings():
    """設置多樣性參數"""
    try:
        data = request.json
        method = data.get('method', 'hybrid')
        level = data.get('level', 0.2)
        
        logger.info(f"設置多樣性參數: method={method}, level={level}")
        
        # 設置 cookie
        response = jsonify({
            'status': 'success',
            'message': '多樣性設置已保存'
        })
        
        response.set_cookie('diversityMethod', method)
        response.set_cookie('diversityLevel', str(level))
        
        return response
    
    except Exception as e:
        logger.error(f"設置多樣性參數時出錯: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"設置多樣性參數時出錯: {str(e)}"
        }), 500

@app.route('/upload_data', methods=['POST'])
def upload_data():
    """上傳數據文件"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': '沒有上傳文件'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': '沒有選擇文件'
            }), 400
        
        # 檢查文件類型
        if not (file.filename.endswith('.xlsx') or file.filename.endswith('.csv')):
            return jsonify({
                'status': 'error',
                'message': '只支持 Excel 或 CSV 文件'
            }), 400
        
        # 保存文件
        file_path = os.path.join(BASE_DIR, 'data', 'uploaded_data.xlsx')
        file.save(file_path)
        
        logger.info(f"數據文件上傳成功: {file_path}")
        
        # 嘗試載入數據
        try:
            fe = LotteryFeatureEngineering(file_path)
            data = fe.load_data()
            total_records = len(data)
            
            logger.info(f"成功載入 {total_records} 條記錄")
            
            return jsonify({
                'status': 'success',
                'message': f'數據文件上傳成功，共 {total_records} 條記錄',
                'total_records': total_records
            })
        
        except Exception as e:
            logger.error(f"載入上傳的數據文件時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            
            return jsonify({
                'status': 'error',
                'message': f'數據文件格式錯誤: {str(e)}'
            }), 400
    
    except Exception as e:
        logger.error(f"上傳數據文件時出錯: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"上傳數據文件時出錯: {str(e)}"
        }), 500

@app.route('/download_predictions', methods=['POST'])
def download_predictions():
    """下載預測結果"""
    try:
        data = request.json
        predictions = data.get('predictions', [])
        
        if not predictions:
            return jsonify({
                'status': 'error',
                'message': '沒有預測結果可下載'
            }), 400
        
        # 創建 DataFrame
        rows = []
        for i, pred_set in enumerate(predictions):
            for j, numbers in enumerate(pred_set):
                rows.append({
                    'set': i + 1,
                    'row': j + 1,
                    'numbers': ', '.join(map(str, numbers))
                })
        
        df = pd.DataFrame(rows)
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'predictions_{timestamp}.xlsx'
        file_path = os.path.join(RESULTS_DIR, filename)
        
        # 保存為 Excel 文件
        df.to_excel(file_path, index=False)
        
        logger.info(f"預測結果已保存至: {file_path}")
        
        # 返回文件下載鏈接
        return jsonify({
            'status': 'success',
            'message': '預測結果已準備好下載',
            'download_url': f'/download/{filename}'
        })
    
    except Exception as e:
        logger.error(f"下載預測結果時出錯: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"下載預測結果時出錯: {str(e)}"
        }), 500

@app.route('/download/<filename>')
def download_file(filename):
    """下載文件"""
    try:
        return send_from_directory(RESULTS_DIR, filename, as_attachment=True)
    except Exception as e:
        logger.error(f"下載文件時出錯: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"下載文件時出錯: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True)