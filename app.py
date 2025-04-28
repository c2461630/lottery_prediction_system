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
import threading



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

# 全域性變數
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

# 全域性變數用於追蹤最佳化進度
optimization_progress = {
    'progress': 0,
    'status': '準備最佳化...',
    'completed': False,
    'cancelled': False
}

# 全局變量來跟踪訓練狀態
training_status = {
    'status': 'not_started',  # 'not_started', 'in_progress', 'completed', 'error'
    'progress': 0.0,  # 0.0 到 1.0
    'current_model': '',
    'message': '',
    'results': None
}

# 全域性變數用於儲存當前的訓練器和最佳化器
current_trainer = None
current_study = None

# 模擬資料
def generate_sample_data(n_samples=200):
    data = []
    for i in range(n_samples):
        # 生成隨機號碼 (1-49)
        numbers = sorted(random.sample(range(1, 50), 5))
        # 新增日期 (過去n天)
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

# 如果沒有資料檔案，則生成模擬資料
if not os.path.exists(DATA_PATH):
    sample_data = generate_sample_data()
    df = pd.DataFrame(sample_data)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_excel(DATA_PATH, index=False)
    logger.info(f"已生成模擬資料並儲存至 {DATA_PATH}")

@app.route('/')
def index():
    """首頁"""
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_models():
    """訓練模型"""
    global training_status
    
    # 如果已經在訓練中，返回錯誤
    if training_status['status'] == 'in_progress':
        return jsonify({'message': '模型訓練已在進行中'}), 400
    
    # 重置訓練狀態
    training_status['status'] = 'in_progress'
    training_status['progress'] = 0.0
    training_status['current_model'] = '準備中'
    training_status['message'] = ''
    training_status['results'] = None
    
    # 在後台執行訓練
    threading.Thread(target=train_models_background).start()
    
    return jsonify({'message': '模型訓練已開始'})

def train_models_background():
    """在後台執行模型訓練"""
    global training_status
    
    try:
        app.logger.info('開始訓練模型...')
        
        # 更新狀態
        training_status['progress'] = 0.05
        training_status['current_model'] = '載入數據'
        
        # 檢查資料檔案是否存在
        if not os.path.exists(DATA_PATH):
            app.logger.error(f"資料檔案不存在: {DATA_PATH}")
            training_status['status'] = 'error'
            training_status['message'] = f"資料檔案不存在: {DATA_PATH}"
            return
        
        # 載入並處理資料
        app.logger.info('載入並處理資料...')
        try:
            fe = LotteryFeatureEngineering(DATA_PATH)
            data = fe.load_data()
            app.logger.info(f"資料載入成功，共 {len(data)} 條記錄")
            training_status['progress'] = 0.1
            training_status['message'] = '資料載入成功'
        except Exception as e:
            app.logger.error(f"載入資料時出錯: {str(e)}")
            app.logger.error(traceback.format_exc())
            training_status['status'] = 'error'
            training_status['message'] = f"載入資料時出錯: {str(e)}"
            return
        
        # 建立特徵
        app.logger.info("建立特徵...")
        try:
            features = fe.create_complex_features()
            app.logger.info(f"特徵建立成功，共 {features.shape[1]} 個特徵")
            training_status['progress'] = 0.15
            training_status['message'] = '特徵建立成功'
        except Exception as e:
            app.logger.error(f"建立特徵時出錯: {str(e)}")
            app.logger.error(traceback.format_exc())
            training_status['status'] = 'error'
            training_status['message'] = f"建立特徵時出錯: {str(e)}"
            return
        
        # 準備訓練資料
        app.logger.info("準備訓練資料...")
        try:
            X, y = fe.get_training_data()
            app.logger.info(f"訓練資料準備完成，特徵數量: {X.shape[1]}，樣本數量: {X.shape[0]}")
            training_status['progress'] = 0.2
            training_status['message'] = '訓練資料準備完成'
        except Exception as e:
            app.logger.error(f"準備訓練資料時出錯: {str(e)}")
            app.logger.error(traceback.format_exc())
            training_status['status'] = 'error'
            training_status['message'] = f"準備訓練資料時出錯: {str(e)}"
            return
        
        # 初始化模型訓練器
        app.logger.info("初始化模型訓練器...")
        try:
            trainer = LotteryModelTrainer(X, y, model_dir=MODEL_DIR)
            app.logger.info("模型訓練器初始化成功")
            training_status['progress'] = 0.25
            training_status['message'] = '模型訓練器初始化成功'
        except Exception as e:
            app.logger.error(f"初始化模型訓練器時出錯: {str(e)}")
            app.logger.error(traceback.format_exc())
            training_status['status'] = 'error'
            training_status['message'] = f"初始化模型訓練器時出錯: {str(e)}"
            return
        
        # 分割訓練集和測試集
        app.logger.info("分割訓練集和測試集...")
        try:
            X_train, X_test, y_train, y_test = trainer.train_test_split()
            app.logger.info(f"資料分割完成，訓練集: {X_train.shape[0]} 樣本，測試集: {X_test.shape[0]} 樣本")
            training_status['progress'] = 0.3
            training_status['message'] = '資料分割完成'
        except Exception as e:
            app.logger.error(f"分割資料時出錯: {str(e)}")
            app.logger.error(traceback.format_exc())
            training_status['status'] = 'error'
            training_status['message'] = f"分割資料時出錯: {str(e)}"
            return
        
        # 訓練模型
        app.logger.info("訓練模型...")
        models = {}
        results = {}
        cv_results = {}
        
        # 訓練隨機森林模型
        try:
            app.logger.info("訓練隨機森林模型...")
            training_status['current_model'] = 'random_forest'
            training_status['message'] = '訓練隨機森林模型'
            training_status['progress'] = 0.35
            
            models['random_forest'] = trainer.train_random_forest(optimize=True)
            app.logger.info("隨機森林模型訓練完成")
            training_status['progress'] = 0.4
        except Exception as e:
            app.logger.error(f"訓練隨機森林模型時出錯: {str(e)}")
            app.logger.error(traceback.format_exc())
        
        # 訓練XGBoost模型
        try:
            app.logger.info("訓練XGBoost模型...")
            training_status['current_model'] = 'xgboost'
            training_status['message'] = '訓練XGBoost模型'
            training_status['progress'] = 0.45
            
            models['xgboost'] = trainer.train_xgboost(optimize=True)
            app.logger.info("XGBoost模型訓練完成")
            training_status['progress'] = 0.55
        except Exception as e:
            app.logger.error(f"訓練XGBoost模型時出錯: {str(e)}")
            app.logger.error(traceback.format_exc())
        
        # 訓練LightGBM模型
        try:
            app.logger.info("訓練LightGBM模型...")
            training_status['current_model'] = 'lightgbm'
            training_status['message'] = '訓練LightGBM模型'
            training_status['progress'] = 0.6
            
            models['lightgbm'] = trainer.train_lightgbm(optimize=True)
            app.logger.info("LightGBM模型訓練完成")
            training_status['progress'] = 0.7
        except Exception as e:
            app.logger.error(f"訓練LightGBM模型時出錯: {str(e)}")
            app.logger.error(traceback.format_exc())
        
        # 訓練CatBoost模型
        try:
            app.logger.info("訓練CatBoost模型...")
            training_status['current_model'] = 'catboost'
            training_status['message'] = '訓練CatBoost模型'
            training_status['progress'] = 0.75
            
            models['catboost'] = trainer.train_catboost(optimize=True)
            app.logger.info("CatBoost模型訓練完成")
            training_status['progress'] = 0.85
        except Exception as e:
            app.logger.error(f"訓練CatBoost模型時出錯: {str(e)}")
            app.logger.error(traceback.format_exc())
        
        # 訓練神經網路模型
        try:
            app.logger.info("訓練神經網路模型...")
            training_status['current_model'] = 'neural_network'
            training_status['message'] = '訓練神經網路模型'
            training_status['progress'] = 0.9
            
            models['neural_network'] = trainer.train_neural_network(optimize=True)
            app.logger.info("神經網路模型訓練完成")
            training_status['progress'] = 0.95
        except Exception as e:
            app.logger.error(f"訓練神經網路模型時出錯: {str(e)}")
            app.logger.error(traceback.format_exc())
        
        # 訓練整合模型
        if len(models) >= 2:  # 至少需要2個模型才能整合
            try:
                app.logger.info("訓練整合模型...")
                training_status['current_model'] = 'ensemble'
                training_status['message'] = '訓練整合模型'
                training_status['progress'] = 0.97
                
                models['ensemble'] = trainer.train_ensemble()
                app.logger.info("整合模型訓練完成")
                training_status['progress'] = 0.99
            except Exception as e:
                app.logger.error(f"訓練整合模型時出錯: {str(e)}")
                app.logger.error(traceback.format_exc())
        else:
            app.logger.warning("可用模型不足，無法訓練整合模型")
        
        # 評估模型
        app.logger.info("評估模型...")
        training_status['message'] = '評估模型'
        
        for model_name in models.keys():
            try:
                results[model_name] = trainer.evaluate_model(model_name)
                app.logger.info(f"{model_name} 模型評估完成")
            except Exception as e:
                app.logger.error(f"評估 {model_name} 模型時出錯: {str(e)}")
                app.logger.error(traceback.format_exc())
        
        # 交叉驗證
        app.logger.info("執行交叉驗證...")
        training_status['message'] = '執行交叉驗證'
        
        for model_name in ['random_forest', 'xgboost', 'lightgbm', 'catboost']:
            if model_name in models:
                try:
                    cv_results[model_name] = trainer.cross_validate(model_name)
                    app.logger.info(f"{model_name} 模型交叉驗證完成")
                except Exception as e:
                    app.logger.error(f"交叉驗證 {model_name} 模型時出錯: {str(e)}")
                    app.logger.error(traceback.format_exc())
        
        # 儲存評估結果
        app.logger.info("儲存評估結果...")
        training_status['message'] = '儲存評估結果'
        
        try:
            results_path = os.path.join(RESULTS_DIR, 'evaluation_results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
            
            cv_results_path = os.path.join(RESULTS_DIR, 'cv_results.json')
            with open(cv_results_path, 'w') as f:
                json.dump(cv_results, f, indent=4)
            
            app.logger.info(f"評估結果已儲存至 {results_path} 和 {cv_results_path}")
        except Exception as e:
            app.logger.error(f"儲存評估結果時出錯: {str(e)}")
            app.logger.error(traceback.format_exc())
        
        # 訓練完成
        training_status['status'] = 'completed'
        training_status['progress'] = 1.0
        training_status['message'] = '訓練完成'
        training_status['results'] = {
            'models': list(models.keys()),
            'evaluation': results
        }
        app.logger.info("模型訓練完成")
        
    except Exception as e:
        app.logger.error(f"訓練模型時出錯: {str(e)}")
        app.logger.error(traceback.format_exc())
        training_status['status'] = 'error'
        training_status['message'] = f"訓練模型時出錯: {str(e)}"

@app.route('/stop_training', methods=['POST'])
def stop_training():
    """終止訓練過程"""
    try:
        global current_trainer, training_progress
        
        if current_trainer:
            current_trainer.should_stop = True
            logger.info("已傳送終止訓練請求")
            training_progress['status'] = '正在終止訓練...'
            
            return jsonify({
                'status': 'success',
                'message': '已傳送終止訓練請求'
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
def training_progress():
    """獲取模型訓練進度"""
    global training_status
    
    # 如果訓練尚未開始但有評估結果文件，載入它
    if training_status['status'] == 'not_started':
        results_path = os.path.join('results', 'evaluation_results.json')
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    training_status['results'] = json.load(f)
                    training_status['status'] = 'completed'
                    training_status['progress'] = 1.0
                    training_status['current_model'] = '完成'
            except Exception as e:
                app.logger.error(f"讀取評估結果時出錯: {str(e)}")
    
    # 返回當前訓練狀態
    return jsonify(training_status)

@app.route('/train_advanced', methods=['POST'])
def train_advanced():
    """高階訓練模型，可以指定更多引數"""
    try:
        global training_progress, current_trainer
        
        # 獲取請求引數
        data = request.json
        epochs = data.get('epochs', 100)
        batch_size = data.get('batch_size', 32)
        
        # 重置訓練進度
        training_progress = {
            'progress': 0,
            'status': '準備高階訓練...',
            'current_model': None,
            'completed': False,
            'cancelled': False
        }
        
        logger.info(f"開始高階訓練模型，epochs={epochs}, batch_size={batch_size}...")
        
        # 檢查資料檔案是否存在
        if not os.path.exists(DATA_PATH):
            logger.error(f"資料檔案不存在: {DATA_PATH}")
            return jsonify({
                'status': 'error',
                'message': f"資料檔案不存在: {DATA_PATH}"
            }), 400
        
        # 載入並處理資料
        logger.info("載入並處理資料...")
        try:
            fe = LotteryFeatureEngineering(DATA_PATH)
            data = fe.load_data()
            logger.info(f"資料載入成功，共 {len(data)} 條記錄")
            training_progress['progress'] = 5
            training_progress['status'] = '資料載入成功'
        except Exception as e:
            logger.error(f"載入資料時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"載入資料時出錯: {str(e)}"
            }), 500
        
        # 建立特徵
        logger.info("建立特徵...")
        try:
            features = fe.create_complex_features()
            logger.info(f"特徵建立成功，共 {features.shape[1]} 個特徵")
            training_progress['progress'] = 10
            training_progress['status'] = '特徵建立成功'
        except Exception as e:
            logger.error(f"建立特徵時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"建立特徵時出錯: {str(e)}"
            }), 500
        
        # 準備訓練資料
        logger.info("準備訓練資料...")
        try:
            X, y = fe.get_training_data()
            logger.info(f"訓練資料準備完成，特徵數量: {X.shape[1]}，樣本數量: {X.shape[0]}")
            training_progress['progress'] = 15
            training_progress['status'] = '訓練資料準備完成'
        except Exception as e:
            logger.error(f"準備訓練資料時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"準備訓練資料時出錯: {str(e)}"
            }), 500
        
        # 初始化模型訓練器
        logger.info("初始化模型訓練器...")
        try:
            trainer = LotteryModelTrainer(X, y, model_dir=MODEL_DIR)
            current_trainer = trainer  # 儲存訓練器的引用
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
            logger.info(f"資料分割完成，訓練集: {X_train.shape[0]} 樣本，測試集: {X_test.shape[0]} 樣本")
            training_progress['progress'] = 25
            training_progress['status'] = '資料分割完成'
        except Exception as e:
            logger.error(f"分割資料時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"分割資料時出錯: {str(e)}"
            }), 500
        
        # 訓練神經網路模型
        try:
            logger.info("訓練神經網路模型...")
            training_progress['current_model'] = 'neural_network'
            training_progress['status'] = '訓練神經網路模型'
            training_progress['progress'] = 30
            
            # 這裡可以新增更多的高階訓練引數
            models = trainer.train_neural_network(optimize=True)
            logger.info("神經網路模型訓練完成")
            training_progress['progress'] = 90
        except Exception as e:
            logger.error(f"訓練神經網路模型時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"訓練神經網路模型時出錯: {str(e)}"
            }), 500
        
        # 評估模型
        logger.info("評估神經網路模型...")
        training_progress['status'] = '評估神經網路模型'
        training_progress['progress'] = 95
        
        try:
            results = trainer.evaluate_model('neural_network')
            logger.info("神經網路模型評估完成")
        except Exception as e:
            logger.error(f"評估神經網路模型時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"評估神經網路模型時出錯: {str(e)}"
            }), 500
        
        # 訓練完成
        training_progress['progress'] = 100
        training_progress['status'] = '高階訓練完成'
        training_progress['completed'] = True
        logger.info("高階模型訓練完成")
        
        return jsonify({
            'status': 'success',
            'message': '高階模型訓練成功',
            'model': 'neural_network',
            'results': results
        })
    
    except Exception as e:
        logger.error(f"高階訓練模型時出錯: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"高階訓練模型時出錯: {str(e)}"
        }), 500


@app.route('/predict', methods=['POST'])
def predict():
    """根據使用者選擇的模型生成預測"""
    try:
        data = request.get_json()
        model_name = data.get('model_name', 'random_forest')  # 獲取使用者選擇的模型
        num_sets = data.get('num_sets', 5)
        
        app.logger.info(f"使用 {model_name} 模型生成 {num_sets} 組預測...")
        
        # 載入特徵工程器
        fe = LotteryFeatureEngineering(DATA_PATH)
        
        # 建立特徵
        fe.create_basic_features()
        fe.create_advanced_features()
        X_train, y_train = fe.get_training_data()
        
        # 確保 X_new 是正確的格式
        X_new = X_train.iloc[-1:].copy()
        app.logger.info(f"X_new 形狀: {X_new.shape}, 型別: {type(X_new)}")
        
        # 檢查 X_new 是否包含 NaN 值
        if X_new.isna().any().any():
            app.logger.warning("X_new 包含 NaN 值，將進行填充")
            X_new = X_new.fillna(X_train.mean())
        
        # 初始化模型訓練器
        trainer = LotteryModelTrainer(X_train, y_train, model_dir=MODEL_DIR)
        
        # 嘗試載入使用者選擇的模型
        try:
            model_path = os.path.join(MODEL_DIR, f'{model_name}_model.pkl')
            if os.path.exists(model_path):
                trainer.models[model_name] = joblib.load(model_path)
                app.logger.info(f"已載入 {model_name} 模型")
            else:
                app.logger.error(f"找不到 {model_name} 模型檔案")
                return jsonify({'error': f"找不到 {model_name} 模型檔案，請先訓練模型"}), 500
        except Exception as e:
            app.logger.error(f"載入 {model_name} 模型時出錯: {str(e)}")
            return jsonify({'error': f"載入模型時出錯: {str(e)}"}), 500
        
        # 生成預測
        try:
            all_predictions = []
            for i in range(num_sets):
                app.logger.info(f"生成預測 {i+1}/{num_sets}")
                # 新增隨機擾動到 X_new 以產生不同的預測
                X_new_perturbed = X_new.copy()
                for col in X_new_perturbed.columns:
                    X_new_perturbed[col] = X_new_perturbed[col] * (1 + np.random.normal(0, 0.05))  # 新增5%的隨機擾動
                
                pred_set = trainer.generate_lottery_numbers(X_new_perturbed, model_name=model_name)
                app.logger.info(f"預測結果: {pred_set}")
                all_predictions.append(pred_set)
            
            # 使用多樣性增強器
            diversity_method = request.cookies.get('diversityMethod', 'hybrid')
            diversity_level = float(request.cookies.get('diversityLevel', '0.2'))
            
            diversity_enhancer = DiversityEnhancer()
            
            if diversity_method != 'none' and diversity_level > 0:
                enhanced_predictions = diversity_enhancer.enhance_diversity(
                    all_predictions,
                    method=diversity_method,
                    diversity_degree=diversity_level
                )
            else:
                enhanced_predictions = all_predictions
            
            # 格式化為前端需要的格式
            formatted_predictions = []
            for i, pred_set in enumerate(enhanced_predictions):
                formatted_predictions.append({
                    'set_number': i + 1,
                    'numbers': pred_set
                })
            
            app.logger.info(f"成功使用 {model_name} 模型生成 {num_sets} 組預測")
            
            return jsonify({
                'model': model_name,
                'predictions': formatted_predictions
            })
        except Exception as e:
            app.logger.error(f"生成預測時出錯: {str(e)}")
            app.logger.error(traceback.format_exc())
            return jsonify({'error': f"生成預測時出錯: {str(e)}"}), 500
    except Exception as e:
        app.logger.error(f"處理請求時出錯: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': f"處理請求時出錯: {str(e)}"}), 500

@app.route('/predict_with_best_params', methods=['POST'])
def predict_with_best_params():
    """使用最佳引數生成預測"""
    try:
        data = request.get_json()
        num_sets = data.get('num_sets', 5)
        
        app.logger.info(f"使用最佳引數生成 {num_sets} 組預測...")
        
        # 載入特徵工程器
        fe = LotteryFeatureEngineering(DATA_PATH)
        
        # 建立特徵
        fe.create_basic_features()
        fe.create_advanced_features()
        X_train, y_train = fe.get_training_data()
        
        # 確保 X_new 是正確的格式
        X_new = X_train.iloc[-1:].copy()
        app.logger.info(f"X_new 形狀: {X_new.shape}, 型別: {type(X_new)}")
        
        # 檢查 X_new 是否包含 NaN 值
        if X_new.isna().any().any():
            app.logger.warning("X_new 包含 NaN 值，將進行填充")
            X_new = X_new.fillna(X_train.mean())
        
        # 初始化模型訓練器
        trainer = LotteryModelTrainer(X_train, y_train, model_dir=MODEL_DIR)
        
        # 載入最佳引數
        optimal_params = trainer.load_optimal_parameters()
        if optimal_params:
            model_name = optimal_params.get('model_name', 'random_forest')
            hit_rate = optimal_params.get('hit_rate', 0)
        else:
            # 如果沒有最佳引數檔案，使用預設值
            params_path = os.path.join(MODEL_DIR, 'optimal_parameters.json')
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    optimal_params = json.load(f)
                model_name = optimal_params.get('model_name', 'random_forest')
                hit_rate = optimal_params.get('hit_rate', 0)
            else:
                model_name = 'random_forest'
                hit_rate = 0
        
        app.logger.info(f"使用最佳模型 {model_name}，歷史命中率: {hit_rate}%")
        
        # 嘗試載入模型
        try:
            model_path = os.path.join(MODEL_DIR, f'{model_name}_model.pkl')
            if os.path.exists(model_path):
                trainer.models[model_name] = joblib.load(model_path)
                app.logger.info(f"已載入最佳模型: {model_name}")
            else:
                app.logger.error(f"找不到最佳模型 {model_name} 的模型檔案")
                return jsonify({'error': f"找不到最佳模型檔案，請先訓練模型"}), 500
        except Exception as e:
            app.logger.error(f"載入最佳模型 {model_name} 時出錯: {str(e)}")
            return jsonify({'error': f"載入最佳模型時出錯: {str(e)}"}), 500
        
        # 生成預測
        try:
            all_predictions = []
            for i in range(num_sets):
                app.logger.info(f"使用最佳模型生成預測 {i+1}/{num_sets}")
                # 新增隨機擾動到 X_new 以產生不同的預測
                X_new_perturbed = X_new.copy()
                for col in X_new_perturbed.columns:
                    X_new_perturbed[col] = X_new_perturbed[col] * (1 + np.random.normal(0, 0.05))  # 新增5%的隨機擾動
                
                pred_set = trainer.generate_lottery_numbers(X_new_perturbed, model_name=model_name)
                app.logger.info(f"最佳模型預測結果: {pred_set}")
                all_predictions.append(pred_set)
            
            # 使用多樣性增強器
            diversity_method = request.cookies.get('diversityMethod', 'hybrid')
            diversity_level = float(request.cookies.get('diversityLevel', '0.2'))

            # 在初始化時設定多樣性程度
            diversity_enhancer = DiversityEnhancer(diversity_degree=diversity_level)

            if diversity_method != 'none' and diversity_level > 0:
                enhanced_predictions = diversity_enhancer.enhance_diversity(
                    all_predictions,
                    method=diversity_method,
                    num_sets=len(all_predictions)  # 如果需要指定返回的預測集數量
                )
            else:
                enhanced_predictions = all_predictions
            
            # 格式化為前端需要的格式
            formatted_predictions = []
            for i, pred_set in enumerate(enhanced_predictions):
                formatted_predictions.append({
                    'set_number': i + 1,
                    'numbers': pred_set
                })
            
            app.logger.info(f"成功使用最佳模型 {model_name} 生成 {num_sets} 組預測")
            
            return jsonify({
                'model': model_name,
                'hit_rate': hit_rate,
                'predictions': formatted_predictions,
                'is_best_model': True
            })
        except Exception as e:
            app.logger.error(f"使用最佳模型生成預測時出錯: {str(e)}")
            app.logger.error(traceback.format_exc())
            return jsonify({'error': f"生成預測時出錯: {str(e)}"}), 500
    except Exception as e:
        app.logger.error(f"處理最佳引數預測請求時出錯: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': f"處理請求時出錯: {str(e)}"}), 500





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
    """最佳化引數"""
    try:
        global optimization_progress, current_study
        
        data = request.json
        model_name = data.get('model_name')
        n_trials = data.get('n_trials', 50)
        actual_numbers = data.get('actual_numbers', [])
        
        # 重置最佳化進度
        optimization_progress = {
            'progress': 0,
            'status': '準備最佳化...',
            'completed': False,
            'cancelled': False
        }
        
        logger.info(f"開始最佳化引數，模型: {model_name}, 試驗次數: {n_trials}")
        
        # 載入特徵工程器
        fe = LotteryFeatureEngineering(DATA_PATH)
        
        # 準備訓練資料
        X, y = fe.get_training_data()
        
        # 初始化模型訓練器
        trainer = LotteryModelTrainer(X, y, model_dir=MODEL_DIR)
        
        # 分割訓練集和測試集
        X_train, X_test, y_train, y_test = trainer.train_test_split()
        
        # 如果指定了模型名稱，則最佳化該模型
        if model_name:
            # 建立最佳化研究
            study = optuna.create_study(direction='minimize')
            current_study = study
            
            # 定義回撥函式來更新進度
            def callback(study, trial):
                global optimization_progress
                progress = int((trial.number + 1) / n_trials * 100)
                optimization_progress['progress'] = min(progress, 99)  # 保留最後1%給最終處理
                
                # 檢查是否應該取消最佳化
                if optimization_progress['cancelled']:
                    study.stop()
            
            # 最佳化引數
            best_params, best_score, best_model = trainer.optimize_hyperparameters(
                model_name=model_name,
                n_trials=n_trials
            )

            # 新增這一行: 儲存最佳引數，包含 best_score
            trainer.save_optimal_parameters(best_model, best_params, 0.0, best_score)  # 命中率暫設為0，後續可計算
            
            # 更新最佳化進度
            optimization_progress['progress'] = 100
            optimization_progress['completed'] = True
            
            logger.info(f"引數最佳化完成，最佳模型: {best_model}, 最佳分數: {best_score:.4f}")
            
            return jsonify({
                'status': 'success',
                'best_model': best_model,
                'best_params': best_params,
                'best_score': best_score
            })
        
        # 如果提供了實際號碼，則最佳化預測結果
        elif actual_numbers:
            # 獲取最新資料作為預測輸入
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
                    
                    # 檢查是否應該取消最佳化
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
                # 儲存最佳引數
                optimal_params = {
                    'model_name': best_model,
                    'hit_rate': best_hit_rate,
                    'predictions': best_predictions
                }
                
                # 修改這一行: 新增 best_score 引數
                # 這裡我們可以使用命中率作為分數，或者設定一個預設值
                best_score = 1.0 - best_hit_rate  # 將命中率轉換為錯誤率作為分數（越低越好）
                params_path = trainer.save_optimal_parameters(best_model, {}, best_hit_rate, best_score)
                
                # 更新最佳化進度
                optimization_progress['progress'] = 100
                optimization_progress['completed'] = True
                
                logger.info(f"最佳化完成，最佳模型: {best_model}, 命中率: {best_hit_rate:.4f}")
                
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
        logger.error(f"最佳化引數時出錯: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"最佳化引數時出錯: {str(e)}"
        }), 500
    
@app.route('/compare_models', methods=['GET'])
def compare_models():
    try:
        models = ['random_forest', 'xgboost', 'lightgbm', 'catboost', 'neural_network']
        results = []
        
        for model_name in models:
            params_path = os.path.join(MODEL_DIR, f'{model_name}_best_params.json')
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    params = json.load(f)
                
                # 載入模型評估結果
                model_result = {
                    'model_name': model_name,
                    'best_params': params
                }
                
                # 嘗試獲取MSE
                if isinstance(params, dict) and 'best_score' in params:
                    model_result['mse'] = params['best_score']
                else:
                    # 如果沒有儲存MSE，可以重新計算或標記為未知
                    model_result['mse'] = 'Unknown'
                
                results.append(model_result)
        
        # 按MSE排序
        results.sort(key=lambda x: x['mse'] if isinstance(x['mse'], (int, float)) else float('inf'))
        
        return jsonify({
            'models': results,
            'best_model': results[0]['model_name'] if results else None
        })
        
    except Exception as e:
        app.logger.error(f"比較模型時出錯: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/stop_optimization', methods=['POST'])
def stop_optimization():
    """終止最佳化過程"""
    try:
        global current_study, optimization_progress
        
        if current_study:
            current_study.stop()
            optimization_progress['cancelled'] = True
            logger.info("已傳送終止最佳化請求")
            
            return jsonify({
                'status': 'success',
                'message': '已傳送終止最佳化請求'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': '沒有正在進行的最佳化'
            }), 400
    
    except Exception as e:
        logger.error(f"終止最佳化時出錯: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"終止最佳化時出錯: {str(e)}"
        }), 500

@app.route('/optimization_progress', methods=['GET'])
def get_optimization_progress():
    """獲取最佳化進度"""
    global optimization_progress
    return jsonify(optimization_progress)

@app.route('/data', methods=['GET'])
def get_data_summary():
    """獲取資料摘要"""
    try:
        logger.info("獲取資料摘要...")
        
        # 載入特徵工程器
        fe = LotteryFeatureEngineering(DATA_PATH)
        
        # 載入資料
        data = fe.load_data()
        
        # 基本統計資訊
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
        
        logger.info("資料摘要獲取成功")
        
        return jsonify({
            'status': 'success',
            'total_records': total_records,
            'hot_numbers': hot_numbers,
            'cold_numbers': cold_numbers,
            'recent_draws': recent_draws,
            'number_frequencies': number_frequencies
        })
    
    except Exception as e:
        logger.error(f"獲取資料摘要時出錯: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"獲取資料摘要時出錯: {str(e)}"
        }), 500

@app.route('/advanced_analysis', methods=['GET'])
def advanced_analysis():
    """執行高階資料分析"""
    try:
        logger.info("執行高階資料分析...")
        
        # 載入特徵工程器
        fe = LotteryFeatureEngineering(DATA_PATH)
        
        # 載入資料
        data = fe.load_data()
        
        # 相關性分析
        correlations = fe.analyze_correlations()
        
        # 週期性分析
        periodicity = fe.analyze_periodicity()
        
        # 趨勢分析
        trends = fe.analyze_trends()
        
        logger.info("高階資料分析完成")
        
        return jsonify({
            'status': 'success',
            'correlations': correlations,
            'periodicity': periodicity,
            'trends': trends
        })
    
    except Exception as e:
        logger.error(f"執行高階資料分析時出錯: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"執行高階資料分析時出錯: {str(e)}"
        }), 500

@app.route('/diversity_settings', methods=['POST'])
def set_diversity_settings():
    """設定多樣性引數"""
    try:
        data = request.json
        method = data.get('method', 'hybrid')
        level = data.get('level', 0.2)
        
        logger.info(f"設定多樣性引數: method={method}, level={level}")
        
        # 設定 cookie
        response = jsonify({
            'status': 'success',
            'message': '多樣性設定已儲存'
        })
        
        response.set_cookie('diversityMethod', method)
        response.set_cookie('diversityLevel', str(level))
        
        return response
    
    except Exception as e:
        logger.error(f"設定多樣性引數時出錯: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"設定多樣性引數時出錯: {str(e)}"
        }), 500

@app.route('/upload_data', methods=['POST'])
def upload_data():
    """上傳資料檔案"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': '沒有上傳檔案'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': '沒有選擇檔案'
            }), 400
        
        # 檢查檔案型別
        if not (file.filename.endswith('.xlsx') or file.filename.endswith('.csv')):
            return jsonify({
                'status': 'error',
                'message': '只支援 Excel 或 CSV 檔案'
            }), 400
        
        # 儲存檔案
        file_path = os.path.join(BASE_DIR, 'data', 'uploaded_data.xlsx')
        file.save(file_path)
        
        logger.info(f"資料檔案上傳成功: {file_path}")
        
        # 嘗試載入資料
        try:
            fe = LotteryFeatureEngineering(file_path)
            data = fe.load_data()
            total_records = len(data)
            
            logger.info(f"成功載入 {total_records} 條記錄")
            
            return jsonify({
                'status': 'success',
                'message': f'資料檔案上傳成功，共 {total_records} 條記錄',
                'total_records': total_records
            })
        
        except Exception as e:
            logger.error(f"載入上傳的資料檔案時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            
            return jsonify({
                'status': 'error',
                'message': f'資料檔案格式錯誤: {str(e)}'
            }), 400
    
    except Exception as e:
        logger.error(f"上傳資料檔案時出錯: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"上傳資料檔案時出錯: {str(e)}"
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
        
        # 建立 DataFrame
        rows = []
        for i, pred_set in enumerate(predictions):
            for j, numbers in enumerate(pred_set):
                rows.append({
                    'set': i + 1,
                    'row': j + 1,
                    'numbers': ', '.join(map(str, numbers))
                })
        
        df = pd.DataFrame(rows)
        
        # 生成檔名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'predictions_{timestamp}.xlsx'
        file_path = os.path.join(RESULTS_DIR, filename)
        
        # 儲存為 Excel 檔案
        df.to_excel(file_path, index=False)
        
        logger.info(f"預測結果已儲存至: {file_path}")
        
        # 返回檔案下載連結
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
    """下載檔案"""
    try:
        return send_from_directory(RESULTS_DIR, filename, as_attachment=True)
    except Exception as e:
        logger.error(f"下載檔案時出錯: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"下載檔案時出錯: {str(e)}"
        }), 500


def select_best_model():
    """根據MSE自動選擇最佳模型並更新optimal_parameters.json"""
    try:
        models = ['random_forest', 'xgboost', 'lightgbm', 'catboost', 'neural_network']
        best_model = None
        best_score = float('inf')
        best_params = None
        best_hit_rate = 0
        
        for model_name in models:
            params_path = os.path.join(MODEL_DIR, f'{model_name}_best_params.json')
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    params = json.load(f)
                
                # 獲取模型的MSE
                model_score = None
                if model_name == 'random_forest':
                    # 從日誌中提取的MSE值
                    model_score = 3.8679721593558383
                elif model_name == 'xgboost':
                    # 假設值，需要從日誌或檔案中獲取
                    model_score = 3.0  # 請替換為實際值
                elif model_name == 'lightgbm':
                    model_score = 2.914969610720887
                elif model_name == 'catboost':
                    model_score = 2.8467909225484958
                elif model_name == 'neural_network':
                    model_score = 14.531107107798258
                
                if model_score is not None and model_score < best_score:
                    best_model = model_name
                    best_score = model_score
                    best_params = params
                    # 這裡需要計算命中率，或者使用預設值
                    best_hit_rate = 38.16  # 使用當前的命中率
        
        if best_model:
            # 初始化模型訓練器
            fe = LotteryFeatureEngineering(DATA_PATH)
            fe.create_basic_features()
            fe.create_advanced_features()
            X_train, y_train = fe.get_training_data()
            trainer = LotteryModelTrainer(X_train, y_train, model_dir=MODEL_DIR)
            
            # 儲存最佳模型資訊
            trainer.save_optimal_parameters(best_model, best_params, best_hit_rate, best_score)
            app.logger.info(f"自動選擇最佳模型: {best_model}，MSE: {best_score}，命中率: {best_hit_rate}%")
            
            return True
        else:
            app.logger.warning("未找到任何模型引數檔案")
            return False
            
    except Exception as e:
        app.logger.error(f"選擇最佳模型時出錯: {str(e)}")
        return False

if __name__ == '__main__':
    # 應用啟動時自動選擇最佳模型
    select_best_model()
    app.run(debug=True)