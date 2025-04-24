import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import joblib
import traceback
import logging
from datetime import datetime
import json
import random
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

# 模擬數據
def generate_sample_data(n_samples=200):
    data = []
    for i in range(n_samples):
        # 生成隨機號碼 (1-49)
        numbers = sorted(random.sample(range(1, 50), 6))
        # 添加日期 (過去n天)
        date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        date = date.replace(day=date.day - i)
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'numbers': numbers
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
            logger.info("模型訓練器初始化成功")
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
        
        # 訓練隨機森林模型
        try:
            logger.info("訓練隨機森林模型...")
            models['random_forest'] = trainer.train_random_forest(optimize=True)
            logger.info("隨機森林模型訓練完成")
        except Exception as e:
            logger.error(f"訓練隨機森林模型時出錯: {str(e)}")
            logger.error(traceback.format_exc())
        
        # 訓練XGBoost模型
        try:
            logger.info("訓練XGBoost模型...")
            models['xgboost'] = trainer.train_xgboost(optimize=True)
            logger.info("XGBoost模型訓練完成")
        except Exception as e:
            logger.error(f"訓練XGBoost模型時出錯: {str(e)}")
            logger.error(traceback.format_exc())
        
        # 訓練LightGBM模型
        try:
            logger.info("訓練LightGBM模型...")
            models['lightgbm'] = trainer.train_lightgbm(optimize=True)
            logger.info("LightGBM模型訓練完成")
        except Exception as e:
            logger.error(f"訓練LightGBM模型時出錯: {str(e)}")
            logger.error(traceback.format_exc())
        
        # 訓練CatBoost模型
        try:
            logger.info("訓練CatBoost模型...")
            models['catboost'] = trainer.train_catboost(optimize=True)
            logger.info("CatBoost模型訓練完成")
        except Exception as e:
            logger.error(f"訓練CatBoost模型時出錯: {str(e)}")
            logger.error(traceback.format_exc())
        
        # 訓練神經網絡模型
        try:
            logger.info("訓練神經網絡模型...")
            models['neural_network'] = trainer.train_neural_network(optimize=True)
            logger.info("神經網絡模型訓練完成")
        except Exception as e:
            logger.error(f"訓練神經網絡模型時出錯: {str(e)}")
            logger.error(traceback.format_exc())
        
        # 訓練集成模型
        if len(models) >= 2:  # 至少需要2個模型才能集成
            try:
                logger.info("訓練集成模型...")
                models['ensemble'] = trainer.train_ensemble()
                logger.info("集成模型訓練完成")
            except Exception as e:
                logger.error(f"訓練集成模型時出錯: {str(e)}")
                logger.error(traceback.format_exc())
        else:
            logger.warning("可用模型不足，無法訓練集成模型")
        
        # 評估模型
        logger.info("評估模型...")
        for model_name in models.keys():
            try:
                results[model_name] = trainer.evaluate_model(model_name)
                logger.info(f"{model_name} 模型評估完成")
            except Exception as e:
                logger.error(f"評估 {model_name} 模型時出錯: {str(e)}")
                logger.error(traceback.format_exc())
                results[model_name] = {"error": str(e)}
        
        # 交叉驗證
        logger.info("進行交叉驗證...")
        for model_name in ['random_forest', 'xgboost', 'lightgbm', 'catboost']:
            if model_name in models:
                try:
                    cv_results[model_name] = trainer.cross_validate(model_name)
                    logger.info(f"{model_name} 模型交叉驗證完成")
                except Exception as e:
                    logger.error(f"{model_name} 模型交叉驗證時出錯: {str(e)}")
                    logger.error(traceback.format_exc())
                    cv_results[model_name] = {"error": str(e)}
        
        logger.info("模型訓練和評估完成")
        return jsonify({
            'status': 'success',
            'message': 'Models trained successfully',
            'models_trained': list(models.keys()),
            'evaluation': results,
            'cross_validation': cv_results
        })
    
    except Exception as e:
        logger.error(f"訓練過程中發生未處理的錯誤: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """預測彩票號碼"""
    try:
        logger.info("開始預測...")
        
        # 獲取請求數據
        data = request.json
        model_name = data.get('model_name', 'ensemble')
        num_sets = data.get('num_sets', 5)
        
        logger.info(f"使用 {model_name} 模型生成 {num_sets} 組預測")
        
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
            lottery_data = fe.load_data()
            features = fe.create_complex_features()
            logger.info("數據處理完成")
        except Exception as e:
            logger.error(f"處理數據時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"處理數據時出錯: {str(e)}"
            }), 500
        
        # 準備訓練數據
        logger.info("準備訓練數據...")
        try:
            X, y = fe.get_training_data()
            logger.info("訓練數據準備完成")
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
            logger.info("模型訓練器初始化成功")
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
            logger.info("數據分割完成")
        except Exception as e:
            logger.error(f"分割數據時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"分割數據時出錯: {str(e)}"
            }), 500
        
        # 載入模型
        logger.info(f"載入 {model_name} 模型...")
        if model_name == 'ensemble':
            model_path = os.path.join(MODEL_DIR, 'ensemble_model.pkl')
            if os.path.exists(model_path):
                try:
                    trainer.ensemble_model = joblib.load(model_path)
                    logger.info("集成模型載入成功")
                except Exception as e:
                    logger.error(f"載入集成模型時出錯: {str(e)}")
                    logger.error(traceback.format_exc())
                    return jsonify({
                        'status': 'error',
                        'message': f"載入集成模型時出錯: {str(e)}"
                    }), 500
            else:
                logger.error(f"集成模型文件不存在: {model_path}")
                return jsonify({
                    'status': 'error',
                    'message': 'Ensemble model not found. Please train the model first.'
                }), 404
        else:
            model_path = os.path.join(MODEL_DIR, f'{model_name}_model.pkl')
            if os.path.exists(model_path):
                try:
                    trainer.models[model_name] = joblib.load(model_path)
                    logger.info(f"{model_name} 模型載入成功")
                except Exception as e:
                    logger.error(f"載入 {model_name} 模型時出錯: {str(e)}")
                    logger.error(traceback.format_exc())
                    return jsonify({
                        'status': 'error',
                        'message': f"載入 {model_name} 模型時出錯: {str(e)}"
                    }), 500
            else:
                logger.error(f"{model_name} 模型文件不存在: {model_path}")
                return jsonify({
                    'status': 'error',
                    'message': f'{model_name} model not found. Please train the model first.'
                }), 404
        
        # 使用最新的數據進行預測
        logger.info("準備最新數據進行預測...")
        try:
            latest_data = X.iloc[-1:].copy()
            logger.info("最新數據準備完成")
        except Exception as e:
            logger.error(f"準備最新數據時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"準備最新數據時出錯: {str(e)}"
            }), 500
        
        # 生成預測
        logger.info("生成預測...")
        try:
            predictions = trainer.generate_lottery_numbers(latest_data, model_name, num_sets)
            logger.info(f"成功生成 {len(predictions)} 組預測")
            
            # 使用多樣性增強器增強預測結果
            logger.info("增強預測結果的多樣性...")
            enhanced_predictions = diversity_enhancer.enhance_diversity(predictions, num_sets)
            logger.info("預測結果多樣性增強完成")
            
        except Exception as e:
            logger.error(f"生成預測時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"生成預測時出錯: {str(e)}"
            }), 500
        
        # 格式化預測結果
        formatted_predictions = []
        for pred_set in enhanced_predictions:
            formatted_set = []
            for numbers in pred_set:
                formatted_set.append(numbers)
            formatted_predictions.append(formatted_set)
        
        logger.info("預測完成")
        return jsonify({
            'status': 'success',
            'predictions': formatted_predictions,
            'model_used': model_name
        })
    
    except Exception as e:
        logger.error(f"預測過程中發生未處理的錯誤: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/predict_with_best_params', methods=['POST'])
def predict_with_best_params():
    """使用最佳參數進行預測"""
    try:
        logger.info("開始使用最佳參數進行預測...")
        
        # 獲取請求數據
        data = request.json
        num_sets = data.get('num_sets', 5)
        
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
            lottery_data = fe.load_data()
            features = fe.create_complex_features()
            logger.info("數據處理完成")
        except Exception as e:
            logger.error(f"處理數據時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"處理數據時出錯: {str(e)}"
            }), 500
        
        # 準備訓練數據
        logger.info("準備訓練數據...")
        try:
            X, y = fe.get_training_data()
            logger.info("訓練數據準備完成")
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
            logger.info("模型訓練器初始化成功")
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
            logger.info("數據分割完成")
        except Exception as e:
            logger.error(f"分割數據時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"分割數據時出錯: {str(e)}"
            }), 500
        
        # 載入最佳參數
        logger.info("載入最佳參數...")
        try:
            best_params = trainer.load_optimal_parameters()
            
            if best_params is None:
                logger.warning("未找到最佳參數，將使用默認模型進行預測")
                model_name = 'ensemble'  # 默認使用ensemble模型
            else:
                logger.info(f"使用最佳模型 {best_params['model_name']} 進行預測，歷史命中率: {best_params['hit_rate']:.2f}%")
                model_name = best_params['model_name']
        except Exception as e:
            logger.error(f"載入最佳參數時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"載入最佳參數時出錯: {str(e)}"
            }), 500
        
        # 載入模型
        logger.info(f"載入 {model_name} 模型...")
        if model_name == 'ensemble':
            model_path = os.path.join(MODEL_DIR, 'ensemble_model.pkl')
            if os.path.exists(model_path):
                try:
                    trainer.ensemble_model = joblib.load(model_path)
                    logger.info("集成模型載入成功")
                except Exception as e:
                    logger.error(f"載入集成模型時出錯: {str(e)}")
                    logger.error(traceback.format_exc())
                    return jsonify({
                        'status': 'error',
                        'message': f"載入集成模型時出錯: {str(e)}"
                    }), 500
            else:
                logger.error(f"集成模型文件不存在: {model_path}")
                return jsonify({
                    'status': 'error',
                    'message': 'Ensemble model not found. Please train the model first.'
                }), 404
        else:
            model_path = os.path.join(MODEL_DIR, f'{model_name}_model.pkl')
            if os.path.exists(model_path):
                try:
                    trainer.models[model_name] = joblib.load(model_path)
                    logger.info(f"{model_name} 模型載入成功")
                except Exception as e:
                    logger.error(f"載入 {model_name} 模型時出錯: {str(e)}")
                    logger.error(traceback.format_exc())
                    return jsonify({
                        'status': 'error',
                        'message': f"載入 {model_name} 模型時出錯: {str(e)}"
                    }), 500
            else:
                logger.error(f"{model_name} 模型文件不存在: {model_path}")
                return jsonify({
                    'status': 'error',
                    'message': f'{model_name} model not found. Please train the model first.'
                }), 404
        
        # 使用最新的數據進行預測
        logger.info("準備最新數據進行預測...")
        try:
            latest_data = X.iloc[-1:].copy()
            logger.info("最新數據準備完成")
        except Exception as e:
            logger.error(f"準備最新數據時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"準備最新數據時出錯: {str(e)}"
            }), 500
        
        # 生成預測
        logger.info("生成預測...")
        try:
            predictions = trainer.generate_lottery_numbers(latest_data, model_name, num_sets)
            logger.info(f"成功生成 {len(predictions)} 組預測")
            
            # 使用多樣性增強器增強預測結果
            logger.info("增強預測結果的多樣性...")
            enhanced_predictions = diversity_enhancer.enhance_diversity(predictions, num_sets)
            logger.info("預測結果多樣性增強完成")
            
        except Exception as e:
            logger.error(f"生成預測時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"生成預測時出錯: {str(e)}"
            }), 500
        
        # 格式化預測結果
        formatted_predictions = []
        for i, pred_set in enumerate(enhanced_predictions):
            formatted_set = []
            for numbers in pred_set:
                formatted_set.append(numbers)
            formatted_predictions.append({
                'set_number': i + 1,
                'numbers': formatted_set
            })
        
        result = {
            'status': 'success',
            'model': model_name,
            'hit_rate': best_params['hit_rate'] if best_params else 'N/A',
            'predictions': formatted_predictions
        }
        
        logger.info("預測完成")
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"預測過程中發生未處理的錯誤: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/evaluate', methods=['POST'])
def evaluate():
    """評估預測結果"""
    try:
        logger.info("開始評估預測結果...")
        
        # 獲取請求數據
        data = request.json
        predictions = data.get('predictions', [])
        actual_numbers = data.get('actual_numbers', [])
        
        if not predictions or not actual_numbers:
            logger.error("缺少預測結果或實際號碼")
            return jsonify({
                'status': 'error',
                'message': "缺少預測結果或實際號碼"
            }), 400
        
        logger.info(f"評估 {len(predictions)} 組預測結果")
        
        # 初始化評估器
        try:
            evaluator = LotteryEvaluator(output_dir=RESULTS_DIR)
            logger.info("評估器初始化成功")
        except Exception as e:
            logger.error(f"初始化評估器時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"初始化評估器時出錯: {str(e)}"
            }), 500
        
        # 計算命中率
        try:
            hit_results = evaluator.calculate_hit_rate(predictions, actual_numbers)
            logger.info("命中率計算完成")
        except Exception as e:
            logger.error(f"計算命中率時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"計算命中率時出錯: {str(e)}"
            }), 500

        # 獲取歷史數據
        try:
            fe = LotteryFeatureEngineering(DATA_PATH)
            historical_data = fe.load_data()
            logger.info(f"歷史數據載入成功，共 {len(historical_data)} 條記錄")
        except Exception as e:
            logger.error(f"載入歷史數據時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"載入歷史數據時出錯: {str(e)}"
            }), 500

        # 生成評估報告
        try:
            report = evaluator.generate_evaluation_report(predictions, actual_numbers, historical_data)
            logger.info("評估報告生成完成")
        except Exception as e:
            logger.error(f"生成評估報告時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"生成評估報告時出錯: {str(e)}"
            }), 500
        
        # 繪製圖表
        try:
            evaluator.plot_hit_distribution(predictions, actual_numbers)
            # 移除對不存在方法的調用
            # evaluator.plot_confusion_matrix(predictions, actual_numbers)
            logger.info("圖表繪製完成")
        except Exception as e:
            logger.error(f"繪製圖表時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            # 繼續執行，不返回錯誤
        
        logger.info("評估完成")
        return jsonify({
            'status': 'success',
            'hit_results': hit_results,
            'evaluation_report': report
        })
    
    except Exception as e:
        logger.error(f"評估過程中發生未處理的錯誤: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/optimize', methods=['POST'])
def optimize_parameters():
    """尋找最佳參數"""
    try:
        logger.info("開始尋找最佳參數...")
        # 獲取請求數據
        data = request.json
        n_trials = data.get('n_trials', 10)
        model_name = data.get('model_name', 'ensemble')  # 添加默認模型名稱
        
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
            lottery_data = fe.load_data()
            features = fe.create_complex_features()
            logger.info("數據處理完成")
        except Exception as e:
            logger.error(f"處理數據時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"處理數據時出錯: {str(e)}"
            }), 500
        
        # 準備訓練數據
        logger.info("準備訓練數據...")
        try:
            X, y = fe.get_training_data()
            logger.info("訓練數據準備完成")
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
            logger.info("模型訓練器初始化成功")
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
            logger.info("數據分割完成")
        except Exception as e:
            logger.error(f"分割數據時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"分割數據時出錯: {str(e)}"
            }), 500
        
        # 使用Optuna進行超參數優化
        logger.info(f"開始使用Optuna進行超參數優化，模型: {model_name}，試驗次數: {n_trials}...")
        try:
            result = trainer.optimize_hyperparameters(model_name, n_trials)
            if isinstance(result, tuple) and len(result) == 3:
                best_params, best_score, best_model_name = result
            else:
                best_params, best_score, best_model_name = result, None, model_name
            logger.info(f"超參數優化完成，最佳模型: {best_model_name}, 最佳參數: {best_params}")
        except Exception as e:
            logger.error(f"超參數優化時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"超參數優化時出錯: {str(e)}"
            }), 500
        
        # 使用最佳參數訓練模型
        logger.info(f"使用最佳參數訓練 {best_model_name} 模型...")
        try:
            trainer.train_model_with_params(best_model_name, best_params)
            logger.info(f"{best_model_name} 模型訓練完成")
        except Exception as e:
            logger.error(f"使用最佳參數訓練模型時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"使用最佳參數訓練模型時出錯: {str(e)}"
            }), 500
        
        # 評估最佳模型
        logger.info("評估最佳模型...")
        try:
            evaluation_results = trainer.evaluate_model(best_model_name, X_test, y_test)
            logger.info("模型評估完成")
        except Exception as e:
            logger.error(f"評估模型時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"評估模型時出錯: {str(e)}"
            }), 500
        
        # 保存最佳參數
        logger.info("保存最佳參數...")
        try:
            trainer.save_optimal_parameters(best_model_name, best_params, best_score)
            logger.info("最佳參數保存完成")
        except Exception as e:
            logger.error(f"保存最佳參數時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"保存最佳參數時出錯: {str(e)}"
            }), 500
        
        logger.info("參數優化完成")
        return jsonify({
            'status': 'success',
            'best_model': best_model_name,
            'best_score': best_score,
            'best_params': best_params,
            'evaluation_results': evaluation_results
        })
    
    except Exception as e:
        logger.error(f"參數優化過程中發生未處理的錯誤: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/update_data', methods=['POST'])
def update_data():
    """更新數據集"""
    try:
        logger.info("開始更新數據集...")
        
        # 獲取請求數據
        data = request.json
        new_data = data.get('new_data', [])
        
        if not new_data:
            logger.error("缺少新數據")
            return jsonify({
                'status': 'error',
                'message': "缺少新數據"
            }), 400
        
        logger.info(f"收到 {len(new_data)} 條新數據")
        
        # 檢查數據文件是否存在
        if not os.path.exists(DATA_PATH):
            logger.warning(f"數據文件不存在，將創建新文件: {DATA_PATH}")
            # 創建目錄
            os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
            
            # 創建新的數據文件
            try:
                df = pd.DataFrame(new_data)
                df.to_csv(DATA_PATH, index=False)
                logger.info(f"創建新數據文件成功: {DATA_PATH}")
            except Exception as e:
                logger.error(f"創建新數據文件時出錯: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({
                    'status': 'error',
                    'message': f"創建新數據文件時出錯: {str(e)}"
                }), 500
        else:
            # 更新現有數據文件
            try:
                # 讀取現有數據
                existing_data = pd.read_csv(DATA_PATH)
                logger.info(f"讀取現有數據成功，共 {len(existing_data)} 條記錄")
                
                # 將新數據轉換為DataFrame
                new_df = pd.DataFrame(new_data)
                
                # 合併數據
                updated_data = pd.concat([existing_data, new_df], ignore_index=True)
                
                # 去除重複項
                updated_data = updated_data.drop_duplicates()
                
                # 保存更新後的數據
                updated_data.to_csv(DATA_PATH, index=False)
                logger.info(f"數據更新成功，現有 {len(updated_data)} 條記錄")
            except Exception as e:
                logger.error(f"更新數據文件時出錯: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({
                    'status': 'error',
                    'message': f"更新數據文件時出錯: {str(e)}"
                }), 500
        
        logger.info("數據集更新完成")
        return jsonify({
            'status': 'success',
            'message': "數據集更新成功"
        })
    
    except Exception as e:
        logger.error(f"更新數據集過程中發生未處理的錯誤: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/data', methods=['GET'])
def get_data():
    """獲取數據摘要"""
    try:
        logger.info("開始獲取數據摘要...")
        
        # 檢查數據文件是否存在
        if not os.path.exists(DATA_PATH):
            logger.error(f"數據文件不存在: {DATA_PATH}")
            return jsonify({
                'status': 'error',
                'message': f"數據文件不存在: {DATA_PATH}"
            }), 400
        
        # 載入數據
        try:
            fe = LotteryFeatureEngineering(DATA_PATH)
            data = fe.load_data()
            logger.info(f"數據載入成功，共 {len(data)} 條記錄")
        except Exception as e:
            logger.error(f"載入數據時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"載入數據時出錯: {str(e)}"
            }), 500
        
        # 獲取數據摘要
        summary = {}
        for col in data.columns:
            if col.startswith('num'):  # 只處理號碼列
                col_data = data[col].dropna()
                summary[col] = {
                    'min': int(col_data.min()),
                    'max': int(col_data.max()),
                    'mean': col_data.mean(),
                    'median': int(col_data.median()),
                    'std': col_data.std(),
                    'most_frequent': int(col_data.value_counts().index[0])
                }
        
        # 獲取最近的記錄
        recent_records = data.head(10).to_dict('records')
        
        # 獲取最近的開獎號碼
        recent_draws = []
        for _, draw in data.head(10).iterrows():
            # 只處理號碼列，跳過日期列
            numbers = [int(draw[col]) for col in data.columns if col.startswith('num') and pd.notna(draw[col])]
            if numbers:
                recent_draws.append(numbers)
        
        # 獲取熱門和冷門號碼
        all_numbers = []
        for col in data.columns:
            if col.startswith('num'):
                all_numbers.extend(data[col].dropna().astype(int).tolist())
        
        number_counts = pd.Series(all_numbers).value_counts()
        hot_numbers = number_counts.head(10).index.tolist()
        cold_numbers = number_counts.tail(10).index.tolist()
        
        # 獲取號碼頻率
        number_frequencies = []
        for num in range(1, 50):  # 假設彩票號碼範圍是1-49
            count = (number_counts.get(num, 0))
            percentage = (count / len(all_numbers)) * 100
            number_frequencies.append({
                'number': num,
                'count': int(count),
                'percentage': percentage
            })
        
        return jsonify({
            'status': 'success',
            'total_records': len(data),
            'summary': summary,
            'recent_records': recent_records,
            'recent_draws': recent_draws,
            'hot_numbers': hot_numbers,
            'cold_numbers': cold_numbers,
            'number_frequencies': number_frequencies
        })
    
    except Exception as e:
        logger.error(f"獲取數據摘要過程中發生未處理的錯誤: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/advanced_analysis', methods=['GET'])
def advanced_analysis():
    """進行高級分析"""
    try:
        logger.info("開始進行高級分析...")
        
        # 檢查數據文件是否存在
        if not os.path.exists(DATA_PATH):
            logger.error(f"數據文件不存在: {DATA_PATH}")
            return jsonify({
                'status': 'error',
                'message': f"數據文件不存在: {DATA_PATH}"
            }), 400
        
        # 載入數據
        try:
            fe = LotteryFeatureEngineering(DATA_PATH)
            data = fe.load_data()
            logger.info(f"數據載入成功，共 {len(data)} 條記錄")
        except Exception as e:
            logger.error(f"載入數據時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"載入數據時出錯: {str(e)}"
            }), 500
        
        # 分析奇偶比例
        odd_even_analysis = {}
        # 分析高低比例
        high_low_analysis = {}
        # 分析和值分佈
        sum_distribution = {}
        # 分析間距分佈
        gap_distribution = {}
        # 分析連號情況
        consecutive_analysis = {}
        
        # 進行奇偶分析
        try:
            all_draws = []
            for i in range(len(data)):
                row = data.iloc[i]
                # 假設號碼在前6列
                numbers = [int(n) for n in row[:6] if pd.notna(n)]
                all_draws.append(numbers)
            
            # 計算每組號碼的奇偶比例
            odd_counts = []
            for draw in all_draws:
                odd_count = sum(1 for num in draw if num % 2 == 1)
                odd_counts.append(odd_count)
            
            # 統計各種奇偶比例的頻率
            for i in range(7):  # 0到6個奇數
                count = odd_counts.count(i)
                odd_even_analysis[f"{i}奇{6-i}偶"] = {
                    'count': count,
                    'percentage': (count / len(all_draws)) * 100
                }
            
            # 計算每組號碼的高低比例 (假設1-24為低，25-49為高)
            high_counts = []
            for draw in all_draws:
                high_count = sum(1 for num in draw if num >= 25)
                high_counts.append(high_count)
            
            # 統計各種高低比例的頻率
            for i in range(7):  # 0到6個高號
                count = high_counts.count(i)
                high_low_analysis[f"{i}高{6-i}低"] = {
                    'count': count,
                    'percentage': (count / len(all_draws)) * 100
                }
            
            # 計算每組號碼的和值
            sums = [sum(draw) for draw in all_draws]
            
            # 將和值分組
            sum_ranges = [(60, 149), (150, 159), (160, 169), (170, 179), 
                          (180, 189), (190, 199), (200, 209), (210, 219), 
                          (220, 229), (230, 239), (240, 300)]
            
            for low, high in sum_ranges:
                count = sum(1 for s in sums if low <= s <= high)
                sum_distribution[f"{low}-{high}"] = {
                    'count': count,
                    'percentage': (count / len(all_draws)) * 100
                }
            
            # 計算每組號碼的間距
            gaps = []
            for draw in all_draws:
                sorted_draw = sorted(draw)
                draw_gaps = [sorted_draw[i+1] - sorted_draw[i] for i in range(len(sorted_draw)-1)]
                gaps.extend(draw_gaps)
            
            # 統計間距分佈
            for gap in range(1, 11):
                count = gaps.count(gap)
                gap_distribution[str(gap)] = {
                    'count': count,
                    'percentage': (count / len(gaps)) * 100
                }
            
            # 分析連號情況
            consecutive_counts = []
            for draw in all_draws:
                sorted_draw = sorted(draw)
                consecutive_count = 0
                for i in range(len(sorted_draw)-1):
                    if sorted_draw[i+1] - sorted_draw[i] == 1:
                        consecutive_count += 1
                consecutive_counts.append(consecutive_count)
            
            # 統計各種連號數量的頻率
            for i in range(6):  # 0到5個連號
                count = consecutive_counts.count(i)
                consecutive_analysis[str(i)] = {
                    'count': count,
                    'percentage': (count / len(all_draws)) * 100
                }
            
        except Exception as e:
            logger.error(f"進行高級分析時出錯: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': f"進行高級分析時出錯: {str(e)}"
            }), 500
        
        logger.info("高級分析完成")
        return jsonify({
            'status': 'success',
            'odd_even_analysis': odd_even_analysis,
            'high_low_analysis': high_low_analysis,
            'sum_distribution': sum_distribution,
            'gap_distribution': gap_distribution,
            'consecutive_analysis': consecutive_analysis
        })
    
    except Exception as e:
        logger.error(f"高級分析過程中發生未處理的錯誤: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/diversity_score', methods=['POST'])
def calculate_diversity_score():
    """計算多樣性分數"""
    try:
        # 獲取請求數據
        data = request.json
        predictions = data.get('predictions', [])
        
        if not predictions:
            return jsonify({
                'status': 'error',
                'message': "缺少預測結果"
            }), 400
        
        # 計算組內多樣性
        intra_set_diversity = diversity_enhancer.calculate_intra_set_diversity(predictions)
        
        # 計算組間多樣性
        inter_set_diversity = diversity_enhancer.calculate_inter_set_diversity(predictions)
        
        # 計算綜合分數
        composite_score = (intra_set_diversity + inter_set_diversity) / 2
        
        return jsonify({
            'status': 'success',
            'diversity_score': {
                'intra_set_diversity': intra_set_diversity,
                'inter_set_diversity': inter_set_diversity
            },
            'composite_score': composite_score
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        data = request.json
        trials = data.get('trials', 100)
        actual_numbers = data.get('actual_numbers', [])
        
        if not actual_numbers:
            return jsonify({
                'status': 'error',
                'message': '請提供實際開獎號碼進行優化'
            }), 400
        
        # 模擬優化過程
        # 在實際應用中，這裡應該有真正的參數優化邏輯
        
        # 模擬進度更新
        progress_updates = []
        for i in range(1, 6):
            progress = i * 20
            progress_updates.append({
                'progress': progress,
                'message': f'優化進度: {progress}%',
                'current_best_score': 0.5 + (i * 0.1)
            })
        
        # 模擬最佳參數
        best_params = {
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 5
            },
            'xgboost': {
                'n_estimators': 150,
                'learning_rate': 0.1,
                'max_depth': 8
            },
            'lightgbm': {
                'n_estimators': 180,
                'learning_rate': 0.08,
                'num_leaves': 31
            },
            'catboost': {
                'iterations': 200,
                'learning_rate': 0.05,
                'depth': 6
            },
            'neural_network': {
                'hidden_layers': 2,
                'neurons_per_layer': 64,
                'dropout_rate': 0.2
            },
            'ensemble': {
                'weights': {
                    'random_forest': 0.25,
                    'xgboost': 0.3,
                    'lightgbm': 0.2,
                    'catboost': 0.15,
                    'neural_network': 0.1
                }
            }
        }
        
        # 使用最佳參數生成預測
        predictions = []
        for _ in range(5):
            # 生成一組預測號碼 (1-49)
            numbers = sorted(random.sample(range(1, 50), 6))
            predictions.append(numbers)
        
        return jsonify({
            'status': 'success',
            'progress_updates': progress_updates,
            'best_params': best_params,
            'optimized_predictions': predictions
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)