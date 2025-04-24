import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import optuna
import joblib
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp

print("所有導入都成功了！")
print(f"TensorFlow 版本: {tf.__version__}")