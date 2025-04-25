from src.models.model_trainer import LotteryModelTrainer
import pandas as pd
import numpy as np

# 創建一些測試數據
X = pd.DataFrame(np.random.random((100, 10)))
y = pd.DataFrame(np.random.random((100, 5)))

# 初始化模型訓練器
trainer = LotteryModelTrainer(X, y)

# 分割數據
trainer.train_test_split()

# 訓練神經網絡模型
model = trainer.train_neural_network(optimize=False)

print("模型訓練成功!")