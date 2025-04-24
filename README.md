# 彩球預測系統

這是一個使用機器學習技術預測彩票號碼的系統。

## 功能特點

- 多種機器學習模型：隨機森林、XGBoost、LightGBM、CatBoost、神經網絡和集成模型
- 自動特徵工程：時間特徵、滯後特徵、滾動統計特徵等
- 超參數優化：使用Optuna和Hyperopt進行模型調優
- 預測評估：命中率計算、混淆矩陣、命中分佈等
- 網頁界面：直觀的用戶界面，方便操作和查看結果

## 安裝

1. 克隆此倉庫：
```bash
git clone https://github.com/yourusername/lottery_prediction_system.git
cd lottery_prediction_system
```

2. 安裝依賴：
```bash
pip install -r requirements.txt
```

3. 準備數據：
   - 將彩票歷史數據保存為Excel文件（lottery_history.xlsx）
   - 放置在`data`目錄下

## 使用方法

1. 啟動系統：
```bash
python run.py
```

2. 在瀏覽器中訪問：http://127.0.0.1:5000

3. 使用流程：
   - 訓練模型：點擊"訓練模型"按鈕
   - 生成預測：選擇模型和組數，點擊"生成預測"
   - 評估預測：輸入實際開獎號碼，點擊"評估預測"
   - 優化參數：設置嘗試次數，點擊"尋找最佳參數"
   - 查看數據：點擊"載入數據摘要"

## 數據格式

數據文件（lottery_history.xlsx）應包含以下列：
- date: 開獎日期
- num1, num2, num3, num4, num5: 開獎號碼

## 系統架構

- `app.py`: 主應用程序和API端點
- `config.py`: 配置參數
- `run.py`: 啟動腳本
- `src/`: 源代碼目錄
  - `features/`: 特徵工程模塊
  - `models/`: 模型訓練和預測模塊
  - `evaluation/`: 評估和分析模塊
  - `utils/`: 工具函數
- `templates/`: HTML模板
- `data/`: 數據目錄
- `models/`: 保存訓練好的模型
- `results/`: 保存評估結果和圖表

## 注意事項

- 彩票預測本質上是隨機的，即使使用先進的機器學習技術，也無法保證準確預測結果
- 本系統僅供學習和研究使用，不建議用於實際投注
- 請理性看待預測結果，不要沉迷於彩票遊戲
```

讓我們再創建一個初始化腳本，用於創建必要的目錄結構：

```python:lottery_prediction_system/init_project.py
import os
import sys

def create_directory_structure():
    """創建項目目錄結構"""
    directories = [
        'data',
        'models',
        'results',
        'src',
        'src/features',
        'src/models',
        'src/evaluation',
        'src/utils',
        'templates'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"創建目錄: {directory}")
    
    # 創建空的__init__.py文件
    init_files = [
        'src/__init__.py',
        'src/features/__init__.py',
        'src/models/__init__.py',
        'src/evaluation/__init__.py',
        'src/utils/__init__.py'
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                pass
            print(f"創建文件: {init_file}")
    
    print("目錄結構創建完成！")

def create_sample_data():
    """創建示例數據文件"""
    import pandas as pd
    import numpy as np
    
    # 檢查數據文件是否已存在
    data_file = 'data/lottery_history.xlsx'
    if os.path.exists(data_file):
        print(f"數據文件 {data_file} 已存在，跳過創建示例數據")
        return
    
    # 創建示例數據
    np.random.seed(42)
    
    # 生成日期序列
    start_date = pd.Timestamp('2010-01-01')
    end_date = pd.Timestamp('2023-12-31')
    
    # 每週一次開獎
    date_range = pd.date_range(start=start_date, end=end_date, freq='W-MON')
    
    # 創建數據框
    data = pd.DataFrame({'date': date_range})
    
    # 生成隨機號碼
    for i in range(1, 6):
        data[f'num{i}'] = [sorted(np.random.choice(range(1, 50), 5, replace=False))[i-1] for _ in range(len(date_range))]
    
    # 保存到Excel
    data.to_excel(data_file, index=False)
    print(f"創建示例數據文件: {data_file}")

if __name__ == '__main__':
    print("初始化彩球預測系統...")
    create_directory_structure()
    
    # 詢問是否創建示例數據
    create_sample = input("是否創建示例數據？(y/n): ").lower() == 'y'
    if create_sample:
        try:
            create_sample_data()
        except Exception as e:
            print(f"創建示例數據時出錯: {e}")
            print("請手動準備數據文件: data/lottery_history.xlsx")
    
    print("\n初始化完成！")
    print("請按照以下步驟使用系統：")
    print("1. 確保 data/lottery_history.xlsx 文件存在")
    print("2. 運行 python run.py 啟動系統")
    print("3. 在瀏覽器中訪問 http://127.0.0.1:5000")
```

最後，創建一個 `.gitignore` 文件：

```text:lottery_prediction_system/.gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Project specific
data/
models/
results/
*.log

# Jupyter Notebook
.ipynb_checkpoints

# OS specific
.DS_Store
Thumbs.db