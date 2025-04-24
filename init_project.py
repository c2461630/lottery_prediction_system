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