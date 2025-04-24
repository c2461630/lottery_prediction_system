import os
import sys
import argparse
from app import app
from src.utils.helpers import set_seed
import config

def main():
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='彩球預測系統')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='主機地址')
    parser.add_argument('--port', type=int, default=5000, help='端口號')
    parser.add_argument('--debug', action='store_true', help='是否開啟調試模式')
    args = parser.parse_args()
    
    # 設置隨機種子
    set_seed(config.DEFAULT_RANDOM_STATE)
    
    # 確保必要的目錄存在
    for directory in [config.DATA_DIR, config.MODEL_DIR, config.RESULTS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # 檢查數據文件是否存在
    if not os.path.exists(config.DATA_FILE):
        print(f"錯誤: 數據文件 {config.DATA_FILE} 不存在")
        print("請將彩票歷史數據文件放置在 data 目錄下，並命名為 lottery_history.xlsx")
        sys.exit(1)
    
    # 啟動應用
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()