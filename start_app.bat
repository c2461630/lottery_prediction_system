@echo off
echo 啟動彩球預測系統...

:: 啟動虛擬環境
call activate tf-env
if %ERRORLEVEL% neq 0 (
    echo 無法啟動 tf-env 虛擬環境，嘗試使用完整路徑...
    call C:\Users\%USERNAME%\Anaconda3\Scripts\activate tf-env
    if %ERRORLEVEL% neq 0 (
        echo 無法啟動虛擬環境，請確保已安裝 Anaconda/Miniconda 並創建了 tf-env 環境。
        pause
        exit /b 1
    )
)

echo 虛擬環境已啟動，正在啟動應用程序...

:: 啟動應用程序
python run.py

:: 如果應用程序異常退出，保持窗口開啟
if %ERRORLEVEL% neq 0 (
    echo 應用程序異常退出，錯誤代碼: %ERRORLEVEL%
    pause
)
