@echo off
echo �Ұʱm�y�w���t��...

:: �Ұʵ�������
call activate tf-env
if %ERRORLEVEL% neq 0 (
    echo �L�k�Ұ� tf-env �������ҡA���ըϥΧ�����|...
    call C:\Users\%USERNAME%\Anaconda3\Scripts\activate tf-env
    if %ERRORLEVEL% neq 0 (
        echo �L�k�Ұʵ������ҡA�нT�O�w�w�� Anaconda/Miniconda �óЫؤF tf-env ���ҡC
        pause
        exit /b 1
    )
)

echo �������Ҥw�ҰʡA���b�Ұ����ε{��...

:: �Ұ����ε{��
python run.py

:: �p�G���ε{�ǲ��`�h�X�A�O�����f�}��
if %ERRORLEVEL% neq 0 (
    echo ���ε{�ǲ��`�h�X�A���~�N�X: %ERRORLEVEL%
    pause
)
