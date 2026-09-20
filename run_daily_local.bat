@echo off
setlocal
cd /d "%~dp0"
if not exist local_data mkdir local_data
set DATA_DIR=%CD%\local_data
set MODEL_NEWS=true
set MODEL_DEEP=false
call .venv\Scripts\activate.bat
python crypto_forecaster_ultimate.py --workdir "%DATA_DIR%\ultimate_run" >> "%DATA_DIR%\daily_model.log" 2>&1
