@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe exit /b 1
if not exist local_data mkdir local_data
set DATA_DIR=%CD%\local_data
set MODEL_NEWS=true
set MODEL_DEEP=false
set LOG=%DATA_DIR%\daily_model.log

echo.>> "%LOG%"
echo ============================================================>> "%LOG%"
echo START %DATE% %TIME%>> "%LOG%"

rem Voorkom slaapstand zolang de modelrun actief is.
".venv\Scripts\python.exe" keep_awake_run.py --workdir "%DATA_DIR%\ultimate_run" >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%

if "%RC%"=="0" (
  echo SUCCES %DATE% %TIME%>> "%LOG%"
) else (
  echo FOUT exitcode=%RC% %DATE% %TIME%>> "%LOG%"
)
exit /b %RC%
