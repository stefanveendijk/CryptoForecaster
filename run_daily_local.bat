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

rem Voorkom slaapstand alleen zolang de modelrun actief is.
start "" /b powercfg /requests >nul 2>nul
powershell -NoProfile -ExecutionPolicy Bypass -Command "$code='[DllImport(\"kernel32.dll\")]public static extern uint SetThreadExecutionState(uint esFlags);'; Add-Type -MemberDefinition $code -Name Power -Namespace Win32; [Win32.Power]::SetThreadExecutionState(0x80000001) | Out-Null; & '.\.venv\Scripts\python.exe' 'crypto_forecaster_ultimate.py' '--workdir' \"$env:DATA_DIR\ultimate_run\"; $rc=$LASTEXITCODE; [Win32.Power]::SetThreadExecutionState(0x80000000) | Out-Null; exit $rc" >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%

if "%RC%"=="0" (
  echo SUCCES %DATE% %TIME%>> "%LOG%"
) else (
  echo FOUT exitcode=%RC% %DATE% %TIME%>> "%LOG%"
)
exit /b %RC%
