@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title CryptoForecaster Local

if not exist ".venv\Scripts\python.exe" (
  echo FOUT: lokale installatie ontbreekt.
  echo Voer eerst install_windows_local.bat uit.
  pause
  exit /b 1
)

if not exist "local_dashboard.html" (
  echo FOUT: local_dashboard.html ontbreekt. Werk de app eerst bij.
  pause
  exit /b 1
)

if not exist "trading_strategy.py" (
  echo FOUT: trading_strategy.py ontbreekt. Werk de app eerst bij.
  pause
  exit /b 1
)

if not exist "strategy_audit.py" (
  echo FOUT: strategy_audit.py ontbreekt. Werk de app eerst bij.
  pause
  exit /b 1
)

if not exist "portfolio_strategy.py" (
  echo FOUT: portfolio_strategy.py ontbreekt. Werk de app eerst bij.
  pause
  exit /b 1
)

if not exist "forecast_proof.py" (
  echo FOUT: forecast_proof.py ontbreekt. Werk de app eerst bij.
  pause
  exit /b 1
)

findstr /C:"Strategie-audit" "local_dashboard.html" >nul 2>nul
if errorlevel 1 (
  echo FOUT: local_dashboard.html is nog een oude versie zonder Strategie-audit.
  echo Voer update_cryptoforecaster.bat opnieuw uit.
  pause
  exit /b 1
)

if not exist local_data mkdir local_data

rem Houd de dagelijkse modelrun automatisch op 06:30 ingesteld.
if exist "setup_daily_task.bat" (
  call setup_daily_task.bat >nul 2>nul
  if errorlevel 1 (
    echo WAARSCHUWING: dagelijkse modelrun kon niet automatisch worden ingesteld.
  )
)

set "DATA_DIR=%CD%\local_data"
set "AUTO_RUN_MODEL=false"
set "LOCAL_MODE=true"
set "MODEL_NEWS=true"
set "MODEL_DEEP=false"
set "LATEST=%DATA_DIR%\ultimate_run\output\latest_forecasts.csv"

rem Herstel eerst eventueel al bestaande, vooraf gemaakte voorspellingen in de bewijslaag.
if exist "repair_proof_local.py" (
  "%CD%\.venv\Scripts\python.exe" "%CD%\repair_proof_local.py" "%DATA_DIR%" >nul 2>nul
)

rem Extra vangnet: vanaf 06:30 moet er iedere kalenderdag een nieuw lokaal
rem latest_forecasts.csv zijn. Als Windows Taakplanner de run heeft gemist,
rem voeren we hem bij het openen van de app alsnog uit. Dit maakt de bewijslaag
rem niet langer afhankelijk van alleen de Taakplanner.
set "NEED_CATCHUP=0"
for /f %%i in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "$now=Get-Date; $due=$now.Date.AddHours(6).AddMinutes(30); if($now -lt $due){'0'} elseif(-not (Test-Path -LiteralPath $env:LATEST)){'1'} else {$p=Get-Item -LiteralPath $env:LATEST; if($p.LastWriteTime.Date -lt $now.Date){'1'} else {'0'}}"') do set "NEED_CATCHUP=%%i"

if "%NEED_CATCHUP%"=="1" (
  echo.
  echo Dagelijkse modelrun van vandaag ontbreekt. Automatische inhaalrun wordt uitgevoerd...
  call run_daily_local.bat
  if errorlevel 1 (
    echo WAARSCHUWING: automatische inhaalrun is mislukt. Zie local_data\daily_model.log.
  ) else (
    echo Inhaalrun gereed en bewijslaag gecontroleerd.
  )
)

echo.
echo Oude lokale server op poort 8080 wordt indien nodig afgesloten...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8080" ^| findstr "LISTENING"') do (
  taskkill /PID %%a /F >nul 2>nul
)
timeout /t 1 /nobreak >nul

echo Nieuwe lokale server wordt gestart...
start "CryptoForecaster Server" /min cmd /k ""%CD%\.venv\Scripts\python.exe" -m uvicorn cloud_server:app --host 127.0.0.1 --port 8080"

echo Wachten totdat de server gereed is...
for /l %%i in (1,1,30) do (
  curl.exe -s --fail "http://127.0.0.1:8080/api/v1/health" >nul 2>nul
  if not errorlevel 1 goto :ready
  timeout /t 1 /nobreak >nul
)

echo.
echo FOUT: de lokale server antwoordt niet binnen 30 seconden.
echo Open het geminimaliseerde venster "CryptoForecaster Server" voor de foutmelding.
pause
exit /b 1

:ready
echo Server gereed.
echo Dashboard wordt geopend op http://127.0.0.1:8080/
set "CACHEBUST=%RANDOM%%RANDOM%"
start "" "http://127.0.0.1:8080/?v=%CACHEBUST%"
exit /b 0
