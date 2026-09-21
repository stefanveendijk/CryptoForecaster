@echo off
setlocal
cd /d "%~dp0"
title CryptoForecaster Controle
echo === CryptoForecaster lokale controle ===
echo.
if not exist .venv\Scripts\python.exe (
  echo FOUT: .venv ontbreekt.
  pause
  exit /b 1
)
set OUT=%CD%\local_data\ultimate_run\output\latest_forecasts.csv
if not exist "%OUT%" (
  echo FOUT: latest_forecasts.csv ontbreekt.
  echo Voer eerst run_model_now.bat uit.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -c "import pandas as pd; p=r'%OUT%'; d=pd.read_csv(p); d=d[d['model'].astype(str).str.upper().eq('META')]; got=sorted((str(r.coin).upper(),int(r.horizon)) for _,r in d.iterrows()); exp=[(c,h) for c in ['BTC','ETH'] for h in [1,7,30,90]]; print('META voorspellingen:'); [print('  ',c,h,'dagen') for c,h in got]; miss=[x for x in exp if x not in got]; print(); print('RESULTAAT: COMPLEET' if not miss else 'ONTBREEKT: '+', '.join(f'{c} {h}d' for c,h in miss)); raise SystemExit(0 if not miss else 2)"
set MODELRC=%ERRORLEVEL%
echo.
schtasks /Query /TN "CryptoForecaster Daily" /FO LIST 2>nul | findstr /I /C:"TaskName" /C:"Taaknaam" /C:"Next Run Time" /C:"Volgende uitvoeringstijd" /C:"Status"
if errorlevel 1 echo WAARSCHUWING: dagelijkse Windows-taak niet gevonden.
echo.
if "%MODELRC%"=="0" (
  echo Lokale modeluitvoer bevat alle 8 verwachte META-voorspellingen.
) else (
  echo Lokale modeluitvoer is nog niet compleet.
)
pause
exit /b %MODELRC%
