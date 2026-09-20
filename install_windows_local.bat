@echo off
setlocal
cd /d "%~dp0"
echo === CryptoForecaster lokale installatie ===
where py >nul 2>nul
if errorlevel 1 (
  echo FOUT: Python launcher niet gevonden.
  echo Installeer Python vanaf python.org en vink "Add Python to PATH" aan.
  pause
  exit /b 1
)
if not exist .venv\Scripts\python.exe (
  echo Lokale Python-omgeving wordt aangemaakt...
  py -m venv .venv
  if errorlevel 1 (
    echo FOUT: lokale Python-omgeving kon niet worden aangemaakt.
    pause
    exit /b 1
  )
)
echo Benodigde pakketten worden lokaal geinstalleerd...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail
if not exist local_data mkdir local_data
schtasks /Create /TN "CryptoForecaster Daily" /TR "\"%CD%\run_daily_local.bat\"" /SC DAILY /ST 05:30 /F
if errorlevel 1 (
  echo WAARSCHUWING: dagelijkse taak kon niet worden aangemaakt.
  echo De app zelf kan wel lokaal draaien.
)
echo.
echo INSTALLATIE GESLAAGD.
echo Lokale Python: %CD%\.venv\Scripts\python.exe
echo Start daarna start_local_app.bat
pause
exit /b 0
:fail
echo.
echo FOUT: installatie is niet voltooid. De dagelijkse taak is niet aangepast.
pause
exit /b 1
