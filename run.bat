@echo off
setlocal
title AIC211 Phishing Email Detection

rem Always run from this project's directory, even when double-clicked elsewhere.
cd /d "%~dp0"

rem Prefer a project-managed virtual environment when it exists.
if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else if exist ".venv-paper\Scripts\python.exe" (
    set "PYTHON=.venv-paper\Scripts\python.exe"
) else (
    where py >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON=py -3"
    ) else (
        set "PYTHON=python"
    )
)

%PYTHON% --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo Python 3 was not found. Please install Python 3, then run this file again.
    echo.
    pause
    exit /b 1
)

rem Check every direct runtime dependency used by the dashboard.  Checking only
rem Streamlit allowed an incomplete system Python to pass and fail later while
rem importing the security modules.
%PYTHON% -c "import cryptography, dns, fasttext, joblib, matplotlib, numpy, pandas, requests, seaborn, sklearn, streamlit, tldextract" >nul 2>&1
if errorlevel 1 (
    echo The selected Python environment is missing one or more dashboard dependencies.
    echo Selected interpreter: %PYTHON%
    echo Install the locked production dependencies first:
    echo   %PYTHON% -m pip install --require-hashes -r requirements.lock
    echo.
    pause
    exit /b 1
)

echo.
echo [T] Local Test: scan your .eml files on this computer using the frozen research model.
echo     This mode is loopback-only and never promotes or retrains a model.
echo [P] Production: require a signed, promoted model and every production control.
choice /c TP /n /m "Choose mode"
if errorlevel 2 goto production

:local_test
set "AIC211_ENV=development"
set "AIC211_ALLOW_UNPROMOTED_RESEARCH=1"
set "AIC211_LAUNCH_BROWSER=1"
set "AIC211_AUTH_IDENTITIES_JSON=[{"subject":"local-test-analyst","token_env":"AIC211_LOCAL_TEST_TOKEN","roles":["viewer","analyst"]}]"
for /f "usebackq delims=" %%A in (`powershell -NoProfile -Command "$b = New-Object byte[] 24; [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($b); [Convert]::ToBase64String($b)"`) do set "AIC211_LOCAL_TEST_TOKEN=%%A"
if not defined AIC211_LOCAL_TEST_TOKEN (
    echo Could not create the temporary local-test token.
    pause
    exit /b 1
)
echo.
echo Starting Local Test dashboard on http://127.0.0.1:8501
echo Enter this one-time access token in the sidebar:
echo %AIC211_LOCAL_TEST_TOKEN%
goto dashboard

:production
rem Production is intentionally fail-closed. Secrets must be injected by the
rem Windows user/service account or its secret manager; never store them here.
set "AIC211_ENV=production"
set "AIC211_LAUNCH_BROWSER="

echo Verifying AIC211 production readiness...
%PYTHON% -m src.production_preflight
if errorlevel 1 (
    echo.
    echo Production startup was blocked. Review the failed preflight checks above.
    echo The dashboard was not started and no unpromoted ML model was loaded.
    echo.
    pause
    exit /b 1
)

echo Production preflight passed. Starting AIC211 dashboard on the loopback listener...

:dashboard
if defined AIC211_LAUNCH_BROWSER (
    rem Streamlit's automatic browser launch is unreliable on some Windows
    rem setups.  Wait until its health endpoint is ready, then open the URL.
    start "" /b powershell.exe -NoProfile -WindowStyle Hidden -Command "$u='http://127.0.0.1:8501'; for ($i=0; $i -lt 60; $i++) { try { $r=Invoke-WebRequest -UseBasicParsing -Uri ($u + '/_stcore/health') -TimeoutSec 1; if ($r.StatusCode -eq 200) { Start-Process $u; break } } catch {}; Start-Sleep -Milliseconds 500 }"
)
%PYTHON% -m streamlit run app.py --server.headless true

if errorlevel 1 (
    echo.
    echo The dashboard stopped with an error.
    pause
)
