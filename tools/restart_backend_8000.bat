@echo off
setlocal

set "PROJECT_ROOT=%~dp0.."
for %%I in ("%PROJECT_ROOT%") do set "PROJECT_ROOT=%%~fI"
set "PYTHON_EXE=D:\Anaconda\python.exe"
set "PORT=8000"
set "VERIFY_FILE_ID=file_85f358c4e8d9"
set "LOG_DIR=%PROJECT_ROOT%\data\backend_logs"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%PORT%" ^| findstr "LISTENING"') do (
  echo Stopping backend process on port %PORT%: PID %%P
  taskkill /PID %%P /F >nul 2>nul
)

timeout /t 2 /nobreak >nul

if not exist "%PYTHON_EXE%" (
  echo Configured Python was not found: %PYTHON_EXE%
  echo Falling back to python on PATH.
  set "PYTHON_EXE=python"
)

set "PYTHONPATH=%PROJECT_ROOT%\src"
echo Starting backend with current source on http://127.0.0.1:%PORT%
start "MLLMProject Backend" /min "%PYTHON_EXE%" -m uvicorn mllmproject.api:app --host 127.0.0.1 --port %PORT% --reload 1>"%LOG_DIR%\uvicorn_8000.out.log" 2>"%LOG_DIR%\uvicorn_8000.err.log"

echo Waiting for backend...
timeout /t 8 /nobreak >nul

echo Files:
powershell -NoProfile -Command "Invoke-RestMethod -Uri 'http://127.0.0.1:%PORT%/api/v1/files' -Method Get | ConvertTo-Json -Depth 5"

echo Chunks for %VERIFY_FILE_ID%:
powershell -NoProfile -Command "$r=Invoke-RestMethod -Uri 'http://127.0.0.1:%PORT%/api/v1/files/%VERIFY_FILE_ID%/chunks?page=1&page_size=100' -Method Get; 'API total chunks: ' + $r.total; $r.chunks | Group-Object source_type | Sort-Object Name | Select-Object Name,Count | Format-Table -AutoSize"

endlocal
