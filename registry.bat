@echo off
rem ======================================================================
rem  The demo vehicle registry and the security console.
rem
rem    registry.bat              service on 8507 + console on 8506
rem    registry.bat --service    just the service (lookups + authority)
rem    registry.bat --console    just the console (reads the store directly)
rem    registry.bat --seed       (re)build the synthetic registry, then stop
rem    registry.bat --test       selftest + end-to-end integration tests
rem
rem  SYNTHETIC. Every record is invented by registry\store.py from a hash of
rem  the plate. Not VAHAN, no real vehicle, owner or case, no government
rem  system contacted. See registry\README.md.
rem
rem  Runs on the MAIN interpreter, not .venv-paddle: nothing here touches
rem  paddle or the GPU, so it can run alongside the cameras without taking
rem  a CUDA context off them.
rem
rem  Nothing in anpr\ core\ api\ sim\ dashboard\ imports this package, and it
rem  opens data\godseye.db read-only. Deleting the registry\ folder, this
rem  file and data\registry.db returns the project to exactly what it was.
rem ======================================================================
setlocal
cd /d "%~dp0"

call "%~dp0_env.bat"
if not defined PY ( echo [registry] No Python 3.11+ found. & exit /b 1 )

if /i "%~1"=="--seed" (
  %PY% -m registry.seed --extra 200 --reset
  exit /b %errorlevel%
)

if /i "%~1"=="--test" (
  %PY% -m registry.selftest
  echo.
  echo [registry] The integration tests need the service running on 8507
  echo [registry] and the phones on 8505. Start them first if either is down.
  %PY% -m registry.integrationtest
  exit /b %errorlevel%
)

rem A port already in use means an old copy is still running. Say so in a
rem second rather than letting uvicorn fail thirty seconds in with a stack
rem trace that names a socket instead of the problem.
if /i not "%~1"=="--console" (
  netstat -ano | findstr /r /c:"LISTENING.*:8507 " >nul 2>&1
  if not errorlevel 1 (
    echo [registry] Port 8507 is already in use - the service is running,
    echo [registry] or a previous copy did not shut down. Run stop.bat.
    exit /b 1
  )
)

if /i "%~1"=="--console" (
  %PY% -m streamlit run registry\console.py --server.port 8506 --server.headless true
  exit /b %errorlevel%
)

if /i "%~1"=="--service" (
  %PY% -m registry.service
  exit /b %errorlevel%
)

echo [registry] service  http://127.0.0.1:8507/docs
echo [registry] console  http://localhost:8506
echo [registry] Close this window or run stop.bat to shut both down.
start "GodsEye registry service" /min %PY% -m registry.service
rem The console reads the store directly, so it comes up whether or not the
rem service is ready - but give the service a moment so its status line in
rem the sidebar is green on the first paint rather than after a refresh.
timeout /t 3 /nobreak >nul
%PY% -m streamlit run registry\console.py --server.port 8506 --server.headless true
endlocal
