@echo off
rem ======================================================================
rem  One click, whole demo. Checks the environment, builds the database if
rem  it is not there, starts the API and the dashboard in their own
rem  windows, and opens the browser.
rem
rem  The API goes up first and on port 8000 specifically: config.API_BASE
rem  defaults to http://127.0.0.1:8000 and the dashboard's live feed and
rem  incident injectors talk to it there. The dashboard still opens if the
rem  API is down, it just loses those two panels.
rem ======================================================================
title GodsEye - demo launcher
cd /d "%~dp0"
call "%~dp0_env.bat"
if not defined PY goto nopython

%PY% demo_check.py
if errorlevel 1 goto checkfailed

rem --- database ---------------------------------------------------------
if exist "data\godseye.db" goto haveDb
echo.
echo   No database yet - generating about six hours of city traffic.
echo   This takes roughly a minute. Leave it alone until it finishes.
echo.
%PY% seed.py
if errorlevel 1 goto seedfailed
:haveDb

rem --- servers ----------------------------------------------------------
rem Both of these run on the CPU, deliberately. This machine has one 4 GB card
rem and multiple CUDA contexts starve each other catastrophically - measured at
rem roughly 50x in RUNBOOK.md, which is why it says this "is not caution, it is
rem a necessity". Without it the API quietly takes the GPU and the next thing
rem that wants it - dashboard_paddle.bat on 8502, or phonecam.bat - appears to
rem hang while paddle loads its model against the contention.
rem
rem Nothing is lost: the CRNN these two run is 16 ms on the CPU against 5.5 ms
rem on the GPU, a difference nobody perceives, and the card stays free for the
rem reader that genuinely needs it.
set "GODSEYE_DEVICE=cpu"

echo.
echo   Starting the API on port 8000 ...
rem --host 0.0.0.0 is what lets a judge's phone reach the console. uvicorn
rem defaults to 127.0.0.1, which accepts connections from this machine only -
rem everything works while you test on the laptop and nothing works from a
rem second device. Windows may ask to allow it through the firewall the first
rem time; say yes for private networks or judges cannot connect.
start "GodsEye API" /d "%~dp0" cmd /k %PY% -m uvicorn api.main:app --host 0.0.0.0 --port 8000

echo   Starting the control room on port 8501 ...
start "GodsEye dashboard" /d "%~dp0" cmd /k %PY% -m streamlit run dashboard/app.py --server.port 8501

rem Streamlit needs a moment before the port answers; opening the browser
rem too early shows a connection error the operator then has to reload past.
rem Poll the port rather than guessing at a sleep - a cold first run loads
rem torch and the YOLO weights and is much slower than a warm one.
echo   Waiting for the dashboard to come up ...
set /a _tries=0
:waitloop
%PY% -c "import socket,sys; sys.exit(0 if socket.socket().connect_ex(('127.0.0.1',8501))==0 else 1)" >nul 2>&1
if not errorlevel 1 goto dashup
set /a _tries+=1
if %_tries% geq 60 goto dashup
ping -n 2 127.0.0.1 >nul
goto waitloop
:dashup
start "" http://localhost:8501

echo.
echo   ==================================================================
echo    Control room : http://localhost:8501
echo    API docs     : http://localhost:8000/docs
echo.
echo    Two windows are now running the servers. Close them, or run
echo    stop.bat, when the demo is over.
echo   ==================================================================
echo.
pause
goto :eof

:checkfailed
echo.
echo   Not starting anything - fix the FAIL lines above first.
echo.
pause
goto :eof

:seedfailed
echo.
echo   seed.py failed, so there is no database to show. Nothing started.
echo.
pause
goto :eof

:nopython
echo.
echo   No Python 3.11+ found on this machine.
echo   Install it from https://www.python.org/downloads/ and tick
echo   "Add python.exe to PATH", then run this again.
echo.
pause
