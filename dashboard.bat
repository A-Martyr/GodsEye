@echo off
rem The control room on its own. Reads the database directly, so it works
rem with the API stopped - the live feed and the injectors are what go
rem missing without it.
title GodsEye - dashboard
cd /d "%~dp0"
call "%~dp0_env.bat"
if not defined PY goto nopython

if not exist "data\godseye.db" goto nodb
%PY% -m streamlit run dashboard/app.py --server.port 8501
goto :eof

:nodb
echo.
echo   No database yet - run seed.bat first (about a minute).
echo.
pause
goto :eof

:nopython
echo.
echo   No Python 3.11+ found. Install it and tick "Add python.exe to PATH".
echo.
pause
