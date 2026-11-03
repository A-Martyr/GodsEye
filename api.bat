@echo off
rem The API on its own, on the port the dashboard expects (config.API_BASE).
title GodsEye - API
cd /d "%~dp0"
call "%~dp0_env.bat"
if not defined PY goto nopython

echo.
echo   API docs: http://localhost:8000/docs
echo.
rem Bound to every interface, not just loopback, so the judge console is
rem reachable from a phone on the same network.
%PY% -m uvicorn api.main:app --host 0.0.0.0 --port 8000
goto :eof

:nopython
echo.
echo   No Python 3.11+ found. Install it and tick "Add python.exe to PATH".
echo.
pause
