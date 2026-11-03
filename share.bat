@echo off
rem ======================================================================
rem  Publishes the judge console on a public HTTPS address, so someone who
rem  is not on this network - another city, a phone on mobile data - can
rem  open it and release a cloned plate into the running demo.
rem
rem  How it works: cloudflared dials OUT to Cloudflare and is handed a
rem  throwaway hostname. Because the connection is outbound, nothing has to
rem  be opened on the firewall and nothing forwarded on the router, which
rem  are the two things nobody can arrange at a venue they walked into an
rem  hour ago. It needs no Cloudflare account.
rem
rem  Two things to know. The address is DIFFERENT every run, so the link
rem  has to be re-sent after a restart. And it stays live until this window
rem  is closed or stop.bat is run - close it when you are done.
rem
rem  The API must already be running (demo.bat or api.bat). The tunnel is
rem  perfectly happy to publish an address that points at nothing, so
rem  starting it first is what makes the link actually work.
rem ======================================================================
title GodsEye - public judge link
cd /d "%~dp0"
call "%~dp0_env.bat"
if not defined PY goto nopython

%PY% -c "import socket,sys; sys.exit(0 if socket.socket().connect_ex(('127.0.0.1',8000))==0 else 1)" >nul 2>&1
if errorlevel 1 goto noapi

echo.
echo   Asking Cloudflare for a public address ...
echo.
%PY% -m core.tunnel
if errorlevel 1 goto failed

echo.
echo   ==================================================================
echo    The link above works from ANY network. Send it to whoever is
echo    releasing the incident.
echo.
echo    Leave this window open. Closing it, or running stop.bat, takes
echo    the link down.
echo   ==================================================================
echo.
pause
goto :eof

:noapi
echo.
echo   Nothing is listening on port 8000, so a public link would point at
echo   nothing. Start the demo first with demo.bat (or api.bat), then run
echo   this again.
echo.
pause
goto :eof

:failed
echo.
echo   Could not create a public link. The usual cause is no internet on
echo   this machine - a tunnel has to reach Cloudflare to be given an
echo   address. The same-network links in the dashboard still work.
echo.
pause
goto :eof

:nopython
echo.
echo   No Python 3.11+ found. Install it and tick "Add python.exe to PATH".
echo.
pause
