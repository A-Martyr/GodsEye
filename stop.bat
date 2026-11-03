@echo off
rem Stops every GodsEye process, whichever launcher started it. Goes by
rem listening port rather than by window title: the title is easy to lose track
rem of once someone has typed in one of those windows, the port is not.
title GodsEye - stop
echo.
echo   Stopping the GodsEye demo ...

call :killport 8501 "control room"
call :killport 8000 "API"
call :killport 8502 "real-photograph reader"

rem The two live-camera demos. Not started by demo.bat - livecam_web.bat and
rem phonecam.bat launch them - but stopped here anyway, because both hold the
rem GPU and leaving one running is what makes the next demo mysteriously slow.
rem phonecam also holds the webcam-facing phones' websockets open.
call :killport 8503 "live webcam reader"
call :killport 8504 "phone junction cameras"
rem 8505 is the same process as 8504 - a loopback-only http listener the
rem control room embeds its video from - so killing 8504 normally takes it
rem with it. Named anyway: a half-dead process holding it would block the
rem next start with an error that names a port nobody has heard of.
call :killport 8505 "control-room camera feed"

rem The demo registry. Two separate processes - registry.bat starts the service
rem in its own window - so both ports are named. Neither holds the GPU, but a
rem stale service on 8507 will block the next registry.bat with a port error.
call :killport 8506 "security console"
call :killport 8507 "demo vehicle registry"

rem The tunnel cannot be found by port. A quick tunnel dials OUT to Cloudflare
rem and listens on nothing, so :killport below - which reads netstat for a
rem LISTENING socket - is structurally blind to it. Left running it keeps a
rem public address pointing at this laptop long after the demo has finished,
rem which is the one thing about the tunnel that must not be left to chance.
tasklist /FI "IMAGENAME eq cloudflared.exe" /NH 2>nul | findstr /I cloudflared.exe >nul
if errorlevel 1 (
    echo     no public tunnel was running
) else (
    taskkill /IM cloudflared.exe /T /F >nul 2>&1
    echo     stopped the public tunnel - the shared link is now dead
)
if exist "%~dp0data\tunnel_url.txt" del /q "%~dp0data\tunnel_url.txt" >nul 2>&1

echo.
echo   Done.
echo.
ping -n 4 127.0.0.1 >nul
goto :eof

:killport
set "_hit="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%~1 " ^| findstr LISTENING') do (
    taskkill /PID %%P /T /F >nul 2>&1
    set "_hit=1"
)
if defined _hit echo     stopped the %~2 on port %~1
if not defined _hit echo     nothing was listening on port %~1
goto :eof
