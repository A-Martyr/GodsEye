@echo off
rem ======================================================================
rem  The 91% real-photograph reader (anpr_ocr / PP-OCRv5).
rem
rem  Runs from .venv-paddle - Python 3.12, paddle, and a CPU-only torch -
rem  because paddle and a CUDA torch clash over compiled symbols inside one
rem  process. In the main environment this backend loads, answers once or
rem  twice, then fails; here it does not. See PADDLE_SETUP.md.
rem
rem  Port 8502, not 8501, so this can run beside demo.bat: the CRNN reader
rem  on the GPU in one window, the real-photograph reader in the other.
rem  Upload a real photo here - that is what this reader is for. On the
rem  synthetic captures the panel generates, the CRNN is still stronger.
rem ======================================================================
title GodsEye - anpr_ocr (real photographs)
cd /d "%~dp0"

set "PYP=%~dp0.venv-paddle\Scripts\python.exe"
if not exist "%PYP%" goto noenv
if not exist "data\godseye.db" goto nodb

rem Make anpr_ocr the reader this dashboard starts on. config.OCR_BACKEND
rem reads this; the Recogniser row can still switch away from it.
set "GODSEYE_OCR_BACKEND=anpr_ocr"

echo.
echo   Real-photograph reader on http://localhost:8502
echo   The first read builds the paddle model and takes about half a minute.
echo.
"%PYP%" -m streamlit run dashboard/app.py --server.port 8502
goto :eof

:noenv
echo.
echo   .venv-paddle is not set up in this folder.
echo   PADDLE_SETUP.md has the steps; briefly:
echo       py -3.12 -m venv .venv-paddle
echo       .venv-paddle\Scripts\python.exe -m pip install -r requirements.txt
echo       .venv-paddle\Scripts\python.exe -m pip install paddlepaddle huggingface_hub safetensors shapely pyclipper scikit-image
echo       .venv-paddle\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
echo.
pause
goto :eof

:nodb
echo.
echo   No database yet - run seed.bat first (about a minute).
echo.
pause
