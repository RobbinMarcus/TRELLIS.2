@echo off
setlocal

REM Run script for TRELLIS.2
REM Usage: run.bat [ARGS]

if not exist .venv (
    echo Error: .venv not found. Please run setup.bat first.
    exit /b 1
)

call .venv\Scripts\activate.bat

python main.py %*
