@echo off
rem Create a pip-only environment in .venv and download the matrix.
rem   setup_env.bat          CPU methods (scipy + MKL PARDISO)
rem   setup_env.bat gpu      + NVIDIA cuDSS (CUDA 12; use gpu13 for CUDA 13)
cd /d "%~dp0"
python -m venv .venv || exit /b 1
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
set EXTRAS=pardiso
if not "%~1"=="" set EXTRAS=pardiso,%~1
pip install -e ".[%EXTRAS%]" || exit /b 1
python get_data.py || exit /b 1
python -m factorbench list
