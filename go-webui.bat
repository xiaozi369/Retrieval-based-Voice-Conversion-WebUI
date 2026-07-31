@echo off
chcp 65001 > nul
cd /d "%~dp0"

set "PYTHON=venv\Scripts\python.exe"
%PYTHON% launcher.py

pause
