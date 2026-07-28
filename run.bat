@echo off
:: ============================================================
::  DevOne — Windows Launcher
::  Double-click this file to start the app.
:: ============================================================

title DevOne — AI Code Assistant

echo.
echo  ============================================================
echo   DevOne  ^|  Local AI Code Analysis ^& Documentation
echo  ============================================================
echo.

:: ── Step 1: Check Python ──────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found.
    echo          Install Python 3.9+ from https://python.org
    echo          Tick "Add Python to PATH" during install.
    pause
    exit /b 1
)
echo  [OK]    Python found.

:: ── Step 2: Create virtual environment if needed ─────────────────────────────
if not exist "myvenv\Scripts\activate.bat" (
    echo  [SETUP] Creating virtual environment...
    python -m venv myvenv
    echo  [OK]    Virtual environment created.
) else (
    echo  [OK]    Virtual environment found.
)

:: ── Step 3: Activate ─────────────────────────────────────────────────────────
call myvenv\Scripts\activate.bat
echo  [OK]    Virtual environment activated.

:: ── Step 4: Install / upgrade dependencies ───────────────────────────────────
echo  [INFO]  Installing dependencies (fast after first run)...
pip install -r requirements.txt -q --upgrade
echo  [OK]    Dependencies ready.

:: ── Step 5: Check Ollama ──────────────────────────────────────────────────────
echo.
ollama --version >nul 2>&1
if errorlevel 1 (
    echo  [WARNING] Ollama not found.
    echo.
    echo            Install from: https://ollama.com/download
    echo            Then open a NEW terminal and run:
    echo              ollama pull llama3.2
    echo              ollama pull mxbai-embed-large
    echo.
) else (
    echo  [OK]    Ollama found.
    echo  [TIP]   If not done yet, open another terminal and run:
    echo            ollama pull llama3.2
    echo            ollama pull mxbai-embed-large
)

:: ── Step 6: Create data directory ────────────────────────────────────────────
if not exist "data"    mkdir data

:: ── Step 7: Launch ────────────────────────────────────────────────────────────
echo.
echo  ============================================================
echo   Launching DevOne in your browser...
echo   URL: http://localhost:8501
echo.
echo   Keep this window open while using the app.
echo   Press CTRL+C to stop.
echo  ============================================================
echo.

streamlit run app.py --server.headless false --browser.gatherUsageStats false

echo.
echo  [INFO] App stopped.
pause
