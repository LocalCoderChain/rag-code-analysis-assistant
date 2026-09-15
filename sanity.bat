@echo off
:: ============================================================
::  DevOne — Sanity Check
::  Verifies your environment is ready BEFORE launching the app.
::  Run this any time something feels broken.
:: ============================================================

title DevOne — Sanity Check

echo.
echo  ============================================================
echo   DevOne  ^|  Environment Sanity Check
echo  ============================================================
echo.

set FAIL=0

:: ── Check 1: Python ────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [FAIL]  Python not found.
    echo          Install Python 3.9+ from https://python.org
    set FAIL=1
) else (
    echo  [OK]    Python found.
)

:: ── Check 2: Virtual environment ───────────────────────────────────────────
if not exist "myvenv\Scripts\activate.bat" (
    echo  [WARN]  Virtual environment not found yet.
    echo          Run run.bat first to create it.
    set FAIL=1
) else (
    echo  [OK]    Virtual environment found.
    call myvenv\Scripts\activate.bat
)

:: ── Check 3: .env file present ─────────────────────────────────────────────
if not exist ".env" (
    echo  [FAIL]  .env file not found.
    echo          Create one with GROQ_API_KEY and GITHUB_PAT.
    set FAIL=1
) else (
    echo  [OK]    .env file found.
)

:: ── Check 4: Required env vars are set (not just present in .env) ─────────
python -c "import os; from dotenv import load_dotenv; load_dotenv(); import sys; missing=[k for k in ('GROQ_API_KEY','GITHUB_PAT') if not os.getenv(k)]; sys.exit(1) if missing else sys.exit(0)" 2>nul
if errorlevel 1 (
    echo  [FAIL]  GROQ_API_KEY or GITHUB_PAT missing/empty in .env.
    set FAIL=1
) else (
    echo  [OK]    GROQ_API_KEY and GITHUB_PAT are set.
)

:: ── Check 5: Core imports don't blow up ────────────────────────────────────
python -c "import streamlit, langchain, langgraph, langchain_groq, sqlite_vec" 2>nul
if errorlevel 1 (
    echo  [FAIL]  One or more core imports failed.
    echo          Run: pip install -r requirements.txt
    set FAIL=1
) else (
    echo  [OK]    Core imports ^(streamlit, langchain, langgraph, sqlite_vec^) OK.
)

:: ── Check 6: Ollama installed ──────────────────────────────────────────────
ollama --version >nul 2>&1
if errorlevel 1 (
    echo  [FAIL]  Ollama not found.
    echo          Install from: https://ollama.com/download
    set FAIL=1
) else (
    echo  [OK]    Ollama found.
)

:: ── Check 7: Required Ollama models are pulled ─────────────────────────────
if not errorlevel 1 (
    ollama list | findstr /C:"llama3.2" >nul 2>&1
    if errorlevel 1 (
        echo  [WARN]  Model 'llama3.2' not found locally.
        echo          Run: ollama pull llama3.2
        set FAIL=1
    ) else (
        echo  [OK]    Model 'llama3.2' is pulled.
    )

    ollama list | findstr /C:"mxbai-embed-large" >nul 2>&1
    if errorlevel 1 (
        echo  [WARN]  Model 'mxbai-embed-large' not found locally.
        echo          Run: ollama pull mxbai-embed-large
        set FAIL=1
    ) else (
        echo  [OK]    Model 'mxbai-embed-large' is pulled.
    )
)

:: ── Check 8: Groq API key is actually valid (live check) ──────────────────
if not errorlevel 1 (
    python -c "import os; from dotenv import load_dotenv; load_dotenv(); from groq import Groq; Groq(api_key=os.getenv('GROQ_API_KEY')).models.list()" 2>nul
    if errorlevel 1 (
        echo  [FAIL]  GROQ_API_KEY did not validate against the Groq API.
        set FAIL=1
    ) else (
        echo  [OK]    GROQ_API_KEY is valid.
    )
)

echo.
echo  ============================================================
if "%FAIL%"=="1" (
    echo   RESULT: One or more checks failed. Fix the items above
    echo           before running run.bat.
    echo  ============================================================
    echo.
    pause
    exit /b 1
) else (
    echo   RESULT: All checks passed. You're good to run run.bat.
    echo  ============================================================
    echo.
    pause
    exit /b 0
)
