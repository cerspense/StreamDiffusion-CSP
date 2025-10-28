@echo off
echo ================================================================
echo StreamDiffusion Frame Capture Debug Mode
echo ================================================================
echo.
echo This script will:
echo   - Skip the first 30 warmup frames
echo   - Capture 4 sequential frames at all pipeline stages
echo   - Save frames to: debug_frames/capture_TIMESTAMP/
echo   - Auto-shutdown after capture completes
echo.
echo Press Ctrl+C to abort, or any key to continue...
pause >nul

cd /d %~dp0
if exist venv (
    call venv\Scripts\activate.bat
    echo Starting with --debug-capture-frames flag...
    venv\Scripts\python.exe streamdiffusionTD\td_main.py --debug-capture-frames
) else (
    call .venv\Scripts\activate.bat
    echo Starting with --debug-capture-frames flag...
    .venv\Scripts\python.exe streamdiffusionTD\td_main.py --debug-capture-frames
)
echo.
echo ================================================================
echo Frame capture complete! Check debug_frames/ folder for results.
echo ================================================================
pause
